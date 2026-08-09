# SPDX-License-Identifier: Apache-2.0
# TQ-native disk serialization by Jinho Jang (eric@jangq.ai) for vMLX.
# Stores TurboQuantKVCache compressed data (EncodedKeys/EncodedValues) directly
# to safetensors — 26x smaller than float16 state. github.com/jjang-ai/vmlx
"""
TQ-native serialization for disk cache.

Stores TurboQuantKVCache compressed data (packed indices, norms, metadata)
directly to safetensors without decompressing to float16 first.

Compression ratio: ~26x vs float16 (40KB vs 1MB per 100 tokens x 8 heads x 128 dim)

Format:
- Safetensors tensors:
  - tq_{i}_ck_indices_packed (uint32) — codebook indices
  - tq_{i}_ck_qjl_packed (uint32) — QJL sign bits
  - tq_{i}_ck_residual_norms (float16) — per-vector residual norms
  - tq_{i}_ck_vector_norms (float16) — per-vector key norms
  - tq_{i}_cv_indices_packed (uint32) — value codebook indices
  - tq_{i}_cv_vector_norms (float16) — per-vector value norms
  - layer_{i}_keys / layer_{i}_values — non-TQ layers (KVCache, standard)
  - layer_{i}_state_{j} — cumulative layers (MambaCache/ArraysCache)
- Safetensors metadata (string key-value):
  - __tq_native__ = "true" — format marker
  - __num_layers__ — total layer count
  - __layer_{i}_class__ — class name per layer
  - __tq_{i}_ck_shape__ / __tq_{i}_cv_shape__ — original shapes (JSON)
  - __tq_{i}_ck_bits__ / __tq_{i}_cv_bits__ — index bit widths
  - __tq_{i}_offset__ — token offset
  - __tq_{i}_key_dim__ / __tq_{i}_value_dim__ — TQ dimensions
  - __tq_{i}_key_bits__ / __tq_{i}_value_bits__ — TQ compression bits
  - __tq_{i}_sink_tokens__ — number of sink tokens
  - __tq_{i}_seed__ — codebook seed used by both encoders
"""

from __future__ import annotations

import json
import logging
import threading
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

_TQ_CLASS_NAME = "TurboQuantKVCache"
_TQ_RESTORE_DTYPES = {
    "bfloat16": mx.bfloat16 if HAS_MLX else None,
    "float16": mx.float16 if HAS_MLX else None,
    "float32": mx.float32 if HAS_MLX else None,
}
_TQ_BLOCK_CODEC_TELEMETRY_LOCK = threading.Lock()
_TQ_BLOCK_CODEC_TELEMETRY: Dict[str, Any] = {
    "sequence": 0,
    "encode_calls": 0,
    "encoded_blocks": 0,
    "encoded_tokens": 0,
    "last_encode": None,
    "decode_calls": 0,
    "decoded_blocks": 0,
    "decoded_tokens": 0,
    "last_decode": None,
}


def _record_tq_block_codec_event(
    operation: str,
    *,
    blocks: int,
    tokens: int,
    metadata: Dict[str, Any],
) -> None:
    """Record one packed block-codec call after its output shape is validated.

    This telemetry belongs to the direct ``tq_disk_store`` codec used by paged
    and L2 cache blocks. It deliberately does not reuse
    ``TurboQuantKVCache._vmlx_last_compress``: that class-level signal describes
    live cache-object compression and is not touched by this storage path.
    """
    if operation not in {"encode", "decode"}:
        raise ValueError(f"unsupported TQ block telemetry operation: {operation}")
    block_count = max(0, int(blocks))
    token_count = max(0, int(tokens))
    with _TQ_BLOCK_CODEC_TELEMETRY_LOCK:
        _TQ_BLOCK_CODEC_TELEMETRY["sequence"] += 1
        sequence = int(_TQ_BLOCK_CODEC_TELEMETRY["sequence"])
        if operation == "encode":
            _TQ_BLOCK_CODEC_TELEMETRY["encode_calls"] += 1
            _TQ_BLOCK_CODEC_TELEMETRY["encoded_blocks"] += block_count
            _TQ_BLOCK_CODEC_TELEMETRY["encoded_tokens"] += token_count
            _TQ_BLOCK_CODEC_TELEMETRY["last_encode"] = {
                "sequence": sequence,
                "blocks": block_count,
                "tokens": token_count,
                **metadata,
            }
        else:
            _TQ_BLOCK_CODEC_TELEMETRY["decode_calls"] += 1
            _TQ_BLOCK_CODEC_TELEMETRY["decoded_blocks"] += block_count
            _TQ_BLOCK_CODEC_TELEMETRY["decoded_tokens"] += token_count
            _TQ_BLOCK_CODEC_TELEMETRY["last_decode"] = {
                "sequence": sequence,
                "blocks": block_count,
                "tokens": token_count,
                **metadata,
            }


def tq_block_codec_telemetry() -> Dict[str, Any]:
    """Return a bounded process snapshot of validated block codec calls."""
    with _TQ_BLOCK_CODEC_TELEMETRY_LOCK:
        last_encode = _TQ_BLOCK_CODEC_TELEMETRY["last_encode"]
        last_decode = _TQ_BLOCK_CODEC_TELEMETRY["last_decode"]
        return {
            "schema": "vmlx-tq-block-codec-v1",
            "encode": {
                "calls": int(_TQ_BLOCK_CODEC_TELEMETRY["encode_calls"]),
                "blocks": int(_TQ_BLOCK_CODEC_TELEMETRY["encoded_blocks"]),
                "tokens": int(_TQ_BLOCK_CODEC_TELEMETRY["encoded_tokens"]),
                "last_event": dict(last_encode) if last_encode is not None else None,
            },
            "decode": {
                "calls": int(_TQ_BLOCK_CODEC_TELEMETRY["decode_calls"]),
                "blocks": int(_TQ_BLOCK_CODEC_TELEMETRY["decoded_blocks"]),
                "tokens": int(_TQ_BLOCK_CODEC_TELEMETRY["decoded_tokens"]),
                "last_event": dict(last_decode) if last_decode is not None else None,
            },
        }


def _canonical_tq_dtype(dtype: Any) -> str:
    """Return the stable wire name for a supported decoded KV dtype."""
    name = str(dtype).rsplit(".", 1)[-1]
    if name not in _TQ_RESTORE_DTYPES:
        raise ValueError(f"unsupported TQ decoded KV dtype: {dtype}")
    return name


def _restore_tq_dtype(value: Any, dtype_name: Any, label: str) -> Any:
    """Restore the attention dtype lost by TurboQuant's float32 decoder."""
    name = str(dtype_name or "")
    target = _TQ_RESTORE_DTYPES.get(name)
    if target is None:
        raise ValueError(f"invalid or missing {label} TQ dtype: {name!r}")
    return value if value.dtype == target else value.astype(target)


@lru_cache(maxsize=256)
def _tq_decoder_pair(
    key_dim: int,
    value_dim: int,
    key_bits: int,
    value_bits: int,
    seed: int,
) -> Tuple[Any, Any]:
    """Return immutable TurboQuant decoder state for one codec configuration.

    Paged-prefix reconstruction may decode thousands of block/layer entries
    with the same dimensions, bit widths, and seed. Rebuilding the identical
    rotation, codebook, and QJL state for each entry is wasteful. Keep a small
    process-local cache of the encoder pair; encode/decode operations remain
    independent because the encoder objects are read-only after initialization.
    """
    from jang_tools.turboquant.pipeline import TurboQuantEncoder

    # Disk/paged-cache serialization only needs the immutable codec state.
    # Constructing a live TurboQuantKVCache here also allocates cache buffers,
    # while ``compress()`` performs an immediate decode that storage discards.
    return (
        TurboQuantEncoder(
            dim=key_dim,
            key_bits=key_bits,
            value_bits=value_bits,
            seed=seed,
        ),
        TurboQuantEncoder(
            dim=value_dim,
            key_bits=key_bits,
            value_bits=value_bits,
            seed=seed + 500,
        ),
    )


def warm_tq_decoder_states(
    cache_layers: List[Any],
    *,
    probe_tokens: int = 64,
    probe_heads: int = 1,
) -> Dict[str, Any]:
    """Materialize the exact per-layer TQ storage decoder state at model start.

    Native TQ cache policies seed each attention layer independently. Large
    models commonly expose 80-120 unique seeds, so a 32-entry decoder LRU
    rebuilt rotation/QJL state on every paged or L2 restore. Worse, the first
    reconstruction's ``mx.eval`` paid that lazy state-initialization cost and
    made a millisecond block decode look like a multi-second disk-cache hit.

    The live cache objects already own the authoritative dimensions, bit widths,
    and seeds. Populate the bounded storage-decoder LRU from those exact values
    and evaluate only its immutable codec tensors. No prompt, KV payload, or
    model output is synthesized, and explicit non-TQ runs never call this path.
    """
    if not HAS_MLX:
        return {
            "configs": 0,
            "arrays": 0,
            "bytes": 0,
            "probe_tokens": 0,
            "probe_heads": 0,
            "codec_probes": 0,
        }

    configs: List[Tuple[int, int, int, int, int]] = []
    seen: set[Tuple[int, int, int, int, int]] = set()

    def visit(layer: Any) -> None:
        if type(layer).__name__ == _TQ_CLASS_NAME:
            try:
                config = (
                    int(layer.key_dim),
                    int(layer.value_dim),
                    int(layer.key_bits),
                    int(layer.value_bits),
                    int(layer._seed),
                )
            except (AttributeError, TypeError, ValueError):
                return
            if config not in seen:
                seen.add(config)
                configs.append(config)
            return
        children = getattr(layer, "caches", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                visit(child)

    for layer in cache_layers or []:
        visit(layer)

    arrays: List[Any] = []
    total_bytes = 0
    for config in configs:
        for encoder in _tq_decoder_pair(*config):
            for value in vars(encoder).values():
                if isinstance(value, mx.array):
                    arrays.append(value)
                    total_bytes += int(value.nbytes)
    if arrays:
        mx.eval(*arrays)

    # Exercise the real packed storage path at the paged block shape while the
    # model is starting. Encoder-state materialization alone does not compile
    # or allocate TurboQuant's packed encode/decode kernels; without this, the
    # first L2 hit still pays that cost inside reconstruct_cache(). Keep the
    # probe independent of prompts/model outputs and use the exact bundle-owned
    # dimensions/bits/seeds discovered above.
    token_count = max(1, int(probe_tokens or 1))
    head_count = max(1, int(probe_heads or 1))
    codec_probes = 0
    probe_arrays: List[Any] = []
    for key_dim, value_dim, key_bits, value_bits, seed in configs:
        keys = mx.zeros((1, head_count, token_count, key_dim), dtype=mx.float16)
        values = mx.zeros((1, head_count, token_count, value_dim), dtype=mx.float16)
        entry = encode_tq_block(
            keys,
            values,
            {
                "key_bits": key_bits,
                "value_bits": value_bits,
                "seed": seed,
            },
            _record_telemetry=False,
        )
        decoded_keys, decoded_values = decode_tq_block(
            entry,
            _record_telemetry=False,
        )
        probe_arrays.extend((decoded_keys, decoded_values))
        codec_probes += 1
    if probe_arrays:
        # Submit all layer-specific codec graphs together. The layer seeds are
        # still distinct, but one worker-stream synchronization avoids paying
        # launch/synchronization overhead once per layer during model start.
        mx.eval(*probe_arrays)
    return {
        "configs": len(configs),
        "arrays": len(arrays),
        "bytes": total_bytes,
        "probe_tokens": token_count if configs else 0,
        "probe_heads": head_count if configs else 0,
        "codec_probes": codec_probes,
    }


def encode_tq_block(
    keys: Any,
    values: Any,
    config: Dict[str, Any],
    *,
    _record_telemetry: bool = True,
) -> Tuple[str, Any, Any, Dict[str, Any]]:
    """Encode one positional paged-cache block as native TurboQuant data."""
    if not HAS_MLX:
        raise RuntimeError("MLX required for TQ block encoding")
    if not hasattr(keys, "shape") or not hasattr(values, "shape"):
        raise ValueError("TQ block keys/values must be tensors")
    if len(keys.shape) != 4 or len(values.shape) != 4:
        raise ValueError(
            f"TQ block requires rank-4 KV tensors, got {keys.shape}/{values.shape}"
        )
    token_count = int(keys.shape[-2])
    if token_count <= 0 or int(values.shape[-2]) != token_count:
        raise ValueError("TQ block key/value token lengths must match and be nonzero")

    from jang_tools.turboquant.pipeline import encode_keys, encode_values

    key_bits = int(config.get("key_bits", 8) or 8)
    value_bits = int(config.get("value_bits", 8) or 8)
    seed = int(config.get("seed", 42) or 42)
    if key_bits not in (2, 3, 4, 8) or value_bits not in (2, 3, 4, 8):
        raise ValueError(
            f"unsupported TQ block codec bits key={key_bits} value={value_bits}"
        )
    key_encoder, value_encoder = _tq_decoder_pair(
        int(keys.shape[-1]),
        int(values.shape[-1]),
        key_bits,
        value_bits,
        seed,
    )
    ck = encode_keys(mx.contiguous(keys), key_encoder)
    cv = encode_values(mx.contiguous(values), value_encoder)
    if (
        ck is None
        or cv is None
        or int(ck.shape[-2]) != token_count
        or int(cv.shape[-2]) != token_count
    ):
        raise ValueError("TQ block encoder did not produce a complete packed payload")
    result = (
        "turboquant_kv",
        ck,
        cv,
        {
            "key_dim": int(keys.shape[-1]),
            "value_dim": int(values.shape[-1]),
            "key_bits": key_bits,
            "value_bits": value_bits,
            "key_dtype": _canonical_tq_dtype(keys.dtype),
            "value_dtype": _canonical_tq_dtype(values.dtype),
            "seed": seed,
            "offset": token_count,
        },
    )
    if _record_telemetry:
        _record_tq_block_codec_event(
            "encode",
            blocks=1,
            tokens=token_count,
            metadata={
                "boundary": "encode_tq_block",
                "key_bits_values": [key_bits],
                "value_bits_values": [value_bits],
                "key_dim_values": [int(keys.shape[-1])],
                "value_dim_values": [int(values.shape[-1])],
                "key_dtype_values": [_canonical_tq_dtype(keys.dtype)],
                "value_dtype_values": [_canonical_tq_dtype(values.dtype)],
                "key_shape": [int(dim) for dim in keys.shape],
                "value_shape": [int(dim) for dim in values.shape],
            },
        )
    return result


def decode_tq_block(
    entry: Tuple[Any, ...],
    *,
    _record_telemetry: bool = True,
) -> Tuple[Any, Any]:
    """Decode one native TurboQuant paged-cache block to attention KV tensors."""
    if not HAS_MLX:
        raise RuntimeError("MLX required for TQ block decoding")
    if not isinstance(entry, (tuple, list)) or len(entry) != 4:
        raise ValueError("malformed TQ block entry")
    tag, encoded_keys, encoded_values, config = entry
    if tag != "turboquant_kv" or not isinstance(config, dict):
        raise ValueError("malformed TQ block tag/config")
    from jang_tools.turboquant.pipeline import decode_keys, decode_values

    key_encoder, value_encoder = _tq_decoder_pair(
        int(config["key_dim"]),
        int(config["value_dim"]),
        int(config["key_bits"]),
        int(config["value_bits"]),
        int(config["seed"]),
    )
    keys = _restore_tq_dtype(
        decode_keys(encoded_keys, key_encoder),
        config.get("key_dtype"),
        "key",
    )
    values = _restore_tq_dtype(
        decode_values(encoded_values, value_encoder),
        config.get("value_dtype"),
        "value",
    )
    expected = int(config["offset"])
    if int(keys.shape[-2]) != expected or int(values.shape[-2]) != expected:
        raise ValueError(
            f"decoded TQ block length mismatch: expected={expected}, "
            f"keys={keys.shape[-2]}, values={values.shape[-2]}"
        )
    if _record_telemetry:
        # _stack_tq_block_entries adds one outer page dimension so a single
        # packed decode invocation can truthfully account for multiple blocks.
        decoded_blocks = int(keys.shape[0]) if len(keys.shape) == 5 else 1
        _record_tq_block_codec_event(
            "decode",
            blocks=decoded_blocks,
            tokens=expected * decoded_blocks,
            metadata={
                "boundary": "decode_tq_block",
                "key_bits_values": [int(config["key_bits"])],
                "value_bits_values": [int(config["value_bits"])],
                "key_dim_values": [int(config["key_dim"])],
                "value_dim_values": [int(config["value_dim"])],
                "key_dtype_values": [str(config["key_dtype"])],
                "value_dtype_values": [str(config["value_dtype"])],
                "key_shape": [int(dim) for dim in keys.shape],
                "value_shape": [int(dim) for dim in values.shape],
            },
        )
    return keys, values


def _tq_block_batch_signature(
    entry: Tuple[Any, ...],
    *,
    include_seed: bool = True,
) -> Optional[Tuple[Any, ...]]:
    """Return a grouping signature for one independently packed TQ page.

    ``include_seed=False`` yields the payload-compatibility signature used by
    the cross-layer batch decode: layers share dimensions, bit widths, dtypes,
    and packed shapes but carry per-layer seeds, so the seed is tracked
    separately and applied through stacked per-layer codec constants.
    """
    if not isinstance(entry, (tuple, list)) or len(entry) != 4:
        return None
    tag, encoded_keys, encoded_values, config = entry
    if tag != "turboquant_kv" or not isinstance(config, dict):
        return None
    try:
        key_shape = tuple(int(dim) for dim in encoded_keys.shape)
        value_shape = tuple(int(dim) for dim in encoded_values.shape)
        if len(key_shape) != 4 or len(value_shape) != 4:
            return None
        if key_shape[-2] != value_shape[-2]:
            return None
        return (
            key_shape,
            value_shape,
            int(config["key_dim"]),
            int(config["value_dim"]),
            int(config["key_bits"]),
            int(config["value_bits"]),
            str(config["key_dtype"]),
            str(config["value_dtype"]),
            int(config["seed"]) if include_seed else None,
            int(encoded_keys.index_bits),
            int(encoded_values.index_bits),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _tq_entry_seed(entry: Tuple[Any, ...]) -> Optional[int]:
    """Return the codec seed of one TQ page entry, or None if malformed."""
    if not isinstance(entry, (tuple, list)) or len(entry) != 4:
        return None
    config = entry[3]
    if not isinstance(config, dict):
        return None
    try:
        return int(config["seed"])
    except (KeyError, TypeError, ValueError):
        return None


def _stack_tq_block_entries(
    entries: List[Tuple[Any, ...]],
    *,
    match_seed: bool = True,
) -> Optional[Tuple[Any, ...]]:
    """Stack equal-shaped independently packed pages as an outer batch.

    TurboQuant packs every page independently and pads its final uint32 word.
    Concatenating those words is lossless only when every page ends exactly on a
    packing boundary.  The added outer dimension preserves the original
    batch/head/token ordering; the caller folds it into the token axis only
    after decoding.  Unusual layouts use the scalar compatibility path.
    """
    if not entries:
        return None
    first = entries[0]
    if not isinstance(first, (tuple, list)) or len(first) != 4:
        return None
    tag, first_keys, first_values, first_config = first
    if tag != "turboquant_kv" or not isinstance(first_config, dict):
        return None

    key_payloads = []
    value_payloads = []
    signature = _tq_block_batch_signature(first, include_seed=match_seed)
    if signature is None:
        return None

    for entry in entries:
        if not isinstance(entry, (tuple, list)) or len(entry) != 4:
            return None
        entry_tag, encoded_keys, encoded_values, config = entry
        if entry_tag != "turboquant_kv" or not isinstance(config, dict):
            return None
        if _tq_block_batch_signature(entry, include_seed=match_seed) != signature:
            return None

        key_shape = tuple(int(dim) for dim in encoded_keys.shape)
        value_shape = tuple(int(dim) for dim in encoded_values.shape)
        if int(config.get("offset", -1)) != key_shape[-2]:
            return None
        key_payloads.append(encoded_keys)
        value_payloads.append(encoded_values)

    merged_key_shape = (len(entries),) + tuple(first_keys.shape)
    merged_value_shape = (len(entries),) + tuple(first_values.shape)
    from jang_tools.turboquant.pipeline import (
        pack_bits,
        pack_signs,
        unpack_bits,
        unpack_signs,
    )

    def _stack_indices(payloads: List[Any], attr: str, bits: int, elements: int):
        packed = [getattr(payload, attr) for payload in payloads]
        if elements % (32 // bits) == 0:
            return mx.concatenate(packed, axis=0)
        unpacked = [unpack_bits(value, bits, elements) for value in packed]
        return pack_bits(mx.concatenate(unpacked, axis=0), bits)

    key_elements = 1
    value_elements = 1
    for dim in first_keys.shape:
        key_elements *= int(dim)
    for dim in first_values.shape:
        value_elements *= int(dim)
    if key_elements % 32 == 0:
        qjl_packed = mx.concatenate(
            [payload.qjl_packed for payload in key_payloads], axis=0
        )
    else:
        qjl_packed = pack_signs(
            mx.concatenate(
                [
                    unpack_signs(payload.qjl_packed, key_elements)
                    for payload in key_payloads
                ],
                axis=0,
            )
        )

    key_type = type(first_keys)
    value_type = type(first_values)
    merged_keys = key_type(
        indices_packed=_stack_indices(
            key_payloads,
            "indices_packed",
            int(first_keys.index_bits),
            key_elements,
        ),
        qjl_packed=qjl_packed,
        residual_norms=mx.stack(
            [payload.residual_norms for payload in key_payloads], axis=0
        ),
        vector_norms=mx.stack(
            [payload.vector_norms for payload in key_payloads], axis=0
        ),
        shape=merged_key_shape,
        index_bits=int(first_keys.index_bits),
    )
    merged_values = value_type(
        indices_packed=_stack_indices(
            value_payloads,
            "indices_packed",
            int(first_values.index_bits),
            value_elements,
        ),
        vector_norms=mx.stack(
            [payload.vector_norms for payload in value_payloads], axis=0
        ),
        shape=merged_value_shape,
        index_bits=int(first_values.index_bits),
    )
    return (
        "turboquant_kv",
        merged_keys,
        merged_values,
        dict(first_config),
    )


def decode_tq_entries(
    entries: List[Tuple[Any, ...]],
    *,
    max_run_entries: Optional[int] = None,
    _stats: Optional[Dict[str, int]] = None,
) -> List[Tuple[Any, Any]]:
    """Decode independent TQ entries while preserving every entry boundary.

    Compatible prompt-cache layers and paged blocks share the same packed
    layout. Decode those entries as an outer batch, then split the decoded
    tensors back into their original entries. Mixed dimensions/codecs and
    malformed layouts keep the scalar compatibility path.
    """
    if not entries:
        raise ValueError("cannot decode an empty TQ block sequence")
    if len(entries) == 1:
        if _stats is not None:
            _stats["scalar_entries"] = _stats.get("scalar_entries", 0) + 1
        return [decode_tq_block(entries[0])]

    decoded = []
    start = 0
    while start < len(entries):
        signature = _tq_block_batch_signature(entries[start])
        end = start + 1
        while (
            signature is not None
            and end < len(entries)
            and (max_run_entries is None or end - start < max_run_entries)
            and _tq_block_batch_signature(entries[end]) == signature
        ):
            end += 1
        run = entries[start:end]
        stacked = _stack_tq_block_entries(run) if len(run) > 1 else None
        if stacked is None:
            if _stats is not None:
                _stats["scalar_entries"] = (
                    _stats.get("scalar_entries", 0) + len(run)
                )
            decoded.extend(decode_tq_block(entry) for entry in run)
        else:
            if _stats is not None:
                _stats["batched_runs"] = _stats.get("batched_runs", 0) + 1
                _stats["batched_entries"] = (
                    _stats.get("batched_entries", 0) + len(run)
                )
            keys, values = decode_tq_block(stacked)
            decoded.extend(
                (keys[index], values[index]) for index in range(len(run))
            )
        start = end

    return decoded


def _tq_decode_timing_enabled() -> bool:
    import os

    return os.environ.get("VMLX_TQ_DECODE_TIMING", "").strip() == "1"


def decode_tq_blocks(entries: List[Tuple[Any, ...]]) -> Tuple[Any, Any]:
    """Decode paged TQ entries and join the independent pages by token axis."""
    timing = _tq_decode_timing_enabled()
    stats: Dict[str, int] = {}
    if timing:
        import time as _time

        t0 = _time.perf_counter()
    stacked = _stack_tq_block_entries(entries) if len(entries) > 1 else None
    if stacked is not None:
        # All pages share one batch signature: decode once and fold the outer
        # page axis into the token axis. Equivalent to per-page split + concat
        # (page-major, token-minor ordering) without materializing ~2N slice
        # nodes per layer (vmlx#91).
        stats["batched_runs"] = 1
        stats["batched_entries"] = len(entries)
        keys, values = decode_tq_block(stacked)
        b, h = int(keys.shape[1]), int(keys.shape[2])
        keys = mx.moveaxis(keys, 0, 2).reshape(b, h, -1, int(keys.shape[-1]))
        values = mx.moveaxis(values, 0, 2).reshape(
            b, h, -1, int(values.shape[-1])
        )
    else:
        decoded = decode_tq_entries(entries, _stats=stats)
        keys = mx.concatenate([k for k, _ in decoded], axis=2)
        values = mx.concatenate([v for _, v in decoded], axis=2)
    if timing:
        # Diagnostic mode only: forcing eval here moves the lazy codec work
        # into this call so the wall time is attributable. The default path
        # stays fully lazy.
        mx.eval(keys, values)
        elapsed = _time.perf_counter() - t0
        logger.info(
            "TQ block decode timing: entries=%d batched_runs=%d "
            "batched_entries=%d scalar_entries=%d tokens=%d wall=%.4fs",
            len(entries),
            stats.get("batched_runs", 0),
            stats.get("batched_entries", 0),
            stats.get("scalar_entries", 0),
            int(keys.shape[2]) if getattr(keys, "ndim", 0) >= 3 else -1,
            elapsed,
        )
    return keys, values


def _decode_tq_stacked_layers(
    stacked: Tuple[Any, ...],
    seeds: List[int],
    per_layer: int,
) -> Tuple[Any, Any]:
    """Decode one multi-layer stacked TQ payload with per-layer codec state.

    ``stacked`` is a ``_stack_tq_block_entries(..., match_seed=False)`` entry
    whose outer axis is ``len(seeds) * per_layer`` pages ordered layer-major.
    Every arithmetic step mirrors ``jang_tools`` ``decode_keys``/
    ``decode_values`` element-for-element; per-layer seeds only swap in stacked
    rotation signs and QJL projections, and codebooks depend solely on
    (dim, bits) so they are shared. Output: per-layer folded KV tensors of
    shape (layers, B, H, per_layer * T, D), fully lazy.
    """
    import math

    from jang_tools.turboquant import pipeline as _tq_pipeline
    from jang_tools.turboquant.codebook import dequantize_scalar
    from jang_tools.turboquant.pipeline import unpack_bits, unpack_signs

    _, merged_keys, merged_values, config = stacked
    layers = len(seeds)
    key_dim = int(config["key_dim"])
    value_dim = int(config["value_dim"])
    key_bits = int(config["key_bits"])
    value_bits = int(config["value_bits"])

    encoder_pairs = [
        _tq_decoder_pair(key_dim, value_dim, key_bits, value_bits, int(seed))
        for seed in seeds
    ]
    key_encoders = [pair[0] for pair in encoder_pairs]
    value_encoders = [pair[1] for pair in encoder_pairs]

    # Use the exact inverse-rotation entry point the installed pipeline's own
    # decode uses (fused Metal butterfly when present, Python butterfly
    # otherwise) so batched output stays bit-identical to per-layer decode.
    hadamard = getattr(_tq_pipeline, "_hadamard_inverse_fast", None)
    if hadamard is None:
        hadamard = _tq_pipeline.hadamard_inverse

    key_shape = tuple(int(dim) for dim in merged_keys.shape)
    value_shape = tuple(int(dim) for dim in merged_values.shape)
    n_key_elements = 1
    for dim in key_shape:
        n_key_elements *= dim
    n_value_elements = 1
    for dim in value_shape:
        n_value_elements *= dim

    # ---- keys: unpack → dequant → QJL correct → inverse rotate → scale ----
    flat_key_indices = unpack_bits(
        merged_keys.indices_packed, int(merged_keys.index_bits), n_key_elements
    ).reshape(-1, key_dim)
    key_mse = dequantize_scalar(flat_key_indices, key_encoders[0].key_codebook)
    flat_qjl = unpack_signs(merged_keys.qjl_packed, n_key_elements).reshape(
        layers, -1, key_dim
    )
    key_res_norms = merged_keys.residual_norms.astype(mx.float32).reshape(
        layers, -1, 1
    )
    key_vec_norms = merged_keys.vector_norms.astype(mx.float32).reshape(
        layers, -1, 1
    )
    qjl_projection = mx.stack(
        [encoder.qjl_S for encoder in key_encoders], axis=0
    )
    qjl_scale = math.sqrt(math.pi / 2.0) / key_dim
    qjl_dequant = qjl_scale * key_res_norms * (flat_qjl @ qjl_projection)
    key_rotated = key_mse.reshape(layers, -1, key_dim) + qjl_dequant
    key_ones = mx.ones(
        (key_dim,), dtype=key_encoders[0].rotation_signs.dtype
    )
    key_signs = mx.stack(
        [encoder.rotation_signs for encoder in key_encoders], axis=0
    ).reshape(layers, 1, key_dim)
    keys = hadamard(key_rotated, key_ones) * key_signs * key_vec_norms
    keys = _restore_tq_dtype(keys, config.get("key_dtype"), "key")

    # ---- values: unpack → dequant → inverse rotate → scale ----
    flat_value_indices = unpack_bits(
        merged_values.indices_packed,
        int(merged_values.index_bits),
        n_value_elements,
    ).reshape(-1, value_dim)
    value_mse = dequantize_scalar(
        flat_value_indices, value_encoders[0].value_codebook
    )
    value_vec_norms = merged_values.vector_norms.astype(mx.float32).reshape(
        layers, -1, 1
    )
    value_ones = mx.ones(
        (value_dim,), dtype=value_encoders[0].rotation_signs.dtype
    )
    value_signs = mx.stack(
        [encoder.rotation_signs for encoder in value_encoders], axis=0
    ).reshape(layers, 1, value_dim)
    values = (
        hadamard(value_mse.reshape(layers, -1, value_dim), value_ones)
        * value_signs
        * value_vec_norms
    )
    values = _restore_tq_dtype(values, config.get("value_dtype"), "value")

    # Fold layer-major pages into each layer's token axis: page-major,
    # token-minor — identical element mapping to decode_tq_blocks' fold.
    keys = keys.reshape(
        layers, per_layer, key_shape[-4], key_shape[-3], key_shape[-2], key_dim
    )
    keys = mx.moveaxis(keys, 1, 3).reshape(
        layers, key_shape[-4], key_shape[-3], per_layer * key_shape[-2], key_dim
    )
    values = values.reshape(
        layers,
        per_layer,
        value_shape[-4],
        value_shape[-3],
        value_shape[-2],
        value_dim,
    )
    values = mx.moveaxis(values, 1, 3).reshape(
        layers,
        value_shape[-4],
        value_shape[-3],
        per_layer * value_shape[-2],
        value_dim,
    )
    return keys, values


def decode_tq_layer_groups(
    groups: Dict[Any, List[Tuple[Any, ...]]],
) -> Dict[Any, Tuple[Any, Any]]:
    """Decode many layers' paged TQ entries as one cross-layer lazy graph.

    ``groups`` maps an opaque caller key (layer index, sub-cache index, …) to
    that layer's ordered page entries. Layers whose per-position payload
    signatures match (everything except the per-layer seed) are stacked and
    decoded together, collapsing L independent per-layer codec graphs into a
    handful of large fused ops with no eval points (vmlx#91). Layers that
    cannot co-batch fall back to :func:`decode_tq_blocks` per layer and stay
    lazy as well. Returns key → (keys, values) with output identical to
    ``decode_tq_blocks(entries)`` for every key.
    """
    timing = _tq_decode_timing_enabled()
    if timing:
        import time as _time

        t0 = _time.perf_counter()

    results: Dict[Any, Tuple[Any, Any]] = {}
    signature_families: Dict[Any, List[Any]] = {}
    fallback_keys: List[Any] = []

    for group_key, entries in groups.items():
        if not entries:
            continue
        signature_seq = tuple(
            _tq_block_batch_signature(entry, include_seed=False)
            for entry in entries
        )
        entry_seeds = {_tq_entry_seed(entry) for entry in entries}
        if (
            any(signature is None for signature in signature_seq)
            or len(entry_seeds) != 1
            or None in entry_seeds
        ):
            # Malformed or mixed-seed layer: per-layer compatibility path.
            fallback_keys.append(group_key)
            continue
        signature_families.setdefault(signature_seq, []).append(group_key)

    batched_runs = 0
    batched_layers = 0
    for signature_seq, group_keys in signature_families.items():
        if len(group_keys) < 2:
            fallback_keys.extend(group_keys)
            continue
        try:
            # Split entry positions into runs of one payload signature (full
            # pages vs a shorter tail page). Positions align across layers
            # because every layer in the family shares the signature sequence.
            run_bounds: List[Tuple[int, int]] = []
            start = 0
            for position in range(1, len(signature_seq) + 1):
                if (
                    position == len(signature_seq)
                    or signature_seq[position] != signature_seq[start]
                ):
                    run_bounds.append((start, position))
                    start = position

            per_key_runs: Dict[Any, List[Any]] = {key: [] for key in group_keys}
            for run_start, run_end in run_bounds:
                per_layer = run_end - run_start
                flat_entries: List[Tuple[Any, ...]] = []
                seeds: List[int] = []
                for group_key in group_keys:
                    layer_entries = groups[group_key][run_start:run_end]
                    flat_entries.extend(layer_entries)
                    seeds.append(_tq_entry_seed(layer_entries[0]))
                stacked = _stack_tq_block_entries(
                    flat_entries, match_seed=False
                )
                if stacked is None:
                    raise ValueError(
                        "cross-layer TQ payload stacking failed for a "
                        "signature-matched run"
                    )
                run_keys, run_values = _decode_tq_stacked_layers(
                    stacked, seeds, per_layer
                )
                first_config = flat_entries[0][3]
                _record_tq_block_codec_event(
                    "decode",
                    blocks=len(flat_entries),
                    tokens=int(first_config["offset"]) * len(flat_entries),
                    metadata={
                        "boundary": "decode_tq_layer_groups",
                        "key_bits_values": [int(first_config["key_bits"])],
                        "value_bits_values": [int(first_config["value_bits"])],
                        "key_dim_values": [int(first_config["key_dim"])],
                        "value_dim_values": [int(first_config["value_dim"])],
                        "key_dtype_values": [str(first_config["key_dtype"])],
                        "value_dtype_values": [str(first_config["value_dtype"])],
                        "key_shape": [int(dim) for dim in run_keys.shape],
                        "value_shape": [int(dim) for dim in run_values.shape],
                    },
                )
                batched_runs += 1
                batched_layers += len(group_keys)
                for index, group_key in enumerate(group_keys):
                    per_key_runs[group_key].append(
                        (run_keys[index], run_values[index])
                    )

            for group_key in group_keys:
                pairs = per_key_runs[group_key]
                if len(pairs) == 1:
                    results[group_key] = pairs[0]
                else:
                    results[group_key] = (
                        mx.concatenate([keys for keys, _ in pairs], axis=2),
                        mx.concatenate([values for _, values in pairs], axis=2),
                    )
        except Exception as exc:  # pragma: no cover - safety net
            logger.warning(
                "cross-layer TQ batch decode failed (%s); falling back to "
                "per-layer decode for %d layer(s)",
                exc,
                len(group_keys),
            )
            for group_key in group_keys:
                results.pop(group_key, None)
            fallback_keys.extend(group_keys)

    for group_key in fallback_keys:
        results[group_key] = decode_tq_blocks(groups[group_key])

    if timing:
        flattened = [tensor for pair in results.values() for tensor in pair]
        if flattened:
            mx.eval(*flattened)
        elapsed = _time.perf_counter() - t0
        logger.info(
            "TQ layer-group decode timing: layers=%d batched_runs=%d "
            "batched_layers=%d fallback_layers=%d wall=%.4fs",
            len(results),
            batched_runs,
            batched_layers,
            len(fallback_keys),
            elapsed,
        )
    return results


def is_tq_compressed_cache(cache: List[Any]) -> bool:
    """Check if any layer is TurboQuantKVCache with compressed data available.

    Returns True if at least one layer has _compressed_keys set, meaning
    compress() has been called and native TQ serialization is possible.
    """
    for c in cache:
        if (type(c).__name__ == _TQ_CLASS_NAME
                and getattr(c, '_compressed_keys', None) is not None
                and getattr(c, '_compressed_values', None) is not None):
            return True
    return False


def has_turboquant_layers(cache: List[Any]) -> bool:
    """Return whether the cache contains any native TQ KV layer."""
    def _has(layer: Any) -> bool:
        if type(layer).__name__ == _TQ_CLASS_NAME:
            return True
        sub_caches = getattr(layer, "caches", None)
        return isinstance(sub_caches, (list, tuple)) and any(
            _has(sub) for sub in sub_caches
        )

    return any(_has(layer) for layer in cache or [])


def canonicalize_tq_cache_for_storage(cache: List[Any]) -> List[Any]:
    """Return a storage-owned cache with every TQ layer fully encoded.

    A live TQ cache may have encoded only ``compress_after`` old tokens while
    retaining sink tokens and a later float window.  Serializing only its
    ``_compressed_*`` fields with the full offset creates a truncated, corrupt
    disk record.  Storage uses a clone with ``sink_tokens=0`` and encodes the
    complete readable state exactly once; non-TQ companion layers are preserved.
    """
    if not HAS_MLX:
        raise RuntimeError("MLX required for TQ storage canonicalization")
    from jang_tools.turboquant.cache import TurboQuantKVCache

    def _canonicalize(layer: Any, label: str) -> Any:
        sub_caches = getattr(layer, "caches", None)
        if (
            type(layer).__name__ != _TQ_CLASS_NAME
            and isinstance(sub_caches, (list, tuple))
        ):
            canonical_subs = [
                _canonicalize(sub, f"{label}/{sub_index}")
                for sub_index, sub in enumerate(sub_caches)
            ]
            try:
                return type(layer)(*canonical_subs)
            except Exception as exc:
                raise ValueError(
                    f"TQ cache list {label} could not be reconstructed: {exc}"
                ) from exc
        if type(layer).__name__ != _TQ_CLASS_NAME:
            return layer
        state = getattr(layer, "state", None)
        if not isinstance(state, (tuple, list)) or len(state) != 2:
            raise ValueError(f"TQ layer {label} has no readable KV state")
        keys, values = state
        if keys is None or values is None:
            raise ValueError(f"TQ layer {label} has empty KV state")
        offset = int(getattr(layer, "offset", 0) or 0)
        seq_len = int(keys.shape[-2])
        if offset != seq_len:
            raise ValueError(
                f"TQ layer {label} offset/state mismatch: offset={offset}, "
                f"state_tokens={seq_len}"
            )
        clone = TurboQuantKVCache(
            key_dim=int(keys.shape[-1]),
            value_dim=int(values.shape[-1]),
            key_bits=int(getattr(layer, "key_bits", 8) or 8),
            value_bits=int(getattr(layer, "value_bits", 8) or 8),
            seed=int(getattr(layer, "_seed", 42) or 42),
            compress_after=0,
            # Sink tokens are a live-attention policy. Disk records must contain
            # one complete packed payload, so encode them with the rest.
            sink_tokens=0,
        )
        clone.keys = mx.contiguous(keys)
        clone.values = mx.contiguous(values)
        clone._vmlx_tq_key_dtype = _canonical_tq_dtype(keys.dtype)
        clone._vmlx_tq_value_dtype = _canonical_tq_dtype(values.dtype)
        clone.offset = offset
        clone.compress()
        return clone

    return [
        _canonicalize(layer, str(index)) for index, layer in enumerate(cache or [])
    ]


def serialize_tq_cache(
    cache: List[Any],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Serialize cache with TQ-native compression for TQ layers.

    For TurboQuantKVCache layers: extracts _compressed_keys/_compressed_values
    directly (26x smaller than .state which decompresses to float16).

    For other layers (KVCache, MambaCache, etc.): uses standard state extraction.

    Args:
        cache: List of cache layer objects from the model.

    Returns:
        (tensors, metadata) — ready for safetensors storage.
        tensors: Dict[str, mx.array] of named tensors.
        metadata: Dict[str, str] of string metadata.
    """
    if not HAS_MLX:
        raise RuntimeError("MLX required for TQ serialization")

    tensors: Dict[str, Any] = {}
    meta: Dict[str, str] = {
        "__tq_native__": "true",
        "__num_layers__": str(len(cache)),
    }
    tq_count = 0
    non_tq_count = 0

    for i, layer in enumerate(cache):
        cls_name = type(layer).__name__
        meta[f"__layer_{i}_class__"] = cls_name

        if (cls_name == _TQ_CLASS_NAME
                and getattr(layer, '_compressed_keys', None) is not None):
            # ─── TQ layer: serialize compressed data directly ───
            _serialize_tq_layer(tensors, meta, i, layer)
            tq_count += 1
        elif hasattr(layer, 'caches') and isinstance(
            getattr(layer, 'caches', None), (list, tuple)
        ):
            # ─── CacheList (MoE models: DeepSeek V3.2, Falcon H1) ───
            # Contains sub-caches that may be TQ or standard KVCache.
            _serialize_cache_list_layer(tensors, meta, i, layer)
            non_tq_count += 1
        elif hasattr(layer, 'state') and hasattr(layer, 'meta_state'):
            # ─── Non-TQ layer: serialize via .state ───
            _serialize_standard_layer(tensors, meta, i, layer, cls_name)
            non_tq_count += 1
        else:
            # Unknown layer — mark as empty
            meta[f"__layer_{i}_empty__"] = "true"

    logger.info(
        f"TQ-native serialize: {tq_count} TQ layers (compressed), "
        f"{non_tq_count} standard layers"
    )
    return tensors, meta


def deserialize_tq_cache(
    tensors: Dict[str, Any],
    metadata: Dict[str, str],
) -> List[Any]:
    """Deserialize TQ-native cache from safetensors data.

    TQ layers are decoded from compressed form to float16 and wrapped in
    KVCache objects. The caller should then call _recompress_to_tq() to
    convert back to TurboQuantKVCache using the model's make_cache() template.

    Non-TQ layers are reconstructed as standard KVCache or placeholder objects.

    Args:
        tensors: Dict of named tensors from mx.load().
        metadata: Dict of string metadata from safetensors header.

    Returns:
        List of cache layer objects (KVCache or placeholders).
    """
    if not HAS_MLX:
        raise RuntimeError("MLX required for TQ deserialization")

    try:
        from .cache_record_validator import validate_tq_native_metadata
    except Exception:
        validate_tq_native_metadata = None
    if validate_tq_native_metadata is not None:
        ok, reason = validate_tq_native_metadata(
            tensors, metadata, source="tq-native-deserialize"
        )
        if not ok:
            raise ValueError(f"unsafe TQ-native metadata: {reason}")

    from mlx_lm.models.cache import KVCache

    num_layers = int(metadata.get("__num_layers__", "0"))
    cache: List[Any] = []

    tq_decoded = 0
    standard_loaded = 0

    # Prompt L2 files commonly contain dozens of attention layers with one
    # identical TQ codec/layout. Decoding them one at a time launches the same
    # transform pipeline once per layer and can cost more than cold prefill.
    # Batch compatible layers in bounded runs while preserving each layer's
    # independent packed payload and output position.
    tq_layers: Dict[int, Any] = {}
    tq_entries: List[Tuple[Any, ...]] = []
    tq_entry_indices: List[int] = []
    for i in range(num_layers):
        if metadata.get(f"__layer_{i}_class__", "") != _TQ_CLASS_NAME:
            continue
        entry = _serialized_tq_layer_entry(tensors, metadata, i)
        if entry is not None:
            tq_entries.append(entry)
            tq_entry_indices.append(i)
    if tq_entries:
        try:
            decoded_entries = decode_tq_entries(tq_entries)
            for i, entry, (keys, values) in zip(
                tq_entry_indices,
                tq_entries,
                decoded_entries,
            ):
                kv = KVCache()
                kv.keys = keys
                kv.values = values
                kv.offset = int(entry[3]["offset"])
                tq_layers[i] = kv
        except Exception as exc:
            logger.warning(
                "Batched prompt TQ decode failed; using scalar layer restore: %s",
                exc,
            )

    for i in range(num_layers):
        cls_name = metadata.get(f"__layer_{i}_class__", "")

        if cls_name == _TQ_CLASS_NAME:
            # ─── TQ layer: decode compressed → KVCache ───
            kv = tq_layers.get(i)
            if kv is None:
                kv = _deserialize_tq_layer(tensors, metadata, i)
            if kv is not None:
                cache.append(kv)
                tq_decoded += 1
            else:
                cache.append(KVCache())
        elif metadata.get(f"__layer_{i}_empty__") == "true":
            cache.append(KVCache())
        elif metadata.get(f"__layer_{i}_cache_list__") == "true":
            # ─── CacheList (MoE models) ───
            layer = _deserialize_cache_list_layer(tensors, metadata, i)
            cache.append(layer)
            standard_loaded += 1
        elif metadata.get(f"__layer_{i}_cumulative__") == "true":
            # ─── Cumulative (SSM) layer ───
            layer = _deserialize_cumulative_layer(tensors, metadata, i)
            cache.append(layer)
            standard_loaded += 1
        elif f"layer_{i}_keys" in tensors:
            # ─── Standard KVCache ───
            kv = _deserialize_standard_kv(tensors, metadata, i)
            cache.append(kv)
            standard_loaded += 1
        elif metadata.get(f"__layer_{i}_quantized__") == "true":
            # ─── QuantizedKVCache ───
            kv = _deserialize_quantized_kv(tensors, metadata, i)
            cache.append(kv)
            standard_loaded += 1
        else:
            cache.append(KVCache())

    logger.info(
        f"TQ-native deserialize: {tq_decoded} TQ decoded, "
        f"{standard_loaded} standard, {num_layers} total layers"
    )
    return cache


# =============================================================================
# Internal: TQ layer serialization
# =============================================================================

def _serialize_tq_layer(
    tensors: Dict[str, Any],
    meta: Dict[str, str],
    i: int,
    layer: Any,
) -> None:
    """Serialize a single TurboQuantKVCache layer's compressed data."""
    ck = layer._compressed_keys   # EncodedKeys namedtuple
    cv = layer._compressed_values  # EncodedValues namedtuple
    offset = int(getattr(layer, "offset", 0) or 0)
    compressed_tokens = int(
        getattr(layer, "_compressed_tokens", 0) or 0
    )
    encoded_key_tokens = int(ck.shape[-2]) if len(ck.shape) >= 2 else 0
    encoded_value_tokens = int(cv.shape[-2]) if len(cv.shape) >= 2 else 0
    if not (
        offset > 0
        and compressed_tokens == offset
        and encoded_key_tokens == offset
        and encoded_value_tokens == offset
        and int(getattr(layer, "sink_tokens", 0) or 0) == 0
    ):
        raise ValueError(
            "TQ layer is not a complete canonical storage payload: "
            f"offset={offset}, compressed={compressed_tokens}, "
            f"key_tokens={encoded_key_tokens}, value_tokens={encoded_value_tokens}, "
            f"sink_tokens={getattr(layer, 'sink_tokens', 0)}"
        )

    # Store EncodedKeys tensors (4 mx.array fields)
    tensors[f"tq_{i}_ck_indices_packed"] = ck.indices_packed
    tensors[f"tq_{i}_ck_qjl_packed"] = ck.qjl_packed
    tensors[f"tq_{i}_ck_residual_norms"] = ck.residual_norms
    tensors[f"tq_{i}_ck_vector_norms"] = ck.vector_norms

    # Store EncodedValues tensors (2 mx.array fields)
    tensors[f"tq_{i}_cv_indices_packed"] = cv.indices_packed
    tensors[f"tq_{i}_cv_vector_norms"] = cv.vector_norms

    # Store metadata (shape tuples, bit widths, TQ config)
    meta[f"__tq_{i}_ck_shape__"] = json.dumps(list(ck.shape))
    meta[f"__tq_{i}_ck_bits__"] = str(ck.index_bits)
    meta[f"__tq_{i}_cv_shape__"] = json.dumps(list(cv.shape))
    meta[f"__tq_{i}_cv_bits__"] = str(cv.index_bits)
    meta[f"__tq_{i}_offset__"] = str(offset)
    meta[f"__tq_{i}_compressed_tokens__"] = str(
        getattr(layer, '_compressed_tokens', layer.offset)
    )
    meta[f"__tq_{i}_key_dim__"] = str(layer.key_dim)
    meta[f"__tq_{i}_value_dim__"] = str(layer.value_dim)
    meta[f"__tq_{i}_key_bits__"] = str(layer.key_bits)
    meta[f"__tq_{i}_value_bits__"] = str(layer.value_bits)
    meta[f"__tq_{i}_key_dtype__"] = _canonical_tq_dtype(
        getattr(layer, "_vmlx_tq_key_dtype", "")
    )
    meta[f"__tq_{i}_value_dtype__"] = _canonical_tq_dtype(
        getattr(layer, "_vmlx_tq_value_dtype", "")
    )
    meta[f"__tq_{i}_sink_tokens__"] = str(getattr(layer, 'sink_tokens', 0))
    meta[f"__tq_{i}_seed__"] = str(getattr(layer, '_seed', 42))


def _serialize_standard_layer(
    tensors: Dict[str, Any],
    meta: Dict[str, str],
    i: int,
    layer: Any,
    cls_name: str,
) -> None:
    """Serialize a non-TQ cache layer via its .state property."""
    state = layer.state
    meta_state = layer.meta_state

    # Detect cumulative (SSM) layers: MambaCache, ArraysCache
    is_cumulative = (
        hasattr(layer, 'cache') and isinstance(getattr(layer, 'cache', None), list)
    )

    if is_cumulative:
        # Store cumulative state arrays
        meta[f"__layer_{i}_cumulative__"] = "true"
        meta[f"__layer_{i}_cumulative_class__"] = cls_name
        if isinstance(state, (list, tuple)):
            for j, arr in enumerate(state):
                if hasattr(arr, 'shape'):
                    tensors[f"layer_{i}_state_{j}"] = arr
            meta[f"__layer_{i}_state_count__"] = str(len(state))
        if meta_state:
            meta[f"__layer_{i}_meta__"] = json.dumps(
                [str(x) for x in meta_state] if isinstance(meta_state, tuple) else str(meta_state)
            )
        return

    if isinstance(state, tuple) and len(state) == 2:
        keys, values = state

        if isinstance(keys, (tuple, list)):
            # QuantizedKVCache: keys/values are tuples of (data, scales, zeros)
            meta[f"__layer_{i}_quantized__"] = "true"
            for j, t in enumerate(keys):
                if hasattr(t, 'shape'):
                    tensors[f"layer_{i}_qk_{j}"] = t
            for j, t in enumerate(values):
                if hasattr(t, 'shape'):
                    tensors[f"layer_{i}_qv_{j}"] = t
            meta[f"__layer_{i}_qk_count__"] = str(len(keys))
            meta[f"__layer_{i}_qv_count__"] = str(len(values))
        elif hasattr(keys, 'shape'):
            # Standard KVCache
            tensors[f"layer_{i}_keys"] = keys
            tensors[f"layer_{i}_values"] = values
            # Cast bfloat16 → float16 (safetensors supports bf16 but numpy doesn't)
            if keys.dtype == mx.bfloat16:
                tensors[f"layer_{i}_keys"] = keys.astype(mx.float16)
                tensors[f"layer_{i}_values"] = values.astype(mx.float16)
                meta[f"__layer_{i}_orig_dtype__"] = "bfloat16"

    # Store meta_state (offset, etc.)
    if meta_state:
        meta[f"__layer_{i}_meta__"] = json.dumps(
            [str(x) for x in meta_state] if isinstance(meta_state, tuple) else str(meta_state)
        )


def _serialize_cache_list_layer(
    tensors: Dict[str, Any],
    meta: Dict[str, str],
    i: int,
    layer: Any,
) -> None:
    """Serialize a CacheList layer (MoE models: DeepSeek V3.2, Falcon H1).

    CacheList wraps a list of sub-caches (.caches attribute). Each sub-cache
    can be TQ, KVCache, or cumulative (MambaCache). We serialize each sub-cache
    independently using the appropriate path.
    """
    meta[f"__layer_{i}_cache_list__"] = "true"
    sub_caches = layer.caches
    meta[f"__layer_{i}_cl_count__"] = str(len(sub_caches))

    for j, sub in enumerate(sub_caches):
        sub_cls = type(sub).__name__
        meta[f"__layer_{i}_cl_{j}_class__"] = sub_cls

        if (sub_cls == _TQ_CLASS_NAME
                and getattr(sub, '_compressed_keys', None) is not None):
            # TQ sub-cache: serialize compressed data
            # Reuse TQ serializer with prefixed keys
            ck = sub._compressed_keys
            cv = sub._compressed_values
            prefix = f"cl_{i}_{j}"
            offset = int(getattr(sub, "offset", 0) or 0)
            compressed_tokens = int(
                getattr(sub, "_compressed_tokens", 0) or 0
            )
            key_tokens = int(ck.shape[-2]) if len(ck.shape) >= 2 else 0
            value_tokens = int(cv.shape[-2]) if len(cv.shape) >= 2 else 0
            if not (
                offset > 0
                and compressed_tokens == offset
                and key_tokens == offset
                and value_tokens == offset
                and int(getattr(sub, "sink_tokens", 0) or 0) == 0
            ):
                raise ValueError(
                    f"CacheList TQ layer {i}/{j} is not a complete canonical "
                    f"storage payload: offset={offset}, "
                    f"compressed={compressed_tokens}, key_tokens={key_tokens}, "
                    f"value_tokens={value_tokens}, "
                    f"sink_tokens={getattr(sub, 'sink_tokens', 0)}"
                )
            tensors[f"{prefix}_ck_indices_packed"] = ck.indices_packed
            tensors[f"{prefix}_ck_qjl_packed"] = ck.qjl_packed
            tensors[f"{prefix}_ck_residual_norms"] = ck.residual_norms
            tensors[f"{prefix}_ck_vector_norms"] = ck.vector_norms
            tensors[f"{prefix}_cv_indices_packed"] = cv.indices_packed
            tensors[f"{prefix}_cv_vector_norms"] = cv.vector_norms
            meta[f"__{prefix}_ck_shape__"] = json.dumps(list(ck.shape))
            meta[f"__{prefix}_ck_bits__"] = str(ck.index_bits)
            meta[f"__{prefix}_cv_shape__"] = json.dumps(list(cv.shape))
            meta[f"__{prefix}_cv_bits__"] = str(cv.index_bits)
            meta[f"__{prefix}_offset__"] = str(offset)
            meta[f"__{prefix}_compressed_tokens__"] = str(compressed_tokens)
            meta[f"__{prefix}_key_dim__"] = str(sub.key_dim)
            meta[f"__{prefix}_value_dim__"] = str(sub.value_dim)
            meta[f"__{prefix}_key_bits__"] = str(sub.key_bits)
            meta[f"__{prefix}_value_bits__"] = str(sub.value_bits)
            meta[f"__{prefix}_key_dtype__"] = _canonical_tq_dtype(
                getattr(sub, "_vmlx_tq_key_dtype", "")
            )
            meta[f"__{prefix}_value_dtype__"] = _canonical_tq_dtype(
                getattr(sub, "_vmlx_tq_value_dtype", "")
            )
            meta[f"__{prefix}_sink_tokens__"] = str(getattr(sub, 'sink_tokens', 0))
            meta[f"__{prefix}_seed__"] = str(getattr(sub, '_seed', 42))
        elif hasattr(sub, 'state') and hasattr(sub, 'meta_state'):
            # Standard sub-cache (KVCache or cumulative)
            state = sub.state
            if isinstance(state, tuple) and len(state) == 2:
                keys, values = state
                if hasattr(keys, 'shape'):
                    tensors[f"cl_{i}_{j}_keys"] = keys
                    tensors[f"cl_{i}_{j}_values"] = values
            sub_meta = sub.meta_state
            if sub_meta:
                meta[f"__cl_{i}_{j}_meta__"] = json.dumps(
                    [str(x) for x in sub_meta] if isinstance(sub_meta, tuple) else str(sub_meta)
                )


def _deserialize_cache_list_layer(
    tensors: Dict[str, Any],
    metadata: Dict[str, str],
    i: int,
) -> Any:
    """Reconstruct a CacheList layer from serialized sub-caches.

    Returns a list of KVCache objects. The caller should wrap this in a
    CacheList if needed, or pass through to _recompress_to_tq().
    """
    from mlx_lm.models.cache import KVCache

    sub_count = _parse_bounded_int(
        metadata,
        f"__layer_{i}_cl_count__",
        default=0,
        lo=0,
        hi=64,
    )
    sub_caches = []

    for j in range(sub_count):
        sub_cls = metadata.get(f"__layer_{i}_cl_{j}_class__", "")
        prefix = f"cl_{i}_{j}"

        if sub_cls == _TQ_CLASS_NAME and f"{prefix}_ck_indices_packed" in tensors:
            # TQ sub-cache — decode same as _deserialize_tq_layer but with cl_ prefix
            kv = KVCache()
            # For now, store as empty KVCache — _recompress_to_tq handles conversion
            # The actual decode requires jang_tools which may not be available
            try:
                from jang_tools.turboquant.cache import EncodedKeys, EncodedValues
                from jang_tools.turboquant.pipeline import decode_keys, decode_values

                ck_shape = tuple(json.loads(metadata.get(f"__{prefix}_ck_shape__", "[]")))
                ck_bits = _parse_bounded_int(metadata, f"__{prefix}_ck_bits__", default=3, lo=1, hi=8)
                cv_shape = tuple(json.loads(metadata.get(f"__{prefix}_cv_shape__", "[]")))
                cv_bits = _parse_bounded_int(metadata, f"__{prefix}_cv_bits__", default=3, lo=1, hi=8)

                encoded_keys = EncodedKeys(
                    indices_packed=tensors[f"{prefix}_ck_indices_packed"],
                    qjl_packed=tensors[f"{prefix}_ck_qjl_packed"],
                    residual_norms=tensors[f"{prefix}_ck_residual_norms"],
                    vector_norms=tensors[f"{prefix}_ck_vector_norms"],
                    shape=ck_shape, index_bits=ck_bits,
                )
                encoded_values = EncodedValues(
                    indices_packed=tensors[f"{prefix}_cv_indices_packed"],
                    vector_norms=tensors[f"{prefix}_cv_vector_norms"],
                    shape=cv_shape, index_bits=cv_bits,
                )
                _key_dim = _parse_bounded_int(metadata, f"__{prefix}_key_dim__", default=128, lo=1, hi=262144)
                _val_dim = _parse_bounded_int(metadata, f"__{prefix}_value_dim__", default=128, lo=1, hi=262144)
                _key_bits = _parse_bounded_int(metadata, f"__{prefix}_key_bits__", default=3, lo=1, hi=8)
                _val_bits = _parse_bounded_int(metadata, f"__{prefix}_value_bits__", default=3, lo=1, hi=8)
                _seed = _parse_bounded_int(
                    metadata,
                    f"__{prefix}_seed__",
                    default=42,
                    lo=0,
                    hi=2_147_483_647,
                )
                _key_encoder, _value_encoder = _tq_decoder_pair(
                    _key_dim,
                    _val_dim,
                    _key_bits,
                    _val_bits,
                    _seed,
                )
                kv.keys = _restore_tq_dtype(
                    decode_keys(encoded_keys, _key_encoder),
                    metadata.get(f"__{prefix}_key_dtype__"),
                    f"CacheList {i}/{j} key",
                )
                kv.values = _restore_tq_dtype(
                    decode_values(encoded_values, _value_encoder),
                    metadata.get(f"__{prefix}_value_dtype__"),
                    f"CacheList {i}/{j} value",
                )
                kv.offset = _parse_bounded_int(metadata, f"__{prefix}_offset__", default=0, lo=0, hi=2_000_000)
            except Exception as e:
                logger.warning("CacheList sub-cache %d/%d TQ decode failed: %s", i, j, e)
            sub_caches.append(kv)
        elif f"{prefix}_keys" in tensors:
            # Standard KVCache sub-cache
            kv = KVCache()
            kv.keys = tensors[f"{prefix}_keys"]
            kv.values = tensors[f"{prefix}_values"]
            sub_meta_str = metadata.get(f"__{prefix}_meta__", "")
            if sub_meta_str:
                try:
                    kv.offset = _parse_meta_offset_str(sub_meta_str, f"{prefix}_meta")
                except (json.JSONDecodeError, ValueError, IndexError):
                    kv.offset = kv.keys.shape[2] if kv.keys is not None and kv.keys.ndim >= 3 else 0
            else:
                kv.offset = kv.keys.shape[2] if kv.keys is not None and kv.keys.ndim >= 3 else 0
            sub_caches.append(kv)
        else:
            sub_caches.append(KVCache())

    # Try to wrap in CacheList if available
    try:
        from mlx_lm.models.cache import CacheList as _CL
        cl = _CL(*sub_caches)
        return cl
    except ImportError:
        # CacheList not available — return raw list
        # The caller should handle this gracefully
        return sub_caches[0] if len(sub_caches) == 1 else KVCache()


# =============================================================================
# Internal: TQ layer deserialization
# =============================================================================

def _serialized_tq_layer_entry(
    tensors: Dict[str, Any],
    metadata: Dict[str, str],
    i: int,
) -> Optional[Tuple[Any, ...]]:
    """Reconstruct one serialized TQ entry without decoding it."""
    try:
        from jang_tools.turboquant.cache import EncodedKeys, EncodedValues
    except ImportError:
        logger.warning("jang_tools not available — cannot restore TQ layer %d", i)
        return None

    prefix = f"tq_{i}"

    # Reconstruct EncodedKeys
    ck_indices = tensors.get(f"{prefix}_ck_indices_packed")
    ck_qjl = tensors.get(f"{prefix}_ck_qjl_packed")
    ck_rnorms = tensors.get(f"{prefix}_ck_residual_norms")
    ck_vnorms = tensors.get(f"{prefix}_ck_vector_norms")

    if ck_indices is None:
        logger.warning("TQ layer %d missing ck_indices_packed", i)
        return None

    try:
        ck_shape = tuple(json.loads(metadata.get(f"__{prefix}_ck_shape__", "[]")))
        ck_bits = _parse_bounded_int(
            metadata, f"__{prefix}_ck_bits__", default=3, lo=1, hi=8
        )
    except (json.JSONDecodeError, ValueError):
        logger.warning("TQ layer %d: invalid ck metadata", i)
        return None

    encoded_keys = EncodedKeys(
        indices_packed=ck_indices,
        qjl_packed=ck_qjl,
        residual_norms=ck_rnorms,
        vector_norms=ck_vnorms,
        shape=ck_shape,
        index_bits=ck_bits,
    )

    # Reconstruct EncodedValues
    cv_indices = tensors.get(f"{prefix}_cv_indices_packed")
    cv_vnorms = tensors.get(f"{prefix}_cv_vector_norms")

    if cv_indices is None:
        logger.warning("TQ layer %d missing cv_indices_packed", i)
        return None

    try:
        cv_shape = tuple(json.loads(metadata.get(f"__{prefix}_cv_shape__", "[]")))
        cv_bits = _parse_bounded_int(
            metadata, f"__{prefix}_cv_bits__", default=3, lo=1, hi=8
        )
    except (json.JSONDecodeError, ValueError):
        logger.warning("TQ layer %d: invalid cv metadata", i)
        return None

    encoded_values = EncodedValues(
        indices_packed=cv_indices,
        vector_norms=cv_vnorms,
        shape=cv_shape,
        index_bits=cv_bits,
    )

    offset = _parse_bounded_int(metadata, f"__{prefix}_offset__", default=0, lo=0, hi=2_000_000)
    key_dim = _parse_bounded_int(metadata, f"__{prefix}_key_dim__", default=128, lo=1, hi=262144)
    value_dim = _parse_bounded_int(metadata, f"__{prefix}_value_dim__", default=128, lo=1, hi=262144)
    key_bits = _parse_bounded_int(metadata, f"__{prefix}_key_bits__", default=3, lo=1, hi=8)
    value_bits = _parse_bounded_int(metadata, f"__{prefix}_value_bits__", default=3, lo=1, hi=8)
    seed = _parse_bounded_int(
        metadata, f"__{prefix}_seed__", default=42, lo=0, hi=2_147_483_647
    )

    return (
        "turboquant_kv",
        encoded_keys,
        encoded_values,
        {
            "key_dim": key_dim,
            "value_dim": value_dim,
            "key_bits": key_bits,
            "value_bits": value_bits,
            "key_dtype": metadata.get(f"__{prefix}_key_dtype__"),
            "value_dtype": metadata.get(f"__{prefix}_value_dtype__"),
            "seed": seed,
            "offset": offset,
        },
    )


def _deserialize_tq_layer(
    tensors: Dict[str, Any],
    metadata: Dict[str, str],
    i: int,
) -> Optional[Any]:
    """Decode one TQ layer through the scalar compatibility path."""
    try:
        from mlx_lm.models.cache import KVCache
    except ImportError:
        return None

    entry = _serialized_tq_layer_entry(tensors, metadata, i)
    if entry is None:
        return None
    try:
        decoded_keys, decoded_values = decode_tq_block(entry)
    except Exception as exc:
        logger.warning("TQ layer %d decode failed: %s", i, exc)
        return None

    # Wrap in KVCache — the caller's _recompress_to_tq() will
    # convert back to TurboQuantKVCache using the model's template.
    kv = KVCache()
    kv.keys = decoded_keys
    kv.values = decoded_values
    kv.offset = int(entry[3]["offset"])

    return kv


def _deserialize_standard_kv(
    tensors: Dict[str, Any],
    metadata: Dict[str, str],
    i: int,
) -> Any:
    """Reconstruct a standard KVCache layer."""
    from mlx_lm.models.cache import KVCache

    kv = KVCache()
    kv.keys = tensors.get(f"layer_{i}_keys")
    kv.values = tensors.get(f"layer_{i}_values")

    # Restore bfloat16 if originally cast
    if metadata.get(f"__layer_{i}_orig_dtype__") == "bfloat16":
        if kv.keys is not None:
            kv.keys = kv.keys.astype(mx.bfloat16)
            kv.values = kv.values.astype(mx.bfloat16)

    # Restore offset from meta_state
    offset = _parse_offset(metadata, i)
    if offset is not None:
        kv.offset = offset
    elif kv.keys is not None and kv.keys.ndim >= 3:
        kv.offset = kv.keys.shape[2]

    return kv


def _deserialize_quantized_kv(
    tensors: Dict[str, Any],
    metadata: Dict[str, str],
    i: int,
) -> Any:
    """Reconstruct a QuantizedKVCache layer.

    Since QuantizedKVCache.from_state() may not be available, we reconstruct
    as a standard KVCache by dequantizing. The caller can re-quantize if needed.
    """
    from mlx_lm.models.cache import KVCache

    try:
        from mlx_lm.models.cache import QuantizedKVCache
        qk_count = _parse_bounded_int(
            metadata, f"__layer_{i}_qk_count__", default=0, lo=0, hi=8
        )
        qv_count = _parse_bounded_int(
            metadata, f"__layer_{i}_qv_count__", default=0, lo=0, hi=8
        )

        keys_tuple = tuple(tensors[f"layer_{i}_qk_{j}"] for j in range(qk_count))
        values_tuple = tuple(tensors[f"layer_{i}_qv_{j}"] for j in range(qv_count))

        # Try to use QuantizedKVCache.from_state if available
        state = (keys_tuple, values_tuple)
        meta_str = metadata.get(f"__layer_{i}_meta__", "")
        if meta_str:
            meta_state = tuple(json.loads(meta_str))
        else:
            meta_state = ()

        try:
            return QuantizedKVCache.from_state(state, meta_state)
        except Exception:
            pass

        # Fallback: dequantize to KVCache
        if len(keys_tuple) >= 3:
            data, scales, zeros = keys_tuple[0], keys_tuple[1], keys_tuple[2]
            keys = mx.dequantize(data, scales, zeros)
        else:
            keys = keys_tuple[0] if keys_tuple else None

        if len(values_tuple) >= 3:
            data, scales, zeros = values_tuple[0], values_tuple[1], values_tuple[2]
            values = mx.dequantize(data, scales, zeros)
        else:
            values = values_tuple[0] if values_tuple else None

        kv = KVCache()
        kv.keys = keys
        kv.values = values
        offset = _parse_offset(metadata, i)
        if offset is not None:
            kv.offset = offset
        elif keys is not None and keys.ndim >= 3:
            kv.offset = keys.shape[2]
        return kv

    except Exception as e:
        logger.warning("Failed to deserialize quantized KV layer %d: %s", i, e)
        return KVCache()


def _deserialize_cumulative_layer(
    tensors: Dict[str, Any],
    metadata: Dict[str, str],
    i: int,
) -> Any:
    """Reconstruct a cumulative (SSM) cache layer."""
    from mlx_lm.models.cache import KVCache

    cls_name = metadata.get(f"__layer_{i}_cumulative_class__", "")
    state_count = _parse_bounded_int(
        metadata, f"__layer_{i}_state_count__", default=0, lo=0, hi=64
    )

    state_arrays = []
    for j in range(state_count):
        arr = tensors.get(f"layer_{i}_state_{j}")
        if arr is not None:
            state_arrays.append(arr)

    if not state_arrays:
        return KVCache()

    # Try to reconstruct the original cache class
    meta_str = metadata.get(f"__layer_{i}_meta__", "")
    meta_state = ()
    if meta_str:
        try:
            meta_state = tuple(json.loads(meta_str))
        except (json.JSONDecodeError, ValueError):
            pass

    try:
        import mlx_lm.models.cache as _cache_mod
        cls = getattr(_cache_mod, cls_name, None)
        if cls is not None and hasattr(cls, 'from_state'):
            return cls.from_state(state_arrays, meta_state)
    except Exception:
        pass

    # Fallback: store as list in a KVCache wrapper
    # (won't work for SSM inference but preserves data)
    kv = KVCache()
    return kv


def _parse_offset(metadata: Dict[str, str], i: int) -> Optional[int]:
    """Parse offset from meta_state metadata."""
    meta_str = metadata.get(f"__layer_{i}_meta__", "")
    if not meta_str:
        return None
    try:
        return _parse_meta_offset_str(meta_str, f"layer {i} meta")
    except (json.JSONDecodeError, ValueError, IndexError):
        pass
    return None


def _parse_meta_offset_str(meta_str: str, label: str) -> int:
    meta_list = json.loads(meta_str)
    if not isinstance(meta_list, list) or not meta_list:
        return 0
    value = int(meta_list[0])
    if not 0 <= value <= 2_000_000:
        raise ValueError(f"{label}: offset {value} outside [0, 2000000]")
    return value


def _parse_bounded_int(
    metadata: Dict[str, str],
    key: str,
    *,
    default: int,
    lo: int,
    hi: int,
) -> int:
    raw = metadata.get(key, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError(f"{key}: invalid integer {raw!r}: {e}") from e
    if value < lo or value > hi:
        raise ValueError(f"{key}: {value} outside [{lo}, {hi}]")
    return value
