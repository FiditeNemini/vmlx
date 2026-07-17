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


@lru_cache(maxsize=32)
def _tq_decoder_pair(
    key_dim: int,
    value_dim: int,
    key_bits: int,
    value_bits: int,
    seed: int,
) -> Tuple[Any, Any]:
    """Return immutable TurboQuant decoder state for one codec configuration.

    Paged-prefix reconstruction may decode thousands of block/layer entries
    with the same dimensions, bit widths, and seed. Constructing a fresh
    ``TurboQuantKVCache`` for each entry also rebuilds identical rotation,
    codebook, and QJL decoder state each time. Keep a small process-local cache
    of the encoder pair; decode operations remain independent because the
    encoder objects are read-only after initialization.
    """
    from jang_tools.turboquant.cache import TurboQuantKVCache

    tq = TurboQuantKVCache(
        key_dim=key_dim,
        value_dim=value_dim,
        key_bits=key_bits,
        value_bits=value_bits,
        seed=seed,
        compress_after=0,
        sink_tokens=0,
    )
    return tq.key_encoder, tq.value_encoder


def encode_tq_block(
    keys: Any,
    values: Any,
    config: Dict[str, Any],
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

    from jang_tools.turboquant.cache import TurboQuantKVCache

    key_bits = int(config.get("key_bits", 8) or 8)
    value_bits = int(config.get("value_bits", 8) or 8)
    seed = int(config.get("seed", 42) or 42)
    if key_bits not in (2, 3, 4, 8) or value_bits not in (2, 3, 4, 8):
        raise ValueError(
            f"unsupported TQ block codec bits key={key_bits} value={value_bits}"
        )
    tq = TurboQuantKVCache(
        key_dim=int(keys.shape[-1]),
        value_dim=int(values.shape[-1]),
        key_bits=key_bits,
        value_bits=value_bits,
        seed=seed,
        compress_after=0,
        sink_tokens=0,
    )
    tq.keys = mx.contiguous(keys)
    tq.values = mx.contiguous(values)
    tq.offset = token_count
    tq.compress()
    ck = tq._compressed_keys
    cv = tq._compressed_values
    if (
        ck is None
        or cv is None
        or int(getattr(tq, "_compressed_tokens", 0) or 0) != token_count
        or int(ck.shape[-2]) != token_count
        or int(cv.shape[-2]) != token_count
    ):
        raise ValueError("TQ block encoder did not produce a complete packed payload")
    return (
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


def decode_tq_block(entry: Tuple[Any, ...]) -> Tuple[Any, Any]:
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
    return keys, values


def _tq_block_batch_signature(entry: Tuple[Any, ...]) -> Optional[Tuple[Any, ...]]:
    """Return a grouping signature for one independently packed TQ page."""
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
            int(config["seed"]),
            int(encoded_keys.index_bits),
            int(encoded_values.index_bits),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _stack_tq_block_entries(entries: List[Tuple[Any, ...]]) -> Optional[Tuple[Any, ...]]:
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
    signature = _tq_block_batch_signature(first)
    if signature is None:
        return None

    for entry in entries:
        if not isinstance(entry, (tuple, list)) or len(entry) != 4:
            return None
        entry_tag, encoded_keys, encoded_values, config = entry
        if entry_tag != "turboquant_kv" or not isinstance(config, dict):
            return None
        if _tq_block_batch_signature(entry) != signature:
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
            decoded.extend(decode_tq_block(entry) for entry in run)
        else:
            keys, values = decode_tq_block(stacked)
            decoded.extend(
                (keys[index], values[index]) for index in range(len(run))
            )
        start = end

    return decoded


def decode_tq_blocks(entries: List[Tuple[Any, ...]]) -> Tuple[Any, Any]:
    """Decode paged TQ entries and join the independent pages by token axis."""
    decoded = decode_tq_entries(entries)

    return (
        mx.concatenate([keys for keys, _ in decoded], axis=2),
        mx.concatenate([values for _, values in decoded], axis=2),
    )


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
