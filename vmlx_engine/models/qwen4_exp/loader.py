"""Memory-bounded qwen4_exp VLM loader with SSD-backed quantized PLE."""

from __future__ import annotations

import copy
import glob
import json
import logging
import re
import struct
from collections.abc import Mapping
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from vmlx_engine.utils.jang_affine_storage import (
    expand_affine1_shard_mlx,
    prepare_affine1_runtime_config,
)

from .register import register_qwen4_exp_runtime
from .table_reader import (
    FileBackedQuantizedNGramTable,
    resolve_jang_bit_map_spec,
)

logger = logging.getLogger("vmlx_engine")

_PLE_TABLE_RE = re.compile(
    r"(?:^|\.)layers\.1\.ple\.(?:ple_embedding\.)?"
    r"ngram_embedding\.(?P<layout>shard_|shards\.)(?P<shard>\d+)\."
    r"(?P<suffix>weight|scales|biases)$"
)
_PLE_BUFFER_RE = re.compile(
    r"(?:^|\.)layers\.1\.ple\.(?:ple_embedding\.)?"
    r"(?P<suffix>layer_multipliers|ngram_heads_vocab_sizes|"
    r"ngram_heads_offsets)$"
)
_PLE_REQUIRED_SHARD_SUFFIXES = ("weight", "scales", "biases")
_HYPER_UP_SUFFIX = ".input_mix_weight_up.weight"
_LOW_PRECISION_FLOAT_DTYPES = {mx.float16, mx.bfloat16}


def is_qwen4_exp_bundle(model_path: str | Path) -> bool:
    try:
        cfg = json.loads((Path(model_path) / "config.json").read_text())
    except Exception:
        return False
    model_types = {
        str(cfg.get("model_type") or "").lower(),
        str((cfg.get("text_config") or {}).get("model_type") or "").lower(),
    }
    return bool(model_types.intersection({"qwen4_exp", "qwen4_exp_text"}))


def _load_runtime_config(
    model_path: Path,
) -> tuple[dict, frozenset[str], dict[str, object] | None, bool]:
    config = json.loads((model_path / "config.json").read_text())
    embedded = config.get("jang_config")
    if embedded is None:
        embedded = config.get("jang")
    if embedded is not None and not isinstance(embedded, dict):
        raise ValueError("qwen4_exp embedded JANG metadata must be an object")
    jang_path = model_path / "jang_config.json"
    sidecar = json.loads(jang_path.read_text()) if jang_path.is_file() else None
    if sidecar is not None and not isinstance(sidecar, dict):
        raise ValueError("qwen4_exp jang_config.json must contain an object")
    if sidecar is not None and embedded is not None and sidecar != embedded:
        raise ValueError(
            "qwen4_exp sidecar and embedded JANG metadata disagree"
        )
    jang_config = sidecar if sidecar is not None else embedded
    if jang_config is None:
        return config, frozenset(), None, False

    norm_convention = str(jang_config.get("norm_convention") or "").strip()
    if norm_convention != "runtime_plus1_applied":
        raise ValueError(
            "qwen4_exp JANG weights require norm_convention="
            "runtime_plus1_applied"
        )

    runtime_config, affine1_modules = prepare_affine1_runtime_config(
        config, jang_config
    )
    raw_bit_map = jang_config.get("bit_map")
    if raw_bit_map is None:
        return runtime_config, affine1_modules, None, True
    if not isinstance(raw_bit_map, Mapping):
        raise ValueError("qwen4_exp JANG bit_map must be an object")
    bit_map = copy.deepcopy(dict(raw_bit_map))
    default_spec = resolve_jang_bit_map_spec("__vmlx_default__", bit_map)
    quantization = runtime_config.get("quantization")
    if quantization is None:
        quantization = {}
    elif not isinstance(quantization, dict):
        raise ValueError("qwen4_exp config quantization must be an object")
    else:
        quantization = copy.deepcopy(quantization)
    for key in ("bits", "group_size", "mode"):
        quantization.setdefault(key, default_spec[key])
    runtime_config["quantization"] = quantization
    return runtime_config, affine1_modules, bit_map, True


def _normalize_jang_hyper_fold_dtypes(
    weights: dict[str, mx.array],
) -> tuple[int, int]:
    """Undo the Qwen4Exp AWQ-fold dtype leak without another value transform.

    The converter's stream fold historically multiplied retained BF16
    mix-down, injection, and grouped-norm tensors by F32 scales without casting
    back. The unaffected mix-up sibling retains the original floating dtype and
    is therefore authoritative. Require the complete F32 norm+down signature
    before changing anything so intentional F32 modules and packed projections
    remain untouched.
    """

    changed_matrices = 0
    changed_norms = 0
    for up_name, up_weight in tuple(weights.items()):
        if not up_name.endswith(_HYPER_UP_SUFFIX):
            continue
        target_dtype = up_weight.dtype
        if target_dtype not in _LOW_PRECISION_FLOAT_DTYPES:
            continue
        base = up_name[: -len(_HYPER_UP_SUFFIX)]
        norm_name = base + ".hc_norm.weight"
        down_name = base + ".input_mix_weight_down.weight"
        norm = weights.get(norm_name)
        down = weights.get(down_name)
        if (
            norm is None
            or down is None
            or norm.dtype != mx.float32
            or down.dtype != mx.float32
        ):
            continue
        weights[norm_name] = norm.astype(target_dtype)
        weights[down_name] = down.astype(target_dtype)
        changed_norms += 1
        changed_matrices += 1
        injection_name = base + ".block_inject_weight.weight"
        injection = weights.get(injection_name)
        if injection is not None and injection.dtype == mx.float32:
            weights[injection_name] = injection.astype(target_dtype)
            changed_matrices += 1
    return changed_matrices, changed_norms


def _classify_ple_tensor(key: str) -> tuple[str, str] | None:
    table_match = _PLE_TABLE_RE.search(key)
    if table_match is not None:
        return "table", table_match.group("suffix")
    buffer_match = _PLE_BUFFER_RE.search(key)
    if buffer_match is not None:
        return "buffer", buffer_match.group("suffix")
    return None


def _is_ple_table_tensor(key: str) -> bool:
    return _classify_ple_tensor(key) is not None


def _resolve_ple_module_key_format(weight_map: dict[str, str], n_shards: int) -> str:
    """Resolve the raw checkpoint's PLE prefix without assuming sanitization.

    Official and converted checkpoints have used both ``ple.ple_embedding``
    and the runtime's shorter ``ple`` nesting, with either ``model`` or
    ``language_model.model`` roots.  MLX converters also use both
    ``shard_N`` and ``shards.N`` spellings.  The index is authoritative:
    accept one complete layout and reject missing or ambiguous tables before
    loading.
    """
    if n_shards <= 0:
        raise ValueError("qwen4_exp split_ngram_parts must be positive")
    candidates: set[str] = set()
    for key in weight_map:
        match = _PLE_TABLE_RE.search(key)
        if match is None or match.group("suffix") != "weight":
            continue
        module_path = key.rsplit(".", 1)[0]
        marker = f"{match.group('layout')}{match.group('shard')}"
        if not module_path.endswith(marker):
            continue
        candidates.add(module_path[: -len(marker)] + match.group("layout") + "{}")

    complete = []
    for key_format in sorted(candidates):
        if all(
            f"{key_format.format(shard)}.{suffix}" in weight_map
            for shard in range(n_shards)
            for suffix in _PLE_REQUIRED_SHARD_SUFFIXES
        ):
            complete.append(key_format)
    if not complete:
        raise ValueError(
            "qwen4_exp bundle has no complete indexed PLE n-gram table for "
            f"{n_shards} shards"
        )
    if len(complete) != 1:
        raise ValueError(
            "qwen4_exp bundle has ambiguous PLE n-gram table layouts: "
            + ", ".join(complete)
        )
    return complete[0]


def _load_bfloat16_tensor(path: Path, key: str) -> mx.array:
    """Read one BF16 tensor when safetensors' MLX adapter cannot expose it."""
    with path.open("rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_len))
        info = header[key]
        if str(info["dtype"]).upper() != "BF16":
            raise TypeError(
                f"selective fallback only supports BF16, got {info['dtype']}"
            )
        shape = tuple(int(dim) for dim in info["shape"])
        start, end = (int(offset) for offset in info["data_offsets"])
        handle.seek(8 + header_len + start)
        raw = handle.read(end - start)
    bits = np.frombuffer(raw, dtype="<u2").copy().reshape(shape)
    fp32 = (bits.astype(np.uint32) << 16).view(np.float32)
    return mx.array(fp32).astype(mx.bfloat16)


def _load_non_table_weight_files(
    weight_files: list[Path],
    affine1_modules: frozenset[str],
    *,
    safe_open_fn,
) -> tuple[dict[str, mx.array], bool, dict[str, mx.array]]:
    """Selectively read non-PLE tensors; never request a 51B table tensor."""
    weights: dict[str, mx.array] = {}
    ple_buffers: dict[str, mx.array] = {}
    is_mlx_format = False
    for file_index, weight_file in enumerate(weight_files):
        with safe_open_fn(str(weight_file), framework="mlx") as handle:
            if file_index == 0:
                is_mlx_format = (handle.metadata() or {}).get("format") == "mlx"
            shard: dict[str, mx.array] = {}
            for key in handle.keys():
                classification = _classify_ple_tensor(str(key))
                if classification is not None and classification[0] == "table":
                    continue
                try:
                    value = handle.get_tensor(key)
                except TypeError as exc:
                    if "bfloat16" not in str(exc).lower():
                        raise
                    value = _load_bfloat16_tensor(weight_file, str(key))
                if classification is not None:
                    suffix = classification[1]
                    if suffix in ple_buffers:
                        raise ValueError(f"duplicate qwen4_exp PLE buffer {suffix}")
                    ple_buffers[suffix] = value
                    continue
                if key in weights or key in shard:
                    raise ValueError(f"duplicate qwen4_exp tensor {key}")
                shard[str(key)] = value
        if affine1_modules:
            affine_aliases = set(affine1_modules)
            for module_path in tuple(affine1_modules):
                affine_aliases.add(
                    module_path.replace(
                        "model.language_model", "language_model.model", 1
                    )
                )
                affine_aliases.add(
                    module_path.replace(
                        "language_model.model", "model.language_model", 1
                    )
                )
                affine_aliases.add(module_path.replace(".ple.ple_embedding.", ".ple."))
            shard, _expanded = expand_affine1_shard_mlx(
                shard, frozenset(affine_aliases)
            )
        weights.update(shard)
    return weights, is_mlx_format, ple_buffers


def _load_non_table_weights(
    model_path: Path,
    affine1_modules: frozenset[str],
) -> tuple[dict[str, mx.array], bool, dict[str, mx.array]]:
    from safetensors import safe_open

    weight_files = sorted(
        Path(path)
        for path in glob.glob(str(model_path / "*.safetensors"))
        if not path.endswith("consolidated.safetensors")
    )
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    return _load_non_table_weight_files(
        weight_files,
        affine1_modules,
        safe_open_fn=safe_open,
    )


def _validate_ple_hash_buffers(model, ple_buffers: dict[str, mx.array]) -> None:
    """Fail closed if bundle hash constants differ from the runtime hasher."""
    import numpy as np

    ple = model.language_model.model.layers[1].ple
    expected = {
        "layer_multipliers": ple.hasher.layer_multipliers,
        "ngram_heads_vocab_sizes": ple.hasher._head_sizes_np,
        "ngram_heads_offsets": ple.hasher._head_offsets_np,
    }
    for suffix, wanted in expected.items():
        value = ple_buffers.get(suffix)
        if value is None:
            raise ValueError(
                f"qwen4_exp bundle is missing required PLE buffer {suffix}"
            )
        mx.eval(value)
        actual = np.asarray(value).astype(np.int64, copy=False)
        if not np.array_equal(actual, wanted):
            raise ValueError(f"qwen4_exp PLE hash buffer mismatch: {suffix}")


def _normalize_runtime_weight_names(
    weights: dict[str, mx.array],
) -> dict[str, mx.array]:
    """Map checkpoint roots to the owning VLM wrapper without hiding extras.

    The official checkpoint is sanitized into wrapper-owned names, while the
    JANG converter writes already-sanitized MLX names without those wrapper
    segments: ``language_model.*``, ``visual.*``, ``lm_head.*``, and
    ``mtp.*``.  Normalize both forms to the instantiated VLM tree without
    hiding extras.  This is deliberately separate from value sanitization:
    JANG's ``runtime_plus1_applied`` stamp means shifted norms and convolution
    layouts must not be transformed a second time.
    """
    normalized: dict[str, mx.array] = {}

    def store(source_key: str, runtime_key: str, value: mx.array) -> None:
        if runtime_key in normalized:
            raise ValueError(
                "qwen4_exp weight-name collision after runtime normalization: "
                f"{source_key} -> {runtime_key}"
            )
        normalized[runtime_key] = value

    for key, value in weights.items():
        runtime_key = key
        if runtime_key.startswith("mtp."):
            runtime_key = f"language_model.{runtime_key}"
        elif runtime_key.startswith("visual."):
            runtime_key = runtime_key.replace("visual.", "vision_tower.", 1)
        elif runtime_key.startswith("lm_head."):
            runtime_key = runtime_key.replace(
                "lm_head.", "language_model.lm_head.", 1
            )
        elif runtime_key.startswith("language_model.") and not runtime_key.startswith(
            ("language_model.model.", "language_model.mtp.", "language_model.lm_head.")
        ):
            runtime_key = runtime_key.replace(
                "language_model.", "language_model.model.", 1
            )
        if (
            runtime_key == "vision_tower.patch_embed.proj.weight"
            and value.ndim == 5
            and value.shape[1] in (1, 3)
            and value.shape[-1] not in (1, 3)
        ):
            # HF Conv3d layout is [out, in, temporal, height, width]; MLX's
            # convolution kernel keeps input channels last.  The JANG converter
            # preserves the HF layout for this passthrough tensor, so perform
            # the same one-time transform used by the established VL converters.
            value = value.transpose(0, 2, 3, 4, 1)
        if runtime_key.endswith(".ple.conv1d.weight"):
            runtime_key = runtime_key.removesuffix(".conv1d.weight") + ".conv1d_weight"
            if value.ndim == 3:
                if value.shape[1] != 1:
                    raise ValueError(
                        "qwen4_exp PLE convolution must have a singleton input "
                        f"channel, got {value.shape}"
                    )
                value = value.squeeze(1)
            if value.ndim != 2:
                raise ValueError(
                    "qwen4_exp PLE convolution must resolve to [channels, taps], "
                    f"got {value.shape}"
                )
        store(key, runtime_key, value)

    # The converter already unfuses every backbone MoE, but preserves the
    # trained top-level MTP head before reaching that transform.  Normalize its
    # fused expert tensors here, including packed affine weight/scales/biases;
    # splitting the expert-intermediate axis preserves each quantization group
    # byte-for-byte.
    fused_suffix = ".experts.gate_up_proj.weight"
    fused_prefixes = [
        key.removesuffix(fused_suffix)
        for key in tuple(normalized)
        if key.startswith("language_model.mtp.") and key.endswith(fused_suffix)
    ]
    for prefix in fused_prefixes:
        gate_up_prefix = f"{prefix}.experts.gate_up_proj"
        down_prefix = f"{prefix}.experts.down_proj"
        for suffix in ("weight", "scales", "biases"):
            gate_up_key = f"{gate_up_prefix}.{suffix}"
            down_key = f"{down_prefix}.{suffix}"
            if gate_up_key not in normalized or down_key not in normalized:
                raise ValueError(
                    "qwen4_exp fused MTP expert tensors are incomplete for "
                    f"{prefix}.{suffix}"
                )
            gate_up = normalized.pop(gate_up_key)
            down = normalized.pop(down_key)
            if gate_up.ndim < 2 or gate_up.shape[-2] % 2:
                raise ValueError(
                    "qwen4_exp fused MTP gate/up tensor must have an even "
                    f"expert-intermediate axis, got {gate_up.shape}"
                )
            midpoint = gate_up.shape[-2] // 2
            store(
                gate_up_key,
                f"{prefix}.switch_mlp.gate_proj.{suffix}",
                gate_up[..., :midpoint, :],
            )
            store(
                gate_up_key,
                f"{prefix}.switch_mlp.up_proj.{suffix}",
                gate_up[..., midpoint:, :],
            )
            store(
                down_key,
                f"{prefix}.switch_mlp.down_proj.{suffix}",
                down,
            )
    return normalized


def _quantize_model(
    model,
    config: dict,
    weights: dict[str, mx.array],
    bit_map: Mapping[str, object] | None = None,
) -> None:
    quantization = config.get("quantization")
    if not isinstance(quantization, dict):
        return
    model_predicate = getattr(model, "quant_predicate", None)

    def predicate(path, module):
        aliases = (
            path,
            path.replace("language_model.model", "language_model", 1),
            path.replace("language_model.mtp", "mtp", 1),
        )
        checkpoint_quantized = any(
            f"{alias}.scales" in weights for alias in aliases
        )
        blocked_by_model = callable(model_predicate) and not model_predicate(
            path, module
        )
        if blocked_by_model:
            # JANG_4M explicitly stores 8-bit router gates.  They cannot be
            # treated as float weights or discarded; instantiate the matching
            # affine module only when the complete prequantized checkpoint
            # tensor is present. Recurrent coefficients remain forbidden.
            if not (
                bit_map is not None
                and checkpoint_quantized
                and path.endswith("mlp.gate")
            ):
                return False
        override = (
            resolve_jang_bit_map_spec(path, bit_map)
            if bit_map is not None
            else quantization.get(path)
        )
        if isinstance(override, dict):
            if checkpoint_quantized:
                return override
            return False
        if not hasattr(module, "to_quantized"):
            return False
        if hasattr(module, "weight") and module.weight.size % 64 != 0:
            return False
        return f"{path}.scales" in weights

    nn.quantize(
        model,
        group_size=int(quantization.get("group_size", 64)),
        bits=int(quantization.get("bits", 4)),
        mode=str(quantization.get("mode", "affine")),
        class_predicate=predicate,
    )


def load_qwen4_exp_vlm_model(model_path: str | Path, *, lazy: bool = False):
    """Load qwen4_exp without ever adding its packed 51B PLE table to weights."""
    from mlx_vlm.utils import (
        get_model_and_args,
        load_image_processor,
        load_processor,
        update_module_configs,
    )

    model_path = Path(model_path)
    if not register_qwen4_exp_runtime():
        raise RuntimeError("qwen4_exp MLX-VLM runtime registration failed")
    config, affine1_modules, bit_map, jang_runtime_sanitized = _load_runtime_config(
        model_path
    )
    model_class, _ = get_model_and_args(config)
    config.setdefault("text_config", config.pop("llm_config", {}))
    config.setdefault("vision_config", {})
    model_config = model_class.ModelConfig.from_dict(config)
    model_config = update_module_configs(
        model_config, model_class, config, ["text", "vision"]
    )
    model = model_class.Model(model_config)

    weights, is_mlx_format, ple_buffers = _load_non_table_weights(
        model_path, affine1_modules
    )
    if not (is_mlx_format or jang_runtime_sanitized):
        weights = model.sanitize(weights)
    weights = _normalize_runtime_weight_names(weights)
    normalized_hyper = (0, 0)
    if jang_runtime_sanitized:
        normalized_hyper = _normalize_jang_hyper_fold_dtypes(weights)
    _quantize_model(model, config, weights, bit_map)
    # The PLE table and its three hash buffers were intentionally removed from
    # the ordinary parameter tree above.  Everything else, including MTP and
    # vision, must match exactly; a permissive load previously allowed an
    # entire top-level MTP head to be silently ignored.
    model.load_weights(list(weights.items()), strict=True)

    from mlx_vlm.models.qwen4_exp.language import (
        compile_hyper_connections,
        fuse_hyper_connection_projections,
    )

    fused_hyper = fuse_hyper_connection_projections(model)
    compiled_hyper = compile_hyper_connections(model)

    from vmlx_engine.metal.qwen4_affine_moe_decode import install_qwen4_affine_moe

    install_qwen4_affine_moe(model)

    _validate_ple_hash_buffers(model, ple_buffers)
    weight_map = json.loads(
        (model_path / "model.safetensors.index.json").read_text()
    ).get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("qwen4_exp safetensors index has no weight_map")
    ngram_shards = int(model_config.text_config.split_ngram_parts)
    ple_key_format = _resolve_ple_module_key_format(weight_map, ngram_shards)
    ngram_heads = (int(model_config.text_config.ngram_size) - 1) * int(
        model_config.text_config.heads_per_ngram
    )
    if ngram_heads <= 0 or int(model_config.text_config.ple_embed_dim) % ngram_heads:
        raise ValueError(
            "qwen4_exp ple_embed_dim must divide evenly across n-gram heads"
        )
    ple_head_dim = int(model_config.text_config.ple_embed_dim) // ngram_heads
    table = FileBackedQuantizedNGramTable(
        model_path,
        ple_key_format,
        ngram_shards,
        expected_head_dim=ple_head_dim,
        bit_map=bit_map,
    )
    model.language_model.model.layers[1].ple.ngram_embedding.set_file_backed(table)

    if not lazy:
        mx.eval(model.parameters())
    model.eval()
    image_processor = load_image_processor(model_path)
    processor = load_processor(
        model_path,
        True,
        eos_token_ids=getattr(model.config, "eos_token_id", None),
    )
    if image_processor is not None:
        processor.image_processor = image_processor
    logger.info(
        "Loaded qwen4_exp with SSD-backed PLE (%s shards, MTP=%s, "
        "hyper_fused=%s, hyper_compiled=%s, hyper_dtype_normalized=%s/%s)",
        model_config.text_config.split_ngram_parts,
        model_config.text_config.mtp_num_hidden_layers,
        fused_hyper,
        compiled_hyper,
        normalized_hyper[0],
        normalized_hyper[1],
    )
    return model, processor
