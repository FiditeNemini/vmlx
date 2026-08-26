"""Memory-bounded qwen4_exp VLM loader with SSD-backed quantized PLE."""

from __future__ import annotations

import glob
import json
import logging
import re
import struct
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from vmlx_engine.utils.jang_affine_storage import (
    expand_affine1_shard_mlx,
    prepare_affine1_runtime_config,
)

from .register import register_qwen4_exp_runtime
from .table_reader import FileBackedQuantizedNGramTable

logger = logging.getLogger("vmlx_engine")

_PLE_TABLE_RE = re.compile(
    r"(?:^|\.)layers\.1\.ple\.(?:ple_embedding\.)?"
    r"ngram_embedding\.shard_(?P<shard>\d+)\."
    r"(?P<suffix>weight|scales|biases)$"
)
_PLE_BUFFER_RE = re.compile(
    r"(?:^|\.)layers\.1\.ple\.(?:ple_embedding\.)?"
    r"(?P<suffix>layer_multipliers|ngram_heads_vocab_sizes|"
    r"ngram_heads_offsets)$"
)
_PLE_REQUIRED_SHARD_SUFFIXES = ("weight", "scales", "biases")


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


def _load_runtime_config(model_path: Path) -> tuple[dict, frozenset[str]]:
    config = json.loads((model_path / "config.json").read_text())
    jang_path = model_path / "jang_config.json"
    if not jang_path.is_file():
        return config, frozenset()
    jang_config = json.loads(jang_path.read_text())
    return prepare_affine1_runtime_config(config, jang_config)


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
    ``language_model.model`` roots.  The index is authoritative: accept one
    complete layout and reject missing or ambiguous tables before loading.
    """
    if n_shards <= 0:
        raise ValueError("qwen4_exp split_ngram_parts must be positive")
    candidates: set[str] = set()
    for key in weight_map:
        match = _PLE_TABLE_RE.search(key)
        if match is None or match.group("suffix") != "weight":
            continue
        module_path = key.rsplit(".", 1)[0]
        marker = f"shard_{match.group('shard')}"
        if not module_path.endswith(marker):
            continue
        candidates.add(module_path[: -len(marker)] + "shard_{}")

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


def _quantize_model(model, config: dict, weights: dict[str, mx.array]) -> None:
    quantization = config.get("quantization")
    if not isinstance(quantization, dict):
        return
    model_predicate = getattr(model, "quant_predicate", None)

    def predicate(path, module):
        if callable(model_predicate) and not model_predicate(path, module):
            return False
        override = quantization.get(path)
        if isinstance(override, dict):
            return override
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
    config, affine1_modules = _load_runtime_config(model_path)
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
    if not is_mlx_format:
        weights = model.sanitize(weights)
    _quantize_model(model, config, weights)
    model.load_weights(list(weights.items()), strict=False)

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
        "Loaded qwen4_exp with SSD-backed PLE (%s shards, MTP=%s)",
        model_config.text_config.split_ngram_parts,
        model_config.text_config.mtp_num_hidden_layers,
    )
    return model, processor
