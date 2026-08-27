"""Row-addressable safetensors reader for Qwen4-Exp's quantized PLE table."""

from __future__ import annotations

import fnmatch
import json
import struct
from collections.abc import Mapping
from pathlib import Path

import mlx.core as mx
import numpy as np

from vmlx_engine.utils.jang_affine_storage import expand_packed_1bit_to_2bit_mlx

_DTYPES = {
    "BF16": np.uint16,
    "F16": np.float16,
    "F32": np.float32,
    "U32": np.uint32,
    "I64": np.int64,
}

_MLX_DTYPES = {
    "BF16": mx.bfloat16,
    "F16": mx.float16,
    "F32": mx.float32,
    "U32": mx.uint32,
    "I64": mx.int64,
}


def resolve_jang_bit_map_spec(
    module_path: str,
    bit_map: Mapping[str, object],
) -> dict:
    """Resolve one converter-style wildcard/prefix quantization rule.

    The Qwen4-Exp converter matches the original tensor name (including its
    ``.weight`` suffix), while the MLX quantizer and PLE reader operate on
    module paths.  Accept both spellings plus the runtime's narrow wrapper
    aliases, choose the longest matching pattern, and reject equally-specific
    conflicting rules instead of depending on JSON insertion order.
    """
    if not isinstance(bit_map, Mapping):
        raise ValueError("qwen4_exp JANG bit_map must be an object")
    default = bit_map.get("default")
    if not isinstance(default, Mapping):
        raise ValueError("qwen4_exp JANG bit_map requires an object default")

    aliases = [
        str(module_path),
        str(module_path).replace("language_model.model", "language_model", 1),
        str(module_path).replace("language_model.mtp", "mtp", 1),
        str(module_path).replace("language_model.lm_head", "lm_head", 1),
    ]
    aliases.extend(f"{alias}.weight" for alias in tuple(aliases))
    aliases = list(dict.fromkeys(aliases))

    matches: list[tuple[int, str, Mapping[str, object]]] = []
    for raw_pattern, raw_spec in bit_map.items():
        pattern = str(raw_pattern)
        if pattern == "default":
            continue
        if not isinstance(raw_spec, Mapping):
            raise ValueError(
                f"qwen4_exp JANG bit_map rule {pattern!r} must be an object"
            )
        if any(
            fnmatch.fnmatch(alias, pattern)
            or alias.startswith(pattern)
            or fnmatch.fnmatch(alias, pattern + "*")
            for alias in aliases
        ):
            matches.append((len(pattern), pattern, raw_spec))

    selected: Mapping[str, object] = default
    if matches:
        best_length = max(length for length, _pattern, _spec in matches)
        best = [item for item in matches if item[0] == best_length]
        selected = best[0][2]
        conflicts = [pattern for _length, pattern, spec in best[1:] if spec != selected]
        if conflicts:
            raise ValueError(
                "conflicting equally-specific qwen4_exp JANG bit_map rules for "
                f"{module_path}: {best[0][1]}, " + ", ".join(conflicts)
            )

    spec = dict(selected)
    try:
        bits = int(spec["bits"])
        group_size = int(spec["group_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid qwen4_exp JANG quantization rule for {module_path}"
        ) from exc
    if bits not in {1, 2, 3, 4, 5, 6, 8} or group_size <= 0:
        raise ValueError(
            f"invalid qwen4_exp JANG quantization rule for {module_path}: "
            f"bits={bits}, group_size={group_size}"
        )
    spec.update({"bits": bits, "group_size": group_size})
    spec.setdefault("mode", "affine")
    return spec


def _module_aliases(module_path: str) -> tuple[str, ...]:
    """Return stable official/runtime aliases without set-order ambiguity."""
    candidates = [
        module_path,
        module_path.replace("model.language_model", "language_model.model", 1),
        module_path.replace("language_model.model", "model.language_model", 1),
        module_path.replace(".ple.ple_embedding.", ".ple."),
    ]
    candidates.extend(
        candidate.replace(".ple.ple_embedding.", ".ple.")
        for candidate in tuple(candidates)
    )
    return tuple(dict.fromkeys(candidates))


def _unique_mapping_override(
    mapping: dict,
    aliases: tuple[str, ...],
    *,
    label: str,
) -> dict | None:
    """Resolve one semantic module override and reject conflicting aliases."""
    matches = [
        (alias, mapping[alias])
        for alias in aliases
        if isinstance(mapping.get(alias), dict)
    ]
    if not matches:
        return None
    first_alias, first_value = matches[0]
    conflicts = [alias for alias, value in matches[1:] if value != first_value]
    if conflicts:
        raise ValueError(
            f"conflicting {label} aliases for {first_alias}: " + ", ".join(conflicts)
        )
    return dict(first_value)


def _validate_affine_layout(
    *,
    weight_shape: tuple[int, ...],
    weight_dtype: str,
    scales_shape: tuple[int, ...],
    biases_shape: tuple[int, ...],
    group_size: int,
    logical_bits: int,
    storage_bits: int,
    head_dim: int,
) -> None:
    if storage_bits not in {1, 2, 3, 4, 5, 6, 8}:
        raise ValueError(f"unsupported packed PLE storage_bits={storage_bits}")
    if logical_bits not in {1, 2, 3, 4, 5, 6, 8}:
        raise ValueError(f"unsupported PLE logical bits={logical_bits}")
    if storage_bits != 1 and storage_bits != logical_bits:
        raise ValueError(
            "PLE storage bits must equal logical bits unless using the lossless "
            f"affine-1 expansion: storage={storage_bits}, logical={logical_bits}"
        )
    if group_size <= 0 or head_dim <= 0:
        raise ValueError("PLE group size and head dimension must be positive")
    if weight_dtype != "U32":
        raise ValueError(f"PLE packed weight must be U32, got {weight_dtype}")
    if scales_shape != biases_shape:
        raise ValueError(
            "PLE scales and biases must have identical shapes: "
            f"{scales_shape} != {biases_shape}"
        )
    if len(weight_shape) != 2 or len(scales_shape) != 2:
        raise ValueError("PLE weight/scales/biases must all be rank 2")
    if not (weight_shape[0] == scales_shape[0] == biases_shape[0]):
        raise ValueError("PLE weight/scales/biases row counts differ")

    runtime_bits = 2 if storage_bits == 1 else logical_bits
    expected_storage_cols = (head_dim * storage_bits + 31) // 32
    expected_runtime_cols = (head_dim * runtime_bits + 31) // 32
    expanded_cols = weight_shape[1] * (2 if storage_bits == 1 else 1)
    expected_scale_cols = (head_dim + group_size - 1) // group_size
    if weight_shape[1] != expected_storage_cols:
        raise ValueError(
            "PLE packed width does not match the configured head dimension: "
            f"got={weight_shape[1]}, expected={expected_storage_cols}"
        )
    if expanded_cols != expected_runtime_cols:
        raise ValueError(
            "PLE runtime packed width does not match affine expansion: "
            f"got={expanded_cols}, expected={expected_runtime_cols}"
        )
    if scales_shape[1] != expected_scale_cols:
        raise ValueError(
            "PLE scale width does not match the configured head dimension: "
            f"got={scales_shape[1]}, expected={expected_scale_cols}"
        )


class SafetensorsRowReader:
    """Memory-map one two-dimensional safetensors tensor and fetch rows only."""

    def __init__(self, path: str | Path, tensor_name: str):
        path = Path(path)
        with path.open("rb") as handle:
            (header_len,) = struct.unpack("<Q", handle.read(8))
            header = json.loads(handle.read(header_len))
        info = header[tensor_name]
        self.dtype_tag = info["dtype"]
        if self.dtype_tag not in _DTYPES:
            raise ValueError(f"unsupported PLE tensor dtype {self.dtype_tag!r}")
        self.shape = tuple(info["shape"])
        if len(self.shape) != 2:
            raise ValueError(f"PLE tensor must be rank 2, got {self.shape}")
        start, _end = info["data_offsets"]
        self.mm = np.memmap(
            path,
            dtype=_DTYPES[self.dtype_tag],
            mode="r",
            offset=8 + header_len + start,
            shape=self.shape,
        )

    def rows(self, indices: np.ndarray) -> np.ndarray:
        values = self.mm[indices]
        if self.dtype_tag == "BF16":
            raw = np.asarray(values, dtype=np.uint16)
            return (raw.astype(np.uint32) << 16).view(np.float32)
        return np.asarray(values)

    @property
    def mlx_dtype(self):
        return _MLX_DTYPES[self.dtype_tag]

    def mlx_rows(self, indices: np.ndarray) -> mx.array:
        """Return rows with the safetensors dtype preserved for MLX kernels."""
        values = self.rows(indices)
        return mx.array(values).astype(self.mlx_dtype)


class _AffineShard:
    def __init__(
        self,
        model_dir: Path,
        weight_map: dict[str, str],
        module_path: str,
        quant_spec: dict,
        storage_bits: int | None,
        expected_head_dim: int,
    ):
        def reader(suffix: str) -> SafetensorsRowReader:
            key = f"{module_path}.{suffix}"
            return SafetensorsRowReader(model_dir / weight_map[key], key)

        self.weight = reader("weight")
        self.scales = reader("scales")
        self.biases = reader("biases")
        self.group_size = int(quant_spec["group_size"])
        self.logical_bits = int(quant_spec["bits"])
        self.storage_bits = int(storage_bits or self.logical_bits)
        self.mode = str(quant_spec.get("mode", "affine"))
        self.head_dim = int(expected_head_dim)
        if self.mode != "affine":
            raise ValueError(f"PLE row reader requires affine mode, got {self.mode!r}")
        if self.scales.dtype_tag != self.biases.dtype_tag:
            raise ValueError(
                "PLE scales and biases must have identical dtypes: "
                f"{self.scales.dtype_tag} != {self.biases.dtype_tag}"
            )
        if self.scales.dtype_tag not in {"BF16", "F16", "F32"}:
            raise ValueError(
                f"PLE affine parameters require a floating dtype, got {self.scales.dtype_tag}"
            )
        _validate_affine_layout(
            weight_shape=self.weight.shape,
            weight_dtype=self.weight.dtype_tag,
            scales_shape=self.scales.shape,
            biases_shape=self.biases.shape,
            group_size=self.group_size,
            logical_bits=self.logical_bits,
            storage_bits=self.storage_bits,
            head_dim=self.head_dim,
        )

    @property
    def rows_count(self) -> int:
        return self.weight.shape[0]

    @property
    def output_dtype(self):
        return self.scales.mlx_dtype

    def gather_mlx(self, rows: np.ndarray) -> mx.array:
        packed = self.weight.mlx_rows(rows)
        runtime_bits = self.logical_bits
        if self.storage_bits == 1:
            packed = expand_packed_1bit_to_2bit_mlx(packed)
            runtime_bits = 2
        scales = self.scales.mlx_rows(rows)
        biases = self.biases.mlx_rows(rows)
        values = mx.dequantize(
            packed,
            scales,
            biases,
            group_size=self.group_size,
            bits=runtime_bits,
            mode=self.mode,
            dtype=self.output_dtype,
        )
        if values.shape[-1] != self.head_dim:
            raise ValueError(
                "PLE dequantized row width differs from the model contract: "
                f"got={values.shape[-1]}, expected={self.head_dim}"
            )
        return values

    def gather(self, rows: np.ndarray) -> np.ndarray:
        values = self.gather_mlx(rows)
        mx.eval(values)
        return np.asarray(values).astype(np.float32, copy=False)


class FileBackedQuantizedNGramTable:
    """Gather/dequantize only requested PLE rows from its 128 SSD shards."""

    def __init__(
        self,
        model_dir: str | Path,
        module_key_format: str,
        n_shards: int,
        expected_head_dim: int,
        index_name: str = "model.safetensors.index.json",
        bit_map: Mapping[str, object] | None = None,
    ):
        model_dir = Path(model_dir)
        config = json.loads((model_dir / "config.json").read_text())
        weight_map = json.loads((model_dir / index_name).read_text())["weight_map"]
        quantization = config.get("quantization") or {}
        default_spec = {
            "bits": quantization.get("bits", 4),
            "group_size": quantization.get("group_size", 64),
            "mode": quantization.get("mode", "affine"),
        }

        storage_manifest = {}
        jang_path = model_dir / "jang_config.json"
        jang = None
        if jang_path.is_file():
            jang = json.loads(jang_path.read_text())
        elif isinstance(config.get("jang_config"), dict):
            jang = config["jang_config"]
        elif isinstance(config.get("jang"), dict):
            jang = config["jang"]
        if isinstance(jang, dict):
            jq = jang.get("quantization") or {}
            storage_manifest = jq.get("tensor_quantization_manifest") or {}

        if n_shards <= 0:
            raise ValueError("PLE n_shards must be positive")
        self.shards = []
        for shard_index in range(n_shards):
            module_path = module_key_format.format(shard_index)
            spec = (
                resolve_jang_bit_map_spec(module_path, bit_map)
                if bit_map is not None
                else dict(default_spec)
            )
            aliases = _module_aliases(module_path)
            override = _unique_mapping_override(
                quantization,
                aliases,
                label="PLE quantization",
            )
            if isinstance(override, dict):
                spec.update(override)
            manifest_spec = _unique_mapping_override(
                storage_manifest,
                aliases,
                label="PLE storage manifest",
            )
            storage_bits = None
            if isinstance(manifest_spec, dict):
                storage_bits = manifest_spec.get(
                    "storage_bits", manifest_spec.get("bits")
                )
            self.shards.append(
                _AffineShard(
                    model_dir,
                    weight_map,
                    module_path,
                    spec,
                    storage_bits,
                    expected_head_dim,
                )
            )
        self.per = self.shards[0].rows_count
        self.head_dim = int(expected_head_dim)
        self.output_dtype = self.shards[0].output_dtype
        if self.per <= 0:
            raise ValueError("PLE shard 0 is empty")
        for shard_index, shard in enumerate(self.shards):
            if shard.head_dim != self.head_dim:
                raise ValueError(f"PLE shard {shard_index} head dimension differs")
            if shard.output_dtype != self.output_dtype:
                raise ValueError(f"PLE shard {shard_index} output dtype differs")
            if shard_index < len(self.shards) - 1 and shard.rows_count != self.per:
                raise ValueError(
                    f"PLE shard {shard_index} has {shard.rows_count} rows; "
                    f"expected {self.per}"
                )
            if shard_index == len(self.shards) - 1 and not (
                0 < shard.rows_count <= self.per
            ):
                raise ValueError("PLE final shard row count is invalid")
        self.total_rows = sum(shard.rows_count for shard in self.shards)

    def gather(self, flat_rows: np.ndarray) -> np.ndarray:
        values = self.gather_mlx(flat_rows)
        mx.eval(values)
        return np.asarray(values).astype(np.float32, copy=False)

    def gather_mlx(self, flat_rows: np.ndarray) -> mx.array:
        """Gather random SSD rows and keep dequantized values on the MLX path."""
        flat_rows = np.asarray(flat_rows, dtype=np.int64).reshape(-1)
        if flat_rows.size and int(flat_rows.min()) < 0:
            raise IndexError("PLE row must be non-negative")
        if flat_rows.size and int(flat_rows.max()) >= self.total_rows:
            raise IndexError("PLE row exceeds the configured n-gram table")
        shard_indices = flat_rows // self.per
        local_rows = flat_rows % self.per
        out = mx.zeros(
            (flat_rows.size, self.head_dim), dtype=self.output_dtype
        )
        for shard_index in np.unique(shard_indices):
            selected = np.nonzero(shard_indices == shard_index)[0]
            selected_mx = mx.array(selected.astype(np.uint32))
            out[selected_mx] = self.shards[int(shard_index)].gather_mlx(
                local_rows[selected]
            )
        return out
