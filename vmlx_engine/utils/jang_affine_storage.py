"""Runtime bridge for JANG affine tensors stored with one-bit codes.

JANG affine always dequantizes as ``code * scale + bias``. MLX does not
currently expose a one-bit QuantizedLinear, so an affine-1 artifact keeps its
honest one-bit storage on disk and widens only the packed codes to two-bit
slots while loading. Codes, scales, and biases are otherwise unchanged.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


AFFINE1_RUNTIME_BITS = 2
AFFINE1_STORAGE_BITS = 1


def affine1_storage_modules(jang_config: Mapping[str, Any]) -> frozenset[str]:
    """Return module paths declared as one-bit affine storage."""
    quantization = jang_config.get("quantization")
    if not isinstance(quantization, Mapping):
        return frozenset()
    manifest = quantization.get("tensor_quantization_manifest")
    if not isinstance(manifest, Mapping):
        return frozenset()
    return frozenset(
        str(module_path)
        for module_path, spec in manifest.items()
        if isinstance(module_path, str)
        and isinstance(spec, Mapping)
        and int(spec.get("storage_bits", spec.get("bits", 0)) or 0)
        == AFFINE1_STORAGE_BITS
    )


def prepare_affine1_runtime_config(
    config: Mapping[str, Any],
    jang_config: Mapping[str, Any],
) -> tuple[dict[str, Any], frozenset[str]]:
    """Return an in-memory config using MLX two-bit modules for affine-1."""
    modules = affine1_storage_modules(jang_config)
    runtime_config = copy.deepcopy(dict(config))
    if not modules:
        return runtime_config, modules

    quantization = runtime_config.get("quantization")
    if not isinstance(quantization, dict):
        raise ValueError("JANG affine-1 bundle is missing config.json quantization")

    if int(quantization.get("bits", 0) or 0) == AFFINE1_STORAGE_BITS:
        quantization["bits"] = AFFINE1_RUNTIME_BITS
    for module_path in modules:
        spec = quantization.get(module_path)
        if not isinstance(spec, dict):
            raise ValueError(
                f"JANG affine-1 module {module_path!r} is missing its config override"
            )
        storage_bits = int(spec.get("storage_bits", spec.get("bits", 0)) or 0)
        if storage_bits != AFFINE1_STORAGE_BITS:
            raise ValueError(
                f"JANG affine-1 module {module_path!r} has storage_bits={storage_bits}"
            )
        spec["storage_bits"] = AFFINE1_STORAGE_BITS
        spec["bits"] = AFFINE1_RUNTIME_BITS
        spec["mode"] = "affine"

    quantization["runtime_expansion"] = {
        "storage_bits": AFFINE1_STORAGE_BITS,
        "runtime_bits": AFFINE1_RUNTIME_BITS,
        "lossless": True,
    }
    runtime_config["quantization"] = quantization
    return runtime_config, modules


def expand_packed_1bit_to_2bit_mlx(packed):
    """Widen uint32-packed one-bit codes to identical two-bit code values."""
    import mlx.core as mx

    source = packed.astype(mx.uint32)
    low = mx.zeros_like(source)
    high = mx.zeros_like(source)
    for index in range(16):
        low = mx.bitwise_or(
            low,
            mx.left_shift(
                mx.bitwise_and(mx.right_shift(source, index), 1), 2 * index
            ),
        )
        high = mx.bitwise_or(
            high,
            mx.left_shift(
                mx.bitwise_and(mx.right_shift(source, index + 16), 1),
                2 * index,
            ),
        )
    return mx.reshape(
        mx.stack((low, high), axis=-1),
        (*source.shape[:-1], source.shape[-1] * 2),
    )


def expand_affine1_shard_mlx(
    weights: Mapping[str, Any],
    storage_modules: frozenset[str] | set[str],
) -> tuple[dict[str, Any], int]:
    """Expand each indexed affine-1 weight present in one loaded shard."""
    if not storage_modules:
        return dict(weights), 0
    expanded = dict(weights)
    count = 0
    for module_path in storage_modules:
        weight_key = f"{module_path}.weight"
        value = expanded.get(weight_key)
        if value is None:
            continue
        expanded[weight_key] = expand_packed_1bit_to_2bit_mlx(value)
        count += 1
    return expanded, count
