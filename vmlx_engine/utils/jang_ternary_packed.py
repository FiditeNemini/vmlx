"""Runtime bridge for JANG ternary bundles stored at PTQ1_0 density.

Storage ``ternary_packed_26b``: per 128-weight group, 26 bytes of base-3
packed trits (5 per byte) plus one float16 scale; no stored bias because a
ternary group's bias is always ``-scale``. At load time each declared module
is expanded losslessly into exactly the uint32 2-bit codes, float16 scales and
float16 biases of a plain affine 2-bit bundle, so the same MLX kernels run and
the output is identical. Only the bytes on disk change (1.75 bits/weight).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import mlx.core as mx

STORAGE_NAME = "ternary_packed_26b"
GROUP = 128
BYTES_PER_GROUP = 26
RUNTIME_BITS = 2


def ternary_packed_modules(jang_config: Mapping[str, Any]) -> frozenset[str]:
    """Return module paths whose manifest entry declares packed-trit storage."""
    quantization = jang_config.get("quantization") if isinstance(jang_config, Mapping) else None
    manifest = quantization.get("tensor_quantization_manifest") if isinstance(quantization, Mapping) else None
    runtime = jang_config.get("runtime", {}) if isinstance(jang_config, Mapping) else {}
    required = isinstance(runtime, Mapping) and runtime.get("requires_jang_ternary_packed_expansion", False)
    manifest = manifest if isinstance(manifest, Mapping) else {}
    modules = frozenset(
        str(path)
        for path, spec in manifest.items()
        if isinstance(spec, Mapping) and spec.get("storage") == STORAGE_NAME
    )
    if required and not modules:
        raise ValueError("packed ternary runtime required but no storage modules declared")
    for path in modules:
        entry = manifest[path]
        if (entry.get("runtime_bits", entry.get("bits")) != RUNTIME_BITS
                or entry.get("group_size") != GROUP or entry.get("mode", "affine") != "affine"):
            raise ValueError(f"unsupported packed ternary quantization for {path}")
    return modules


def expand_ternary_packed_mlx(packed, scales):
    """uint8 [rows, groups*26] + float16 scales -> (uint32 2-bit words, scales, biases)."""
    if packed.dtype != mx.uint8 or packed.ndim != 2:
        raise ValueError(f"ternary_packed weight must be 2-D uint8, got {packed.dtype} {packed.shape}")
    rows, cols = packed.shape
    if not rows or not cols or cols % BYTES_PER_GROUP:
        raise ValueError(f"ternary_packed columns {cols} not divisible by {BYTES_PER_GROUP}")
    groups = cols // BYTES_PER_GROUP
    if tuple(scales.shape) != (rows, groups):
        raise ValueError(f"scales shape {scales.shape} != {(rows, groups)}")
    if scales.dtype != mx.float16:
        raise ValueError("packed ternary scales must be float16 (no implicit precision conversion)")
    # Bound the trit intermediate, particularly the vocabulary-sized lm_head.
    # Eagerly materialize each block so lazy graphs do not retain every expanded
    # [rows, input_width] temporary until the final concatenation.
    rows_per_chunk = max(1, (1 << 20) // (groups * GROUP))
    if rows > rows_per_chunk:
        chunks = []
        for start in range(0, rows, rows_per_chunk):
            words, _, _ = expand_ternary_packed_mlx(
                packed[start:start + rows_per_chunk], scales[start:start + rows_per_chunk]
            )
            mx.eval(words)
            chunks.append(words)
        words = mx.concatenate(chunks, axis=0)
        biases = -scales
        mx.eval(words, biases)
        return words, scales, biases
    b = packed.astype(mx.uint32).reshape(rows, groups, BYTES_PER_GROUP)
    head = b[:, :, :25]
    tail = b[:, :, 25:26]
    if not (mx.all(head <= 242) & mx.all(tail <= 26) & mx.all(mx.isfinite(scales))).item():
        raise ValueError("invalid packed ternary trits or non-finite scales")
    head_codes = mx.stack(
        [mx.remainder(mx.floor_divide(head, 3**k), 3) for k in range(5)], axis=-1
    ).reshape(rows, groups, 125)
    tail_codes = mx.stack(
        [mx.remainder(mx.floor_divide(tail, 3**k), 3) for k in range(3)], axis=-1
    ).reshape(rows, groups, 3)
    codes = mx.concatenate([head_codes, tail_codes], axis=-1).reshape(rows, groups * GROUP // 16, 16)
    shifts = mx.array([2 * lane for lane in range(16)], dtype=mx.uint32)
    words = mx.sum(mx.left_shift(codes, shifts), axis=-1).astype(mx.uint32)
    return words, scales, -scales


def expand_ternary_packed_shard_mlx(
    weights: Mapping[str, Any],
    storage_modules: frozenset[str] | set[str],
) -> tuple[dict[str, Any], int]:
    """Expand every declared module present in one loaded shard."""
    if not storage_modules:
        return dict(weights), 0
    expanded = dict(weights)
    count = 0
    for module_path in storage_modules:
        weight_key = f"{module_path}.weight"
        value = expanded.get(weight_key)
        if value is None:
            continue
        scales = expanded.get(f"{module_path}.scales")
        if scales is None:
            raise ValueError(f"ternary_packed module {module_path!r} is missing its scales")
        if f"{module_path}.biases" in expanded:
            raise ValueError(f"ternary_packed module {module_path!r} must not store biases")
        words, scales16, biases16 = expand_ternary_packed_mlx(value, scales)
        expanded[weight_key] = words
        expanded[f"{module_path}.scales"] = scales16
        expanded[f"{module_path}.biases"] = biases16
        count += 1
    return expanded, count
