"""Normalize JANG affine quantization metadata to the model compute dtype.

JANG affine conversion stores per-group ``scales``/``biases`` as FP16 under
its current MLX ABI convention. A family whose activation contract is BF16
(GLM-5.3 Flash) then mixes FP16 metadata with BF16 activations inside every
quantized matmul, and MLX promotes FP16+BF16 to FP32 because neither 16-bit
format can represent the other. The FP32 result propagates through the whole
activation path and disqualifies every FP16/BF16-only fused decode kernel
(KDA conv/step, KDA grouped q/k/v, fused MoE pair), which is the dominant
cost of the promotion.

This helper casts affine ``scales``/``biases`` to the resolved 16-bit compute
dtype at the load boundary. It is a runtime representation fix only: packed
integer codes, codebook/JANGTQ metadata, non-affine modes, integer metadata,
and the on-disk bundle are never touched. Families whose compute contract is
FP16 (DSV4 asserts FP16 metadata) are unaffected because their metadata
already matches. Intentionally localized FP32 accumulators inside kernels are
out of scope: this changes operand dtypes, not accumulation contracts.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

logger = logging.getLogger("vmlx_engine")

_SIXTEEN_BIT = (mx.float16, mx.bfloat16)


def normalize_affine_metadata_dtype(model: Any, compute_dtype: Any) -> int:
    """Cast affine scales/biases to ``compute_dtype``; return modules changed.

    Only quantized modules with packed ``uint32`` codes, affine mode, and
    16-bit floating metadata are eligible. Metadata already matching the
    compute dtype is left untouched, as is anything that is not clearly the
    JANG affine layout.
    """

    if compute_dtype not in _SIXTEEN_BIT:
        return 0
    changed = 0
    pending: list[mx.array] = []
    for _name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        scales = getattr(module, "scales", None)
        biases = getattr(module, "biases", None)
        if weight is None or scales is None or biases is None:
            continue
        if not isinstance(weight, mx.array) or not isinstance(scales, mx.array):
            continue
        if not isinstance(biases, mx.array):
            continue
        if weight.dtype != mx.uint32:
            continue
        if str(getattr(module, "mode", "affine")) != "affine":
            continue
        if scales.dtype not in _SIXTEEN_BIT or biases.dtype not in _SIXTEEN_BIT:
            continue
        if scales.dtype == compute_dtype and biases.dtype == compute_dtype:
            continue
        module.scales = scales.astype(compute_dtype)
        module.biases = biases.astype(compute_dtype)
        pending.extend((module.scales, module.biases))
        changed += 1
    if pending:
        mx.eval(*pending)
        logger.info(
            "Normalized affine scales/biases to %s on %d quantized modules",
            compute_dtype,
            changed,
        )
    return changed


__all__ = ["normalize_affine_metadata_dtype"]
