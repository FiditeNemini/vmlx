# SPDX-License-Identifier: Apache-2.0
"""Opt-in GLM AR output norm with the native FP32 rounding boundaries.

This is not the shared approximate gated-RMS kernel. Match the qualified
MLX 0.32.2 width-128 sum tree, mean, epsilon, norm/weight multiplication,
sigmoid and final BF16 cast. Other shapes, runtimes, training and speculative
verification retain their existing path. No activations/state are retained.
"""

from functools import lru_cache
import importlib.metadata
import logging
import math

import mlx.core as mx

from .affine_moe_pair_decode import affine_moe_ar_scope_active

logger = logging.getLogger(__name__)
_FAILED = False
_OBSERVED = False
_CALL_COUNT = 0

_SOURCE = r"""
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    uint row = thread_position_in_grid.x / 32u;
    uint lane = thread_index_in_simdgroup;
    if (row >= 64u) return;
    size_t base = size_t(row) * 128u;
    float total = 0.0f;
    for (uint d = 0; d < 4; ++d) {
        float value = x[base + lane * 4u + d];
        float product = value * value;
        total = product + total;
    }
    total = metal::simd_sum(total);
    float inv = metal::precise::rsqrt(total / 128.0f + eps[0]);
    for (uint d = 0; d < 4; ++d) {
        uint col = lane * 4u + d;
        size_t i = base + col;
        float normalized = x[i] * inv;
        float weighted = float(weight[col]) * normalized;
        float g = float(gate[i]);
        float e = metal::precise::exp(metal::abs(g));
        float y = 1.0f / (1.0f + e);
        float sigmoid = g < 0.0f ? y : 1.0f - y;
        out[i] = bfloat(weighted * sigmoid);
    }
"""


@lru_cache(maxsize=1)
def _compatible_runtime() -> bool:
    try:
        return (importlib.metadata.version("mlx") == "0.32.2"
                and "Apple M5" in mx.device_info().get("device_name", ""))
    except (importlib.metadata.PackageNotFoundError, RuntimeError):
        return False


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="vmlx_glm5_exact_output_norm_v1",
        input_names=["x", "gate", "weight", "eps"], output_names=["out"],
        source=_SOURCE, ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=16)
def _epsilon(value: float):
    scalar = mx.array([value], dtype=mx.float32)
    mx.eval(scalar)
    return scalar


def glm5_output_norm(x, gate, weight, eps: float, *, output_dtype,
                     enabled: bool, training: bool = False):
    """Return the bounded AR result, or None for the original expression."""
    global _FAILED, _OBSERVED, _CALL_COUNT
    if (not enabled or training or _FAILED or not affine_moe_ar_scope_active()
            or x.shape != (1, 1, 64, 128) or x.dtype != mx.float32
            or gate.shape != x.shape or gate.dtype != mx.bfloat16
            or weight.shape != (128,) or weight.dtype != mx.bfloat16
            or output_dtype != mx.bfloat16
            or not math.isfinite(eps) or eps <= 0
            or mx.default_device() != mx.gpu or not mx.metal.is_available()
            or not _compatible_runtime()):
        return None
    try:
        result = _kernel()(
            inputs=[x, gate, weight, _epsilon(float(eps))],
            grid=(64 * 32, 1, 1), threadgroup=(128, 1, 1),
            output_shapes=[x.shape], output_dtypes=[mx.bfloat16],
        )[0]
        if not _OBSERVED:
            mx.eval(result)
            _OBSERVED = True
            logger.info("GLM exact output norm active: heads=64 width=128 "
                        "input=fp32 gate=bf16 weight=bf16 output=bf16 "
                        "scope=productive_ar math=mlx0322_fp32_row128_bf16_v1")
        _CALL_COUNT += 1
        return result
    except (RuntimeError, ValueError):
        _FAILED = True
        logger.exception("GLM output norm failed; retaining stock FP32 expression")
        return None
