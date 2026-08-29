"""Small-row fused sigmoid-gated RMSNorm for hybrid decode paths.

Qwen3.8-Flash-Next GDN and GLM-5.3 KDA both finish their recurrent update
with the same independent per-head epilogue::

    rms_norm(x, weight, eps) * sigmoid(gate)

At autoregressive and native-MTP verification widths the eager expression
spends more time materializing intermediates and launching kernels than it
does on arithmetic.  This Metal kernel keeps the fp32 reduction and gate math
in one dispatch.  It deliberately owns only batch-one, one-to-eight-row
shapes; prefill and every unsupported dtype/geometry stay on the stock MLX
path.

The lane is diagnostic by default.  ``VMLX_FUSED_GATED_RMSNORM=1`` enables
it at model construction time; a same-bundle full-model A/B must demonstrate
an answer-preserving median win before the default changes.
"""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx


_MAX_ROWS = 8
_KERNEL: Any | None = None
_EPS_SCALARS: dict[float, mx.array] = {}

_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

_SOURCE = """
    uint tid = thread_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    uint row = tid / 32u;
    if (row >= uint(ROWS)) return;

    size_t base = (size_t)row * (size_t)D;
    float sumsq = 0.0f;
    for (uint col = lane; col < uint(D); col += 32u) {
        float value = float(x[base + col]);
        sumsq += value * value;
    }
    sumsq = simd_sum(sumsq);
    float inv_rms = metal::rsqrt(sumsq / float(D) + float(eps[0]));

    for (uint col = lane; col < uint(D); col += 32u) {
        float gate_value = float(gate[base + col]);
        float magnitude = metal::exp(-metal::abs(gate_value));
        float sigmoid = gate_value < 0.0f
            ? magnitude / (1.0f + magnitude)
            : 1.0f / (1.0f + magnitude);
        float normalized = float(x[base + col]) * inv_rms * float(weight[col]);
        out[base + col] = T(normalized * sigmoid);
    }
"""


def fused_gated_rmsnorm_requested() -> bool:
    """Resolve the process-start diagnostic switch once per model instance."""

    value = os.environ.get("VMLX_FUSED_GATED_RMSNORM", "0").strip().lower()
    return value not in {"", "0", "false", "off", "no"}


def _kernel() -> Any:
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="vmlx_sigmoid_gated_rmsnorm_small_rows",
            input_names=["x", "gate", "weight", "eps"],
            output_names=["out"],
            header=_HEADER,
            source=_SOURCE,
        )
    return _KERNEL


def _eps_scalar(value: float) -> mx.array:
    scalar = _EPS_SCALARS.get(value)
    if scalar is None:
        scalar = mx.array([value], dtype=mx.float32)
        mx.eval(scalar)
        _EPS_SCALARS[value] = scalar
    return scalar


def sigmoid_gated_rmsnorm_small_rows(
    x: mx.array,
    gate: mx.array,
    weight: mx.array,
    eps: float,
    *,
    output_dtype: mx.Dtype | None = None,
    enabled: bool,
) -> mx.array | None:
    """Return the fused epilogue, or ``None`` for the stock-path fallback."""

    if not enabled or x.ndim != 4 or gate.shape != x.shape:
        return None
    batch, rows, heads, dims = (int(value) for value in x.shape)
    if batch != 1 or rows < 1 or rows > _MAX_ROWS or heads < 1 or dims < 1:
        return None
    if weight.ndim != 1 or int(weight.shape[0]) != dims:
        return None
    supported = (mx.float16, mx.bfloat16, mx.float32)
    if x.dtype not in supported or gate.dtype not in supported or weight.dtype not in supported:
        return None
    dtype = x.dtype if output_dtype is None else output_dtype
    if dtype not in supported:
        return None

    flat_rows = rows * heads
    return _kernel()(
        inputs=[x, gate, weight, _eps_scalar(float(eps))],
        template=[("T", dtype), ("D", dims), ("ROWS", flat_rows)],
        grid=(32 * flat_rows, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[tuple(x.shape)],
        output_dtypes=[dtype],
    )[0]


__all__ = [
    "fused_gated_rmsnorm_requested",
    "sigmoid_gated_rmsnorm_small_rows",
]
