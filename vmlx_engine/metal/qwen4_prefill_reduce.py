"""Experimental FP16 Flash prefill epilogue; native projections are unchanged."""

from __future__ import annotations

import copy
import functools
import importlib.metadata
import os

import mlx.core as mx
from mlx_lm.models.switch_layers import (
    QuantizedSwitchLinear,
    SwitchGLU,
    _gather_sort,
)


_STATUS = {
    "observation": "submitted_only",
    "dispatch_calls": 0,
    "dispatched_rows": 0,
    "maximum_rows": 0,
}


def status() -> dict:
    """Detached process-lifetime submission metadata, not GPU completion."""
    return {**copy.deepcopy(_STATUS), "requested": requested()}


def _record_submission(switch, x, scores, output):
    rows = x.size // 2560
    _STATUS.update(
        dispatch_calls=_STATUS["dispatch_calls"] + 1,
        dispatched_rows=_STATUS["dispatched_rows"] + rows,
        maximum_rows=max(_STATUS["maximum_rows"], rows),
        last_rows=rows,
        last_input_dtype=str(x.dtype),
        last_score_dtype=str(scores.dtype),
        last_output_dtype=str(output.dtype),
        projections={
            name: {"bits": getattr(switch, name).bits,
                   "group_size": getattr(switch, name).group_size}
            for name in ("up_proj", "gate_proj", "down_proj")
        },
    )


def requested() -> bool:
    return os.environ.get("VMLX_QWEN4_PREFILL_REDUCE", "0") == "1"


def _available() -> bool:
    try:
        return (
            importlib.metadata.version("mlx") == "0.32.3"
            and mx.metal.is_available()
            and mx.default_device() == mx.gpu
            and mx.device_info().get("device_name") == "Apple M5 Max"
        )
    except (importlib.metadata.PackageNotFoundError, RuntimeError):
        return False


@functools.lru_cache(maxsize=1)
def _kernel():
    # Native col_reduce_small has eight row lanes for top10. Each multiply
    # rounds to FP16 before the FP16 lane accumulation and sequential merge.
    # This candidate still requires on-device raw-bit qualification.
    return mx.fast.metal_kernel(
        name="vmlx_qwen4_prefill_reduce_f16_top10_v1",
        input_names=["Y", "INV", "S"],
        output_names=["OUT"],
        compile_options={"math_mode": "safe"},
        source=r"""
const uint i = thread_position_in_grid.x;
if (i >= ROWS * 2560) return;
const uint row = i / 2560, col = i % 2560;
half parts[8];
for (uint lane = 0; lane < 8; ++lane) parts[lane] = half(0.0f);
for (uint route = 0; route < 10; ++route) {
    const uint source = INV[row * 10 + route];
    if (source >= ROWS * 10) { OUT[i] = half(NAN); return; }
    const half product = half(float(Y[size_t(source) * 2560 + col])
                                 * float(S[row * 10 + route]));
    const uint lane = route % 8;
    parts[lane] = half(float(product) + float(parts[lane]));
}
half total = parts[0];
for (uint lane = 1; lane < 8; ++lane)
    total = half(float(parts[lane]) + float(total));
OUT[i] = total;
""",
    )


def weighted_unsort(y, inverse, scores):
    """Internal sorted-down ABI; callers establish bounds before projecting."""
    rows = scores.size // 10
    return _kernel()(
        inputs=[y, inverse, scores],
        template=[("ROWS", rows)],
        grid=(rows * 2560, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(*scores.shape[:-1], 2560)],
        output_dtypes=[mx.float16],
    )[0]


def prefill_reduce(switch, x, indices, scores):
    """Return candidate weighted output, or None before any tensor operation.

    Only the default native affine SwitchGLU route is admitted. Decode, custom
    JANGH, training, mixed activation dtypes and unknown layouts stay native.
    """
    if not requested() or not _available():
        return None
    if (
        type(switch) is not SwitchGLU
        or switch.training
        or x.ndim not in (2, 3)
        or x.shape[-1] != 2560
        or x.size // 2560 <= 32
        or x.size >= 2**31
        or x.dtype != mx.float16
        or indices.shape != (*x.shape[:-1], 10)
        or indices.dtype not in (mx.int32, mx.uint32)
        or scores.shape != indices.shape
        or scores.dtype != mx.float16
    ):
        return None
    for projection, input_dims, output_dims in (
        (switch.up_proj, 2560, 640),
        (switch.gate_proj, 2560, 640),
        (switch.down_proj, 640, 2560),
    ):
        if (
            type(projection) is not QuantizedSwitchLinear
            or projection.mode != "affine"
            or "bias" in projection
            or projection.weight.dtype != mx.uint32
            or projection.weight.ndim != 3
            or projection.weight.shape[:2] != (512, output_dims)
            or projection.input_dims != input_dims
            or projection.output_dims != output_dims
            or projection.scales.dtype != mx.float16
            or projection.biases is None
            or projection.biases.dtype != mx.float16
        ):
            return None
    expanded, sorted_ids, inverse = _gather_sort(mx.expand_dims(x, (-2, -3)), indices)
    up = switch.up_proj(expanded, sorted_ids, sorted_indices=True)
    gate = switch.gate_proj(expanded, sorted_ids, sorted_indices=True)
    down = switch.down_proj(
        switch.activation(up, gate), sorted_ids, sorted_indices=True
    )
    output = weighted_unsort(down, inverse, scores)
    _record_submission(switch, x, scores, output)
    return output
