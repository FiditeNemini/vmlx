"""Two-dispatch GLM-5.3 mHC transform for decode and MTP verification.

The portable graph launches FP32 RMS normalization, a 24-row wide-input
projection, sigmoid/softmax, 20 Sinkhorn normalization rounds, and stream
collapse separately. GLM invokes that graph twice in every decoder layer.
This diagnostic path evaluates the 24 projection rows as independent SIMD
groups, then fuses every small-matrix operation and stream collapse into one
epilogue dispatch. Prefill and unsupported layouts stay on stock MLX.

The same per-token transform also handles the 2-4 row slabs produced by
native-MTP D1-D3 verification. Longer prefill and unsupported layouts stay on
stock MLX. The candidate is enabled by default after exact-bundle output/TPS,
syntax, native-MTP rollback, and terminal-finalization proof. Set
``VMLX_GLM5_FUSED_MHC=0`` for an explicit stock-path rollback. Unsupported
shapes, including prefill and multi-token MTP verification, still fall back to
stock MLX automatically.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import mlx.core as mx

_PROJECTION_THREADS = 128
_PROJECTION_SIMDGROUPS = _PROJECTION_THREADS // 32
_EPILOGUE_THREADS = 256
_OBSERVED = False

_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

_PROJECTION_SOURCE = """
    uint group = thread_position_in_grid.x / PROJECTION_THREADS;
    uint token = group / MIX_SIZE;
    uint row = group % MIX_SIZE;
    uint tid = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint simd_id = simdgroup_index_in_threadgroup;
    if (token >= ROWS || row >= MIX_SIZE) return;

    threadgroup float partials[PROJECTION_SIMDGROUPS];
    threadgroup float inv_rms;

    float sumsq = 0.0f;
    for (uint col = tid; col < FEATURES; col += PROJECTION_THREADS) {
        float value = float(streams[(size_t)token * FEATURES + col]);
        sumsq += value * value;
    }
    sumsq = simd_sum(sumsq);
    if (lane == 0u) partials[simd_id] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = 0.0f;
        for (uint index = 0u; index < PROJECTION_SIMDGROUPS; ++index) {
            total += partials[index];
        }
        inv_rms = metal::rsqrt(
            total / float(FEATURES) + float(rms_eps[0])
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float value = 0.0f;
    size_t weight_base = (size_t)row * (size_t)FEATURES;
    for (uint col = tid; col < FEATURES; col += PROJECTION_THREADS) {
        value += float(streams[(size_t)token * FEATURES + col]) * inv_rms *
            float(hc_fn[weight_base + col]);
    }
    value = simd_sum(value);
    if (lane == 0u) partials[simd_id] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = 0.0f;
        for (uint index = 0u; index < PROJECTION_SIMDGROUPS; ++index) {
            total += partials[index];
        }
        mix[(size_t)token * MIX_SIZE + row] = total;
    }
"""

_EPILOGUE_SOURCE = """
    uint tid = thread_index_in_threadgroup;
    uint token = thread_position_in_grid.x / EPILOGUE_THREADS;
    if (token >= ROWS) return;
    size_t streams_base = (size_t)token * H * D;
    size_t mix_base = (size_t)token * MIX_SIZE;
    size_t post_base = (size_t)token * H;
    size_t comb_base = (size_t)token * H * H;
    size_t collapsed_base = (size_t)token * D;
    threadgroup float pre_values[H];
    threadgroup float post_values[H];
    threadgroup float matrix[H * H];

    if (tid == 0u) {
        float scale_pre = float(hc_scale[0]);
        float scale_post = float(hc_scale[1]);
        float scale_comb = float(hc_scale[2]);
        float epsilon = float(sink_eps[0]);

        for (uint index = 0u; index < H; ++index) {
            float pre_x = float(mix[mix_base + index]) * scale_pre +
                float(hc_base[index]);
            pre_values[index] = 1.0f / (1.0f + metal::exp(-pre_x)) + epsilon;
            float post_x = float(mix[mix_base + H + index]) * scale_post +
                float(hc_base[H + index]);
            post_values[index] = 2.0f / (1.0f + metal::exp(-post_x));
        }

        for (uint row = 0u; row < H; ++row) {
            float row_max = -INFINITY;
            for (uint col = 0u; col < H; ++col) {
                uint index = row * H + col;
                float item = float(mix[mix_base + 2u * H + index]) * scale_comb +
                    float(hc_base[2u * H + index]);
                matrix[index] = item;
                row_max = metal::max(row_max, item);
            }
            float row_sum = 0.0f;
            for (uint col = 0u; col < H; ++col) {
                uint index = row * H + col;
                float item = metal::exp(matrix[index] - row_max);
                matrix[index] = item;
                row_sum += item;
            }
            for (uint col = 0u; col < H; ++col) {
                uint index = row * H + col;
                matrix[index] = matrix[index] / row_sum + epsilon;
            }
        }

        for (uint col = 0u; col < H; ++col) {
            float col_sum = 0.0f;
            for (uint row = 0u; row < H; ++row) {
                col_sum += matrix[row * H + col];
            }
            for (uint row = 0u; row < H; ++row) {
                uint index = row * H + col;
                matrix[index] /= col_sum + epsilon;
            }
        }

        for (uint iteration = 1u; iteration < ITERS; ++iteration) {
            for (uint row = 0u; row < H; ++row) {
                float row_sum = 0.0f;
                for (uint col = 0u; col < H; ++col) {
                    row_sum += matrix[row * H + col];
                }
                for (uint col = 0u; col < H; ++col) {
                    uint index = row * H + col;
                    matrix[index] /= row_sum + epsilon;
                }
            }
            for (uint col = 0u; col < H; ++col) {
                float col_sum = 0.0f;
                for (uint row = 0u; row < H; ++row) {
                    col_sum += matrix[row * H + col];
                }
                for (uint row = 0u; row < H; ++row) {
                    uint index = row * H + col;
                    matrix[index] /= col_sum + epsilon;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint dim = tid; dim < D; dim += EPILOGUE_THREADS) {
        float value = 0.0f;
        for (uint stream = 0u; stream < H; ++stream) {
            value += pre_values[stream] *
                float(streams[streams_base + stream * D + dim]);
        }
        collapsed[collapsed_base + dim] = T(value);
    }
    for (uint index = tid; index < H; index += EPILOGUE_THREADS) {
        post[post_base + index] = post_values[index];
    }
    for (uint index = tid; index < H * H; index += EPILOGUE_THREADS) {
        comb[comb_base + index] = matrix[index];
    }
"""


def fused_glm5_mhc_requested() -> bool:
    value = os.environ.get("VMLX_GLM5_FUSED_MHC", "1").strip().lower()
    return value not in {"", "0", "false", "off", "no"}


@lru_cache(maxsize=1)
def _projection_kernel() -> Any:
    return mx.fast.metal_kernel(
        name="vmlx_glm5_mhc_projection",
        input_names=["streams", "hc_fn", "rms_eps"],
        output_names=["mix"],
        header=_HEADER,
        source=_PROJECTION_SOURCE,
    )


@lru_cache(maxsize=1)
def _epilogue_kernel() -> Any:
    return mx.fast.metal_kernel(
        name="vmlx_glm5_mhc_epilogue",
        input_names=["streams", "mix", "hc_base", "hc_scale", "sink_eps"],
        output_names=["post", "comb", "collapsed"],
        header=_HEADER,
        source=_EPILOGUE_SOURCE,
    )


@lru_cache(maxsize=16)
def _scalar(value: float) -> mx.array:
    result = mx.array([value], dtype=mx.float32)
    mx.eval(result)
    return result


def glm5_mhc_decode(
    streams: mx.array,
    hc_fn: mx.array,
    hc_base: mx.array,
    hc_scale: mx.array,
    *,
    rms_eps: float,
    sink_eps: float,
    iterations: int,
    enabled: bool | None = None,
) -> tuple[mx.array, mx.array, mx.array] | None:
    """Return fused ``(post, comb, collapsed)`` or the stock-path sentinel."""

    if enabled is None:
        enabled = fused_glm5_mhc_requested()
    if not enabled or streams.ndim != 4:
        return None
    batch, rows, streams_count, hidden = (int(value) for value in streams.shape)
    if batch != 1 or not 1 <= rows <= 4 or streams_count != 4 or hidden <= 0:
        return None
    mix_size = (2 + streams_count) * streams_count
    features = streams_count * hidden
    if tuple(hc_fn.shape) != (mix_size, features):
        return None
    if tuple(hc_base.shape) != (mix_size,) or tuple(hc_scale.shape) != (3,):
        return None
    if iterations < 1 or iterations > 64:
        return None
    supported = (mx.float16, mx.bfloat16, mx.float32)
    if any(
        value.dtype not in supported
        for value in (streams, hc_fn, hc_base, hc_scale)
    ):
        return None

    mix = _projection_kernel()(
        inputs=[streams, hc_fn, _scalar(float(rms_eps))],
        template=[
            ("FEATURES", features),
            ("MIX_SIZE", mix_size),
            ("ROWS", rows),
            ("PROJECTION_THREADS", _PROJECTION_THREADS),
            ("PROJECTION_SIMDGROUPS", _PROJECTION_SIMDGROUPS),
        ],
        grid=(_PROJECTION_THREADS * mix_size * rows, 1, 1),
        threadgroup=(_PROJECTION_THREADS, 1, 1),
        output_shapes=[(rows, mix_size)],
        output_dtypes=[mx.float32],
    )[0]
    output = tuple(
        _epilogue_kernel()(
            inputs=[streams, mix, hc_base, hc_scale, _scalar(float(sink_eps))],
            template=[
                ("T", streams.dtype),
                ("H", streams_count),
                ("D", hidden),
                ("ROWS", rows),
                ("MIX_SIZE", mix_size),
                ("ITERS", iterations),
                ("EPILOGUE_THREADS", _EPILOGUE_THREADS),
            ],
            grid=(_EPILOGUE_THREADS * rows, 1, 1),
            threadgroup=(_EPILOGUE_THREADS, 1, 1),
            output_shapes=[
                (1, rows, streams_count),
                (1, rows, streams_count, streams_count),
                (1, rows, hidden),
            ],
            output_dtypes=[mx.float32, mx.float32, streams.dtype],
        )
    )
    global _OBSERVED
    if not _OBSERVED:
        _OBSERVED = True
    return output


def glm5_mhc_status() -> dict[str, object]:
    return {
        "installed": _OBSERVED,
        "observed_calls": int(_OBSERVED),
        "reason": None,
    }


__all__ = ["fused_glm5_mhc_requested", "glm5_mhc_decode", "glm5_mhc_status"]
