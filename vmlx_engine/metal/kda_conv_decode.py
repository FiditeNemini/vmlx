"""Batch-one GLM KDA q/k/v short-convolution fusion."""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

_OBSERVED = False


def fused_kda_conv_requested() -> bool:
    value = os.environ.get("VMLX_GLM5_FUSED_KDA_CONV", "0").strip().lower()
    return value not in {"", "0", "false", "off", "no"}


@lru_cache(maxsize=16)
def _kernel(channels: int, kernel_size: int):
    def lane(prefix: str) -> str:
        return f"""
            float {prefix}_acc = 0.0f;
            for (uint tap = 0; tap < {kernel_size - 1}u; ++tap) {{
                float value = (float){prefix}_state[(size_t)tap * {channels}u + channel];
                float coeff = (float){prefix}_weight[(size_t)channel * {kernel_size}u + tap];
                {prefix}_acc += value * coeff;
            }}
            float {prefix}_current = (float){prefix}_token[channel];
            {prefix}_acc += {prefix}_current *
                (float){prefix}_weight[(size_t)channel * {kernel_size}u + {kernel_size - 1}u];
            for (uint tap = 0; tap < {kernel_size - 2}u; ++tap) {{
                {prefix}_next[(size_t)tap * {channels}u + channel] =
                    {prefix}_state[(size_t)(tap + 1u) * {channels}u + channel];
            }}
            {prefix}_next[(size_t){kernel_size - 2}u * {channels}u + channel] =
                (T){prefix}_current;
            {prefix}_out[channel] =
                (T)({prefix}_acc / (1.0f + metal::exp(-{prefix}_acc)));
"""

    source = f"""
        uint channel = thread_position_in_grid.x;
        if (channel >= {channels}u) return;
        {lane('q')}
        {lane('k')}
        {lane('v')}
"""
    return mx.fast.metal_kernel(
        name=f"vmlx_glm5_kda_qkv_conv_k{kernel_size}_c{channels}",
        input_names=[
            "q_token",
            "k_token",
            "v_token",
            "q_state",
            "k_state",
            "v_state",
            "q_weight",
            "k_weight",
            "v_weight",
        ],
        output_names=[
            "q_out",
            "k_out",
            "v_out",
            "q_next",
            "k_next",
            "v_next",
        ],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=source,
    )


def glm5_kda_conv_decode(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    q_state: mx.array | None,
    k_state: mx.array | None,
    v_state: mx.array | None,
    q_weight: mx.array,
    k_weight: mx.array,
    v_weight: mx.array,
    *,
    enabled: bool | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array] | None:
    """Return convolved q/k/v followed by their exact shifted states."""

    if enabled is None:
        enabled = fused_kda_conv_requested()
    if not enabled:
        return None
    if any(array.ndim != 3 for array in (q, k, v)):
        return None
    if any(tuple(array.shape[:2]) != (1, 1) for array in (q, k, v)):
        return None
    if not (q.shape == k.shape == v.shape):
        return None
    channels = int(q.shape[-1])
    weights = (q_weight, k_weight, v_weight)
    if any(weight.ndim != 2 or int(weight.shape[0]) != channels for weight in weights):
        return None
    kernel_size = int(q_weight.shape[1])
    if kernel_size < 2 or any(tuple(weight.shape) != (channels, kernel_size) for weight in weights):
        return None
    if q.dtype not in (mx.float16, mx.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        return None
    states = [q_state, k_state, v_state]
    for index, state in enumerate(states):
        if state is None:
            states[index] = mx.zeros(
                (1, kernel_size - 1, channels), dtype=q.dtype
            )
        elif tuple(state.shape) != (1, kernel_size - 1, channels) or state.dtype != q.dtype:
            return None

    outputs = _kernel(channels, kernel_size)(
        inputs=[q, k, v, *states, q_weight, k_weight, v_weight],
        template=[("T", q.dtype)],
        grid=(channels, 1, 1),
        threadgroup=(min(256, channels), 1, 1),
        output_shapes=[
            tuple(q.shape),
            tuple(k.shape),
            tuple(v.shape),
            tuple(states[0].shape),
            tuple(states[1].shape),
            tuple(states[2].shape),
        ],
        output_dtypes=[q.dtype] * 6,
    )
    global _OBSERVED
    if not _OBSERVED:
        _OBSERVED = True
    return tuple(outputs)


def glm5_kda_conv_status() -> dict[str, object]:
    return {
        "installed": _OBSERVED,
        "observed_calls": int(_OBSERVED),
        "reason": None,
    }


__all__ = [
    "fused_kda_conv_requested",
    "glm5_kda_conv_decode",
    "glm5_kda_conv_status",
]
