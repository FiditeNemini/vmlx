"""Single-token Qwen GDN depthwise-convolution/state-shift fusion."""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

_OBSERVED = False


def fused_gdn_conv_requested() -> bool:
    value = os.environ.get("VMLX_QWEN4_FUSED_GDN_CONV", "0").strip().lower()
    return value not in {"", "0", "false", "off", "no"}


@lru_cache(maxsize=16)
def _kernel(channels: int, kernel_size: int):
    source = f"""
        uint channel = thread_position_in_grid.x;
        if (channel >= {channels}u) return;

        float acc = 0.0f;
        for (uint tap = 0; tap < {kernel_size - 1}u; ++tap) {{
            float value = (float)state[(size_t)tap * {channels}u + channel];
            float coeff = (float)weight[(size_t)channel * {kernel_size}u + tap];
            acc += value * coeff;
        }}
        float current = (float)token[channel];
        acc += current * (float)weight[(size_t)channel * {kernel_size}u + {kernel_size - 1}u];

        for (uint tap = 0; tap < {kernel_size - 2}u; ++tap) {{
            next_state[(size_t)tap * {channels}u + channel] =
                state[(size_t)(tap + 1u) * {channels}u + channel];
        }}
        next_state[(size_t){kernel_size - 2}u * {channels}u + channel] = (T)current;
        convolved[channel] = (T)(acc / (1.0f + metal::exp(-acc)));
"""
    return mx.fast.metal_kernel(
        name=f"vmlx_qwen4_gdn_conv_k{kernel_size}_c{channels}",
        input_names=["state", "token", "weight"],
        output_names=["next_state", "convolved"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=source,
    )


def qwen4_gdn_conv_decode(
    qkv: mx.array,
    state: mx.array,
    weight: mx.array,
    *,
    enabled: bool | None = None,
) -> tuple[mx.array, mx.array] | None:
    """Return ``(silu(conv), next_state)`` for the exact AR decode shape."""

    if enabled is None:
        enabled = fused_gdn_conv_requested()
    if not enabled:
        return None
    if qkv.ndim != 3 or tuple(qkv.shape[:2]) != (1, 1):
        return None
    channels = int(qkv.shape[-1])
    if state.ndim != 3 or tuple(state.shape[:1]) != (1,):
        return None
    kernel_size = int(state.shape[1]) + 1
    if kernel_size < 2 or int(state.shape[2]) != channels:
        return None
    if tuple(weight.shape) != (channels, kernel_size, 1):
        return None
    if qkv.dtype not in (mx.float16, mx.bfloat16):
        return None
    if state.dtype != qkv.dtype or weight.dtype != qkv.dtype:
        return None

    next_state, convolved = _kernel(channels, kernel_size)(
        inputs=[state, qkv, weight],
        template=[("T", qkv.dtype)],
        grid=(channels, 1, 1),
        threadgroup=(min(256, channels), 1, 1),
        output_shapes=[tuple(state.shape), tuple(qkv.shape)],
        output_dtypes=[qkv.dtype, qkv.dtype],
    )
    global _OBSERVED
    if not _OBSERVED:
        _OBSERVED = True
    return convolved, next_state


def qwen4_gdn_conv_status() -> dict[str, object]:
    return {
        "installed": _OBSERVED,
        "observed_calls": int(_OBSERVED),
        "reason": None,
    }


__all__ = [
    "fused_gdn_conv_requested",
    "qwen4_gdn_conv_decode",
    "qwen4_gdn_conv_status",
]
