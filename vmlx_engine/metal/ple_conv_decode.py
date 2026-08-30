"""Single-token Qwen4 PLE dilated-convolution/state-shift fusion."""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

_OBSERVED = False


def fused_ple_conv_requested() -> bool:
    value = os.environ.get("VMLX_QWEN4_FUSED_PLE_CONV", "0").strip().lower()
    return value not in {"", "0", "false", "off", "no"}


@lru_cache(maxsize=16)
def _kernel(channels: int, kernel_size: int, dilation: int):
    state_length = (kernel_size - 1) * dilation
    source = f"""
        uint channel = thread_position_in_grid.x;
        if (channel >= {channels}u) return;

        float acc = 0.0f;
        for (uint tap = 0u; tap < {kernel_size - 1}u; ++tap) {{
            size_t state_offset = (size_t)(tap * {dilation}u) *
                {channels}u + channel;
            size_t weight_offset = (size_t)channel * {kernel_size}u + tap;
            acc += float(state[state_offset]) * float(weight[weight_offset]);
        }}
        float current = float(token[channel]);
        acc += current * float(
            weight[(size_t)channel * {kernel_size}u + {kernel_size - 1}u]
        );

        for (uint position = 0u; position < {state_length - 1}u; ++position) {{
            next_state[(size_t)position * {channels}u + channel] =
                state[(size_t)(position + 1u) * {channels}u + channel];
        }}
        next_state[(size_t){state_length - 1}u * {channels}u + channel] =
            T(current);
        convolved[channel] = T(acc / (1.0f + metal::exp(-acc)));
"""
    return mx.fast.metal_kernel(
        name=(
            f"vmlx_qwen4_ple_conv_k{kernel_size}_d{dilation}_c{channels}"
        ),
        input_names=["state", "token", "weight"],
        output_names=["next_state", "convolved"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=source,
    )


def qwen4_ple_conv_decode(
    x: mx.array,
    state: mx.array,
    weight: mx.array,
    *,
    dilation: int,
    enabled: bool | None = None,
) -> tuple[mx.array, mx.array] | None:
    """Return ``(silu(conv), next_state)`` for the exact AR PLE shape."""

    if enabled is None:
        enabled = fused_ple_conv_requested()
    if not enabled or x.ndim != 3 or tuple(x.shape[:2]) != (1, 1):
        return None
    channels = int(x.shape[-1])
    if state.ndim != 3 or tuple(state.shape[:1]) != (1,):
        return None
    if weight.ndim != 2 or int(weight.shape[0]) != channels:
        return None
    kernel_size = int(weight.shape[1])
    if kernel_size < 2 or dilation < 1:
        return None
    state_length = (kernel_size - 1) * dilation
    if tuple(state.shape[1:]) != (state_length, channels):
        return None
    if x.dtype not in (mx.float16, mx.bfloat16):
        return None
    if state.dtype != x.dtype or weight.dtype != x.dtype:
        return None

    next_state, convolved = _kernel(channels, kernel_size, dilation)(
        inputs=[state, x, weight],
        template=[("T", x.dtype)],
        grid=(channels, 1, 1),
        threadgroup=(min(256, channels), 1, 1),
        output_shapes=[tuple(state.shape), tuple(x.shape)],
        output_dtypes=[x.dtype, x.dtype],
    )
    global _OBSERVED
    if not _OBSERVED:
        _OBSERVED = True
    return convolved, next_state


def qwen4_ple_conv_status() -> dict[str, object]:
    return {
        "installed": _OBSERVED,
        "observed_calls": int(_OBSERVED),
        "reason": None,
    }


__all__ = [
    "fused_ple_conv_requested",
    "qwen4_ple_conv_decode",
    "qwen4_ple_conv_status",
]
