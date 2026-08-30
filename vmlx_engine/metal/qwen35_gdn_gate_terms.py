"""Exact joint gate preparation for single-token Qwen3.5 GDN decode."""

from __future__ import annotations

import os
from functools import partial

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gated_delta import gated_delta_kernel

_OBSERVED = False


def fused_qwen35_gdn_gates_requested() -> bool:
    value = os.environ.get("VMLX_QWEN35_FUSED_GDN_GATES", "0").strip().lower()
    return value not in {"", "0", "false", "off", "no"}


@partial(mx.compile, shapeless=True)
def _joint_gate_terms(A_log, a, dt_bias, b):
    beta = mx.sigmoid(b)
    g = mx.exp(
        -mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias)
    )
    return g, beta


def qwen35_gated_delta_decode(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: mx.array | None,
    mask: mx.array | None,
    *,
    enabled: bool | None = None,
) -> tuple[mx.array, mx.array] | None:
    """Run the stock recurrent kernel with jointly compiled gate terms."""

    if enabled is None:
        enabled = fused_qwen35_gdn_gates_requested()
    if not enabled or mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None
    if q.ndim != 4 or tuple(q.shape[:2]) != (1, 1):
        return None
    if k.ndim != 4 or tuple(k.shape[:2]) != (1, 1):
        return None
    if v.ndim != 4 or tuple(v.shape[:2]) != (1, 1):
        return None
    if a.ndim != 3 or b.ndim != 3 or tuple(a.shape[:2]) != (1, 1):
        return None
    if tuple(b.shape) != tuple(a.shape):
        return None
    if q.dtype not in (mx.float16, mx.bfloat16):
        return None
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return None

    key_heads = int(q.shape[-2])
    key_dim = int(q.shape[-1])
    value_heads = int(v.shape[-2])
    value_dim = int(v.shape[-1])
    if key_dim < 32 or key_dim % 32 != 0:
        return None
    if value_heads % key_heads != 0:
        return None
    if tuple(k.shape[-2:]) != (key_heads, key_dim):
        return None
    if tuple(a.shape) != (1, 1, value_heads):
        return None
    if tuple(A_log.shape) != (value_heads,) or tuple(dt_bias.shape) != (
        value_heads,
    ):
        return None

    if state is None:
        state = mx.zeros(
            (1, value_heads, value_dim, key_dim), dtype=mx.float32
        )
    elif tuple(state.shape) != (1, value_heads, value_dim, key_dim):
        return None

    g, beta = _joint_gate_terms(A_log, a, dt_bias, b)
    result = gated_delta_kernel(q, k, v, g, beta, state, mask)
    global _OBSERVED
    _OBSERVED = True
    return result


def qwen35_gdn_gate_terms_status() -> dict[str, object]:
    return {
        "installed": _OBSERVED,
        "observed_calls": int(_OBSERVED),
        "reason": None,
    }


__all__ = [
    "fused_qwen35_gdn_gates_requested",
    "qwen35_gated_delta_decode",
    "qwen35_gdn_gate_terms_status",
]
