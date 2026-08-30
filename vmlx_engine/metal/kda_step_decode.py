"""Single-dispatch GLM KDA recurrent update for batch-one decode."""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

_OBSERVED = False


def fused_kda_step_requested() -> bool:
    value = os.environ.get("VMLX_GLM5_FUSED_KDA_STEP", "0").strip().lower()
    return value not in {"", "0", "false", "off", "no"}


@lru_cache(maxsize=8)
def _kernel(heads: int, key_dim: int, value_dim: int):
    chunks = (key_dim + 31) // 32
    source = f"""
        uint global_simd = thread_position_in_grid.x / 32u;
        uint lane = thread_index_in_simdgroup;
        uint head = global_simd / {value_dim}u;
        uint value_index = global_simd % {value_dim}u;
        if (head >= {heads}u) return;

        float decayed[{chunks}];
        float key_values[{chunks}];
        float ks_partial = 0.0f;
        for (uint chunk = 0; chunk < {chunks}u; ++chunk) {{
            uint key_index = chunk * 32u + lane;
            float state_value = 0.0f;
            float key_value = 0.0f;
            if (key_index < {key_dim}u) {{
                size_t key_offset = (size_t)head * {key_dim}u + key_index;
                size_t state_offset = key_offset * {value_dim}u + value_index;
                key_value = (float)key[key_offset];
                state_value = (float)state[state_offset] *
                    metal::exp((float)gate[key_offset]);
                ks_partial += key_value * state_value;
            }}
            decayed[chunk] = state_value;
            key_values[chunk] = key_value;
        }}
        float key_state = simd_sum(ks_partial);
        float correction = (float)value[(size_t)head * {value_dim}u + value_index]
            - key_state;
        float beta_value = (float)beta[head];
        float output_partial = 0.0f;
        for (uint chunk = 0; chunk < {chunks}u; ++chunk) {{
            uint key_index = chunk * 32u + lane;
            if (key_index < {key_dim}u) {{
                size_t key_offset = (size_t)head * {key_dim}u + key_index;
                size_t state_offset = key_offset * {value_dim}u + value_index;
                float next = decayed[chunk] +
                    beta_value * key_values[chunk] * correction;
                next_state[state_offset] = next;
                output_partial += (float)query[key_offset] * {key_dim ** -0.5:.12f}f * next;
            }}
        }}
        float output_value = simd_sum(output_partial);
        if (lane == 0u) {{
            output[(size_t)head * {value_dim}u + value_index] = output_value;
        }}
"""
    return mx.fast.metal_kernel(
        name=f"vmlx_glm5_kda_step_h{heads}_k{key_dim}_v{value_dim}",
        input_names=["query", "key", "value", "gate", "beta", "state"],
        output_names=["output", "next_state"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=source,
    )


def glm5_kda_step_decode(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    *,
    enabled: bool | None = None,
) -> tuple[mx.array, mx.array] | None:
    """Return the exact-shape KDA decode output/state candidate."""

    if enabled is None:
        enabled = fused_kda_step_requested()
    if not enabled:
        return None
    if q.ndim != 3 or int(q.shape[0]) != 1:
        return None
    if q.shape != k.shape or q.shape != g.shape:
        return None
    heads, key_dim = int(q.shape[1]), int(q.shape[2])
    if v.ndim != 3 or tuple(v.shape[:2]) != (1, heads):
        return None
    value_dim = int(v.shape[2])
    if tuple(beta.shape) != (1, heads):
        return None
    if tuple(state.shape) != (1, heads, key_dim, value_dim):
        return None
    if state.dtype != mx.float32:
        return None
    supported = (mx.float16, mx.bfloat16, mx.float32)
    if any(array.dtype not in supported for array in (q, k, v, g, beta)):
        return None
    if key_dim <= 0 or value_dim <= 0 or key_dim > 256:
        return None

    output, next_state = _kernel(heads, key_dim, value_dim)(
        inputs=[q, k, v, g, beta, state],
        grid=(32 * heads * value_dim, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, heads, value_dim), tuple(state.shape)],
        output_dtypes=[mx.float32, mx.float32],
    )
    global _OBSERVED
    if not _OBSERVED:
        _OBSERVED = True
    return output, next_state


def glm5_kda_step_status() -> dict[str, object]:
    return {
        "installed": _OBSERVED,
        "observed_calls": int(_OBSERVED),
        "reason": None,
    }


__all__ = [
    "fused_kda_step_requested",
    "glm5_kda_step_decode",
    "glm5_kda_step_status",
]
