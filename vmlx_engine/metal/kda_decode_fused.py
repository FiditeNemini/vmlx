"""Single-dispatch GLM KDA convolution and recurrent decode candidate.

The older diagnostic kernels split the K4 q/k/v convolution and the KDA
recurrent update into separate eager Metal dispatches.  Both primitives win in
isolation but lose in the complete GLM graph because they interrupt MLX's lazy
fusion boundary.  This candidate owns the full architecture-native decode
unit: short convolution, SiLU, q/k normalization, delta-rule recurrence,
output reduction, and all convolution/recurrent state writes.

It is opt-in and batch-one/single-token only.  Prefill, unsupported geometry,
or any disabled request returns ``None`` so the source runtime retains the
stock MLX graph.
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

_THREADS = 256
_SIMDGROUPS = _THREADS // 32
_OBSERVED = False


def fused_kda_decode_requested() -> bool:
    value = os.environ.get("VMLX_GLM5_FUSED_KDA_DECODE", "0").strip().lower()
    return value not in {"", "0", "false", "off", "no"}


@lru_cache(maxsize=16)
def _kernel(heads: int, key_dim: int, value_dim: int, kernel_size: int):
    key_chunks = (key_dim + 31) // 32
    source = f"""
        uint tid = thread_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint simd_id = simdgroup_index_in_threadgroup;
        uint head = thread_position_in_grid.x / {_THREADS}u;
        if (head >= {heads}u) return;

        threadgroup float q_values[{key_dim}];
        threadgroup float k_values[{key_dim}];
        threadgroup float v_values[{value_dim}];
        threadgroup float q_partials[{_SIMDGROUPS}];
        threadgroup float k_partials[{_SIMDGROUPS}];
        threadgroup float q_inv_norm;
        threadgroup float k_inv_norm;

        if (tid < {key_dim}u) {{
            uint channel = head * {key_dim}u + tid;
            float q_acc = 0.0f;
            float k_acc = 0.0f;
            for (uint tap = 0u; tap < {kernel_size - 1}u; ++tap) {{
                size_t state_offset = (size_t)tap * {heads * key_dim}u + channel;
                size_t weight_offset = (size_t)channel * {kernel_size}u + tap;
                q_acc += float(q_state[state_offset]) * float(q_weight[weight_offset]);
                k_acc += float(k_state[state_offset]) * float(k_weight[weight_offset]);
            }}
            float q_current = float(q_token[channel]);
            float k_current = float(k_token[channel]);
            q_acc += q_current * float(q_weight[(size_t)channel * {kernel_size}u + {kernel_size - 1}u]);
            k_acc += k_current * float(k_weight[(size_t)channel * {kernel_size}u + {kernel_size - 1}u]);
            q_values[tid] = float(T(q_acc / (1.0f + metal::exp(-q_acc))));
            k_values[tid] = float(T(k_acc / (1.0f + metal::exp(-k_acc))));
            for (uint tap = 0u; tap < {kernel_size - 2}u; ++tap) {{
                size_t dst = (size_t)tap * {heads * key_dim}u + channel;
                size_t src = (size_t)(tap + 1u) * {heads * key_dim}u + channel;
                q_next[dst] = q_state[src];
                k_next[dst] = k_state[src];
            }}
            size_t tail = (size_t){kernel_size - 2}u * {heads * key_dim}u + channel;
            q_next[tail] = T(q_current);
            k_next[tail] = T(k_current);
        }}
        if (tid < {value_dim}u) {{
            uint channel = head * {value_dim}u + tid;
            float v_acc = 0.0f;
            for (uint tap = 0u; tap < {kernel_size - 1}u; ++tap) {{
                size_t state_offset = (size_t)tap * {heads * value_dim}u + channel;
                size_t weight_offset = (size_t)channel * {kernel_size}u + tap;
                v_acc += float(v_state[state_offset]) * float(v_weight[weight_offset]);
            }}
            float v_current = float(v_token[channel]);
            v_acc += v_current * float(v_weight[(size_t)channel * {kernel_size}u + {kernel_size - 1}u]);
            v_values[tid] = float(T(v_acc / (1.0f + metal::exp(-v_acc))));
            for (uint tap = 0u; tap < {kernel_size - 2}u; ++tap) {{
                size_t dst = (size_t)tap * {heads * value_dim}u + channel;
                size_t src = (size_t)(tap + 1u) * {heads * value_dim}u + channel;
                v_next[dst] = v_state[src];
            }}
            size_t tail = (size_t){kernel_size - 2}u * {heads * value_dim}u + channel;
            v_next[tail] = T(v_current);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float q_sumsq = tid < {key_dim}u ? q_values[tid] * q_values[tid] : 0.0f;
        float k_sumsq = tid < {key_dim}u ? k_values[tid] * k_values[tid] : 0.0f;
        q_sumsq = simd_sum(q_sumsq);
        k_sumsq = simd_sum(k_sumsq);
        if (lane == 0u) {{
            q_partials[simd_id] = q_sumsq;
            k_partials[simd_id] = k_sumsq;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {{
            float q_total = 0.0f;
            float k_total = 0.0f;
            for (uint index = 0u; index < {_SIMDGROUPS}u; ++index) {{
                q_total += q_partials[index];
                k_total += k_partials[index];
            }}
            q_inv_norm = metal::rsqrt(q_total + 1.0e-6f);
            k_inv_norm = metal::rsqrt(k_total + 1.0e-6f);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < {key_dim}u) {{
            q_values[tid] *= q_inv_norm;
            k_values[tid] *= k_inv_norm;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint value_index = simd_id; value_index < {value_dim}u;
             value_index += {_SIMDGROUPS}u) {{
            float decayed[{key_chunks}];
            float keys[{key_chunks}];
            float key_state_partial = 0.0f;
            for (uint chunk = 0u; chunk < {key_chunks}u; ++chunk) {{
                uint key_index = chunk * 32u + lane;
                float decayed_value = 0.0f;
                float key_value = 0.0f;
                if (key_index < {key_dim}u) {{
                    size_t key_offset = (size_t)head * {key_dim}u + key_index;
                    size_t state_offset = key_offset * {value_dim}u + value_index;
                    key_value = k_values[key_index];
                    decayed_value = float(state[state_offset]) *
                        metal::exp(float(gate[key_offset]));
                    key_state_partial += key_value * decayed_value;
                }}
                decayed[chunk] = decayed_value;
                keys[chunk] = key_value;
            }}
            float key_state = simd_sum(key_state_partial);
            float correction = v_values[value_index] - key_state;
            float beta_value = float(beta[head]);
            float output_partial = 0.0f;
            for (uint chunk = 0u; chunk < {key_chunks}u; ++chunk) {{
                uint key_index = chunk * 32u + lane;
                if (key_index < {key_dim}u) {{
                    size_t key_offset = (size_t)head * {key_dim}u + key_index;
                    size_t state_offset = key_offset * {value_dim}u + value_index;
                    float next = decayed[chunk] +
                        beta_value * keys[chunk] * correction;
                    next_state[state_offset] = next;
                    output_partial += q_values[key_index] *
                        {key_dim ** -0.5:.12f}f * next;
                }}
            }}
            float output_value = simd_sum(output_partial);
            if (lane == 0u) {{
                output[(size_t)head * {value_dim}u + value_index] = output_value;
            }}
        }}
"""
    return mx.fast.metal_kernel(
        name=(
            f"vmlx_glm5_kda_decode_h{heads}_k{key_dim}_v{value_dim}_"
            f"c{kernel_size}"
        ),
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
            "gate",
            "beta",
            "state",
        ],
        output_names=[
            "output",
            "next_state",
            "q_next",
            "k_next",
            "v_next",
        ],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=source,
    )


def glm5_kda_decode(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    q_state: mx.array | None,
    k_state: mx.array | None,
    v_state: mx.array | None,
    q_weight: mx.array,
    k_weight: mx.array,
    v_weight: mx.array,
    gate: mx.array,
    beta: mx.array,
    state: mx.array | None,
    *,
    enabled: bool | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array] | None:
    """Return output plus KDA recurrent and convolution states."""

    if enabled is None:
        enabled = fused_kda_decode_requested()
    if not enabled or state is None:
        return None
    if any(array.ndim != 3 for array in (q, k, v)):
        return None
    if any(tuple(array.shape[:2]) != (1, 1) for array in (q, k, v)):
        return None
    if q.dtype not in (mx.float16, mx.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        return None
    if state.ndim != 4 or int(state.shape[0]) != 1 or state.dtype != mx.float32:
        return None
    heads, key_dim, value_dim = (int(value) for value in state.shape[1:])
    if key_dim <= 0 or value_dim <= 0 or key_dim > 256:
        return None
    if int(q.shape[-1]) != heads * key_dim or int(k.shape[-1]) != heads * key_dim:
        return None
    if int(v.shape[-1]) != heads * value_dim:
        return None
    if tuple(gate.shape) != (1, 1, heads, key_dim) or gate.dtype != mx.float32:
        return None
    if tuple(beta.shape) != (1, 1, heads) or beta.dtype != mx.float32:
        return None

    weights = (q_weight, k_weight, v_weight)
    channels = (heads * key_dim, heads * key_dim, heads * value_dim)
    if any(weight.ndim != 2 for weight in weights):
        return None
    kernel_size = int(q_weight.shape[1])
    if kernel_size < 2 or kernel_size > 8:
        return None
    if any(tuple(weight.shape) != (channel, kernel_size) for weight, channel in zip(weights, channels)):
        return None
    states = [q_state, k_state, v_state]
    for index, (conv_state, channel) in enumerate(zip(states, channels)):
        expected = (1, kernel_size - 1, channel)
        if conv_state is None:
            states[index] = mx.zeros(expected, dtype=q.dtype)
        elif tuple(conv_state.shape) != expected or conv_state.dtype != q.dtype:
            return None

    output, next_state, q_next, k_next, v_next = _kernel(
        heads, key_dim, value_dim, kernel_size
    )(
        inputs=[
            q,
            k,
            v,
            *states,
            *weights,
            gate,
            beta,
            state,
        ],
        template=[("T", q.dtype)],
        grid=(heads * _THREADS, 1, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[
            (1, heads, value_dim),
            tuple(state.shape),
            tuple(states[0].shape),
            tuple(states[1].shape),
            tuple(states[2].shape),
        ],
        output_dtypes=[mx.float32, mx.float32, q.dtype, q.dtype, q.dtype],
    )
    global _OBSERVED
    if not _OBSERVED:
        _OBSERVED = True
    return output, next_state, q_next, k_next, v_next


def glm5_kda_decode_status() -> dict[str, object]:
    return {
        "installed": _OBSERVED,
        "observed_calls": int(_OBSERVED),
        "reason": None,
    }


__all__ = [
    "fused_kda_decode_requested",
    "glm5_kda_decode",
    "glm5_kda_decode_status",
]
