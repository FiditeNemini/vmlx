# SPDX-License-Identifier: MIT
# Arithmetic adapted from MLX v0.32.2 gemv.h and quantized.h.
# Copyright (c) 2023-2026 Apple Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Guarded GLM AR router + existing affine shared gate/up dispatch.

No repacking, converted weights, selection, activation, down projection or cache
mutation. Exact arithmetic is a qualification requirement, not a source claim.
"""
from functools import lru_cache
import os

import mlx.core as mx

from vmlx_engine.metal.affine_moe_pair_decode import affine_moe_ar_scope_active
from vmlx_engine.metal.glm5_router_matvec import _compatible_runtime as _router_runtime
from vmlx_engine.metal.quantized_projection_group import QuantizedProjectionGroup

_GRAPH_CALLS = 0

# q8 specializations of load_vector and qdot retain the original float loop
# order and expression tree. Do not expand scale/bias into per-weight BF16.
_HEADER = r'''
template <typename T>
inline float rs_load8(const device T* x, thread float* values) {
    float sum = 0;
    for (int i = 0; i < 8; i++) {
        sum += x[i];
        values[i] = x[i];
    }
    return sum;
}
inline float rs_qdot8(const device uint8_t* w, const thread float* values,
                     float scale, float bias, float sum) {
    float accum = 0;
    for (int i = 0; i < 8; i++) {
        accum += values[i] * w[i];
    }
    return scale * accum + sum * bias;
}
'''

_SOURCE = r'''
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint group = threadgroup_position_in_grid.x;
    if (group < 18u) {
        // Unchanged existing router's native FP32 GEMV tree, BF16 loads.
        uint row = group * 16u + sg * 4u;
        float accum[4] = {0};
        for (uint base = 0u; base < 4096u; base += 128u) {
            uint k = base + lane * 4u;
            float v[4];
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 4u; ++j) v[j] = float(x[k+j]);
            #pragma clang loop unroll(full)
            for (uint r = 0u; r < 4u; ++r) {
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 4u; ++j)
                    accum[r] += float(router_weight[(row+r)*4096u+k+j])*v[j];
            }
        }
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < 4u; ++r) {
            #pragma clang loop unroll(full)
            for (ushort offset = 16u; offset >= 1u; offset >>= 1u)
                accum[r] += simd_shuffle_down(accum[r], offset);
            if (lane == 0u) logits[row+r] = accum[r];
        }
    } else {
        // Two native 64-thread QMV groups share a physical 128-thread group.
        // There is no cross-SIMD reduction/barrier in native qmv_fast_impl.
        uint virtual_group = (group - 18u) * 2u + sg / 2u;
        uint virtual_sg = sg % 2u;
        int out_row = virtual_group * 8u + virtual_sg * 4u;
        const device uint8_t* ws = (const device uint8_t*)packed_weight;
        ws += out_row * 4096 + lane * 8;
        const device T* scales = packed_scales + out_row * 64 + lane / 8;
        const device T* biases = packed_biases + out_row * 64 + lane / 8;
        const device T* xv = x + lane * 8;
        thread float values[8];
        thread float result[4] = {0};
        for (int k = 0; k < 4096; k += 256) {
            float sum = rs_load8<T>(xv, values);
            for (int row = 0; row < 4; row++) {
                auto wl = (const device uint8_t*)(ws + row * 4096);
                const device T* sl = scales + row * 64;
                const device T* bl = biases + row * 64;
                float s = sl[0];
                float b = bl[0];
                result[row] += rs_qdot8(wl, values, s, b, sum);
            }
            ws += 256;
            scales += 4;
            biases += 4;
            xv += 256;
        }
        for (int row = 0; row < 4; row++) {
            result[row] = simd_sum(result[row]);
            if (lane == 0) gate_up[out_row + row] = static_cast<T>(result[row]);
        }
    }
'''


def glm5_router_shared_requested():
    return os.environ.get("VMLX_GLM5_ROUTER_SHARED", "0") == "1"


def glm5_router_shared_status():
    return {"requested": glm5_router_shared_requested(), "graph_calls": _GRAPH_CALLS}


@lru_cache(maxsize=1)
def _compatible_runtime():
    return _router_runtime() and mx.device_info().get("device_name") == "Apple M5 Max"


def admission_reason(x, router_weight, group, *, enabled, training=False):
    if not enabled:
        return "disabled"
    if training or not affine_moe_ar_scope_active():
        return "not productive inference AR"
    if x.shape != (1, 1, 4096) or x.dtype != mx.bfloat16:
        return "activation shape/dtype"
    if router_weight.shape != (288, 4096) or router_weight.dtype != mx.bfloat16:
        return "router shape/dtype"
    if type(group) is not QuantizedProjectionGroup:
        return "shared group is not prepared native affine group"
    if (group.input_dims, group.output_dims, group.bits, group.group_size,
            group.mode, group.split_indices) != (4096, 4096, 8, 64, "affine", (2048,)):
        return "shared group geometry/format/split"
    if group.weight.shape != (4096, 1024) or group.weight.dtype != mx.uint32:
        return "shared packed weight"
    if (group.scales.shape != (4096, 64) or group.biases is None
            or group.biases.shape != (4096, 64)
            or group.scales.dtype != mx.bfloat16 or group.biases.dtype != mx.bfloat16):
        return "shared hydrated coefficients"
    if (mx.default_device() != mx.gpu or not mx.metal.is_available()
            or not _compatible_runtime()):
        return "runtime"
    return None


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="vmlx_glm5_router_shared_q8g64_bf16_v1",
        input_names=["x", "router_weight", "packed_weight", "packed_scales", "packed_biases"],
        output_names=["logits", "gate_up"], header=_HEADER, source=_SOURCE,
        ensure_row_contiguous=True,
    )


def try_glm5_router_shared(x, router_weight, group, *, enabled, training=False):
    """Return (FP32 logits, BF16 gate, BF16 up), or decline before dispatch.

    A dispatch/evaluation error propagates: no replay of an already-mutated
    outer model cache. Counter means graph construction, not GPU execution.
    """
    global _GRAPH_CALLS
    if admission_reason(x, router_weight, group, enabled=enabled, training=training):
        return None
    logits, gate_up = _kernel()(
        inputs=[x, router_weight, group.weight, group.scales, group.biases],
        template=[("T", mx.bfloat16)], grid=(274 * 128, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(1, 1, 288), (1, 1, 4096)],
        output_dtypes=[mx.float32, mx.bfloat16],
    )
    gate, up = mx.split(gate_up, group.split_indices, axis=-1)
    _GRAPH_CALLS += 1
    return logits, gate, up
