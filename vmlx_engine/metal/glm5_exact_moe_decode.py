# SPDX-License-Identifier: MIT
# Projection instruction patterns derived from MLX v0.32.2 quantized.h.
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
"""Opt-in GLM AR expert fusion preserving native q2/BF16 arithmetic.

This is not the older float SwiGLU/weighted-down candidate. Preserve MLX's
qmv_fast accumulation tree, stock activation, BF16 projection and weighted
product rounding, and BF16 col_reduce_small additions. No tensor repacking,
retained arrays, cross-group atomics, or change to router/native cache state.
"""

from functools import lru_cache
import importlib.metadata
import logging

import mlx.core as mx

from .affine_moe_pair_decode import affine_moe_ar_scope_active, _projection_reason

logger = logging.getLogger(__name__)
_FAILED = False
_OBSERVED = False
_CALL_COUNT = 0

_HEADER = r'''
#include <metal_stdlib>
using namespace metal;
template<typename T>
inline float glm_q2_load(const device T* x, thread float* xt) {
    float sum=0;
    for(int i=0;i<16;i+=4) {
        sum += x[i]+x[i+1]+x[i+2]+x[i+3];
        xt[i]=x[i]; xt[i+1]=x[i+1]/4.0f;
        xt[i+2]=x[i+2]/16.0f; xt[i+3]=x[i+3]/64.0f;
    }
    return sum;
}
inline float glm_q2_dot(const device uint8_t* w,const thread float* xt,
                       float scale,float bias,float sum) {
    float accum=0;
    for(int i=0;i<4;++i) {
        accum += (xt[4*i]*(w[i]&0x03)+xt[4*i+1]*(w[i]&0x0c)
                  +xt[4*i+2]*(w[i]&0x30)+xt[4*i+3]*(w[i]&0xc0));
    }
    return scale*accum+sum*bias;
}
'''

_PAIR = r'''
    uint group=thread_position_in_grid.x/64u;
    uint route=group/(N/8u),tile=group%(N/8u);
    uint lane=thread_index_in_simdgroup;
    uint first_row=tile*8u+simdgroup_index_in_threadgroup*4u;
    uint expert=expert_ids[route];
    const device uint8_t* wg=(const device uint8_t*)gate_weight;
    const device uint8_t* wu=(const device uint8_t*)up_weight;
    float ga[4]={0},ua[4]={0},xt[16];
    for(int k=0;k<K;k+=512) {
        uint input_start=k+lane*16u;
        float sum=glm_q2_load(x+input_start,xt);
        for(int r=0;r<4;++r) {
            size_t row=(size_t)expert*N+first_row+r;
            size_t packed=row*(K/4u)+input_start/4u;
            size_t meta=row*(K/128u)+input_start/128u;
            ga[r]+=glm_q2_dot(wg+packed,xt,float(gate_scales[meta]),float(gate_biases[meta]),sum);
            ua[r]+=glm_q2_dot(wu+packed,xt,float(up_scales[meta]),float(up_biases[meta]),sum);
        }
    }
    for(int r=0;r<4;++r) {
        ga[r]=simd_sum(ga[r]);ua[r]=simd_sum(ua[r]);
        if(lane==0u) {
            size_t out=(size_t)route*N+first_row+r;
            gate_out[out]=T(ga[r]);up_out[out]=T(ua[r]);
        }
    }
'''

_DOWN = r'''
    uint tile=threadgroup_position_in_grid.x;
    uint route=simdgroup_index_in_threadgroup;
    uint lane=thread_index_in_simdgroup;
    uint first_row=tile*4u,expert=expert_ids[route];
    const device uint8_t* wb=(const device uint8_t*)weight;
    float accum[4]={0},xt[16];
    for(int k=0;k<K;k+=512) {
        uint input_start=k+lane*16u;
        float sum=glm_q2_load(activated+(size_t)route*K+input_start,xt);
        for(int r=0;r<4;++r) {
            size_t row=(size_t)expert*N+first_row+r;
            size_t packed=row*(K/4u)+input_start/4u;
            size_t meta=row*(K/128u)+input_start/128u;
            accum[r]+=glm_q2_dot(wb+packed,xt,float(scales[meta]),float(biases[meta]),sum);
        }
    }
    threadgroup T partial[8*4];
    for(int r=0;r<4;++r) {
        accum[r]=simd_sum(accum[r]);
        if(lane==0u) {
            T projection=T(accum[r]);
            T weighted=T(projection*route_scores[route]);
            partial[route*4u+r]=T(weighted+T(0));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(route==0u && lane<4u) {
        T total=partial[lane];
        for(uint i=1u;i<8u;++i) total=T(partial[i*4u+lane]+total);
        output[first_row+lane]=total;
    }
'''


@lru_cache(maxsize=1)
def _compatible_runtime() -> bool:
    try:
        return (importlib.metadata.version("mlx") == "0.32.2"
                and "Apple M5" in mx.device_info().get("device_name", ""))
    except (importlib.metadata.PackageNotFoundError, RuntimeError):
        return False


@lru_cache(maxsize=1)
def _kernels():
    pair = mx.fast.metal_kernel(
        name="vmlx_glm5_exact_q2_pair_v1", header=_HEADER, source=_PAIR,
        input_names=["x", "gate_weight", "gate_scales", "gate_biases",
                     "up_weight", "up_scales", "up_biases", "expert_ids"],
        output_names=["gate_out", "up_out"], ensure_row_contiguous=True)
    down = mx.fast.metal_kernel(
        name="vmlx_glm5_exact_q2_down_v1", header=_HEADER, source=_DOWN,
        input_names=["activated", "weight", "scales", "biases",
                     "expert_ids", "route_scores"],
        output_names=["output"], ensure_row_contiguous=True)
    return pair, down


def _eligible(switch, x, indices, scores) -> bool:
    if (x.shape != (1, 1, 4096) or x.dtype != mx.bfloat16
            or indices.shape != (1, 1, 8) or scores.shape != indices.shape
            or indices.dtype not in (mx.int32, mx.uint32, mx.int64, mx.uint64)
            or scores.dtype not in (mx.bfloat16, mx.float32)
            or bool(getattr(switch, "training", False))
            or switch.activation.__class__.__name__ != "ClampedSwiGLU"
            or getattr(switch.activation, "_limit", None) != 10.0):
        return False
    experts = None
    for name, hidden, intermediate in (("gate_proj", 4096, 2048),
                                       ("up_proj", 4096, 2048),
                                       ("down_proj", 2048, 4096)):
        proj = getattr(switch, name, None)
        if proj is None or _projection_reason(proj, hidden=hidden, intermediate=intermediate):
            return False
        if (proj.bits != 2 or proj.group_size != 128
                or proj.scales.dtype != mx.bfloat16 or proj.biases.dtype != mx.bfloat16):
            return False
        if experts is None:
            experts = proj.weight.shape[0]
        if proj.weight.shape[0] != experts or experts < 8:
            return False
    return True


def glm5_exact_moe_output(switch, x, indices, scores, *, enabled: bool):
    """Return a qualified productive-AR result, or None for stock SwitchGLU.

    Eligibility uses actual per-projection layout, not the bundle name. The
    outer flag is snapshotted when the GLM layer is constructed. Prefill,
    batching, MTP seed/draft/verify and other architectures remain unchanged.
    """
    global _FAILED, _OBSERVED, _CALL_COUNT
    if (not enabled or _FAILED or not affine_moe_ar_scope_active()
            or mx.default_device() != mx.gpu or not mx.metal.is_available()
            or not _compatible_runtime() or not _eligible(switch, x, indices, scores)):
        return None
    try:
        pair, down = _kernels()
        gate, up = switch.gate_proj, switch.up_proj
        ids = indices.reshape(-1).astype(mx.uint32)
        g, u = pair(inputs=[x.reshape(-1), gate.weight, gate.scales, gate.biases,
                             up.weight, up.scales, up.biases, ids],
                    template=[("T", x.dtype), ("K", 4096), ("N", 2048)],
                    grid=(64*256*8, 1, 1), threadgroup=(64, 1, 1),
                    output_shapes=[(1, 1, 8, 1, 2048)]*2, output_dtypes=[x.dtype]*2)
        activated = switch.activation(u, g)
        proj = switch.down_proj
        output = down(inputs=[activated.reshape(8, 2048), proj.weight,
                              proj.scales, proj.biases, ids,
                              scores.reshape(-1).astype(x.dtype)],
                      template=[("T", x.dtype), ("K", 2048), ("N", 4096)],
                      grid=(256*1024, 1, 1), threadgroup=(256, 1, 1),
                      output_shapes=[x.shape], output_dtypes=[x.dtype])[0]
        if not _OBSERVED:
            mx.eval(output)
            _OBSERVED = True
            logger.info("GLM exact MoE decode active: q2/g128 bf16 hidden=4096 "
                        "intermediate=2048 top_k=8 scope=productive_ar "
                        "math=mlx0322_q2g128_bf16_v1")
        _CALL_COUNT += 1
        return output
    except (RuntimeError, ValueError):
        _FAILED = True
        logger.exception("GLM exact MoE decode failed; retaining stock expert path")
        return None
