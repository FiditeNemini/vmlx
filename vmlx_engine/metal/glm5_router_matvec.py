# SPDX-License-Identifier: MIT
# Instruction layout derived from MLX v0.32.2 GEMVKernel in gemv.h.
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
"""GLM AR router: original BF16 storage with native FP32 GEMV arithmetic.

Fold the weight/input casts into their loads without retaining an FP32 weight
copy. Preserve MLX's BM4/BN1/SM1/SN32/TM4/TN4 product/reduction order. Selection,
bias, shared experts, prefill, batching, MTP and other families stay stock.
"""

from functools import lru_cache
import importlib.metadata
import logging

import mlx.core as mx

from .affine_moe_pair_decode import affine_moe_ar_scope_active

logger = logging.getLogger(__name__)
_FAILED = False
_OBSERVED = False
_CALL_COUNT = 0

_SOURCE = r'''
    uint lane=thread_index_in_simdgroup;
    uint row=threadgroup_position_in_grid.x*16u
             +simdgroup_index_in_threadgroup*4u;
    float accum[4]={0};
    for(uint base=0u;base<4096u;base+=128u) {
        uint k=base+lane*4u;
        float v[4];
        #pragma clang loop unroll(full)
        for(uint j=0u;j<4u;++j) v[j]=float(x[k+j]);
        #pragma clang loop unroll(full)
        for(uint r=0u;r<4u;++r) {
            #pragma clang loop unroll(full)
            for(uint j=0u;j<4u;++j)
                accum[r]+=float(weight[(row+r)*4096u+k+j])*v[j];
        }
    }
    #pragma clang loop unroll(full)
    for(uint r=0u;r<4u;++r) {
        #pragma clang loop unroll(full)
        for(ushort offset=16u;offset>=1u;offset>>=1u)
            accum[r]+=simd_shuffle_down(accum[r],offset);
        if(lane==0u) output[row+r]=accum[r];
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
def _kernel():
    return mx.fast.metal_kernel(
        name="vmlx_glm5_router_bf16_storage_fp32_math_v1",
        input_names=["x", "weight"], output_names=["output"],
        source=_SOURCE, ensure_row_contiguous=True,
    )


def glm5_router_logits(x, weight, *, enabled: bool, training: bool = False):
    """Return the qualified AR logits, or None to keep the original matmul.

    Match real tensors, never quant labels. Own no model/state arrays or
    persistent converted weights. The owning GLM layer snapshots the opt-in.
    """
    global _FAILED, _OBSERVED, _CALL_COUNT
    if (not enabled or training or _FAILED or not affine_moe_ar_scope_active()
            or x.shape != (1, 1, 4096) or x.dtype != mx.bfloat16
            or weight.shape != (288, 4096) or weight.dtype != mx.bfloat16
            or mx.default_device() != mx.gpu or not mx.metal.is_available()
            or not _compatible_runtime()):
        return None
    try:
        result = _kernel()(
            inputs=[x, weight], grid=(18 * 128, 1, 1), threadgroup=(128, 1, 1),
            output_shapes=[(1, 1, 288)], output_dtypes=[mx.float32],
        )[0]
        if not _OBSERVED:
            mx.eval(result)
            _OBSERVED = True
            logger.info("GLM router matvec active: storage=bf16 compute=fp32 "
                        "hidden=4096 experts=288 retained_fp32_weights=0 "
                        "scope=productive_ar math=mlx0322_gemv_fp32_bf16load_v1")
        _CALL_COUNT += 1
        return result
    except (RuntimeError, ValueError):
        _FAILED = True
        logger.exception("GLM router matvec failed; retaining stock FP32 matmul")
        return None
