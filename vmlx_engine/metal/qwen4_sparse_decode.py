"""Experimental QSA AR traversal preserving MLX's logical SDPA partitions.

No default/runtime admission. The bitmap removes repeated masked traversal,
not any selected token or reduction lane. Context bounds are dispatch guards.
"""

from functools import lru_cache
import os

import mlx.core as mx


@lru_cache(maxsize=1)
def _hardware_allowed():
    return (mx.__version__ == "0.32.2"
            and mx.device_info().get("architecture") == "applegpu_g17s")


@lru_cache(maxsize=1)
def _bitmap():
    return mx.fast.metal_kernel(
        name="vmlx_qsa_ar_bitmap1024", input_names=["mask"],
        output_names=["bitmap"], ensure_row_contiguous=False,
        source="""
        uint part = thread_position_in_grid.x;
        if (part >= 1024) return;
        uint bits[4] = {0, 0, 0, 0};
        uint step = 0;
        for (uint p = part; p < uint(mask_shape[3]); p += 1024, ++step) {
            if (float(mask[p * mask_strides[3]]) >= -65504.0f)
                bits[step / 32] |= (1u << (step % 32));
        }
        for (uint w = 0; w < 4; ++w) bitmap[part * 4 + w] = bits[w];
        """,
    )


@lru_cache(maxsize=1)
def _selected_bitmap():
    """Construct the identical partition bits without a full-context mask."""
    return mx.fast.metal_kernel(
        name="vmlx_qsa_ar_selected_bitmap1024",
        input_names=["selected", "valid", "length"], output_names=["bitmap"],
        atomic_outputs=True, ensure_row_contiguous=False,
        source="""
        uint i = thread_position_in_grid.x;
        uint n = uint(length[0]);
        uint complete = n / 4u;
        if (i < uint(selected_shape[1])) {
            int block = selected[i * selected_strides[1]];
            if (valid[i * valid_strides[1]] && block >= 0 && uint(block) < complete) {
                for (uint j = 0; j < 4u; ++j) {
                    uint pos = uint(block) * 4u + j;
                    uint step = pos / 1024u;
                    atomic_fetch_or_explicit(&bitmap[(pos % 1024u) * 4u + step / 32u],
                                             1u << (step % 32u), memory_order_relaxed);
                }
            }
        }
        if (i == uint(selected_shape[1])) {
            for (uint pos = complete * 4u; pos < n; ++pos) {
                uint step = pos / 1024u;
                atomic_fetch_or_explicit(&bitmap[(pos % 1024u) * 4u + step / 32u],
                                         1u << (step % 32u), memory_order_relaxed);
            }
        }
        """,
    )


# The two-pass arithmetic is adapted from MLX v0.32.2 sdpa_vector.h.
# Copyright (c) 2024 Apple Inc. Licensed under the MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
@lru_cache(maxsize=1)
def _partials():
    return mx.fast.metal_kernel(
        name="vmlx_qsa_ar_partials1024", input_names=["q", "k", "v", "mask", "bitmap"],
        output_names=["partials", "sums", "maxs"], ensure_row_contiguous=False,
        source="""
        uint kv = threadgroup_position_in_grid.x;
        uint part = threadgroup_position_in_grid.z;
        uint head = kv * 12 + thread_position_in_threadgroup.y;
        uint lane = thread_index_in_simdgroup;
        float query[8], acc[8] = {0};
        for (uint j = 0; j < 8; ++j)
            query[j] = 0.0625f * float(q[head * q_strides[1] + (lane * 8 + j) * q_strides[3]]);
        float max_score = -3.402823466e+38f;
        float sum_exp_score = 0;
        for (uint word = 0; word < 4; ++word) {
            uint bits = bitmap[part * 4 + word];
            while (bits != 0) {
                uint bit = metal::ctz(bits);
                uint pos = part + (word * 32 + bit) * 1024;
                bits &= bits - 1;
                size_t ki = kv * k_strides[1] + size_t(pos) * k_strides[2];
                size_t vi = kv * v_strides[1] + size_t(pos) * v_strides[2];
                float score = 0;
                for (uint j = 0; j < 8; ++j)
                    score += query[j] * k[ki + (lane * 8 + j) * k_strides[3]];
                score = simd_sum(score);
                score += mask[pos * mask_strides[3]];
                float new_max = metal::max(max_score, score);
                float factor = metal::fast::exp(max_score - new_max);
                float exp_score = metal::fast::exp(score - new_max);
                max_score = new_max;
                sum_exp_score = sum_exp_score * factor + exp_score;
                for (uint j = 0; j < 8; ++j)
                    acc[j] = acc[j] * factor + exp_score * v[vi + (lane * 8 + j) * v_strides[3]];
            }
        }
        if (lane == 0) {
            sums[head * 1024 + part] = sum_exp_score;
            maxs[head * 1024 + part] = max_score;
        }
        for (uint j = 0; j < 8; ++j)
            partials[(head * 1024 + part) * 256 + lane * 8 + j] = acc[j];
        """,
    )


@lru_cache(maxsize=1)
def _finish():
    return mx.fast.metal_kernel(
        name="vmlx_qsa_ar_finish1024", input_names=["partials", "sums", "maxs"],
        output_names=["out"],
        source="""
        uint head = threadgroup_position_in_grid.x;
        uint group = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        threadgroup float shared[1024];
        float acc[8] = {0};
        float denominator = 0, maximum = -3.402823466e+38f;
        for (uint b = 0; b < 32; ++b)
            maximum = metal::max(maximum, maxs[head * 1024 + lane + 32 * b]);
        maximum = simd_max(maximum);
        for (uint b = 0; b < 32; ++b) {
            uint index = head * 1024 + lane + 32 * b;
            float factor = metal::fast::exp(maxs[index] - maximum);
            denominator += factor * sums[index];
        }
        denominator = simd_sum(denominator);
        for (uint b = 0; b < 32; ++b) {
            uint index = head * 1024 + group + 32 * b;
            float factor = metal::fast::exp(maxs[index] - maximum);
            for (uint j = 0; j < 8; ++j)
                acc[j] += factor * float(partials[index * 256 + lane * 8 + j]);
        }
        for (uint j = 0; j < 8; ++j) {
            shared[lane * 32 + group] = acc[j];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            acc[j] = simd_sum(shared[group * 32 + lane]);
            acc[j] = denominator == 0 ? acc[j] : acc[j] / denominator;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0)
            for (uint j = 0; j < 8; ++j) out[head * 256 + group * 8 + j] = acc[j];
        """,
    )


def supported(q, k, v, *, scale):
    """Shape-only admission before the caller chooses an indexer output form."""
    return not (q.shape != (1, 24, 1, 256)
            or k.ndim != 4 or k.shape[:2] != (1, 2) or k.shape[3] != 256
            or v.shape != k.shape or not 65537 <= k.shape[2] <= 131072
            or q.dtype != mx.float16 or k.dtype != q.dtype or v.dtype != q.dtype
            or scale != 0.0625 or os.environ.get("MLX_SDPA_BLOCKS")
            or mx.default_device() != mx.gpu or not _hardware_allowed())


def attention(q, k, v, mask, *, scale, enabled=False):
    """Return the optional AR result, or None for the unmodified caller path."""
    if (not enabled or not supported(q, k, v, scale=scale)
            or mask is None or mask.shape != (1, 1, 1, k.shape[2]) or mask.dtype != q.dtype):
        return None
    bitmap = _bitmap()(inputs=[mask], grid=(1024, 1, 1), threadgroup=(256, 1, 1),
                       output_shapes=[(1024, 4)], output_dtypes=[mx.uint32])[0]
    return _attention_from_bitmap(q, k, v, mask, bitmap)


def attention_from_blocks(q, k, v, selected, valid, *, scale, enabled=False):
    """QSA's selected four-token blocks plus its incomplete tail, AR only.

    The bitmap is order independent; attention still traverses absolute keys
    in the original partition order. No cache arrays or quantization change.
    Only the exact QSA zero/-infinity mask contract is represented here; the
    general additive-mask path above retains finite biases and empty masks.
    """
    if (not enabled or not supported(q, k, v, scale=scale)
            or selected.shape != (1, 512) or selected.dtype != mx.int32
            or valid.shape != selected.shape or valid.dtype != mx.bool_):
        return None
    tokens = k.shape[2]
    bitmap = _selected_bitmap()(
        inputs=[selected, valid, mx.array([tokens], dtype=mx.uint32)],
        grid=(513, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(1024, 4)], output_dtypes=[mx.uint32], init_value=0,
    )[0]
    # Only selected keys are visited. A stride-zero view supplies their exact
    # additive +0 without allocating/casting/scanning a context-sized mask.
    mask = mx.broadcast_to(mx.zeros((1,), dtype=q.dtype), (1, 1, 1, tokens))
    return _attention_from_bitmap(q, k, v, mask, bitmap)


def _attention_from_bitmap(q, k, v, mask, bitmap):
    partials, sums, maxs = _partials()(
        inputs=[q, k, v, mask, bitmap], grid=(64, 12, 1024), threadgroup=(32, 12, 1),
        output_shapes=[(24, 1024, 256), (24, 1024), (24, 1024)],
        output_dtypes=[q.dtype, mx.float32, mx.float32])
    return _finish()(inputs=[partials, sums, maxs], grid=(24 * 1024, 1, 1),
                     threadgroup=(1024, 1, 1), output_shapes=[q.shape],
                     output_dtypes=[q.dtype])[0]
