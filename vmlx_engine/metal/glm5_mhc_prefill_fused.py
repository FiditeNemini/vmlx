"""GLM-5.3 mHC for prefill rows: (mix + Sinkhorn + collapse + RMSNorm) and placement, one threadgroup per token.

The stock prefill graph per call (x90 per chunk): fp32 copy of the 4 streams, fp32 RMS, a (T x 16384) @ (16384 x 24)
fp32 matmul, sigmoid/softmax, Sinkhorn, collapse, then the layer RMSNorm; placement = bf16 post*out + a per-token
batched (4x4)^T @ (4x4096) matmul. Measured at T=2048: 3.76 ms + 1.39 ms per call (2026-10-10) for work that moves
~80 MB. Here:

  mix_norm: 512 threads per token hold 32 of the 16384 stream values each; RMS -> 24 projection partials (hc_fn row
            reads hit cache across tokens) -> SIMD-parallel 4x4 Sinkhorn (same formulas as the decode v3 kernel) ->
            collapse rounded to T exactly like the stock cast -> RMSNorm with MLX's rounding (normalize, round to T,
            then weight multiply). Outputs post (T,4) f32, comb (T,4,4) f32, normalized (T,D) T.
  place:    new[s,d] = T( T(T(post[s]) * out[d]) + T(sum_r T(comb[r,s]) * res[r,d]) ): the stock bf16 rounding sequence
            (bf16 multiply, matmul accumulated in fp32 then rounded, bf16 add).
VMLX_GLM5_MHC_PREFILL_FUSED=0 restores the stock graph.
"""
from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

_OBSERVED = {"mix": 0, "place": 0}


def mhc_prefill_fused_requested() -> bool:
    return os.environ.get("VMLX_GLM5_MHC_PREFILL_FUSED", "1").strip().lower() not in {"", "0", "false", "off", "no"}


@lru_cache(maxsize=4)
def _mix_kernel(H: int, D: int, iters: int):
    assert H == 4
    F = H * D
    TH = 512
    PER = F // TH
    assert F % TH == 0 and D % TH == 0
    src = f"""
    const uint t = threadgroup_position_in_grid.x;
    const uint tid = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    threadgroup float red[{TH // 32}][24];
    threadgroup float mix[24];
    threadgroup float pre_v[4], post_v[4], mat[16];
    threadgroup float scal[2];
    device const T* st = streams + (size_t)t * {F}u;
    float v[{PER}];
    float ss = 0.0f;
    for (uint j = 0; j < {PER}u; ++j) {{ v[j] = (float)st[tid + j * {TH}u]; ss += v[j] * v[j]; }}
    ss = simd_sum(ss);
    if (lane == 0u) red[sg][0] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {{
        float tot = 0.0f;
        for (uint g = 0; g < {TH // 32}u; ++g) tot += red[g][0];
        scal[0] = metal::rsqrt(tot / {float(F)!r}f + rms_eps[0]);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = scal[0];
    float acc[24];
    for (uint r = 0; r < 24u; ++r) acc[r] = 0.0f;
    for (uint j = 0; j < {PER}u; ++j) {{
        const uint f = tid + j * {TH}u;
        const float x = v[j] * inv;
        for (uint r = 0; r < 24u; ++r) acc[r] += x * (float)hc_fn[(size_t)r * {F}u + f];
    }}
    for (uint r = 0; r < 24u; ++r) {{
        float p = simd_sum(acc[r]);
        if (lane == 0u) red[sg][r] = p;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 24u) {{
        float tot = 0.0f;
        for (uint g = 0; g < {TH // 32}u; ++g) tot += red[g][tid];
        mix[tid] = tot;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {{
        const float s0 = (float)hc_scale[0], s1 = (float)hc_scale[1], s2 = (float)hc_scale[2], eps = sink_eps[0];
        if (lane < 4u) {{
            pre_v[lane] = 1.0f / (1.0f + metal::exp(-(mix[lane] * s0 + (float)hc_base[lane]))) + eps;
            post_v[lane] = 2.0f / (1.0f + metal::exp(-(mix[4u + lane] * s1 + (float)hc_base[4u + lane])));
        }}
        const uint e = lane & 15u;
        float item = mix[8u + e] * s2 + (float)hc_base[8u + e];
        float m = metal::max(item, simd_shuffle_xor(item, 1u)); m = metal::max(m, simd_shuffle_xor(m, 2u));
        float ex = metal::exp(item - m);
        float rs = ex + simd_shuffle_xor(ex, 1u); rs = rs + simd_shuffle_xor(rs, 2u);
        float c = ex / rs + eps;
        float cs = c + simd_shuffle_xor(c, 4u); cs = cs + simd_shuffle_xor(cs, 8u);
        c = c / (cs + eps);
        for (uint it = 1u; it < {iters}u; ++it) {{
            rs = c + simd_shuffle_xor(c, 1u); rs = rs + simd_shuffle_xor(rs, 2u); c = c / (rs + eps);
            cs = c + simd_shuffle_xor(c, 4u); cs = cs + simd_shuffle_xor(cs, 8u); c = c / (cs + eps);
        }}
        if (lane < 16u) mat[lane] = c;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 4u) post[(size_t)t * 4u + tid] = post_v[tid];
    if (tid < 16u) comb[(size_t)t * 16u + tid] = mat[tid];
    // collapse (stock: fp32 sum over streams, cast to T) + RMSNorm (MLX: normalize, round to T, weight multiply)
    float xc[{D // TH}];
    float s2 = 0.0f;
    for (uint j = 0; j < {D // TH}u; ++j) {{
        const uint d = tid + j * {TH}u;
        float a = 0.0f;
        for (uint s = 0; s < 4u; ++s) a += pre_v[s] * (float)st[s * {D}u + d];
        xc[j] = (float)(T)a;
        s2 += xc[j] * xc[j];
    }}
    s2 = simd_sum(s2);
    if (lane == 0u) red[sg][0] = s2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {{
        float tot = 0.0f;
        for (uint g = 0; g < {TH // 32}u; ++g) tot += red[g][0];
        scal[1] = metal::precise::rsqrt(tot / {float(D)!r}f + norm_eps[0]);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < {D // TH}u; ++j) {{
        const uint d = tid + j * {TH}u;
        normalized[(size_t)t * {D}u + d] = norm_weight[d] * static_cast<T>(xc[j] * scal[1]);
    }}
"""
    return mx.fast.metal_kernel(
        name=f"vmlx_glm5_mhc_mix_norm_prefill_h{H}_d{D}_i{iters}",
        input_names=["streams", "hc_fn", "hc_base", "hc_scale", "rms_eps", "sink_eps", "norm_weight", "norm_eps"],
        output_names=["post", "comb", "normalized"], header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=src, ensure_row_contiguous=True)


@lru_cache(maxsize=8)
def _mix_kernel_multi(H: int, D: int, iters: int, TOK: int):
    """_mix_kernel with TOK tokens per threadgroup (2026-10-10): each hc_fn element is loaded once per threadgroup and
    used for TOK tokens (the one-token kernel re-reads the 24 x 4D projection for every token). Every per-token
    operation keeps the one-token kernel's order, so outputs are bit-identical. A tail threadgroup masks rows >= T."""
    assert H == 4 and TOK >= 2
    F = H * D
    TH = 512
    PER = F // TH
    assert F % TH == 0 and D % TH == 0 and TOK <= TH // 32
    src = f"""
    const uint t0 = threadgroup_position_in_grid.x * {TOK}u;
    const uint tid = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint nr = (uint)n_rows[0];
    threadgroup float red[{TOK}][{TH // 32}][24];
    threadgroup float mix[{TOK}][24];
    threadgroup float pre_v[{TOK}][4], post_v[{TOK}][4], mat[{TOK}][16];
    threadgroup float scal[{TOK}][2];
    float v[{TOK}][{PER}];
    for (uint k = 0; k < {TOK}u; ++k) {{
        const bool ok = t0 + k < nr;
        device const T* st = streams + (size_t)(ok ? t0 + k : 0u) * {F}u;
        float ss = 0.0f;
        for (uint j = 0; j < {PER}u; ++j) {{ v[k][j] = ok ? (float)st[tid + j * {TH}u] : 0.0f; ss += v[k][j] * v[k][j]; }}
        ss = simd_sum(ss);
        if (lane == 0u) red[k][sg][0] = ss;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < {TOK}u) {{
        float tot = 0.0f;
        for (uint g = 0; g < {TH // 32}u; ++g) tot += red[tid][g][0];
        scal[tid][0] = metal::rsqrt(tot / {float(F)!r}f + rms_eps[0]);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv[{TOK}];
    for (uint k = 0; k < {TOK}u; ++k) inv[k] = scal[k][0];
    float acc[{TOK}][24];
    for (uint k = 0; k < {TOK}u; ++k) for (uint r = 0; r < 24u; ++r) acc[k][r] = 0.0f;
    for (uint j = 0; j < {PER}u; ++j) {{
        const uint f = tid + j * {TH}u;
        float x[{TOK}];
        for (uint k = 0; k < {TOK}u; ++k) x[k] = v[k][j] * inv[k];
        for (uint r = 0; r < 24u; ++r) {{
            const float w = (float)hc_fn[(size_t)r * {F}u + f];
            for (uint k = 0; k < {TOK}u; ++k) acc[k][r] += x[k] * w;
        }}
    }}
    for (uint k = 0; k < {TOK}u; ++k) for (uint r = 0; r < 24u; ++r) {{
        float p = simd_sum(acc[k][r]);
        if (lane == 0u) red[k][sg][r] = p;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < {TOK * 24}u) {{
        const uint k = tid / 24u, r = tid % 24u;
        float tot = 0.0f;
        for (uint g = 0; g < {TH // 32}u; ++g) tot += red[k][g][r];
        mix[k][r] = tot;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg < {TOK}u) {{
        const uint k = sg;
        const float s0 = (float)hc_scale[0], s1 = (float)hc_scale[1], s2 = (float)hc_scale[2], eps = sink_eps[0];
        if (lane < 4u) {{
            pre_v[k][lane] = 1.0f / (1.0f + metal::exp(-(mix[k][lane] * s0 + (float)hc_base[lane]))) + eps;
            post_v[k][lane] = 2.0f / (1.0f + metal::exp(-(mix[k][4u + lane] * s1 + (float)hc_base[4u + lane])));
        }}
        const uint e = lane & 15u;
        float item = mix[k][8u + e] * s2 + (float)hc_base[8u + e];
        float m = metal::max(item, simd_shuffle_xor(item, 1u)); m = metal::max(m, simd_shuffle_xor(m, 2u));
        float ex = metal::exp(item - m);
        float rs = ex + simd_shuffle_xor(ex, 1u); rs = rs + simd_shuffle_xor(rs, 2u);
        float c = ex / rs + eps;
        float cs = c + simd_shuffle_xor(c, 4u); cs = cs + simd_shuffle_xor(cs, 8u);
        c = c / (cs + eps);
        for (uint it = 1u; it < {iters}u; ++it) {{
            rs = c + simd_shuffle_xor(c, 1u); rs = rs + simd_shuffle_xor(rs, 2u); c = c / (rs + eps);
            cs = c + simd_shuffle_xor(c, 4u); cs = cs + simd_shuffle_xor(cs, 8u); c = c / (cs + eps);
        }}
        if (lane < 16u) mat[k][lane] = c;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < {TOK}u; ++k) {{
        if (t0 + k < nr) {{
            if (tid < 4u) post[(size_t)(t0 + k) * 4u + tid] = post_v[k][tid];
            if (tid < 16u) comb[(size_t)(t0 + k) * 16u + tid] = mat[k][tid];
        }}
    }}
    float xc[{TOK}][{D // TH}];
    for (uint k = 0; k < {TOK}u; ++k) {{
        const bool ok = t0 + k < nr;
        device const T* st = streams + (size_t)(ok ? t0 + k : 0u) * {F}u;
        float s2 = 0.0f;
        for (uint j = 0; j < {D // TH}u; ++j) {{
            const uint d = tid + j * {TH}u;
            float a = 0.0f;
            for (uint s = 0; s < 4u; ++s) a += pre_v[k][s] * (ok ? (float)st[s * {D}u + d] : 0.0f);
            xc[k][j] = (float)(T)a;
            s2 += xc[k][j] * xc[k][j];
        }}
        s2 = simd_sum(s2);
        if (lane == 0u) red[k][sg][0] = s2;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < {TOK}u) {{
        float tot = 0.0f;
        for (uint g = 0; g < {TH // 32}u; ++g) tot += red[tid][g][0];
        scal[tid][1] = metal::precise::rsqrt(tot / {float(D)!r}f + norm_eps[0]);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < {TOK}u; ++k) {{
        if (t0 + k >= nr) continue;
        for (uint j = 0; j < {D // TH}u; ++j) {{
            const uint d = tid + j * {TH}u;
            normalized[(size_t)(t0 + k) * {D}u + d] = norm_weight[d] * static_cast<T>(xc[k][j] * scal[k][1]);
        }}
    }}
"""
    return mx.fast.metal_kernel(
        name=f"vmlx_glm5_mhc_mix_norm_prefill_h{H}_d{D}_i{iters}_t{TOK}",
        input_names=["streams", "hc_fn", "hc_base", "hc_scale", "rms_eps", "sink_eps", "norm_weight", "norm_eps", "n_rows"],
        output_names=["post", "comb", "normalized"], header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=src, ensure_row_contiguous=True)


def mhc_prefill_tokens_per_group() -> int:
    """VMLX_GLM5_MHC_PREFILL_TOK: tokens per threadgroup of the mix kernel (default 2: 1.30 -> 1.01 ms per T=2048 call,
    bit-identical; 4 spills registers, 1.97 ms; 1 = the one-token kernel); read per call."""
    try:
        return max(1, int(os.environ.get("VMLX_GLM5_MHC_PREFILL_TOK", "2")))
    except ValueError:
        return 1


@lru_cache(maxsize=4)
def _place_kernel(H: int, D: int):
    src = f"""
    const uint gid = thread_position_in_grid.x;
    const uint t = gid / {D}u, d = gid % {D}u;
    if (t >= (uint)n_rows[0]) return;
    const float o = (float)out[(size_t)t * {D}u + d];
    float r[{H}];
    for (uint q = 0; q < {H}u; ++q) r[q] = (float)residual[((size_t)t * {H}u + q) * {D}u + d];
    for (uint s = 0; s < {H}u; ++s) {{
        const float p = (float)(T)post[(size_t)t * {H}u + s];
        const float a = (float)(T)(p * o);
        float m = 0.0f;
        for (uint q = 0; q < {H}u; ++q) m += (float)(T)comb[((size_t)t * {H}u + q) * {H}u + s] * r[q];
        new_streams[((size_t)t * {H}u + s) * {D}u + d] = (T)(a + (float)(T)m);
    }}
"""
    return mx.fast.metal_kernel(name=f"vmlx_glm5_hc_place_prefill_h{H}_d{D}",
                                input_names=["post", "comb", "out", "residual", "n_rows"], output_names=["new_streams"],
                                header="#include <metal_stdlib>\nusing namespace metal;\n", source=src,
                                ensure_row_contiguous=True)


def glm5_mhc_mix_norm_prefill(streams, hc, norm):
    """streams (1, T, 4, D) -> (post (1,T,4) f32, comb (1,T,4,4) f32, normalized (1,T,D)) or None."""
    if streams.ndim != 4 or streams.shape[0] != 1 or streams.shape[2] != 4 or streams.dtype not in (mx.bfloat16, mx.float16):
        return None
    T, D = int(streams.shape[1]), int(streams.shape[3])
    if D % 512 or tuple(hc.hc_fn.shape) != (24, 4 * D) or norm.weight.shape != (D,) or norm.weight.dtype != streams.dtype:
        return None
    ins = [streams, hc.hc_fn, hc.hc_base, hc.hc_scale, mx.array([hc.rms_eps], mx.float32),
           mx.array([hc.eps], mx.float32), norm.weight, mx.array([norm.eps], mx.float32)]
    outs = dict(template=[("T", streams.dtype)], threadgroup=(512, 1, 1),
                output_shapes=[(1, T, 4), (1, T, 4, 4), (1, T, D)], output_dtypes=[mx.float32, mx.float32, streams.dtype])
    tok = mhc_prefill_tokens_per_group()
    if tok > 1 and T > 1:
        post, comb, normalized = _mix_kernel_multi(4, D, int(hc.iters), tok)(
            inputs=ins + [mx.array([T], mx.int32)], grid=(((T + tok - 1) // tok) * 512, 1, 1), **outs)
    else:
        post, comb, normalized = _mix_kernel(4, D, int(hc.iters))(inputs=ins, grid=(T * 512, 1, 1), **outs)
    _OBSERVED["mix"] += 1
    return post, comb, normalized


def glm5_hc_place_prefill(post, comb, out, residual):
    if residual.ndim != 4 or residual.shape[0] != 1 or residual.shape[2] != 4 or out.dtype != residual.dtype:
        return None
    T, D = int(residual.shape[1]), int(residual.shape[3])
    if out.shape != (1, T, D) or post.shape != (1, T, 4) or comb.shape != (1, T, 4, 4):
        return None
    y = _place_kernel(4, D)(inputs=[post, comb, out, residual, mx.array([T], mx.int32)], template=[("T", residual.dtype)],
                            grid=(T * D, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(1, T, 4, D)],
                            output_dtypes=[residual.dtype])[0]
    _OBSERVED["place"] += 1
    return y


__all__ = ["mhc_prefill_fused_requested", "glm5_mhc_mix_norm_prefill", "glm5_hc_place_prefill"]
