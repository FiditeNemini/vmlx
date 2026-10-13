"""One-dispatch GLM KDA decode middle: conv + l2norm + gates + recurrent update + gated RMSNorm (2026-10-10).

The batch-one decode path between the q/k/v projection and o_proj used ~20 dependent dispatches per KDA layer
(fused conv, two l2norms, gate/beta elementwise, the recurrent step, the unfused gated RMSNorm). In GLM's dependent
decode chain that cost ~4.9 ms per token over 34 layers (glmt.decode_profile sub-ablation), and the step kernel read
the state column-wise (512-byte lane stride). Here one threadgroup per head does all of it:

  phase 1 (384 threads): q/k/v short conv + silu, rounded to the activation dtype exactly like kda_conv_decode;
           shifted conv states; decay gate lower_bound*sigmoid(exp(A_log)*(f+dt_bias)) and beta = sigmoid(b);
           sums q.q, k.k, q.k for the l2norms (fp32, eps 1e-6) and the output identity below.
  phase 2 (512 threads = 128 value columns x 4 key slices): S' = S*exp(g) read ONCE, coalesced along V, 32 values per
           thread in registers; kS and q.S' partials reduced through threadgroup memory; S_new = S' + beta*k*(v-kS)
           written once; o = q.S' + beta*(v-kS)*(q.k)  (= q.S_new, algebraically).
  phase 3: o * rsqrt(mean(o^2)+eps) * o_norm * sigmoid(gate) -> activation dtype.
Same formulas as kda.kda_step / short_conv / the stock gated norm; summation order differs (fp32).
VMLX_GLM5_FUSED_KDA_DECODE=0 restores the multi-dispatch path.
"""
from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

_OBSERVED = 0


def fused_kda_decode_requested() -> bool:
    return os.environ.get("VMLX_GLM5_FUSED_KDA_DECODE", "1").strip().lower() not in {"", "0", "false", "off", "no"}


@lru_cache(maxsize=8)
def _kernel(H: int, K: int, W: int, lower_bound: float, rms_eps: float):
    assert K == 128 and W >= 2
    C = H * K
    src = f"""
    const uint h = threadgroup_position_in_grid.x;
    const uint tid = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    threadgroup float qv[{K}], kv[{K}], vv[{K}], eg[{K}];
    threadgroup float red[16][3];
    threadgroup float scal[4];
    threadgroup float pA[4][{K}], pB[4][{K}];
    threadgroup float ov[{K}];

    // ---- phase 1: conv (+silu, rounded to T), gates
    if (tid < {3 * K}u) {{
        const uint which = tid / {K}u;           // 0 q, 1 k, 2 v
        const uint i = tid % {K}u;
        const uint c = h * {K}u + i;
        device const T* tok = which == 0u ? q_tok : (which == 1u ? k_tok : v_tok);
        device const T* st = which == 0u ? q_state : (which == 1u ? k_state : v_state);
        device const T* wt = which == 0u ? q_w : (which == 1u ? k_w : v_w);
        device T* nx = which == 0u ? q_next : (which == 1u ? k_next : v_next);
        float acc = 0.0f;
        for (uint tap = 0; tap < {W - 1}u; ++tap)
            acc += (float)st[(size_t)tap * {C}u + c] * (float)wt[(size_t)c * {W}u + tap];
        float cur = (float)tok[c];
        acc += cur * (float)wt[(size_t)c * {W}u + {W - 1}u];
        for (uint tap = 0; tap < {W - 2}u; ++tap)
            nx[(size_t)tap * {C}u + c] = st[(size_t)(tap + 1u) * {C}u + c];
        nx[(size_t){W - 2}u * {C}u + c] = tok[c];
        float y = (float)(T)(acc / (1.0f + metal::exp(-acc)));
        if (which == 0u) qv[i] = y; else if (which == 1u) kv[i] = y; else vv[i] = y;
    }} else {{
        const uint i = tid - {3 * K}u;          // 0..127: decay gate
        const uint c = h * {K}u + i;
        float f = (float)f_raw[c] + (float)dt_bias[c];
        float rate = metal::exp((float)A_log[h]);
        float g = {float(lower_bound)!r}f * (1.0f / (1.0f + metal::exp(-(rate * f))));
        eg[i] = metal::exp(g);
        if (i == 0u) scal[0] = 1.0f / (1.0f + metal::exp(-(float)b_raw[h]));   // beta
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // sums q.q, k.k, q.k (threads 0..127, 4 simdgroups)
    if (tid < {K}u) {{
        float a = qv[tid], b = kv[tid];
        float s0 = simd_sum(a * a), s1 = simd_sum(b * b), s2 = simd_sum(a * b);
        if (lane == 0u) {{ red[sg][0] = s0; red[sg][1] = s1; red[sg][2] = s2; }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {{
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
        for (uint j = 0; j < {K // 32}u; ++j) {{ s0 += red[j][0]; s1 += red[j][1]; s2 += red[j][2]; }}
        float iq = metal::rsqrt(s0 + 1e-6f), ik = metal::rsqrt(s1 + 1e-6f);
        scal[1] = iq * {float(K ** -0.5)!r}f;        // q scale (l2norm * K^-1/2)
        scal[2] = ik;                            // k l2norm
        scal[3] = s2 * iq * {float(K ** -0.5)!r}f * ik;   // (q_scaled . k_normed)
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- phase 2: recurrent update; thread = (value column vi, key slice ks)
    const uint vi = tid % {K}u;
    const uint ks = tid / {K}u;                 // 0..3
    const float qs = scal[1], kn = scal[2], beta = scal[0];
    float Sd[32];
    float a_part = 0.0f, b_part = 0.0f;
    const size_t sbase = (size_t)h * {K * K}u;
    for (uint j = 0; j < 32u; ++j) {{
        const uint kk = ks * 32u + j;
        float s = state[sbase + (size_t)kk * {K}u + vi] * eg[kk];
        Sd[j] = s;
        a_part += kv[kk] * kn * s;
        b_part += qv[kk] * qs * s;
    }}
    pA[ks][vi] = a_part; pB[ks][vi] = b_part;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float kS = pA[0][vi] + pA[1][vi] + pA[2][vi] + pA[3][vi];
    const float corr = vv[vi] - kS;
    for (uint j = 0; j < 32u; ++j) {{
        const uint kk = ks * 32u + j;
        next_state[sbase + (size_t)kk * {K}u + vi] = Sd[j] + beta * kv[kk] * kn * corr;
    }}
    if (ks == 0u) ov[vi] = (pB[0][vi] + pB[1][vi] + pB[2][vi] + pB[3][vi]) + beta * corr * scal[3];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- phase 3: gated RMSNorm over the head
    if (tid < {K}u) {{
        float o = ov[tid];
        float s = simd_sum(o * o);
        if (lane == 0u) red[sg][0] = s;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < {K}u) {{
        float s = 0.0f;
        for (uint j = 0; j < {K // 32}u; ++j) s += red[j][0];
        float inv = metal::rsqrt(s / {float(K)}f + {float(rms_eps)!r}f);
        const uint c = h * {K}u + tid;
        float gte = (float)gate_raw[c];
        float y = (float)o_norm[tid] * (ov[tid] * inv) * (1.0f / (1.0f + metal::exp(-gte)));
        gated[c] = (T)y;
    }}
"""
    return mx.fast.metal_kernel(
        name=f"vmlx_glm5_kda_fused_decode_h{H}_k{K}_w{W}",
        input_names=["q_tok", "k_tok", "v_tok", "q_state", "k_state", "v_state", "q_w", "k_w", "v_w",
                     "f_raw", "dt_bias", "A_log", "b_raw", "gate_raw", "o_norm", "state"],
        output_names=["gated", "next_state", "q_next", "k_next", "v_next"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=src,
        ensure_row_contiguous=True,
    )


def glm5_kda_fused_decode(q, k, v, cq, ck, cv, wq, wk, wv, f_raw, dt_bias, A_log, b_raw, gate_raw, o_norm, state, *,
                          heads: int, key_dim: int, lower_bound: float, rms_eps: float, enabled: bool = True):
    """Return (gated [1,1,H*K] in q.dtype, next_state, cq', ck', cv') or None (caller keeps the stock path)."""
    if not enabled or key_dim != 128 or state is None or cq is None or ck is None or cv is None:
        return None
    C = heads * key_dim
    if q.shape != (1, 1, C) or k.shape != q.shape or v.shape != q.shape or q.dtype != k.dtype or q.dtype != v.dtype:
        return None
    if q.dtype not in (mx.bfloat16, mx.float16) or state.dtype != mx.float32 or tuple(state.shape) != (1, heads, key_dim, key_dim):
        return None
    if wq.ndim == 3:
        wq, wk, wv = (w_.reshape(w_.shape[0], -1) for w_ in (wq, wk, wv))
    W = int(wq.shape[1])
    if tuple(wq.shape) != (C, W) or W < 2 or tuple(cq.shape) != (1, W - 1, C) or cq.dtype != q.dtype:
        return None
    if f_raw.size != C or gate_raw.size != C or b_raw.size != heads or o_norm.size != key_dim or dt_bias.size != C or A_log.size != heads:
        return None
    out = _kernel(heads, key_dim, W, float(lower_bound), float(rms_eps))(
        inputs=[q, k, v, cq, ck, cv, wq, wk, wv, f_raw, dt_bias, A_log, b_raw, gate_raw, o_norm, state],
        template=[("T", q.dtype)],
        grid=(heads * 512, 1, 1), threadgroup=(512, 1, 1),
        output_shapes=[(1, 1, C), tuple(state.shape), tuple(cq.shape), tuple(ck.shape), tuple(cv.shape)],
        output_dtypes=[q.dtype, mx.float32, cq.dtype, ck.dtype, cv.dtype],
    )
    global _OBSERVED
    _OBSERVED += 1
    return tuple(out)


__all__ = ["fused_kda_decode_requested", "glm5_kda_fused_decode"]


# ---------------------------------------------------------------------------------------------------------------------
# Multi-row variant for speculative-verify slabs (2026-10-10). Same math per position as the decode kernel, looped over
# the T slab rows inside one threadgroup per head; the recurrent state stays in registers across rows and the state
# after EVERY row is written (rollback needs each accepted-prefix boundary). Conv tails per row are slices of the padded
# input and stay with the caller. Replaces short_conv_with_states + l2norm + gates + kda_recurrent_with_states + the
# gated norm of the vectorized verify path (27 ms of a 4-row GLM verify forward, measured).
def _verify_source(H: int, K: int, W: int, lower_bound: float, rms_eps: float, records: bool) -> str:
    """The verify kernel's source. ``records`` swaps STORES only (every arithmetic statement is the same text): instead
    of the 4 MiB state after every row it writes the row's replay record (decay factors, normalized-k inputs, beta,
    k-norm, value correction) and the final state once. With records=False the text is byte-identical to the per-row-
    state kernel (tests pin both the source hash and the bits)."""
    assert K == 128 and W >= 2
    C = H * K
    def rec(text: str) -> str:
        return text if records else ""
    rec_kv = rec("\n            if (which == 1u) rec_kv[((size_t)t * %du + h) * %du + i] = y;" % (H, K))
    rec_eg = rec(" rec_eg[((size_t)t * %du + h) * %du + i] = eg[i];" % (H, K))
    rec_s = rec("\n            rec_s[((size_t)t * %du + h) * 2u] = scal[0]; rec_s[((size_t)t * %du + h) * 2u + 1u] = ik;" % (H, H))
    final_store = rec("    for (uint j = 0; j < 32u; ++j) final_state[sbase + (size_t)(ks * 32u + j) * %du + vi] = S[j];\n" % K)
    row_state_store = "" if records else "            states_out[obase + (size_t)kk * {K}u + vi] = S[j];\n".format(K=K)
    obase_line = ("        if (ks == 0u) rec_corr[((size_t)t * {H}u + h) * {K}u + vi] = corr;\n".format(H=H, K=K) if records
                  else "        const size_t obase = (size_t)t * {HKK}u + sbase;\n".format(HKK=H * K * K))
    return f"""
    const uint h = threadgroup_position_in_grid.x;
    const uint tid = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint T_ = (uint)n_rows[0];
    threadgroup float qv[{K}], kv[{K}], vv[{K}], eg[{K}];
    threadgroup float red[16][3];
    threadgroup float scal[4];
    threadgroup float pA[4][{K}], pB[4][{K}];
    threadgroup float ov[{K}];
    const uint vi = tid % {K}u;
    const uint ks = tid / {K}u;
    const size_t sbase = (size_t)h * {K * K}u;
    float S[32];
    for (uint j = 0; j < 32u; ++j) S[j] = state[sbase + (size_t)(ks * 32u + j) * {K}u + vi];
    for (uint t = 0; t < T_; ++t) {{
        if (tid < {3 * K}u) {{
            const uint which = tid / {K}u, i = tid % {K}u, c = h * {K}u + i;
            device const T* tok = which == 0u ? q_tok : (which == 1u ? k_tok : v_tok);
            device const T* st = which == 0u ? q_state : (which == 1u ? k_state : v_state);
            device const T* wt = which == 0u ? q_w : (which == 1u ? k_w : v_w);
            float acc = 0.0f;
            for (uint tap = 0; tap < {W}u; ++tap) {{
                // padded[t + tap] with padded = [state (W-1 rows), tok (T rows)]
                const int pos = (int)t + (int)tap - {W - 1};
                float val = pos < 0 ? (float)st[(size_t)(pos + {W - 1}) * {C}u + c] : (float)tok[(size_t)pos * {C}u + c];
                acc += val * (float)wt[(size_t)c * {W}u + tap];
            }}
            float y = (float)(T)(acc / (1.0f + metal::exp(-acc)));
            if (which == 0u) qv[i] = y; else if (which == 1u) kv[i] = y; else vv[i] = y;{rec_kv}
        }} else {{
            const uint i = tid - {3 * K}u, c = h * {K}u + i;
            float f = (float)f_raw[(size_t)t * {C}u + c] + (float)dt_bias[c];
            float g = {float(lower_bound)!r}f * (1.0f / (1.0f + metal::exp(-(metal::exp((float)A_log[h]) * f))));
            eg[i] = metal::exp(g);{rec_eg}
            if (i == 0u) scal[0] = 1.0f / (1.0f + metal::exp(-(float)b_raw[(size_t)t * {H}u + h]));
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < {K}u) {{
            float a = qv[tid], b = kv[tid];
            float s0 = simd_sum(a * a), s1 = simd_sum(b * b), s2 = simd_sum(a * b);
            if (lane == 0u) {{ red[sg][0] = s0; red[sg][1] = s1; red[sg][2] = s2; }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {{
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
            for (uint j = 0; j < {K // 32}u; ++j) {{ s0 += red[j][0]; s1 += red[j][1]; s2 += red[j][2]; }}
            float iq = metal::rsqrt(s0 + 1e-6f), ik = metal::rsqrt(s1 + 1e-6f);
            scal[1] = iq * {float(K ** -0.5)!r}f; scal[2] = ik; scal[3] = s2 * iq * {float(K ** -0.5)!r}f * ik;{rec_s}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float qs = scal[1], kn = scal[2], beta = scal[0];
        float a_part = 0.0f, b_part = 0.0f;
        for (uint j = 0; j < 32u; ++j) {{
            const uint kk = ks * 32u + j;
            S[j] *= eg[kk];
            a_part += kv[kk] * kn * S[j];
            b_part += qv[kk] * qs * S[j];
        }}
        pA[ks][vi] = a_part; pB[ks][vi] = b_part;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float kS = pA[0][vi] + pA[1][vi] + pA[2][vi] + pA[3][vi];
        const float corr = vv[vi] - kS;
{obase_line}        for (uint j = 0; j < 32u; ++j) {{
            const uint kk = ks * 32u + j;
            S[j] += beta * kv[kk] * kn * corr;
{row_state_store}        }}
        if (ks == 0u) ov[vi] = (pB[0][vi] + pB[1][vi] + pB[2][vi] + pB[3][vi]) + beta * corr * scal[3];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < {K}u) {{
            float o = ov[tid];
            float s = simd_sum(o * o);
            if (lane == 0u) red[sg][0] = s;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < {K}u) {{
            float s = 0.0f;
            for (uint j = 0; j < {K // 32}u; ++j) s += red[j][0];
            float inv = metal::rsqrt(s / {float(K)}f + {float(rms_eps)!r}f);
            const uint c = h * {K}u + tid;
            float gte = (float)gate_raw[(size_t)t * {C}u + c];
            gated[(size_t)t * {C}u + c] = (T)((float)o_norm[tid] * (ov[tid] * inv) * (1.0f / (1.0f + metal::exp(-gte))));
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
{final_store}"""


_VERIFY_INPUTS = ["q_tok", "k_tok", "v_tok", "q_state", "k_state", "v_state", "q_w", "k_w", "v_w",
                  "f_raw", "dt_bias", "A_log", "b_raw", "gate_raw", "o_norm", "state", "n_rows"]


@lru_cache(maxsize=8)
def _kernel_verify(H: int, K: int, W: int, lower_bound: float, rms_eps: float):
    return mx.fast.metal_kernel(
        name=f"vmlx_glm5_kda_fused_verify_h{H}_k{K}_w{W}",
        input_names=_VERIFY_INPUTS,
        output_names=["gated", "states_out"], header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=_verify_source(H, K, W, lower_bound, rms_eps, records=False), ensure_row_contiguous=True)


@lru_cache(maxsize=8)
def _kernel_verify_records(H: int, K: int, W: int, lower_bound: float, rms_eps: float):
    return mx.fast.metal_kernel(
        name=f"vmlx_glm5_kda_fused_verify_records_h{H}_k{K}_w{W}",
        input_names=_VERIFY_INPUTS,
        output_names=["gated", "rec_eg", "rec_kv", "rec_s", "rec_corr", "final_state"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=_verify_source(H, K, W, lower_bound, rms_eps, records=True), ensure_row_contiguous=True)


def glm5_kda_fused_verify(q, k, v, cq, ck, cv, wq, wk, wv, f_raw, dt_bias, A_log, b_raw, gate_raw, o_norm, state, *,
                          heads: int, key_dim: int, lower_bound: float, rms_eps: float):
    """q/k/v (1, T, H*K) raw projections; returns (gated (1,T,H*K), states (T, 1, H, K, K) f32) or None."""
    if key_dim != 128 or state is None or cq is None or ck is None or cv is None:
        return None
    C = heads * key_dim
    if q.ndim != 3 or q.shape[0] != 1 or q.shape[2] != C or k.shape != q.shape or v.shape != q.shape:
        return None
    T = int(q.shape[1])
    if T < 1 or T > 64 or q.dtype not in (mx.bfloat16, mx.float16) or state.dtype != mx.float32 \
            or tuple(state.shape) != (1, heads, key_dim, key_dim):
        return None
    if wq.ndim == 3:
        wq, wk, wv = (w_.reshape(w_.shape[0], -1) for w_ in (wq, wk, wv))
    W = int(wq.shape[1])
    if tuple(cq.shape) != (1, W - 1, C) or cq.dtype != q.dtype or f_raw.size != T * C or gate_raw.size != T * C \
            or b_raw.size != T * heads:
        return None
    gated, states = _kernel_verify(heads, key_dim, W, float(lower_bound), float(rms_eps))(
        inputs=[q, k, v, cq, ck, cv, wq, wk, wv, f_raw, dt_bias, A_log, b_raw, gate_raw, o_norm, state,
                mx.array([T], mx.int32)],
        template=[("T", q.dtype)], grid=(heads * 512, 1, 1), threadgroup=(512, 1, 1),
        output_shapes=[(1, T, C), (T, 1, heads, key_dim, key_dim)], output_dtypes=[q.dtype, mx.float32])
    global _OBSERVED
    _OBSERVED += 1
    return gated, states


# ---------------------------------------------------------------------------------------------------------------------
# Replay rollback for wide verify slabs (2026-10-12, the "cache inputs, not state" idea of ReplaySSM, arXiv 2610.13778,
# reduced to what is bit-exact here). The per-row-state kernel above writes the 4 MiB state of every head-layer after
# EVERY row: T x 142.6 MB per GLM-5.3 round (2.28 GB at T=16), and at T >= 8 those stores dominate the kernel
# (34 layers, M5 Max synthetic: T=8 3.44 ms, T=16 6.10 ms). The records variant writes, per row and head, the inputs of
# the state update instead — decay factors exp(g) [K], the conv'd k [K], beta, the k l2-norm scale, the value correction
# [K] (1.5 KB vs 64 KB) — and the final state once. A rollback to the state after row a replays rows 0..a from the
# pre-verify state with the SAME per-element operations in the same order (S *= exp(g); S += beta * k * kn * corr), so
# every replayed state is bit-identical to the per-row kernel's (tests: all rows, T 8/16, 3 seeds). Measured, 34 layers:
# T=8 1.88 ms (+0.87 replay on a partial accept) vs 3.44; T=16 2.88 (+0.91) vs 6.10; T=4 2.27 vs 2.14 and T=2 2.15 vs
# 1.34 -> slabs below REPLAY_MIN_ROWS keep the per-row states. VMLX_GLM5_KDA_VERIFY_REPLAY=0 keeps them at every width.
REPLAY_MIN_ROWS = 8


def kda_verify_replay_requested() -> bool:
    return os.environ.get("VMLX_GLM5_KDA_VERIFY_REPLAY", "1").strip().lower() not in {"", "0", "false", "off", "no"}


@lru_cache(maxsize=8)
def _kernel_replay(H: int, K: int):
    src = f"""
    const uint h = threadgroup_position_in_grid.x;
    const uint tid = thread_index_in_threadgroup;
    const uint vi = tid % {K}u;
    const uint ks = tid / {K}u;
    const size_t sbase = (size_t)h * {K * K}u;
    const uint n = (uint)n_rows[0];
    float S[32];
    for (uint j = 0; j < 32u; ++j) S[j] = state[sbase + (size_t)(ks * 32u + j) * {K}u + vi];
    for (uint t = 0; t < n; ++t) {{
        const size_t rb = ((size_t)t * {H}u + h);
        const float beta = rec_s[rb * 2u], kn = rec_s[rb * 2u + 1u];
        const float corr = rec_corr[rb * {K}u + vi];
        for (uint j = 0; j < 32u; ++j) {{
            const uint kk = ks * 32u + j;
            S[j] *= rec_eg[rb * {K}u + kk];
        }}
        for (uint j = 0; j < 32u; ++j) {{
            const uint kk = ks * 32u + j;
            const float kvv = rec_kv[rb * {K}u + kk];
            S[j] += beta * kvv * kn * corr;
        }}
    }}
    for (uint j = 0; j < 32u; ++j) out_state[sbase + (size_t)(ks * 32u + j) * {K}u + vi] = S[j];
"""
    return mx.fast.metal_kernel(
        name=f"vmlx_glm5_kda_verify_replay_h{H}_k{K}",
        input_names=["state", "rec_eg", "rec_kv", "rec_s", "rec_corr", "n_rows"], output_names=["out_state"],
        header="#include <metal_stdlib>\nusing namespace metal;\n", source=src, ensure_row_contiguous=True)


def glm5_kda_fused_verify_records(q, k, v, cq, ck, cv, wq, wk, wv, f_raw, dt_bias, A_log, b_raw, gate_raw, o_norm, state,
                                  *, heads: int, key_dim: int, lower_bound: float, rms_eps: float):
    """As glm5_kda_fused_verify, but returns (gated, records, final_state): records replay any row's state."""
    if key_dim != 128 or state is None or cq is None or ck is None or cv is None:
        return None
    C = heads * key_dim
    if q.ndim != 3 or q.shape[0] != 1 or q.shape[2] != C or k.shape != q.shape or v.shape != q.shape:
        return None
    T = int(q.shape[1])
    if T < 1 or T > 64 or q.dtype not in (mx.bfloat16, mx.float16) or state.dtype != mx.float32 \
            or tuple(state.shape) != (1, heads, key_dim, key_dim):
        return None
    if wq.ndim == 3:
        wq, wk, wv = (w_.reshape(w_.shape[0], -1) for w_ in (wq, wk, wv))
    W = int(wq.shape[1])
    if tuple(cq.shape) != (1, W - 1, C) or cq.dtype != q.dtype or f_raw.size != T * C or gate_raw.size != T * C \
            or b_raw.size != T * heads:
        return None
    gated, eg, kvr, sc, corr, final = _kernel_verify_records(heads, key_dim, W, float(lower_bound), float(rms_eps))(
        inputs=[q, k, v, cq, ck, cv, wq, wk, wv, f_raw, dt_bias, A_log, b_raw, gate_raw, o_norm, state,
                mx.array([T], mx.int32)],
        template=[("T", q.dtype)], grid=(heads * 512, 1, 1), threadgroup=(512, 1, 1),
        output_shapes=[(1, T, C), (T, heads, key_dim), (T, heads, key_dim), (T, heads, 2), (T, heads, key_dim),
                       tuple(state.shape)],
        output_dtypes=[q.dtype, mx.float32, mx.float32, mx.float32, mx.float32, mx.float32])
    global _OBSERVED
    _OBSERVED += 1
    return gated, (eg, kvr, sc, corr), final


def glm5_kda_replay(state, records, rows: int):
    """The recurrent state after ``rows`` verify rows, replayed from the pre-verify ``state`` (bit-identical)."""
    eg, kvr, sc, corr = records
    heads, key_dim = int(state.shape[1]), int(state.shape[2])
    return _kernel_replay(heads, key_dim)(
        inputs=[state, eg, kvr, sc, corr, mx.array([int(rows)], mx.int32)], grid=(heads * 512, 1, 1),
        threadgroup=(512, 1, 1), output_shapes=[tuple(state.shape)], output_dtypes=[mx.float32])[0]


class ReplayedKDAStates:
    """Sequence of (conv_q, conv_k, conv_v, state) after each verify row, as ``Glm5KDACache`` keeps for rollback. The
    last row's state is the kernel's final state; an earlier row's is replayed from the pre-verify state on access
    (lazy MLX op, evaluated with the next forward). Conv tails are slices of the padded inputs, as before."""

    def __init__(self, tails_q, tails_k, tails_v, pre_state, records, final_state):
        self._tq, self._tk, self._tv = tails_q, tails_k, tails_v
        self._pre, self._records, self._final = pre_state, records, final_state

    def __len__(self) -> int:
        return len(self._tq)

    def __bool__(self) -> bool:
        return len(self._tq) > 0

    def __getitem__(self, row: int):
        n = len(self._tq)
        if row < 0:
            row += n
        if not 0 <= row < n:
            raise IndexError(row)
        state = self._final if row == n - 1 else glm5_kda_replay(self._pre, self._records, row + 1)
        return (self._tq[row], self._tk[row], self._tv[row], state)
