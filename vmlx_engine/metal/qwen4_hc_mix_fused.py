"""One-dispatch hyper-connection mix for Qwen3.8 Flash-Next (qwen4_exp) GatedResidual.

What it replaces (``GatedResidual._forward_normed`` after the fused down/inject GEMV):

    combined   = W_inj @ normed                       # [S, R + C]   (kept: one GEMV, row-invariant)
    mix        = combined[..., :R]                     # R = hc_lowrank = 320
    injection  = combined[..., R:]                     # C = hc_count  = 4
    s          = silu(mix / C)                         # fp16 per op
    up         = sigmoid(W_up @ s)                     # W_up [C*H, R], H = hidden = 2560
    mixed      = mean_g( up[g*H + d] * normed[g*H + d] ) over g = 0..C-1      -> [S, H]
    inject_w   = 2 * sigmoid(injection / C)                                    -> [S, C]

That tail is ~12 dependent MLX launches per call (take x2, divide, silu, GEMV, sigmoid, astype, multiply, mean, astype, divide,
sigmoid, multiply) on tiny tensors, so it is latency-bound, not bandwidth-bound (measured 2026-10-05: quantizing the HC weights to
q8/q6/q4 gave no speedup and cost 6-17 % top-1 agreement; rejected). This module computes the whole tail in ONE Metal dispatch.

Kernel layout: grid = (32 * H, S); one SIMD group per output column d of one row r. Each lane recomputes its slice of s (10 of 320
values, from the row's combined[:R]) — cheaper than a second dispatch. For each stream g the 32 lanes dot W_up[g*H + d, :] with s
(fp32 accumulation, simd_sum), lane 0 applies the per-op fp16 rounding of the original graph:
    dot -> half (GEMV output dtype)  ->  sigmoid computed in float, rounded once to half (MLX 0.32.3 sigmoid)
    -> half product with normed  ->  float sum over the C streams -> / C -> half
Lanes of the SIMD group with d == 0 also write inject_w = half(2 * half(sigmoid(half(inj / C)))).

Exactness contract: NOT bit-identical to the composed MLX graph (the up-GEMV reduction order differs from MLX's GEMV), so it changes
fp16 rounding of `mixed` by at most a few ULP. It IS row-invariant (every row is computed by the same code independent of S), and it
is used for decode (S=1) and native-MTP verify rows (S<=8) alike, so greedy Adaptive == AR is preserved. Quality is gated by
teacher-forced top-1 agreement / KL against the stock path on every supported bundle before the default may change.

Eligibility (else ``None`` -> stock path): Apple GPU, float16/bfloat16 activations, dense same-dtype W_up of shape [C*H, R],
R % 32 == 0, rows 1..8, C == 4. ``VMLX_QWEN4_HC_FUSED_MIX`` default ON; ``=0`` restores the composed graph.

Proof (M5 Max, MLX 0.32.3, 2026-10-05): in-process S=1 decode 4M 19.36 -> 18.17 ms, JANGH4 20.39 -> 19.16, Allosaurus 17.14 ->
15.90 (+6-8 % tok/s); served Adaptive at matched GPU clock 4M prose 57.7 -> 61.8, easy code 96.1 -> 99.5, JANGH4 prose
55.6 -> 57.5 / 56.2 -> 57.5, code 79.4 -> 83.2 / 82.9 -> 84.8; greedy Adaptive == AR 10/10 on 4M and JANGH4; distance to a float32
reference of the same tail equal to the stock graph's (closer in 4/6 prose/code cells).
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import mlx.core as mx

logger = logging.getLogger(__name__)

_MAX_ROWS = 8
_STATUS = {"calls": 0, "rejected": None, "observed": False}


def hc_fused_mix_requested() -> bool:
    return os.environ.get("VMLX_QWEN4_HC_FUSED_MIX", "1").strip().lower() not in {"0", "false", "no", "off"}


_SOURCE = r"""
    // grid.x = 32 * H (one SIMD group per output column d), grid.y = rows.
    uint lane = thread_index_in_simdgroup;
    uint d = thread_position_in_grid.x / 32u;
    uint r = thread_position_in_grid.y;
    if (d >= H) return;
    const device T* zr = combined + (size_t)r * (R + C);
    const device T* nr = normed + (size_t)r * (C * H);

    // s_k = silu(half(z_k / C)) with fp16 rounding after every op, for this lane's slice of k.
    constexpr uint PER_LANE = R / 32u;
    T s[PER_LANE];
    for (uint j = 0; j < PER_LANE; ++j) {
        uint k = lane + 32u * j;
        T t = T(zr[k] / T(C));
        T sg = T(1.0f / (1.0f + metal::precise::exp(-float(t))));
        s[j] = T(t * sg);
    }
    float acc = 0.0f;
    for (uint g = 0; g < C; ++g) {
        const device T* wrow = w_up + ((size_t)g * H + d) * R;
        float dot = 0.0f;
        for (uint j = 0; j < PER_LANE; ++j) {
            dot += float(wrow[lane + 32u * j]) * float(s[j]);
        }
        dot = simd_sum(dot);
        T up = T(dot);
        T sg = T(1.0f / (1.0f + metal::precise::exp(-float(up))));
        T prod = T(sg * nr[(size_t)g * H + d]);
        acc += float(prod);
    }
    if (lane == 0u) {
        mixed[(size_t)r * H + d] = T(acc / float(C));
    }
    if (d == 0u && lane < C) {
        T t = T(zr[R + lane] / T(C));
        T sg = T(1.0f / (1.0f + metal::precise::exp(-float(t))));
        inject[(size_t)r * C + lane] = T(T(2.0f) * sg);
    }
"""


@lru_cache(maxsize=4)
def _kernel():
    return mx.fast.metal_kernel(
        name="vmlx_qwen4_hc_mix_fused_v1",
        input_names=["combined", "normed", "w_up"],
        output_names=["mixed", "inject"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=_SOURCE,
        ensure_row_contiguous=True,
    )


def _reason(combined, normed, w_up, hc_count, lowrank, hidden):
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return "no metal gpu"
    if hc_count != 4 or lowrank % 32 != 0:
        return f"geometry C={hc_count} R={lowrank}"
    if normed.dtype not in (mx.float16, mx.bfloat16) or combined.dtype != normed.dtype or w_up.dtype != normed.dtype:
        return f"dtype normed={normed.dtype} combined={combined.dtype} w_up={w_up.dtype}"
    if tuple(w_up.shape) != (hc_count * hidden, lowrank):
        return f"w_up shape {tuple(w_up.shape)}"
    if normed.shape[-1] != hc_count * hidden or combined.shape[-1] != lowrank + hc_count:
        return "feature widths"
    rows = 1
    for dim in normed.shape[:-1]:
        rows *= int(dim)
    if not 1 <= rows <= _MAX_ROWS:
        return f"rows={rows}"
    return None


def hc_mix_fused(combined, normed, w_up, *, hc_count: int, lowrank: int, hidden: int):
    """Return (mixed [..., H], inject_w [..., C]) from one Metal dispatch, or None (caller keeps the stock graph)."""
    reason = _reason(combined, normed, w_up, hc_count, lowrank, hidden)
    if reason is not None:
        if _STATUS["rejected"] != reason:
            _STATUS["rejected"] = reason
            logger.info("Qwen4 fused HC mix not used: %s", reason)
        return None
    lead = normed.shape[:-1]
    rows = 1
    for dim in lead:
        rows *= int(dim)
    mixed, inject = _kernel()(
        inputs=[combined.reshape(rows, lowrank + hc_count), normed.reshape(rows, hc_count * hidden), w_up],
        template=[("T", normed.dtype), ("H", hidden), ("R", lowrank), ("C", hc_count)],
        grid=(32 * hidden, rows, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, hidden), (rows, hc_count)],
        output_dtypes=[normed.dtype, normed.dtype],
    )
    _STATUS["calls"] += 1
    if not _STATUS["observed"]:
        _STATUS["observed"] = True
        logger.info("Qwen4 fused HC mix active: H=%d R=%d C=%d dtype=%s rows<=%d", hidden, lowrank, hc_count, normed.dtype, _MAX_ROWS)
    return mixed.reshape(*lead, hidden), inject.reshape(*lead, hc_count)


def hc_mix_fused_status() -> dict:
    return {"requested": hc_fused_mix_requested(), **_STATUS}
