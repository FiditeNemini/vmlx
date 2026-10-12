# SPDX-License-Identifier: Apache-2.0
"""Fused DSA indexer scores for prefill: out[q, k] = sum_h w[q, h] * relu(Q[h, q, :] . K[k, :]).

WHY (Naive N0.5 Flash, 2026-10-12): the indexer materialized the (heads, queries, keys) fp32 score tensor and ran
relu, the head-weight product and the head sum over it as separate passes. At a 2048-query prefill chunk the q.k
matmul ran at 14-16 TFLOPS (10.7 ms at 20k keys) while the passes over the intermediate took ~80% of the indexer
(52 ms per DSA layer per chunk at 20k, 105 ms at 40k; 12 DSA layers). This kernel keeps each (query, key) tile in
registers: per head an 8x8 simdgroup-MMA dot product over head_dim, then relu, weight and accumulation, and writes the
(queries, keys) result once.

Arithmetic: dot products accumulate over head_dim in 8-wide simdgroup-MMA steps, head 0..H-1 sequential with the
product rounded before the add (no FMA contraction) — the order of the stock matmul + maximum + multiply + sum path,
which the tests check bit for bit; callers fall back to the stock path for anything not admitted.
"""
from __future__ import annotations

from functools import lru_cache
import os

import mlx.core as mx



def dsa_index_scores_requested() -> bool:
    return os.environ.get("VMLX_DSA_FUSED_INDEX_SCORES", "1") == "1"


_SOURCE = r"""
    #pragma clang fp contract(off)
    const int Lq = int(q_shape[2]);
    const int Lk = int(k_shape[2]);
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int c0 = min(int(threadgroup_position_in_grid.x) * TN, Lk - TN);
    const int r0 = min(int(threadgroup_position_in_grid.y) * TM, Lq - TM);
    const int sr = r0 + int(sg / 2) * (FI * 8);
    const int sc = c0 + int(sg % 2) * (FJ * 8);
    const short qid = short(lane / 4);
    const short fm = (qid & 4) + short((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + short(lane % 2) * 2;
    float out_acc[FI][FJ][2];
    for (int i = 0; i < FI; ++i) for (int j = 0; j < FJ; ++j) { out_acc[i][j][0] = 0.0f; out_acc[i][j][1] = 0.0f; }
    for (int h = 0; h < H; ++h) {
        simdgroup_float8x8 acc[FI][FJ];
        for (int i = 0; i < FI; ++i) for (int j = 0; j < FJ; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
        const device float* qh = q + (size_t)h * (size_t)Lq * D;
        for (int d = 0; d < D; d += 8) {
            simdgroup_float8x8 a[FI];
            simdgroup_float8x8 b[FJ];
            for (int i = 0; i < FI; ++i)
                simdgroup_load(a[i], qh + (size_t)(sr + i * 8) * D + d, D);
            for (int j = 0; j < FJ; ++j)
                simdgroup_load(b[j], k + (size_t)(sc + j * 8) * D + d, D, ulong2(0, 0), true);
            for (int i = 0; i < FI; ++i)
                for (int j = 0; j < FJ; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        for (int i = 0; i < FI; ++i) {
            const float wv = w[(size_t)(sr + i * 8 + fm) * H + h];
            for (int j = 0; j < FJ; ++j) {
                thread auto& e = acc[i][j].thread_elements();
                float p0 = metal::max(e[0], 0.0f) * wv;
                float p1 = metal::max(e[1], 0.0f) * wv;
                out_acc[i][j][0] = out_acc[i][j][0] + p0;
                out_acc[i][j][1] = out_acc[i][j][1] + p1;
            }
        }
    }
    for (int i = 0; i < FI; ++i) {
        const size_t row = (size_t)(sr + i * 8 + fm);
        for (int j = 0; j < FJ; ++j) {
            const size_t col = (size_t)(sc + j * 8 + fn);
            out[row * (size_t)Lk + col] = out_acc[i][j][0];
            out[row * (size_t)Lk + col + 1] = out_acc[i][j][1];
        }
    }
"""


@lru_cache(maxsize=None)
def _kernel():
    return mx.fast.metal_kernel(
        name="vmlx_dsa_fused_index_scores_v1",
        input_names=["q", "k", "w"],
        output_names=["out"],
        source=_SOURCE,
        header="#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\nusing namespace metal;\n",
        ensure_row_contiguous=True,
    )


# Fragments per simdgroup (8x8 each): 4 x 2 -> a 64 x 32 tile per threadgroup of 4 simdgroups. Measured (max2, 16 heads
# x 128, 2048 queries): 2x2 1.6-2.0x of stock, 4x2 2.1-2.5x, 2x4 2.0-2.3x, 4x4 0.25x (register spill).
FI, FJ = 4, 2


def dsa_index_scores(q: mx.array, k: mx.array, w: mx.array, *, enabled: bool = True):
    """q (1, H, L, D) fp32, k (1, 1, Lk, D) fp32, w (1, L, H) fp32 -> (1, L, Lk) fp32; None when not admitted.

    Admitted: batch 1, one shared key head, fp32 operands, D a multiple of 8, 2 <= L, Lk >= 32 (the last tile
    overlaps the previous one instead of reading past the end; overlapping cells are computed identically; L below
    one tile is zero-padded). Decode (L == 1) returns None."""
    if not enabled or mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None
    if q.ndim != 4 or k.ndim != 4 or w.ndim != 3:
        return None
    B, H, L, D = q.shape
    if B != 1 or k.shape[0] != 1 or k.shape[1] != 1 or k.shape[3] != D or tuple(w.shape) != (1, L, H):
        return None
    Lk = int(k.shape[2])
    TM, TN = 2 * FI * 8, 2 * FJ * 8
    if D % 8 or L < 2 or Lk < TN or any(a.dtype != mx.float32 for a in (q, k, w)):
        return None                         # decode (L == 1) keeps the stock GEMV path
    if max(L, TM) * Lk >= 2**31:
        return None
    if L < TM:
        # A short prefill tail (a warm turn's few new tokens) takes the same arithmetic as the cold chunk did:
        # zero-pad the queries to one tile and keep the first L rows (each row is computed independently).
        out = dsa_index_scores(
            mx.pad(q, [(0, 0), (0, 0), (0, TM - L), (0, 0)]),
            k,
            mx.pad(w, [(0, 0), (0, TM - L), (0, 0)]),
            enabled=enabled,
        )
        return None if out is None else out[:, :L]
    grid_x = -(-Lk // TN)
    grid_y = -(-L // TM)
    return _kernel()(
        inputs=[q, k, w],
        template=[("H", int(H)), ("D", int(D)), ("FI", FI), ("FJ", FJ), ("TM", TM), ("TN", TN)],
        grid=(grid_x * 128, grid_y, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, L, Lk)],
        output_dtypes=[mx.float32],
    )[0]
