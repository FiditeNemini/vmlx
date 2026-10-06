# SPDX-License-Identifier: Apache-2.0
# Portions adapted from TensorFold 0.6.6 (https://github.com/ashhart/TensorFold, Apache-2.0;
# Copyright 2026 TensorFold contributors),
# src/tensorfold/kernels/qwen/dense/v1/lane_qmm.py and lane_widen.py at commit cb2ebf0. The Metal kernel
# sources are verbatim; the Python wrapper is reduced to what vMLX uses.
"""Row-independent affine quantized matmul for 1..128 rows on the M5 matrix units ("lane matmul").

WHY (Qwen3.8-27B DFlash2 verify, R2-21)
---------------------------------------
MLX's ``quantized_matmul`` picks a different kernel per row count (qmv at 1 row, qmv variants for a few
rows, a qmm tile from 6 rows).  On the 27B JANG_4D target that is 43 ms for 1 row, 53 ms for 5, 69 ms for
6, 79 ms for 8 and 115 ms for 12.  This kernel always runs a 16-row ``matmul2d`` on the matrix units
(Metal 4 MetalPerformancePrimitives), so 1..16 rows cost about the same, and because the op shape never
depends on M, each output row's bits are independent of how many rows share the call.

HOW
---
* The matrix unit multiplies bf16 activations by the RAW unsigned 4-bit codes (``uint4b_format``) of one
  weight group (GS = 64 values of K) at a time, fp32 accumulate.  The affine dequant is applied after the
  dot product:  C += s[g,n] * (X_g @ Q_g)[m,n] + b[g,n] * XS[g,m],  XS[g,m] = sum of x over group g
  (``_XSUM``).  This is algebraically x @ (s*q + b)^T.
* 5/6/8-bit weights are widened to bytes in threadgroup memory and fed to a uint8 op (``BYTES``); 2/3-bit
  to nibbles (``NIBBLES``).  4-bit goes to the op as stored.
* K is split across SK simdgroups (fixed by the weight shape, ``split_k``); slices are summed in slice
  order, so the reduction order is a property of the weight, not of M.
* Scales/biases are packed once per weight as (K/GS, N, 2) bf16 pairs (``pack_scales``).

TRADE-OFFS
----------
* Not bit-identical to MLX's own qmv/qmm (different summation order), but identical across row counts.
* Only affine weights with bf16 scales, GS 64 (or 32 for 4-bit), K % 64 == 0, N % 4 == 0 (``supports``).
* Needs Metal 4 tensor ops (M5-class GPU); callers must check ``available()``.
"""
from __future__ import annotations

import hashlib

import mlx.core as mx

MAX_ROWS = 128
ROW_BLOCK = 32
NT = 32
BITS = (2, 3, 4, 5, 6, 8)

_HEADER = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""

_XSUM = r"""
  const int M = mdims[0], MP = mdims[1];
  const uint m = thread_position_in_grid.y;
  const uint g = thread_position_in_grid.x;
  if (g >= K / GS || int(m) >= MP) return;
  float acc = 0.0f;
  if (int(m) < M) for (int i = 0; i < GS; i++) acc += float(X[m * K + g * GS + i]);
  XS[g * MP + m] = acc;
"""

_MAIN = r"""
  const ushort lane = thread_index_in_simdgroup;
  const ushort sg = simdgroup_index_in_threadgroup;     // K slice
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);       // fragment row of this lane (and fm + 8)
  const short fn = ((qid & 2) | (lane & 1)) * 4;        // first of its four fragment columns
  const int M = mdims[0], MP = mdims[1];
  constexpr int KG = K / GS;
  constexpr int NF = NT / 16;
  const int n0 = threadgroup_position_in_grid.x * NT;
  const int rb = threadgroup_position_in_grid.y * 16 * TMR;   // first row of this threadgroup's row block
  const int g_begin = (sg * KG) / SK;
  const int g_end = ((sg + 1) * KG) / SK;

  // one op for all TMR 16-row blocks: each row gets the 16-row op's bits
  constexpr auto desc = matmul2d_descriptor(16 * TMR, NT, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X + (int64_t)rb * K, dextents<int32_t, 2>(K, M - rb));
  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tB((device uchar*)Wq, dextents<int32_t, 2>(K, N));

  float C[TMR][NF * 8];
  for (int t = 0; t < TMR; t++) for (int i = 0; i < NF * 8; i++) C[t][i] = 0.0f;
  const device uint4* sbv = (const device uint4*)SBt;   // (s, b) bf16 pairs, [g][n]
  bool colok[NF];
  for (int f = 0; f < NF; f++) colok[f] = n0 + f * 16 + fn < N;
  for (int g = g_begin; g < g_end; g++) {
    float s[NF][4], bb[NF][4];
    for (int f = 0; f < NF; f++) {
      const uint4 q = colok[f] ? sbv[(g * N + n0 + f * 16 + fn) / 4] : uint4(0);
      const vec<bfloat, 8> v = as_type<vec<bfloat, 8>>(q);
      for (int j = 0; j < 4; j++) { s[f][j] = float(v[2 * j]); bb[f][j] = float(v[2 * j + 1]); }
    }
    auto a = tA.slice(g * GS, 0);
    auto b = tB.slice(g * GS, n0);
    auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    op.run(a, b, P);
    for (int t = 0; t < TMR; t++) {
      const bool live = !EDGE || rb + t * 16 < MP;     // EDGE: the last 32-row block passes MP, where XS ends
      const float xs0 = live ? XS[g * MP + rb + t * 16 + fm] : 0.0f;
      const float xs1 = live ? XS[g * MP + rb + t * 16 + fm + 8] : 0.0f;
      for (int f = 0; f < NF; f++)
        for (int r = 0; r < 2; r++)
          for (int j = 0; j < 4; j++) {
            const int i = f * 8 + r * 4 + j;
            C[t][i] = fma(s[f][j], P[t * NF * 8 + i], fma(bb[f][j], r ? xs1 : xs0, C[t][i]));
          }
    }
  }
  // K slices are added in slice order, one 16-row block at a time
  threadgroup float part[(SK > 1 ? SK - 1 : 1) * NF * 8 * 32];
  for (int t = 0; t < TMR; t++) {
    if (SK > 1) {
      if (sg > 0) for (int i = 0; i < NF * 8; i++) part[((sg - 1) * NF * 8 + i) * 32 + lane] = C[t][i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0)
        for (int s2 = 1; s2 < SK; s2++) for (int i = 0; i < NF * 8; i++) C[t][i] += part[((s2 - 1) * NF * 8 + i) * 32 + lane];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg == 0)
      for (int f = 0; f < NF; f++)
        for (int r = 0; r < 2; r++) {
          const int m = rb + t * 16 + fm + 8 * r;
          const int n = n0 + f * 16 + fn;
          if (m < M && n < N)
            for (int j = 0; j < 4; j++) Y[m * N + n + j] = static_cast<bfloat>(C[t][f * 8 + r * 4 + j]);
        }
  }
"""

# Tiled weights (``tile_weight``): column tile t's group g is one contiguous NT x GS block. Same values, same op.
_MAIN_TILED = _MAIN.replace(
    "    auto b = tB.slice(g * GS, n0);\n",
    "    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b(\n"
    "        (device uchar*)Wq + (int64_t)(threadgroup_position_in_grid.x * KG + g) * (NT * GS / 2), dextents<int32_t, 2>(GS, NT));\n")
assert _MAIN_TILED != _MAIN

NIBBLES = r"""
  static_assert(NT == 32, "one column per lane");
  static_assert(BITS == 2 || BITS == 3, "4-bit weights go to the tensor op as they are");
  const ushort lane = thread_index_in_simdgroup;
  const ushort sg = simdgroup_index_in_threadgroup;     // K slice
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  const int M = mdims[0], MP = mdims[1];
  constexpr int KG = K / 64;
  constexpr int NF = NT / 16;
  constexpr int WPG = 2 * BITS;                          // words per column per group: 64 values x BITS bits
  constexpr int KW = K * BITS / 32;                      // words per column
  const int n0 = threadgroup_position_in_grid.x * NT;
  const int rb = threadgroup_position_in_grid.y * 16 * TMR;
  const int g_begin = (sg * KG) / SK;
  const int g_end = ((sg + 1) * KG) / SK;
  constexpr auto desc = matmul2d_descriptor(16 * TMR, NT, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X + (int64_t)rb * K, dextents<int32_t, 2>(K, M - rb));
  threadgroup uint stage_all[SK * NT * 8];               // per K slice: NT columns x 64 nibbles
  threadgroup uint* stage = stage_all + sg * NT * 8;
  tensor<threadgroup uint4b_format, dextents<int32_t, 2>, tensor_inline> b((threadgroup uchar*)stage, dextents<int32_t, 2>(64, NT));
  const device uint* Wv = (const device uint*)Wq;
  const int n = n0 + lane;

  float C[TMR][NF * 8];
  for (int t = 0; t < TMR; t++) for (int i = 0; i < NF * 8; i++) C[t][i] = 0.0f;
  const device uint4* sbv = (const device uint4*)SBt;
  bool colok[NF];
  for (int f = 0; f < NF; f++) colok[f] = n0 + f * 16 + fn < N;
  for (int g = g_begin; g < g_end; g++) {
    uint w[WPG + 1];
    for (int i = 0; i <= WPG; i++) w[i] = 0;
    if (n < N) {
      const device uint* src = TILED ? Wv + ((int64_t)(threadgroup_position_in_grid.x * KG + g) * NT + lane) * WPG
                                     : Wv + (int64_t)n * KW + g * WPG;
      for (int i = 0; i < WPG; i++) w[i] = src[i];
    }
    if (BITS == 3) {
      for (int c = 0; c < 8; c++) {
        const int bit = 24 * c, i = bit >> 5, sh = bit & 31;
        uint pack = w[i] >> sh;
        if (sh > 8) pack |= w[i + 1] << (32 - sh);
        uint nib = 0;
        for (int j = 0; j < 8; j++) nib |= ((pack >> (3 * j)) & 7u) << (4 * j);
        stage[lane * 8 + c] = nib;
      }
    } else {
      for (int c = 0; c < 8; c++) {                     // word c/2's half c%2: 8 values of 2 bits -> 8 nibbles
        uint v = (w[c >> 1] >> (16 * (c & 1))) & 0xFFFFu;
        v = (v | (v << 8)) & 0x00FF00FFu;
        v = (v | (v << 4)) & 0x0F0F0F0Fu;
        v = (v | (v << 2)) & 0x33333333u;
        stage[lane * 8 + c] = v;
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float s[NF][4], bb[NF][4];
    for (int f = 0; f < NF; f++) {
      const uint4 q = colok[f] ? sbv[(g * N + n0 + f * 16 + fn) / 4] : uint4(0);
      const vec<bfloat, 8> v = as_type<vec<bfloat, 8>>(q);
      for (int j = 0; j < 4; j++) { s[f][j] = float(v[2 * j]); bb[f][j] = float(v[2 * j + 1]); }
    }
    auto a = tA.slice(g * 64, 0);
    auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    op.run(a, b, P);
    simdgroup_barrier(mem_flags::mem_threadgroup);   // the op has read the stage before the next group's widening
    for (int t = 0; t < TMR; t++) {
      // the last row block can run past MP (MP % 32 == 16): those rows are never stored, and XS ends at MP
      const bool live = rb + t * 16 < MP;
      const float xs0 = live ? XS[g * MP + rb + t * 16 + fm] : 0.0f;
      const float xs1 = live ? XS[g * MP + rb + t * 16 + fm + 8] : 0.0f;
      for (int f = 0; f < NF; f++)
        for (int r = 0; r < 2; r++)
          for (int j = 0; j < 4; j++) {
            const int i = f * 8 + r * 4 + j;
            C[t][i] = fma(s[f][j], P[t * NF * 8 + i], fma(bb[f][j], r ? xs1 : xs0, C[t][i]));
          }
    }
  }
  threadgroup float part[(SK > 1 ? SK - 1 : 1) * NF * 8 * 32];
  for (int t = 0; t < TMR; t++) {
    if (SK > 1) {
      if (sg > 0) for (int i = 0; i < NF * 8; i++) part[((sg - 1) * NF * 8 + i) * 32 + lane] = C[t][i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0)
        for (int s2 = 1; s2 < SK; s2++) for (int i = 0; i < NF * 8; i++) C[t][i] += part[((s2 - 1) * NF * 8 + i) * 32 + lane];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg == 0)
      for (int f = 0; f < NF; f++)
        for (int r = 0; r < 2; r++) {
          const int m = rb + t * 16 + fm + 8 * r;
          const int nn = n0 + f * 16 + fn;
          if (m < M && nn < N)
            for (int j = 0; j < 4; j++) Y[m * N + nn + j] = static_cast<bfloat>(C[t][f * 8 + r * 4 + j]);
        }
  }
"""

_NIBBLE_WIDENING = NIBBLES[NIBBLES.index("    if (BITS == 3) {"):
                           NIBBLES.index("    simdgroup_barrier(mem_flags::mem_threadgroup);\n    float s[NF][4]")]
_BYTE_WIDENING = """    for (int c = 0; c < 16; c++) {
      const int bit = 4 * BITS * c, i = bit >> 5, sh = bit & 31;
      uint word = w[i] >> sh;
      if (sh + 4 * BITS > 32) word |= w[i + 1] << (32 - sh);
      word = (word & ((1u << (2 * BITS)) - 1u)) | (((word >> (2 * BITS)) & ((1u << (2 * BITS)) - 1u)) << 16);
      word = (word & ((0x10001u << BITS) - 0x10001u)) | (((word >> BITS) & ((0x10001u << BITS) - 0x10001u)) << 8);
      stage[lane * 16 + c] = word;
    }
"""


def _bytes(source: str) -> str:
    for old, new in (
        ('static_assert(BITS == 2 || BITS == 3, "4-bit weights go to the tensor op as they are");',
         'static_assert(BITS == 5 || BITS == 6 || BITS == 8, "bytes for 5-, 6- and 8-bit weights");'),
        ("threadgroup uint stage_all[SK * NT * 8];               // per K slice: NT columns x 64 nibbles",
         "threadgroup uint stage_all[SK * NT * 16];              // per K slice: NT columns x 64 bytes"),
        ("threadgroup uint* stage = stage_all + sg * NT * 8;", "threadgroup uint* stage = stage_all + sg * NT * 16;"),
        ("tensor<threadgroup uint4b_format, dextents<int32_t, 2>, tensor_inline> b((threadgroup uchar*)stage,",
         "tensor<threadgroup uint8_t, dextents<int32_t, 2>, tensor_inline> b((threadgroup uint8_t*)stage,"),
        (_NIBBLE_WIDENING, _BYTE_WIDENING),
    ):
        if source.count(old) != 1:
            raise AssertionError(f"lane_widen: the nibble kernel changed ({old[:60]!r})")
        source = source.replace(old, new)
    return source


BYTES = _bytes(NIBBLES)


class _Baked:
    """Template integers baked into the source; one compiled kernel per constant set (named by source hash)."""

    def __init__(self, base: str, body: str, inputs: list, outputs: list) -> None:
        self.base, self.body, self.inputs, self.outputs = base, body, inputs, outputs
        self.compiled: dict = {}

    def __call__(self, *, template=(), **kwargs):
        key = tuple(template)
        run = self.compiled.get(key)
        if run is None:
            source = "".join(f"  constexpr int {k} = {int(v)};\n" for k, v in key) + self.body
            name = f"{self.base}_{hashlib.sha256((_HEADER + source).encode()).hexdigest()[:16]}"
            run = self.compiled[key] = mx.fast.metal_kernel(
                name=name, input_names=self.inputs, output_names=self.outputs, source=source, header=_HEADER)
        return run(**kwargs)


_K = {
    "xsum": _Baked("vmlx_lane_xsum", _XSUM, ["X", "mdims"], ["XS"]),
    "main": _Baked("vmlx_lane_main", _MAIN, ["X", "XS", "Wq", "SBt", "mdims"], ["Y"]),
    "main_tiled": _Baked("vmlx_lane_main_tiled", _MAIN_TILED, ["X", "XS", "Wq", "SBt", "mdims"], ["Y"]),
    "lowbit": _Baked("vmlx_lane_lowbit", NIBBLES, ["X", "XS", "Wq", "SBt", "mdims"], ["Y"]),
    "bytes": _Baked("vmlx_lane_bytes", BYTES, ["X", "XS", "Wq", "SBt", "mdims"], ["Y"]),
}
_MDIMS: dict = {}


def _mdims(m: int, mp: int) -> mx.array:
    a = _MDIMS.get((m, mp))
    if a is None:
        a = _MDIMS[(m, mp)] = mx.array([m, mp] + [0] * 14, dtype=mx.int32)
    return a


def split_k(n: int, k: int) -> int:
    """K slices for an (n, k) weight: fixed by the shape, never by the row count."""
    tiles = -(-n // NT)
    sk = 1
    while sk < 8 and tiles * sk < 1024 and (k // 64) // (sk * 2) >= 8:
        sk *= 2
    return sk


def tile_weight(weight: mx.array, *, bits: int, group: int = 64) -> mx.array:
    """MLX's packed (N, K*bits/32) weight regrouped [N/NT][K/group][NT columns x a group's words]: same bytes."""
    n, kw, w = int(weight.shape[0]), int(weight.shape[1]), group * bits // 32
    return mx.contiguous(weight.reshape(n // NT, NT, kw // w, w).transpose(0, 2, 1, 3).reshape(n, kw))


def untile_weight(weight: mx.array, *, bits: int, group: int = 64) -> mx.array:
    """``tile_weight`` undone: MLX's packed layout again."""
    n, kw, w = int(weight.shape[0]), int(weight.shape[1]), group * bits // 32
    return mx.contiguous(weight.reshape(n // NT, kw // w, NT, w).transpose(0, 2, 1, 3).reshape(n, kw))


def pack_scales(scales: mx.array, biases: mx.array) -> mx.array:
    """(N, K/GS) scales and biases -> (K/GS, N, 2) bf16 pairs, group-major."""
    return mx.stack([scales.T, biases.T], axis=-1).astype(mx.bfloat16)


def supports(weight: mx.array, scales: mx.array, x: mx.array, bits: int, group_size: int, mode: str = "affine") -> bool:
    if mode != "affine" or bits not in BITS or group_size not in ((32, 64) if bits == 4 else (64,)):
        return False
    if x.dtype != mx.bfloat16 or scales.dtype != mx.bfloat16 or weight.dtype != mx.uint32 or weight.ndim != 2:
        return False
    k = int(x.shape[-1])
    return k % 64 == 0 and int(weight.shape[1]) * 32 == k * bits and int(weight.shape[0]) % 4 == 0


_AVAILABLE = None


def available() -> bool:
    """Metal 4 tensor ops compile here and give the right answer (checked once per process)."""
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            w = mx.random.normal((64, 128)).astype(mx.bfloat16)
            q, s, b = mx.quantize(w, group_size=64, bits=4)
            x = mx.random.normal((3, 128)).astype(mx.bfloat16)
            y = lane_matmul(x, q, pack_scales(s, b), bits=4, group=64)
            ref = x.astype(mx.float32) @ mx.dequantize(q, s, b, group_size=64, bits=4).astype(mx.float32).T
            _AVAILABLE = bool(mx.allclose(y.astype(mx.float32), ref, atol=0.25, rtol=0.05).item())
        except Exception:
            _AVAILABLE = False
    return _AVAILABLE


def lane_matmul(x: mx.array, weight: mx.array, sbt: mx.array, *, bits: int, group: int = 64,
                tiled: bool = False) -> mx.array:
    """x (..., K) bf16 times packed ``weight`` (N, K*bits/32) transposed; at most MAX_ROWS rows; MLX layout."""
    K = int(x.shape[-1])
    N = int(weight.shape[0])
    lead = x.shape[:-1]
    x2 = x.reshape(-1, K)
    M = int(x2.shape[0])
    if M > MAX_ROWS:
        raise ValueError(f"lane_matmul takes at most {MAX_ROWS} rows, got {M}")
    MP = 16 * ((M + 15) // 16)
    KG = K // group
    mdims = _mdims(M, MP)
    xs = _K["xsum"](inputs=[x2, mdims], template=[("K", K), ("GS", group)], grid=(KG, MP, 1),
                    threadgroup=(min(KG, 256), 1, 1), output_shapes=[(KG, MP)], output_dtypes=[mx.float32])[0]
    sk = split_k(N, K)
    block = MP if MP <= ROW_BLOCK else ROW_BLOCK
    edge = int(MP % block != 0)
    if bits != 4:
        if group != 64:
            raise ValueError("non-4-bit lane weights need groups of 64")
        y = _K["lowbit" if bits < 4 else "bytes"](
            inputs=[x2, xs, weight, sbt, mdims],
            template=[("TMR", block // 16), ("N", N), ("K", K), ("NT", NT), ("SK", sk), ("BITS", bits), ("TILED", int(tiled))],
            grid=(-(-N // NT) * 32 * sk, -(-MP // block), 1), threadgroup=(32 * sk, 1, 1),
            output_shapes=[(M, N)], output_dtypes=[mx.bfloat16])[0]
    else:
        y = _K["main_tiled" if tiled else "main"](
            inputs=[x2, xs, weight, sbt, mdims],
            template=[("TMR", block // 16), ("N", N), ("K", K), ("NT", NT), ("SK", sk), ("GS", group), ("EDGE", edge)],
            grid=(-(-N // NT) * 32 * sk, -(-MP // block), 1), threadgroup=(32 * sk, 1, 1),
            output_shapes=[(M, N)], output_dtypes=[mx.bfloat16])[0]
    return y.reshape(*lead, N)


# -- routing a model's projections ----------------------------------------------------------------
#
# Installed per module (class swap), never by patching nn.QuantizedLinear.__call__: the drafter and
# every other model in the process keep MLX's kernels.  Calls of <= MAX_ROWS bf16 rows (decode, DFlash2
# verify, short resume deltas) take the lane kernel; wider calls (prompt prefill chunks) and other
# dtypes take MLX's quantized_matmul on the MLX-layout weight (rebuilt per call when tiled: one weight
# copy per call, ~1-2 % of a 2048-row chunk's compute).

import mlx.nn as _nn


class LaneQuantizedLinear(_nn.QuantizedLinear):
    def __call__(self, x: mx.array) -> mx.array:
        rows = 1
        for d in x.shape[:-1]:
            rows *= int(d)
        tiled = getattr(self, "_lane_tiled", False)
        if rows <= MAX_ROWS and x.dtype == mx.bfloat16:
            y = lane_matmul(x, self["weight"], self._lane_sbt, bits=self.bits, group=self.group_size, tiled=tiled)
        else:
            weight = untile_weight(self["weight"], bits=self.bits, group=self.group_size) if tiled else self["weight"]
            y = mx.quantized_matmul(x, weight, self["scales"], self["biases"], transpose=True,
                                    group_size=self.group_size, bits=self.bits)
        if "bias" in self:
            y = y + self["bias"]
        return y


def _takes(module) -> bool:
    w, s = module["weight"], module["scales"]
    k = int(w.shape[1]) * 32 // int(module.bits) if w.ndim == 2 else 0
    return (type(module) is _nn.QuantizedLinear and getattr(module, "mode", "affine") == "affine"
            and module.bits in BITS and module.group_size in ((32, 64) if module.bits == 4 else (64,))
            and s.dtype == mx.bfloat16 and w.dtype == mx.uint32 and w.ndim == 2
            and k % 64 == 0 and int(w.shape[0]) % 4 == 0)


def install(model, *, tile: bool = True) -> dict:
    """Route ``model``'s plain affine QuantizedLinear layers through the lane matmul.

    Returns counts {"lane": n, "tiled": n, "skipped": {kind: n}} for the load log.
    """
    if not available():
        return {"lane": 0, "tiled": 0, "skipped": {"no Metal 4 tensor ops": 1}}
    counts = {"lane": 0, "tiled": 0, "skipped": {}}
    built, pending = [], 0
    for _name, module in model.named_modules():
        if not isinstance(module, _nn.QuantizedLinear) or isinstance(module, LaneQuantizedLinear):
            continue
        if not _takes(module):
            kind = f"{type(module).__name__} {module.bits}-bit g{module.group_size} {module['scales'].dtype}"
            counts["skipped"][kind] = counts["skipped"].get(kind, 0) + 1
            continue
        sbt = pack_scales(module["scales"], module["biases"])
        object.__setattr__(module, "_lane_sbt", sbt)
        built.append(sbt)
        pending += sbt.nbytes
        w = module["weight"]
        if tile and int(w.shape[0]) % NT == 0 and (module.bits == 4 or module.group_size == 64):
            module.weight = tile_weight(w, bits=module.bits, group=module.group_size)
            object.__setattr__(module, "_lane_tiled", True)
            built.append(module["weight"])
            pending += 2 * w.nbytes          # old layout lives until this batch is evaluated
            counts["tiled"] += 1
        module.__class__ = LaneQuantizedLinear
        counts["lane"] += 1
        if pending >= 2 * 1024**3:
            mx.eval(built)
            built, pending = [], 0
    if built:
        mx.eval(built)
    mx.clear_cache()
    return counts
