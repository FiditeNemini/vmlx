// SPDX-License-Identifier: Apache-2.0
// Sparse-GQA organization follows the vendored oMLX kernel. NAX fragments,
// QK/PV operand types and softmax reduction follow MLX 0.32.2 Steel attention
// (Copyright 2025 Apple Inc.; see MLX_LICENSE.txt). Experimental, not exact.

#pragma once

template <typename T>
[[kernel, max_total_threads_per_threadgroup(64)]]
void qwen4_sparse_online_nax(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    const device uint* Ids [[buffer(3)]],
    device T* O [[buffer(4)]],
    constant Qwen4QSASparseGQAParams* p [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint d_half [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  using namespace mlx::steel;
  const int row = int(group.x), kv = int(group.y);
  const int absolute = p->q_offset + row;
  const int complete = (absolute + 1) / 4;
  const int valid_blocks = min(512, complete);
  const short2 coord = BaseNAXFrag::get_coord();
  threadgroup int tile_base;
  threadgroup uint tile_bits;
  threadgroup float exchange[2][32 * 16];
  NAXTile<float, 1, 8> output;
  output.clear();
  metal::vec<float, 2> maximum{Limits<float>::finite_min, Limits<float>::finite_min};
  metal::vec<float, 2> denominator{0};
  const float scale = p->scale * 1.44269504089f;

  NAXTile<T, 1, 1> query_tiles[8];
  STEEL_PRAGMA_UNROLL
  for (short id = 0; id < 8; ++id) {
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < 8; ++e) {
      const int head = coord.y + (e / 4) * 8;
      const int dim = int(d_half) * 128 + id * 16 + coord.x + e % 4;
      query_tiles[id].frag_at(0, 0)[e] = head < 12
          ? Q[size_t(kv * 12 + head) * p->Q_strides[1] + size_t(row) * p->Q_strides[2] + dim]
          : T(0);
    }
  }

  // Match MLX head-256 NAX: BK32 and two independent 128-wide QK partials.
  // Keep each selected key in its ORIGINAL 32-key tile. Packing keys changes
  // both online-softmax and PV reduction order even when selection is equal.
  // Skip only tiles with no selected keys (stock P is zero there).
  int cursor = 0;
  bool tail_pending = (absolute + 1) % 4 != 0;
  for (int tile = 0; tile < 513; ++tile) {
    if (d_half == 0 && lane == 0) {
      tile_base = -1;
      tile_bits = 0;
      if (cursor < valid_blocks) {
        tile_base = int(Ids[size_t(row) * p->Topk_strides[2] + cursor] / 8) * 32;
        while (cursor < valid_blocks) {
          const uint block = Ids[size_t(row) * p->Topk_strides[2] + cursor];
          if (int(block / 8) * 32 != tile_base) break;
          tile_bits |= 15u << ((block % 8) * 4);
          ++cursor;
        }
      } else if (tail_pending) {
        tile_base = complete * 4 / 32 * 32;
      }
      if (tail_pending && tile_base == complete * 4 / 32 * 32) {
        tile_bits |= ((1u << ((absolute + 1) % 4)) - 1u) << (complete * 4 % 32);
        tail_pending = false;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tile_base < 0) break;
    NAXTile<float, 1, 2> scores;
    scores.clear();
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < 2; ik += 2) {
      STEEL_PRAGMA_UNROLL
      for (short d = 0; d < 128; d += 16) {
        NAXTile<T, 2, 1> k;
        const device T* keys = K + size_t(kv) * p->K_strides[1]
            + size_t(tile_base) * p->K_strides[2] + int(d_half) * 128 + d;
        if (tile_base + 32 <= p->kL) k.load(keys, int(p->K_strides[2]));
        else k.load_rows(keys, int(p->K_strides[2]), p->kL - tile_base);
        BaseNAXFrag::mma(scores.frag_at(0, ik), scores.frag_at(0, ik + 1),
                         query_tiles[d / 16].frag_at(0, 0), metal::false_type{},
                         k.frag_at(0, 0), k.frag_at(1, 0), metal::true_type{});
      }
    }
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < 8; ++e) {
      exchange[d_half][lane * 16 + e] = scores.frag_at(0, 0)[e];
      exchange[d_half][lane * 16 + 8 + e] = scores.frag_at(0, 1)[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < 8; ++e) {
      scores.frag_at(0, 0)[e] += exchange[1 - d_half][lane * 16 + e];
      scores.frag_at(0, 1)[e] += exchange[1 - d_half][lane * 16 + 8 + e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < 2; ++ik) {
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < 8; ++e) {
        const int col = ik * 16 + coord.x + e % 4;
        const int token = tile_base + col;
        const bool visible = ((tile_bits >> col) & 1u) && token < p->kL && token <= absolute;
        scores.frag_at(0, ik)[e] = visible ? scores.frag_at(0, ik)[e] * scale : -INFINITY;
      }
    }
    metal::vec<float, 2> next_max{maximum[0], maximum[1]};
    scores.template row_reduce<Qwen4SparseMaxOp>(next_max);
    scores.template row_bin_op<Qwen4SparseExpSubOp>(next_max);
    metal::vec<float, 2> factor;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; ++i) {
      factor[i] = fast::exp2(maximum[i] - next_max[i]);
      maximum[i] = next_max[i];
      denominator[i] *= factor[i];
    }
    scores.template row_reduce<Qwen4SparseSumOp>(denominator);
    output.template row_bin_op<Qwen4SparseMulOp>(factor);
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < 8; id += 2) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < 2; ++ik) {
        NAXTile<T, 1, 2> v;
        const device T* values = V + size_t(kv) * p->V_strides[1]
            + size_t(tile_base + ik * 16) * p->V_strides[2] + int(d_half) * 128 + id * 16;
        if (tile_base + (ik + 1) * 16 <= p->kL) v.load(values, int(p->V_strides[2]));
        else v.load_rows(values, int(p->V_strides[2]), p->kL - tile_base - ik * 16);
        BaseNAXFrag::mma(output.frag_at(0, id), output.frag_at(0, id + 1),
                         scores.frag_at(0, ik), metal::false_type{},
                         v.frag_at(0, 0), v.frag_at(0, 1), metal::false_type{});
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  metal::vec<float, 2> reciprocal{1.f / denominator[0], 1.f / denominator[1]};
  output.template row_bin_op<Qwen4SparseMulOp>(reciprocal);
  STEEL_PRAGMA_UNROLL
  for (short id = 0; id < 8; ++id) {
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < 8; ++e) {
      const int head = coord.y + (e / 4) * 8;
      const int dim = int(d_half) * 128 + id * 16 + coord.x + e % 4;
      if (head < 12)
        O[size_t(kv * 12 + head) * p->O_strides[1] + size_t(row) * p->O_strides[2] + dim]
            = T(output.frag_at(0, id)[e]);
    }
  }
}
