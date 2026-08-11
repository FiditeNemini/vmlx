# SPDX-License-Identifier: Apache-2.0
"""Muse Glimmer vision tower and adapter.

Ground truth is the working Swift port,
``vmlx-swift/Libraries/MLXVLM/Models/MuseGlimmerVision.swift``.

Read off the checkpoint, and deliberately NOT modelled on the in-tree Qwen-VL
tower, which differs in every one of these respects:

* ``norm1``/``norm2`` and ``ln_pre``/``ln_post`` all carry a bias, so they are
  LayerNorm, not RMSNorm.
* Position information comes from a learned ``position_embedding_table`` of
  ``pos_emb_height * pos_emb_width`` (32*32 = 1024) rows, BILINEARLY RESAMPLED
  onto the actual patch grid — it is not a per-index lookup, and it is not 2D
  RoPE (there is separate rotary on top).
* Attention keeps separate ``q_proj``/``k_proj``/``v_proj`` (each with bias) and
  names its output ``proj``, not a fused qkv with ``o_proj``.
* ``patch_embedding`` takes 1176 inputs = 2*3*14*14, i.e. a temporal patch of 2
  frames — this is a video encoder, not a still-image one.

FOUR ORDERING HAZARDS live in here. Every one of them leaves shapes valid and
the model fluent, so none of them announce themselves:

1. The processor emits rows MERGE-BLOCK-major; positions, rotary and the window
   partition are all defined over RASTER order. ``_raster_order`` converts.
2. Each 1176-vector arrives ``[C][T][h][w]`` but ``patch_embedding`` was trained
   on ``[T][C][h][w]``. Invisible in greyscale; scrambles colour.
3. Rotary coordinates are ``(column, row)`` — in that order — and 1-BASED.
4. The 2x2 merge is FEATURE-major (transpose before the flatten), not four
   patch vectors concatenated.
"""

import math

import numpy as np
from typing import Any, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn


def _raster_order(grid_t: int, grid_h: int, grid_w: int, merge: int) -> mx.array:
    """Row indices that turn merge-block-major patch rows into raster order.

    The processor groups each ``merge x merge`` block contiguously. Everything
    downstream — the position grid, the rotary coordinates, the window
    partition — indexes patches as plain ``(row, col)``, so undo the grouping
    once, up front.
    """
    rows = []
    blocks_w = grid_w // merge
    per_frame = grid_h * grid_w
    for t in range(grid_t):
        base = t * per_frame
        for row in range(grid_h):
            for col in range(grid_w):
                block = (row // merge) * blocks_w + (col // merge)
                within = (row % merge) * merge + (col % merge)
                rows.append(base + block * merge * merge + within)
    return mx.array(rows, dtype=mx.int32)


def _bilinear_position_rows(
    table: mx.array, grid_h: int, grid_w: int, side: int
) -> mx.array:
    """Resample the learned ``side x side`` position table onto ``grid_h x grid_w``.

    This is ``F.grid_sample(align_corners=False, padding_mode="zeros")`` written
    out. The zeros padding is load-bearing: an out-of-range corner contributes
    weight ZERO. Clamping the index and keeping its weight instead smears the
    border inward, which looks fine and is wrong.
    """

    def axis(n: int) -> Tuple[List[int], List[int], List[float], List[float]]:
        lo_idx, hi_idx, lo_w, hi_w = [], [], [], []
        for i in range(n):
            g = (i + 0.5) * (side / n) - 0.5
            f = math.floor(g)
            frac = g - f
            fi = int(f)
            lo_idx.append(min(max(fi, 0), side - 1))
            hi_idx.append(min(max(fi + 1, 0), side - 1))
            lo_w.append((1.0 - frac) if 0 <= fi <= side - 1 else 0.0)
            hi_w.append(frac if 0 <= fi + 1 <= side - 1 else 0.0)
        return lo_idx, hi_idx, lo_w, hi_w

    r_lo, r_hi, r_wlo, r_whi = axis(grid_h)
    c_lo, c_hi, c_wlo, c_whi = axis(grid_w)

    table32 = table.astype(mx.float32)
    rows = []
    for i in range(grid_h):
        for j in range(grid_w):
            acc = None
            for r, wr in ((r_lo[i], r_wlo[i]), (r_hi[i], r_whi[i])):
                for c, wc in ((c_lo[j], c_wlo[j]), (c_hi[j], c_whi[j])):
                    w = wr * wc
                    if w == 0.0:
                        continue
                    # Table is indexed ROW-MAJOR. A transposed index (c*side+r)
                    # is the documented failure signature.
                    term = table32[r * side + c] * w
                    acc = term if acc is None else acc + term
            rows.append(acc if acc is not None else mx.zeros_like(table32[0]))
    return mx.stack(rows, axis=0)


def _rope_tables(grid_h: int, grid_w: int, head_dim: int, theta: float) -> mx.array:
    """Per-patch rotary frequencies, laid out ``[col_freqs, row_freqs]``.

    Coordinates are (column, row) — in that order — and 1-based.
    """
    half = head_dim // 2
    dims = half // 2
    inv_freq = mx.exp(
        -mx.arange(0, dims, dtype=mx.float32) * (2.0 * math.log(theta) / (2 * dims))
    )
    max_side = max(grid_h, grid_w) + 2
    table = mx.outer(mx.arange(max_side, dtype=mx.float32), inv_freq)

    cols, rows = [], []
    for row in range(grid_h):
        for col in range(grid_w):
            cols.append(col + 1)
            rows.append(row + 1)
    col_f = table[mx.array(cols, dtype=mx.int32)]
    row_f = table[mx.array(rows, dtype=mx.int32)]
    return mx.concatenate([col_f, row_f], axis=-1)


def _apply_rope(x: mx.array, freqs: mx.array) -> mx.array:
    """NeoX split-half rotation: element i pairs with i + head_dim/2."""
    emb = mx.concatenate([freqs, freqs], axis=-1)
    cos = mx.cos(emb)[None, None, :, :]
    sin = mx.sin(emb)[None, None, :, :]
    half = x.shape[-1] // 2
    rotated = mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos + rotated * sin


def _window_index_and_bounds(
    grid_t: int, grid_h: int, grid_w: int, window_patches: int
) -> Tuple[mx.array, List[int]]:
    """Window-major visiting order over PATCHES (not merged cells).

    Cells past the grid edge are SKIPPED, not padded, so edge windows are
    simply smaller and the segment bounds are the real per-window counts.
    """
    order: List[int] = []
    bounds: List[int] = [0]
    per_frame = grid_h * grid_w
    for t in range(grid_t):
        base = t * per_frame
        for wh in range(0, grid_h, window_patches):
            for ww in range(0, grid_w, window_patches):
                count = 0
                for r in range(wh, min(wh + window_patches, grid_h)):
                    for c in range(ww, min(ww + window_patches, grid_w)):
                        order.append(base + r * grid_w + c)
                        count += 1
                if count:
                    bounds.append(bounds[-1] + count)
    return mx.array(order, dtype=mx.int32), bounds


def _segment_mask(segment_ids: Sequence[int]) -> mx.array:
    """Boolean mask, True = may attend. Non-causal, block-diagonal."""
    ids = mx.array(list(segment_ids), dtype=mx.int32)
    return (ids[:, None] == ids[None, :])[None, None, :, :]


def _block_diagonal_mask(bounds: Sequence[int], length: int) -> mx.array:
    """Segment ids from cumulative bounds, then a block-diagonal mask."""
    seg = [0] * length
    for i in range(len(bounds) - 1):
        for p in range(bounds[i], min(bounds[i + 1], length)):
            seg[p] = i
    return _segment_mask(seg)


class MuseVisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = dim // self.n_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        # Named `proj` in the checkpoint, not `o_proj`.
        self.proj = nn.Linear(dim, dim, bias=True)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        freqs: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)

        if freqs is not None:
            q = _apply_rope(q.astype(mx.float32), freqs).astype(x.dtype)
            k = _apply_rope(k.astype(mx.float32), freqs).astype(x.dtype)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.proj(out)


class MuseVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class MuseVisionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        eps = config.layer_norm_eps
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=eps)
        self.attn = MuseVisionAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=eps)
        self.mlp = MuseVisionMLP(config)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        freqs: Optional[mx.array] = None,
    ) -> mx.array:
        x = x + self.attn(self.norm1(x), mask=mask, freqs=freqs)
        return x + self.mlp(self.norm2(x))


class MusePatchEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = int(config.patch_size)
        self.temporal = int(config.patch_temporal)
        patch_dim = self.temporal * 3 * self.patch_size * self.patch_size
        self.patch_embedding = nn.Linear(patch_dim, config.hidden_size, bias=False)
        self.position_embedding_table = nn.Embedding(
            config.position_table_size, config.hidden_size
        )

    def temporal_major(self, patches: mx.array) -> mx.array:
        """``[C][T][h][w]`` -> ``[T][C][h][w]``.

        The processor emits channel-major (that is what a Qwen-style patchify
        produces) but this checkpoint's patch_embedding was trained on
        temporal-major. Getting it wrong is invisible in shapes and in
        greyscale, and scrambles colour.
        """
        spatial = self.patch_size * self.patch_size
        L = patches.shape[0]
        return (
            patches.reshape(L, 3, self.temporal, spatial)
            .transpose(0, 2, 1, 3)
            .reshape(L, 3 * self.temporal * spatial)
        )

    def __call__(self, patches: mx.array, positions: mx.array) -> mx.array:
        h = self.patch_embedding(self.temporal_major(patches))
        return h + positions.astype(h.dtype)


class VisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_type = config.model_type
        self.config = config
        self.patch_embedder = MusePatchEmbedder(config)
        self.ln_pre = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = [MuseVisionLayer(config) for _ in range(config.num_hidden_layers)]
        self.ln_post = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # ---- geometry ---------------------------------------------------------

    def _window_patches(self) -> int:
        """Window side in PATCHES. 32*14 = 448px, partitioned over patches."""
        return int(getattr(self.config, "pos_emb_height", 32) or 32)

    def _layer_is_windowed(self, index: int) -> bool:
        types = getattr(self.config, "vision_layer_types", None)
        if types and index < len(types):
            return str(types[index]) != "full_attention"
        # The shipped 50-layer list breaks the x4 period at the tail
        # (full at 47 then 49), so never rely on the derived cadence when the
        # checkpoint gives an explicit list.
        return (index + 1) % 4 != 0

    def __call__(
        self,
        patches: mx.array,
        grid_thw: Optional[Sequence[Tuple[int, int, int]]] = None,
        **kwargs: Any,
    ) -> mx.array:
        if patches.ndim == 3:
            patches = patches.reshape(-1, patches.shape[-1])
        if not grid_thw:
            raise ValueError(
                "Muse Glimmer vision tower needs grid_thw; positions, rotary and "
                "the window partition are all defined over the patch grid."
            )
        grid_t, grid_h, grid_w = (int(v) for v in grid_thw[0])
        merge = int(self.config.merge_size)
        side = int(getattr(self.config, "pos_emb_height", 32) or 32)

        # 1. merge-block-major -> raster
        patches = patches[_raster_order(grid_t, grid_h, grid_w, merge)]

        # 2. positions: bilinear resample, same rows repeated for every t
        pos = _bilinear_position_rows(
            self.patch_embedder.position_embedding_table.weight, grid_h, grid_w, side
        )
        if grid_t > 1:
            pos = mx.concatenate([pos] * grid_t, axis=0)

        h = self.ln_pre(self.patch_embedder(patches, pos))[None]

        # 3. rotary + masks over the raster grid
        freqs = _rope_tables(
            grid_h, grid_w, self.patch_embedder.patch_embedding.weight.shape[0]
            // self.config.num_attention_heads,
            float(getattr(self.config, "rope_theta", 10000.0) or 10000.0),
        )
        if grid_t > 1:
            freqs = mx.concatenate([freqs] * grid_t, axis=0)

        length = grid_t * grid_h * grid_w
        window_index, window_bounds = _window_index_and_bounds(
            grid_t, grid_h, grid_w, self._window_patches()
        )

        # 4. Permute into window order ONCE, run all layers there, invert once.
        # Both masks therefore live in window-permuted coordinates. Full layers
        # still segment per t-slice — frames never attend across each other,
        # even on a "full" layer — so their segment id is the t-slice that each
        # permuted position came from.
        inverse = mx.argsort(window_index)
        order = np.asarray(window_index).tolist()
        per_frame = grid_h * grid_w
        window_mask = _block_diagonal_mask(window_bounds, length)
        full_mask = _segment_mask([p // per_frame for p in order])

        h = h[:, window_index]
        freqs = freqs[window_index]
        for index, layer in enumerate(self.layers):
            mask = window_mask if self._layer_is_windowed(index) else full_mask
            h = layer(h, mask=mask, freqs=freqs)
        h = h[:, inverse]

        return self.ln_post(h)[0]


class MuseVisionAdapter(nn.Module):
    """Merge a merge_size x merge_size patch neighbourhood, then project.

    ``fc1`` accepting vision_hidden * merge_size**2 is what pins the merge
    factor: 1536 * 4 = 6144 for the shipped bundles.

    The merged vector is FEATURE-major — all four block values of feature 0,
    then feature 1, and so on — NOT four patch vectors concatenated. Both
    produce a 6144-wide tensor.
    """

    def __init__(self, model_config):
        super().__init__()
        vision = model_config.vision_config
        merged = vision.hidden_size * (int(vision.merge_size) ** 2)
        hidden = int(model_config.projector_hidden_size)
        self.merge_size = int(vision.merge_size)
        self.fc1 = nn.Linear(merged, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, hidden, bias=False)

    def pixel_shuffle(
        self, features: mx.array, grid_thw: Sequence[Tuple[int, int, int]]
    ) -> mx.array:
        """Raster patches -> merged cells, feature-major within each cell."""
        merge = self.merge_size
        grid_t, grid_h, grid_w = (int(v) for v in grid_thw[0])
        dim = features.shape[-1]
        per_frame = grid_h * grid_w

        rows = []
        for t in range(grid_t):
            base = t * per_frame
            for hb in range(grid_h // merge):
                for wb in range(grid_w // merge):
                    for di in range(merge):
                        for dj in range(merge):
                            r = hb * merge + di
                            c = wb * merge + dj
                            rows.append(base + r * grid_w + c)
        gathered = features[mx.array(rows, dtype=mx.int32)]
        cells = gathered.shape[0] // (merge * merge)
        return gathered.reshape(cells, merge * merge, dim).transpose(0, 2, 1).reshape(
            cells, dim * merge * merge
        )

    def __call__(
        self,
        features: mx.array,
        grid_thw: Optional[Sequence[Tuple[int, int, int]]] = None,
    ) -> mx.array:
        if features.ndim == 3:
            features = features.reshape(-1, features.shape[-1])
        if grid_thw:
            features = self.pixel_shuffle(features, grid_thw)
        else:
            group = self.merge_size**2
            usable = (features.shape[0] // group) * group
            features = features[:usable].reshape(usable // group, group * features.shape[-1])
        return self.fc2(nn.gelu(self.fc1(features)))
