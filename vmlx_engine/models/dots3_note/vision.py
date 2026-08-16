# SPDX-License-Identifier: Apache-2.0
"""dots3_note vision tower (MoE-ViT) as an nn.Module tree.

Math is a verbatim port of the verified functional reference
(``dots3-ref/towers.py``: cos 0.999999-1.0 vs the transformers PR at capture),
re-hosted on mlx.nn modules so the parameter paths match the checkpoint when
this tower is mounted as ``vision_encoder`` by the outer model.

Traps that shaped this file:

- 2D rope is BLOCK-MAJOR over the 2x2 merge blocks (``vision_position_ids``),
  cos/sin built from ``(N, 2)`` pos ids x ``inv_freq`` then DUPLICATED — this is
  rotate_half over the FULL head_dim, NOT the LM's interleaved rope.
- Attention is segmented PER FRAME via cu_seqlens; one flat SDPA over the whole
  patch sequence attends across media boundaries and stays fluent while wrong.
- The sigmoid router adds the f32 ``router_bias`` for SELECTION only; returned
  weights come from the UNBIASED probs, renormalized by their own sum, and the
  accumulated output is divided by ``wsum`` again exactly as the reference does.
- ``patch_embed.proj`` weights live in MLX Conv2d layout (O, kh, kw, I); the
  checkpoint ships OIHW — the outer model's sanitize owns that transpose.
- Deep pyramid MoE blocks (block ~38) have violent inter-expert cancellation:
  per-expert outputs of +-250 sum to +-5, so bf16-class compute shows 5-25%
  block-output noise there. This is EXPECTED and tolerated — do not "fix" it,
  it is the model, not the port.
"""

import math
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import VisionConfig


def vision_position_ids(grid_thw: np.ndarray, merge: int) -> np.ndarray:
    """(total_patches, 2) h/w ids, block-major over merge x merge blocks."""
    out = []
    for t, h, w in np.asarray(grid_thw).reshape(-1, 3).tolist():
        hh, ww = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        shape = (h // merge, merge, w // merge, merge)
        hh = hh.reshape(shape).transpose(0, 2, 1, 3).reshape(-1)
        ww = ww.reshape(shape).transpose(0, 2, 1, 3).reshape(-1)
        pos = np.stack([hh, ww], -1)
        out.append(np.tile(pos, (t, 1)))
    return np.concatenate(out, 0)


def vision_cu_seqlens(grid_thw: np.ndarray) -> List[int]:
    """Per-frame attention segments (merge_temporal=False convention)."""
    seq = []
    for t, h, w in np.asarray(grid_thw).reshape(-1, 3).tolist():
        seq.extend([h * w] * t)
    cu = [0]
    for s in seq:
        cu.append(cu[-1] + s)
    return cu


def _rotate_half(x: mx.array) -> mx.array:
    a, b = mx.split(x, 2, axis=-1)
    return mx.concatenate([-b, a], axis=-1)


def _gelu_erf(x: mx.array) -> mx.array:
    # Exact-erf GELU; the tanh approximation drifts the adapter output.
    return 0.5 * x * (1 + mx.erf(x / math.sqrt(2.0)))


class VisionPatchEmbed(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.embed_dim = config.embed_dim
        self.proj = nn.Conv2d(
            config.num_channels,
            config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True,
        )
        self.norm = nn.RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        p = self.patch_size
        pv = pixel_values.reshape(-1, self.num_channels, p, p)
        # MLX conv is NHWC; rows arrive channel-major (C, p, p) per patch.
        x = self.proj(pv.transpose(0, 2, 3, 1))
        return self.norm(x.reshape(-1, self.embed_dim))


class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.heads = config.num_attention_heads
        self.embed = config.embed_dim
        self.head_dim = self.embed // self.heads
        self.qkv = nn.Linear(self.embed, 3 * self.embed, bias=config.use_bias)
        self.proj = nn.Linear(self.embed, self.embed, bias=config.use_bias)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(
        self, x: mx.array, cu: List[int], cos: mx.array, sin: mx.array
    ) -> mx.array:
        N = x.shape[0]
        qkv = self.qkv(x).reshape(N, 3, self.heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        # q/k RMSNorm FIRST, then rope in fp32 on the full head_dim.
        q = self.q_norm(q)
        k = self.k_norm(k)
        cs, sn = cos[:, None, :], sin[:, None, :]
        qf, kf = q.astype(mx.float32), k.astype(mx.float32)
        q = (qf * cs + _rotate_half(qf) * sn).astype(q.dtype)
        k = (kf * cs + _rotate_half(kf) * sn).astype(k.dtype)
        outs = []
        scale = self.head_dim ** -0.5
        # One SDPA per frame segment — cross-frame attention is wrong-but-fluent.
        for s0, s1 in zip(cu[:-1], cu[1:]):
            o = mx.fast.scaled_dot_product_attention(
                q[s0:s1].transpose(1, 0, 2)[None],
                k[s0:s1].transpose(1, 0, 2)[None],
                v[s0:s1].transpose(1, 0, 2)[None],
                scale=scale,
                mask=None,
            )
            outs.append(o[0].transpose(1, 0, 2).reshape(s1 - s0, self.embed))
        return self.proj(mx.concatenate(outs, 0))


class VisionMLP(nn.Module):
    """swiglu: fc2(silu(fc1(x)) * fc3(x))."""

    def __init__(self, config: VisionConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        inter = intermediate_size or config.intermediate_size
        self.fc1 = nn.Linear(config.embed_dim, inter, bias=config.use_bias)
        self.fc2 = nn.Linear(inter, config.embed_dim, bias=config.use_bias)
        self.fc3 = nn.Linear(config.embed_dim, inter, bias=config.use_bias)

    def __call__(self, x: mx.array) -> mx.array:
        g = self.fc1(x)
        return self.fc2((g * mx.sigmoid(g)) * self.fc3(x))


class VisionMoE(nn.Module):
    """Pyramid MoE block: sigmoid top-2 router with an f32 bias buffer.

    ``gate_weight`` is a PLAIN parameter (no Linear wrapper) so the checkpoint
    key ``blocks.{i}.mlp.gate_weight`` resolves directly; ``router_bias`` is the
    f32 selection-bias buffer. Bias shifts WHICH experts win, never the weights.
    """

    def __init__(self, config: VisionConfig, layer_idx: int):
        super().__init__()
        n_exp = config.pyramid_num_routed[layer_idx]
        if n_exp < 1:
            raise ValueError(f"block {layer_idx} is dense, not MoE")
        if config.router_scoring_func != "sigmoid":
            raise ValueError(
                f"unsupported router_scoring_func {config.router_scoring_func!r}"
            )
        self.num_experts = n_exp
        self.top_k = min(int(config.capacity_factor), n_exp)
        self.gate_weight = mx.zeros((n_exp, config.embed_dim), dtype=mx.bfloat16)
        self.router_bias = mx.zeros((n_exp,), dtype=mx.float32)
        self.experts = [
            VisionMLP(config, config.moe_intermediate_size) for _ in range(n_exp)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        gate_w = self.gate_weight.astype(mx.float32)
        bias = self.router_bias.astype(mx.float32)
        logits = x.astype(mx.float32) @ gate_w.T
        probs = mx.sigmoid(logits)
        k = self.top_k
        # Bias enters SELECTION only; routed weights come from unbiased probs.
        inds = mx.argpartition(-(probs + bias[None]), kth=k - 1, axis=-1)[:, :k]
        rw = mx.take_along_axis(probs, inds, axis=-1)
        rw = rw / (rw.sum(-1, keepdims=True) + 1e-9)
        out = mx.zeros_like(x.astype(mx.float32))
        wsum = mx.zeros((x.shape[0],), mx.float32)
        # Per-expert gather loop, np only for the index np.where (reference
        # does the same); token indices are unique per expert so plain
        # scatter-assign is safe. Correct beats fast here.
        inds_np = np.asarray(inds)
        rw_np = np.asarray(rw)
        for e in range(self.num_experts):
            tok, slot = np.where(inds_np == e)
            if tok.size == 0:
                continue
            t = mx.array(tok)
            we = mx.array(rw_np[tok, slot])
            y = self.experts[e](x[t])
            out[t] = out[t] + y.astype(mx.float32) * we[:, None]
            wsum[t] = wsum[t] + we
        # Final wsum renormalization is part of the reference math — keep it
        # even though rw already sums to ~1 per token.
        out = out / (wsum[:, None] + 1e-9)
        return out.astype(x.dtype)


class VisionBlock(nn.Module):
    def __init__(self, config: VisionConfig, layer_idx: int):
        super().__init__()
        self.norm_1 = nn.RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.norm_2 = nn.RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.attn = VisionAttention(config)
        if config.pyramid_num_routed[layer_idx] >= 1:
            self.mlp = VisionMoE(config, layer_idx)
        else:
            self.mlp = VisionMLP(config)

    def __call__(
        self, x: mx.array, cu: List[int], cos: mx.array, sin: mx.array
    ) -> mx.array:
        x = x + self.attn(self.norm_1(x), cu, cos, sin)
        return x + self.mlp(self.norm_2(x))


class VisionAdapter(nn.Module):
    """LN -> 2x2 merge -> Linear -> GELU -> Linear.

    ``mlp`` is a plain 3-item list so checkpoint keys ``adapter.mlp.0.*`` and
    ``adapter.mlp.2.*`` resolve; index 1 is the parameterless GELU placeholder.
    ``ln_q`` runs in fp32 at eps 1e-6 (NOT the LayerNorm default 1e-5).
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        merged = config.adapter_in_dim * config.adapter_merge_size ** 2
        self.merged_size = merged
        self.ln_q = nn.LayerNorm(config.adapter_in_dim, eps=1e-6, bias=True)
        self.mlp = [
            nn.Linear(merged, merged, bias=True),
            nn.GELU(),  # exact erf form; placeholder keeps index 2 aligned
            nn.Linear(merged, config.adapter_out_dim, bias=True),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        h = mx.fast.layer_norm(
            x.astype(mx.float32),
            self.ln_q.weight.astype(mx.float32),
            self.ln_q.bias.astype(mx.float32),
            1e-6,
        )
        h = h.reshape(-1, self.merged_size)
        h = self.mlp[0](h)
        h = _gelu_erf(h)
        return self.mlp[2](h)


class VisionModel(nn.Module):
    """dots3_note MoE-ViT: 25 dense + 17 pyramid-MoE blocks, 2x2 patch merger.

    ``__call__(pixel_values, grid_thw)``:
      pixel_values [n_patches, C*patch*patch] (block-major rows from the
      processor), grid_thw int [n_media, 3] as (t, h, w) in PATCHES.
      Returns [sum(t*h*w)/merge^2, adapter_out_dim].
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.merge = config.spatial_merge_size
        head_dim = config.embed_dim // config.num_attention_heads
        self.patch_embed = VisionPatchEmbed(config)
        self.blocks = [
            VisionBlock(config, i) for i in range(config.num_hidden_layers)
        ]
        self.post_trunk_norm = nn.RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.adapter = VisionAdapter(config)
        # np (NOT mx) so it never enters the parameter tree. (head_dim/4,)
        # frequencies; the (N,2) pos ids expand to (N, head_dim/2), then the
        # duplication below fills the full head_dim.
        self._inv_freq = (
            1.0
            / (
                10000.0
                ** (
                    np.arange(0, head_dim // 2, 2, dtype=np.float64)
                    / (head_dim // 2)
                )
            )
        ).astype(np.float32)

    def _rope(self, pos_ids: np.ndarray):
        emb = (pos_ids[:, :, None] * self._inv_freq[None, None, :]).reshape(
            pos_ids.shape[0], -1
        )
        emb = np.concatenate([emb, emb], -1).astype(np.float32)
        return mx.array(np.cos(emb)), mx.array(np.sin(emb))

    def __call__(self, pixel_values, grid_thw) -> mx.array:
        grid = np.asarray(grid_thw, dtype=np.int64).reshape(-1, 3)
        pv = pixel_values if isinstance(pixel_values, mx.array) else mx.array(
            np.asarray(pixel_values)
        )
        x = self.patch_embed(pv)
        pos = vision_position_ids(grid, self.merge)
        cu = vision_cu_seqlens(grid)
        cos, sin = self._rope(pos)
        for block in self.blocks:
            x = block(x, cu, cos, sin)
        x = self.post_trunk_norm(x)
        return self.adapter(x)
