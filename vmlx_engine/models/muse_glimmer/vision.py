# SPDX-License-Identifier: Apache-2.0
"""Muse Glimmer vision tower and adapter.

Read off the checkpoint, and deliberately NOT modelled on the in-tree Qwen-VL
tower, which differs in every one of these respects:

* ``norm1``/``norm2`` and ``ln_pre``/``ln_post`` all carry a bias, so they are
  LayerNorm, not RMSNorm.
* Position information comes from a learned ``position_embedding_table`` of
  ``pos_emb_height * pos_emb_width`` (32*32 = 1024) rows, matching the
  (1024, 1536) tensor exactly — not 2D RoPE.
* Attention keeps separate ``q_proj``/``k_proj``/``v_proj`` (each with bias) and
  names its output ``proj``, not a fused qkv with ``o_proj``.
* ``patch_embedding`` takes 1176 inputs = 2*3*14*14, i.e. a temporal patch of 2
  frames — this is a video encoder, not a still-image one.

The adapter merges a 2x2 spatial neighbourhood (merge_size 2) before projecting
into the text hidden size, which is why ``fc1`` accepts 1536*4 = 6144.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


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

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)

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

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        x = x + self.attn(self.norm1(x), mask=mask)
        return x + self.mlp(self.norm2(x))


class MusePatchEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        patch_dim = (
            int(config.patch_temporal) * 3 * int(config.patch_size) * int(config.patch_size)
        )
        self.patch_embedding = nn.Linear(patch_dim, config.hidden_size, bias=False)
        self.position_embedding_table = nn.Embedding(
            config.position_table_size, config.hidden_size
        )

    def __call__(self, patches: mx.array, position_ids: Optional[mx.array] = None) -> mx.array:
        h = self.patch_embedding(patches)
        if position_ids is None:
            L = h.shape[1]
            position_ids = mx.arange(L)[None, :]
        return h + self.position_embedding_table(position_ids)


class VisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_type = config.model_type
        self.config = config
        self.patch_embedder = MusePatchEmbedder(config)
        self.ln_pre = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = [MuseVisionLayer(config) for _ in range(config.num_hidden_layers)]
        self.ln_post = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(
        self,
        patches: mx.array,
        position_ids: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.ln_pre(self.patch_embedder(patches, position_ids))
        for layer in self.layers:
            h = layer(h, mask=mask)
        return self.ln_post(h)


class MuseVisionAdapter(nn.Module):
    """Merge a merge_size x merge_size patch neighbourhood, then project.

    ``fc1`` accepting vision_hidden * merge_size**2 is what pins the merge
    factor: 1536 * 4 = 6144 for the shipped bundles.
    """

    def __init__(self, model_config):
        super().__init__()
        vision = model_config.vision_config
        merged = vision.hidden_size * (int(vision.merge_size) ** 2)
        hidden = int(model_config.projector_hidden_size)
        self.merge_size = int(vision.merge_size)
        self.fc1 = nn.Linear(merged, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, hidden, bias=False)

    def __call__(self, features: mx.array) -> mx.array:
        B, L, D = features.shape
        group = self.merge_size**2
        if group > 1:
            usable = (L // group) * group
            features = features[:, :usable, :].reshape(B, usable // group, group * D)
        return self.fc2(nn.gelu(self.fc1(features)))
