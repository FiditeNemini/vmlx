from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.glm_ocr.vision import (
    GlmOcrVisionAttention,
    GlmOcrVisionPatchEmbed,
    GlmOcrVisionRotaryEmbedding,
    check_array_shape,
)

from .config import VisionConfig


def _clamped_swiglu(gate, up, limit: float):
    gate = mx.minimum(gate, limit)
    up = mx.clip(up, -limit, limit)
    return nn.silu(gate) * up


class Glm5NextVisionMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.limit = config.swiglu_limit
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.attention_bias
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.attention_bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.attention_bias
        )

    def __call__(self, x):
        return self.down_proj(
            _clamped_swiglu(self.gate_proj(x), self.up_proj(x), self.limit)
        )


class Glm5NextVisionPatchMerger(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        dim = config.out_hidden_size
        context_dim = config.projection_intermediate_size
        self.limit = config.swiglu_limit
        self.proj = nn.Linear(dim, dim, bias=False)
        self.post_projection_norm = nn.LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, context_dim, bias=False)
        self.up_proj = nn.Linear(dim, context_dim, bias=False)
        self.down_proj = nn.Linear(context_dim, dim, bias=False)

    def __call__(self, x):
        x = nn.gelu(self.post_projection_norm(self.proj(x)))
        return self.down_proj(
            _clamped_swiglu(self.gate_proj(x), self.up_proj(x), self.limit)
        )


class Glm5NextVisionBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = GlmOcrVisionAttention(config)
        self.mlp = Glm5NextVisionMLP(config)

    def __call__(self, hidden_states, cu_seqlens, position_embeddings):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.patch_embed = GlmOcrVisionPatchEmbed(config)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = GlmOcrVisionRotaryEmbedding(head_dim // 2)
        self.blocks = [Glm5NextVisionBlock(config) for _ in range(config.depth)]
        self.merger = Glm5NextVisionPatchMerger(config)
        self.downsample = nn.Conv2d(
            in_channels=config.hidden_size,
            out_channels=config.out_hidden_size,
            kernel_size=config.spatial_merge_size,
            stride=config.spatial_merge_size,
            bias=True,
        )
        self.post_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def _position_embeddings(self, grid_thw):
        pos_ids = []
        merge = self.spatial_merge_size
        for t, h, w in grid_thw.tolist():
            h_ids = mx.repeat(mx.arange(h)[:, None], w, axis=1)
            w_ids = mx.repeat(mx.arange(w)[None, :], h, axis=0)
            h_ids = h_ids.reshape(h // merge, merge, w // merge, merge)
            w_ids = w_ids.reshape(h // merge, merge, w // merge, merge)
            h_ids = h_ids.transpose(0, 2, 1, 3).flatten()
            w_ids = w_ids.transpose(0, 2, 1, 3).flatten()
            pos_ids.append(mx.tile(mx.stack([h_ids, w_ids], axis=-1), (t, 1)))
        pos_ids = mx.concatenate(pos_ids, axis=0)
        rotary = self.rotary_pos_emb(int(mx.max(grid_thw[:, 1:]).item()))
        rotary = rotary[pos_ids].reshape(pos_ids.shape[0], -1)
        emb = mx.concatenate([rotary, rotary], axis=-1)
        return mx.cos(emb), mx.sin(emb)

    def __call__(
        self,
        hidden_states,
        grid_thw,
        output_hidden_states: bool | None = None,
    ):
        del output_hidden_states
        hidden_states = self.patch_embed(hidden_states)
        position_embeddings = self._position_embeddings(grid_thw)
        repeated_lengths = []
        for t, h, w in grid_thw.tolist():
            repeated_lengths.extend([h * w] * t)
        cu_seqlens = mx.pad(mx.cumsum(mx.array(repeated_lengths)), (1, 0))
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = hidden_states.reshape(
            -1,
            self.spatial_merge_size,
            self.spatial_merge_size,
            hidden_states.shape[-1],
        )
        hidden_states = self.downsample(hidden_states).reshape(
            -1, self.config.out_hidden_size
        )
        return self.merger(hidden_states)

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if "position_ids" in key:
                continue
            if key.endswith(
                ("patch_embed.proj.weight", "downsample.weight")
            ) and not check_array_shape(value):
                if value.ndim == 5:
                    value = value.transpose(0, 2, 3, 4, 1)
                elif value.ndim == 4:
                    value = value.transpose(0, 2, 3, 1)
            sanitized[key] = value
        return sanitized
