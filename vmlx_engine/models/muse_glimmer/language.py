# SPDX-License-Identifier: Apache-2.0
"""Muse Glimmer text backbone.

Layer shape, verified against the checkpoint's tensor names and shapes:

    input_layernorm -> self_attn -> post_attention_layernorm  (+residual)
    pre_feedforward_layernorm -> mlp -> post_feedforward_layernorm (+residual)

That four-norm sandwich is Gemma's. Everything else is not:

* ``self_attn.gate_proj`` (heads*head_dim, hidden) gates the attention output
  elementwise before ``o_proj``. Gemma has no such projection.
* There is no ``q_norm``/``k_norm`` anywhere in the checkpoint, unlike Gemma 3/4.
* Attention logits use the bundle's ``qk_scale_factor`` (3.87), NOT
  ``1/sqrt(head_dim)`` (0.0884 at head_dim 128).
* Full-attention layers declare ``layer_rope_theta = 0`` and take NO rotary
  embedding; only the sliding layers are position-encoded.
* The hidden state is multiplied by ``output_multiplier`` before the LM head,
  and the logits are then tanh-softcapped at 20.0.
"""

from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn


class MuseAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        # Declared multiplier, not the conventional inverse-sqrt.
        self.scale = float(config.qk_scale_factor)
        self.is_sliding = config.layer_is_sliding(layer_idx)
        self.sliding_window = int(config.sliding_window or 0)

        dim = config.hidden_size
        q_out = self.n_heads * self.head_dim
        kv_out = self.n_kv_heads * self.head_dim
        bias = bool(config.attention_bias)

        self.q_proj = nn.Linear(dim, q_out, bias=bias)
        self.k_proj = nn.Linear(dim, kv_out, bias=bias)
        self.v_proj = nn.Linear(dim, kv_out, bias=bias)
        # Gates the attention output; same width as the concatenated heads.
        self.gate_proj = nn.Linear(dim, q_out, bias=bias)
        self.o_proj = nn.Linear(q_out, dim, bias=bias)

        self.uses_rope = config.layer_uses_rope(layer_idx)
        self.rope = (
            nn.RoPE(self.head_dim, traditional=False, base=config.layer_rope_base(layer_idx))
            if self.uses_rope
            else None
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if self.rope is not None:
            offset = cache.offset if cache is not None else 0
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        out = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)

        # Elementwise gate on the attention output, before the output
        # projection. Sigmoid gating matches the checkpoint's single
        # (heads*head_dim, hidden) projection with no extra activation weights.
        out = out * mx.sigmoid(self.gate_proj(x))
        return self.o_proj(out)


class MuseMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim, hidden = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MuseDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = MuseAttention(config, layer_idx)
        self.mlp = MuseMLP(config)
        eps, post_eps = config.rms_norm_eps, config.post_norm_eps
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=post_eps)
        self.pre_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.post_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=post_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.post_attention_layernorm(
            self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        )
        x = x + h
        h = self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))
        return x + h


class MuseTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            MuseDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, layer_cache in zip(self.layers, cache):
            h = layer(h, mask=mask, cache=layer_cache)
        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = MuseTextModel(config)
        # tie_word_embeddings is False in every shipped bundle and the
        # checkpoint carries its own lm_head tensor.
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        h = self.model(inputs, inputs_embeds=inputs_embeds, mask=mask, cache=cache)
        h = h * self.config.output_multiplier
        logits = self.lm_head(h)
        cap = float(self.config.final_logit_softcapping or 0.0)
        if cap > 0:
            logits = mx.tanh(logits / cap) * cap
        return logits

    @property
    def layers(self):
        return self.model.layers
