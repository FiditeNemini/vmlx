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


def _resolve_qk_scale(config) -> float:
    """Attention scale for Muse.

    VMLX_MUSE_QK_SCALE_MODE selects the interpretation while the correct one is
    being established against real output:
      inv_sqrt_head (default) -- conventional 1/sqrt(head_dim)
      pre_attn_scalar         -- qk_scale_factor**-0.5, the Gemma convention
      raw                     -- qk_scale_factor used directly (produces
                                 degenerate repetition; kept only for A/B)
    """
    import os

    head_dim = int(config.head_dim)
    factor = float(getattr(config, "qk_scale_factor", 0.0) or 0.0)
    mode = os.environ.get("VMLX_MUSE_QK_SCALE_MODE", "inv_sqrt_head").strip().lower()
    if mode == "raw" and factor > 0:
        return factor
    if mode == "pre_attn_scalar" and factor > 0:
        return factor ** -0.5
    return head_dim ** -0.5


class MuseAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        # qk_scale_factor is NOT the attention scale itself. Using 3.87
        # directly (vs 1/sqrt(128) = 0.0884, a 44x difference) saturates
        # softmax and the model emits degenerate repetition -- observed live as
        # "the the the the S D D D D...". Treat it the way Gemma treats
        # query_pre_attn_scalar: an inverse-sqrt pre-attention scalar.
        self.scale = _resolve_qk_scale(config)
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

    def make_cache(self):
        """One cache per text layer, matching each layer's attention type.

        Without this the loader falls back to a plain KVCache for all 52
        layers. That is not merely wasteful — the 39 sliding layers would grow
        unbounded instead of ringing at sliding_window — it also makes the
        mixed-SWA store reject the result, so the family silently gets NO
        prefix caching at all (observed live: cached=0 on every repeat,
        "clean mixed_swa_kv_v1 prompt prefill unavailable").
        """
        from mlx_lm.models.cache import KVCache, RotatingKVCache

        window = int(getattr(self.config, "sliding_window", 0) or 0)
        caches = []
        for index in range(self.config.num_hidden_layers):
            if window > 0 and self.config.layer_is_sliding(index):
                caches.append(RotatingKVCache(max_size=window, keep=0))
            else:
                caches.append(KVCache())
        return caches

    @property
    def layers(self):
        return self.model.layers
