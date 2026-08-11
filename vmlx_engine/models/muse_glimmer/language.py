# SPDX-License-Identifier: Apache-2.0
"""Muse Glimmer text backbone.

Layer shape, verified against the checkpoint's tensor names and shapes:

    input_layernorm -> self_attn -> post_attention_layernorm  (+residual)
    pre_feedforward_layernorm -> mlp -> post_feedforward_layernorm (+residual)

That four-norm sandwich is Gemma's. Everything else is not:

* ``self_attn.gate_proj`` (heads*head_dim, hidden) gates the attention output
  elementwise before ``o_proj``. Gemma has no such projection.
* Q and K pass through a *weightless* RMSNorm over ``head_dim`` before attention,
  and ``qk_scale_factor`` (3.87) then multiplies Q ONLY. The checkpoint carries no
  ``q_norm``/``k_norm`` tensors because that norm has no learned gain -- the
  absent tensor does not mean the absent operation. SDPA still runs at the
  conventional ``head_dim ** -0.5``.
* Every RMSNorm gain in the text tower is stored ZERO-CENTERED: the mathematical
  weight is ``1 + w``. The ``+1`` is folded in once at load (``Model.sanitize``).
* The embedding lookup is itself RMSNormed (weightless). This stands in place of
  Gemma's ``sqrt(hidden_size)`` scale -- Muse has no such scale.
* Full-attention layers declare ``layer_rope_theta = 0`` and take NO rotary
  embedding; only the sliding layers are position-encoded.
* Logits are multiplied by ``output_multiplier`` and then tanh-softcapped at 20.0.

Ground truth for every one of the above is the working Swift port,
``vmlx-swift/Libraries/MLXLLM/Models/MuseGlimmerText.swift``.
"""

from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask

# One unit-gain vector per (dim, dtype). Q/K norm runs 2x per layer across 52
# layers on every token; rebuilding the ones tensor there would add ~104
# allocations per decode step for a value that never changes.
_UNIT_GAIN: Dict[tuple, mx.array] = {}


def _scaleless_rms_norm(x: mx.array, eps: float) -> mx.array:
    """RMSNorm with no learned gain, over the last axis."""
    dim = x.shape[-1]
    key = (dim, x.dtype)
    gain = _UNIT_GAIN.get(key)
    if gain is None:
        gain = mx.ones((dim,), dtype=x.dtype)
        _UNIT_GAIN[key] = gain
    return mx.fast.rms_norm(x, gain, eps)


class MuseAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        # The conventional scale goes to SDPA; qk_scale_factor rides on top of
        # it, on Q only, after the weightless QK-norm. No single scalar can
        # reproduce normalize-then-scale, which is why substituting 3.87,
        # 3.87**-0.5 and 1/sqrt(128) into SDPA all produced identical garbage.
        self.scale = float(config.head_dim) ** -0.5
        self.qk_scale_factor = float(getattr(config, "qk_scale_factor", 1.0) or 1.0)
        self.rms_norm_eps = float(config.rms_norm_eps)
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

        # Weightless per-head QK-norm, then the asymmetric factor on Q alone.
        # Both must precede RoPE.
        queries = _scaleless_rms_norm(queries, self.rms_norm_eps) * self.qk_scale_factor
        keys = _scaleless_rms_norm(keys, self.rms_norm_eps)

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

    def embed(self, inputs: mx.array) -> mx.array:
        """Normed embedding: a weightless RMSNorm over the raw table lookup.

        Muse has no ``sqrt(hidden_size)`` embedding scale -- this norm is what
        stands in its place, and it is deliberately kept outside the weight
        matrix so it can never be folded in. Feeding layer 0 the raw lookup
        leaves the residual stream at the wrong magnitude and the model emits
        fluent-looking token soup.
        """
        return _scaleless_rms_norm(
            self.embed_tokens(inputs), float(self.config.rms_norm_eps)
        )

    def _make_masks(self, h: mx.array, cache: List[Any]) -> List[Any]:
        """One mask per attention kind, mapped onto the layers that use it.

        Sliding and full layers cannot share a mask: reusing the full-attention
        mask on the 39 sliding layers lets them attend past the 2048-token
        window, which only diverges once the context outgrows that window.
        """
        window = int(getattr(self.config, "sliding_window", 0) or 0)
        built: Dict[bool, Any] = {}
        masks = []
        for index, layer_cache in zip(range(len(self.layers)), cache):
            sliding = window > 0 and self.config.layer_is_sliding(index)
            if sliding not in built:
                built[sliding] = create_attention_mask(
                    h, layer_cache, window_size=window if sliding else None
                )
            masks.append(built[sliding])
        return masks

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        # Callers that pass inputs_embeds have already applied the embed norm
        # (see Model.get_input_embeddings); media features scattered into that
        # stream stay raw and must not be normed a second time.
        h = self.embed(inputs) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * len(self.layers)
        # A caller-supplied mask is only usable if it already distinguishes the
        # two attention kinds. A single array applied uniformly re-creates the
        # bug the per-type masks exist to fix: the 39 sliding layers would see
        # past their 2048-token window. Accept a per-layer sequence or a
        # {layer_type: mask} mapping; otherwise build our own.
        layer_masks = self._resolve_layer_masks(h, cache, mask)
        for index, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            h = layer(h, mask=layer_masks[index], cache=layer_cache)
        return self.norm(h)

    def _resolve_layer_masks(self, h, cache, mask):
        count = len(self.layers)
        if mask is None:
            return self._make_masks(h, cache)
        if isinstance(mask, dict):
            return [
                mask.get(
                    "sliding_attention"
                    if self.config.layer_is_sliding(i)
                    else "full_attention"
                )
                for i in range(count)
            ]
        if isinstance(mask, (list, tuple)) and len(mask) == count:
            return list(mask)
        # A bare mask cannot be correct for both kinds. Honour it only where it
        # is safe (the full-attention layers) and build the windowed mask for
        # the sliding ones rather than silently widening their attention.
        built = self._make_masks(h, cache)
        return [
            built[i] if self.config.layer_is_sliding(i) else mask
            for i in range(count)
        ]


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
        # Multiplier lands on the logits, not the hidden state. The scalar
        # commutes through a bias-free lm_head, but the head is quantized here
        # so scaling after it keeps the reference's rounding.
        logits = self.lm_head(h) * self.config.output_multiplier
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
                # step sizes the block RotatingKVCache allocates at a time, and
                # the update writes the whole incoming chunk into it. A prefill
                # chunk larger than step runs off the end of the freshly
                # allocated region and trips a scatter precondition; sizing step
                # to the window means any chunk the prefill loop can produce
                # always fits. It is a class attribute here, not a ctor kwarg.
                slot = RotatingKVCache(max_size=window, keep=0)
                slot.step = window
                caches.append(slot)
            else:
                caches.append(KVCache())
        return caches

    @property
    def layers(self):
        return self.model.layers
