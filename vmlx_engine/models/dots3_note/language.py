# SPDX-License-Identifier: Apache-2.0
"""dots3_note language model (46-layer hybrid MLA MoE + MTP layer 46).

Numerics ported from the conversion pipeline's verified core
(``jang_tools/dots3/ops.py``, cos 1.000000 vs the transformers PR reference
at S=10 and S=245) and the runtime handoff. The traps that shaped this file:

- TWO MLA geometries: full/DSA layers (128 heads, kv_lora 512, nope 128,
  θ=8e7) vs sliding layers (64 heads, kv_lora 1024, nope 192, θ=5e4,
  window 513). A single-geometry implementation produces plausible garbage.
- The low-rank rescale ``sqrt(hidden/rank)`` applies AFTER the q_a/kv_a
  RMSNorms; ``k_rope_only_layernorm`` applies BEFORE rope; rope is
  GPT-J INTERLEAVED (``mx.fast.rope(..., traditional=True)``); the rope key
  is a single MQA head broadcast across all heads.
- Norm weights have small means (0.002-0.49) but are NOT zero-centered:
  plain multiply, NO +1 shift.
- Router logits run in the activation dtype (bf16), selection adds the F32
  ``e_score_correction_bias``, returned weights come from UNBIASED scores.
- ``model.layers.46`` is the MTP layer (enorm/hnorm/eh_proj + full-geometry
  MLA without indexer + dense FFN + shared_head.norm, sharing the backbone
  lm_head, with its OWN ``model.mtp.embed_tokens``). The backbone forward
  runs layers[:46] only.

Cache contract (stage A): every backbone layer uses a plain, unbounded
KVCache holding MATERIALIZED per-head keys/values; sliding layers enforce
their window purely through the additive mask. This is exact and simple but
caps practical context (~2K) — the latent/absorbed cache replaces it behind
an equivalence A/B (task #198). DSA indexer selection is not applied: for
sequences <= index_topk (2048) top-k selects every causal position, so dense
attention is mathematically identical; longer prompts are refused loudly.
"""

import math
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.switch_layers import SwitchGLU

from .config import AttnGeom, ModelConfig, TextConfig


def _sliding_causal_mask(
    seq_len: int,
    past: int,
    window: Optional[int],
    dtype,
) -> mx.array:
    """Additive mask [S, past+S]: causal, optionally sliding.

    HF semantics: key visible iff k <= q and (q - k) < window — window 513
    means self + 512 past. Off-by-one here is invisible on short prompts.
    """
    q = (past + mx.arange(seq_len))[:, None]
    k = mx.arange(past + seq_len)[None, :]
    allowed = k <= q
    if window is not None:
        allowed = mx.logical_and(allowed, (q - k) < window)
    return mx.where(
        allowed, mx.array(0.0, dtype=dtype), mx.array(-mx.inf, dtype=dtype)
    )


class Dots3Indexer(nn.Module):
    """DSA indexer projections (full/DSA layers only).

    Weights must load so the bundle's strict weight map resolves; the
    top-2048 selection itself engages only past ``index_topk`` (task #199).
    ``k_norm`` is a LayerNorm WITH bias run in fp32 — the checkpoint ships a
    ``.bias`` tensor; RMSNorm here is wrong.
    """

    def __init__(self, config: TextConfig):
        super().__init__()
        self.wq_b = nn.Linear(
            config.q_lora_rank,
            config.index_n_heads * config.index_head_dim,
            bias=False,
        )
        self.wk = nn.Linear(config.hidden_size, config.index_head_dim, bias=False)
        self.weights_proj = nn.Linear(
            config.hidden_size, config.index_n_heads, bias=False
        )
        self.k_norm = nn.LayerNorm(config.index_head_dim, bias=True)


class Dots3MLAAttention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        g = config.geom(layer_idx)
        self.geom = g
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.apply_rescale = config.apply_mla_qkv_lora_rescale

        self.q_a_proj = nn.Linear(config.hidden_size, g.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.RMSNorm(g.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            g.q_lora_rank, g.num_heads * g.qk_head_dim, bias=False
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size, g.kv_lora_rank + g.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = nn.RMSNorm(g.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            g.kv_lora_rank,
            g.num_heads * (g.qk_nope_head_dim + g.v_head_dim),
            bias=False,
        )
        self.k_rope_only_layernorm = nn.RMSNorm(
            g.qk_rope_head_dim, eps=config.rms_norm_eps
        )
        self.o_proj = nn.Linear(
            g.num_heads * g.v_head_dim, config.hidden_size, bias=False
        )
        self.g_proj = nn.Linear(config.hidden_size, g.num_heads, bias=False)
        if not config.is_sliding(layer_idx) and layer_idx < config.num_hidden_layers:
            self.indexer = Dots3Indexer(config)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        g = self.geom
        B, S, H = x.shape
        past = int(cache.offset) if cache is not None else 0

        q_lora = self.q_a_layernorm(self.q_a_proj(x))
        if self.apply_rescale:
            q_lora = q_lora * math.sqrt(H / g.q_lora_rank)
        q = self.q_b_proj(q_lora)
        q = q.reshape(B, S, g.num_heads, g.qk_head_dim).transpose(0, 2, 1, 3)
        q_nope = q[..., : g.qk_nope_head_dim]
        q_pe = q[..., g.qk_nope_head_dim :]

        latent = self.kv_a_proj_with_mqa(x)
        kv_a = self.kv_a_layernorm(latent[..., : g.kv_lora_rank])
        if self.apply_rescale:
            kv_a = kv_a * math.sqrt(H / g.kv_lora_rank)
        kv = self.kv_b_proj(kv_a)
        kv = kv.reshape(
            B, S, g.num_heads, g.qk_nope_head_dim + g.v_head_dim
        ).transpose(0, 2, 1, 3)
        k_nope = kv[..., : g.qk_nope_head_dim]
        values = kv[..., g.qk_nope_head_dim :]

        k_pe = latent[..., g.kv_lora_rank :].reshape(B, 1, S, g.qk_rope_head_dim)
        k_pe = self.k_rope_only_layernorm(k_pe)

        q_pe = mx.fast.rope(
            q_pe,
            g.qk_rope_head_dim,
            traditional=True,
            base=g.rope_theta,
            scale=1.0,
            offset=past,
        )
        k_pe = mx.fast.rope(
            k_pe,
            g.qk_rope_head_dim,
            traditional=True,
            base=g.rope_theta,
            scale=1.0,
            offset=past,
        )

        queries = mx.concatenate([q_nope, q_pe], axis=-1)
        keys = mx.concatenate(
            [
                k_nope,
                mx.broadcast_to(k_pe, (B, g.num_heads, S, g.qk_rope_head_dim)),
            ],
            axis=-1,
        )

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        if mask is None:
            eff_past = keys.shape[2] - S
            mask = _sliding_causal_mask(
                S, eff_past, g.sliding_window, mx.float32
            )

        out = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=g.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3)  # [B, S, heads, v]

        gate = mx.sigmoid(self.g_proj(x))
        out = out * gate[..., None]  # headwise

        out = out.reshape(B, S, g.num_heads * g.v_head_dim)
        return self.o_proj(out)


class Dots3TopkRouter(nn.Module):
    """noaux_tc router with n_group=1 => plain top-k.

    Selection adds ``e_score_correction_bias`` (kept F32); the returned
    weights come from the UNBIASED sigmoid scores, renormalized. Mixing the
    two up is a quiet quality regression, not a crash. ``weight`` stays an
    unquantized bf16 [E, H] (2.5 MB/layer).
    """

    def __init__(self, config: TextConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.weight = mx.zeros((config.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((config.n_routed_experts,))

    def __call__(self, x2d: mx.array) -> tuple:
        # Router logits in the ACTIVATION dtype (bf16 deploy) — matching the
        # reference's model-dtype F.linear; fp32 here shifts top-8 on ties.
        logits = x2d @ self.weight.astype(x2d.dtype).T
        scores = mx.sigmoid(logits).astype(mx.float32)
        choice = scores + self.e_score_correction_bias.astype(mx.float32)[None, :]
        inds = mx.argpartition(-choice, kth=self.top_k - 1, axis=-1)[
            ..., : self.top_k
        ]
        w = mx.take_along_axis(scores, inds, axis=-1)
        if self.norm_topk_prob:
            w = w / (w.sum(axis=-1, keepdims=True) + 1e-20)
        return inds.astype(mx.uint32), w * self.routed_scaling_factor


class Dots3DenseMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Dots3MoE(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.gate = Dots3TopkRouter(config)
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
        )
        self.shared_experts = Dots3DenseMLP(
            config.hidden_size,
            config.moe_intermediate_size * config.n_shared_experts,
        )

    def __call__(self, x: mx.array) -> mx.array:
        inds, weights = self.gate(x.reshape(-1, x.shape[-1]))
        inds = inds.reshape(x.shape[0], x.shape[1], -1)
        weights = weights.reshape(x.shape[0], x.shape[1], -1)
        routed = self.switch_mlp(x, inds)
        out = (routed * weights[..., None].astype(routed.dtype)).sum(axis=-2)
        return out + self.shared_experts(x)


class Dots3DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attn = Dots3MLAAttention(config, layer_idx)
        if config.is_moe(layer_idx):
            self.mlp = Dots3MoE(config)
        else:
            self.mlp = Dots3DenseMLP(config.hidden_size, config.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class Dots3MTPLayer(nn.Module):
    """MTP decoder = ``model.layers.46``.

    Structure only in stage A so the strict weight map loads; the
    speculative decode loop wires it up in task #200. Full-attention MLA
    geometry, NO indexer, dense FFN, ``shared_head.norm`` before the SHARED
    backbone lm_head. Fusion inputs: eh_proj(cat(enorm(embed(next_tok)),
    hnorm(prev_hidden))).
    """

    def __init__(self, config: TextConfig):
        super().__init__()
        self.enorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(
            2 * config.hidden_size, config.hidden_size, bias=False
        )
        self.input_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # Full geometry, indexer-free: build with an index past the backbone
        # so config.geom() resolves full and is_sliding() is False, while the
        # Dots3MLAAttention constructor's indexer condition (idx <
        # num_hidden_layers) skips the indexer.
        self.self_attn = Dots3MLAAttention(config, config.num_hidden_layers)
        self.mlp = Dots3DenseMLP(config.hidden_size, config.intermediate_size)

        class _SharedHead(nn.Module):
            def __init__(self, hidden_size: int, eps: float):
                super().__init__()
                self.norm = nn.RMSNorm(hidden_size, eps=eps)

        self.shared_head = _SharedHead(config.hidden_size, config.rms_norm_eps)

    def __call__(
        self,
        embed_next: mx.array,
        prev_hidden: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        x = self.eh_proj(
            mx.concatenate(
                [self.enorm(embed_next), self.hnorm(prev_hidden)], axis=-1
            )
        )
        x = x + self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return self.shared_head.norm(x)


class Dots3MTPEmbeddings(nn.Module):
    """``model.mtp.embed_tokens`` — the MTP layer's OWN embedding table.

    Verified NOT byte-identical to the backbone embedding; aliasing them is
    a silent quality bug.
    """

    def __init__(self, config: TextConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)


class Dots3NoteModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Dots3DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ] + [Dots3MTPLayer(config)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mtp = Dots3MTPEmbeddings(config)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds

        n_backbone = self.config.num_hidden_layers
        past = 0
        if cache is not None and cache[0] is not None:
            past = int(cache[0].offset)
        seq_len = h.shape[1]
        if past + seq_len > self.config.index_topk:
            raise ValueError(
                f"dots3_note context {past + seq_len} exceeds the DSA "
                f"dense-equivalence bound ({self.config.index_topk}); the "
                "sparse indexer path is not wired yet"
            )

        if cache is None:
            cache = [None] * n_backbone

        full_mask = None
        swa_mask = None
        if seq_len > 1 or mask is not None:
            if mask is None:
                full_mask = _sliding_causal_mask(seq_len, past, None, mx.float32)
                swa_mask = _sliding_causal_mask(
                    seq_len, past, self.config.sliding_window_size, mx.float32
                )
            else:
                full_mask = mask
                swa_mask = mask
        for i in range(n_backbone):
            layer_mask = (
                swa_mask if self.config.is_sliding(i) else full_mask
            )
            h = self.layers[i](h, mask=layer_mask, cache=cache[i])
        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        text = config if isinstance(config, TextConfig) else config.text_config
        self.text_config = text
        self.model_type = text.model_type
        self.model = Dots3NoteModel(text)
        self.lm_head = nn.Linear(text.hidden_size, text.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
        **kwargs,
    ) -> mx.array:
        h = self.model(inputs, inputs_embeds=inputs_embeds, mask=mask, cache=cache)
        logits = self.lm_head(h)
        if kwargs.get("return_hidden"):
            return logits, h
        return logits

    def make_cache(self):
        """One plain KVCache per BACKBONE layer (stage A: materialized).

        Sliding layers enforce their window through the additive mask, so a
        plain cache is exact. The MTP layer (index 46) is NOT part of this
        list — the speculative path owns its private cache.
        """
        from mlx_lm.models.cache import KVCache

        return [KVCache() for _ in range(self.text_config.num_hidden_layers)]

    @property
    def layers(self):
        return self.model.layers[: self.text_config.num_hidden_layers]
