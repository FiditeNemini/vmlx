# SPDX-License-Identifier: Apache-2.0
"""Naive-N0.5-Flash (model_type naive_n05_flash) — mlx-lm runtime module, vendored by vMLX (INTERNAL draft).

Math mirrors NaiveAI's reference modeling_naive_n05_flash.py @0235b3b (see jangh-n05/n05/model.py, parity-tested):
  * hybrid_layer_pattern 1 = SWA (window 128, sink bias, 8 kv heads, theta 1e4) -> RotatingKVCache(128)
  * 0 = DSA (4 kv heads, theta 1e7, no sink) + lightweight indexer -> CacheList(KVCache kv, KVCache indexer-keys)
      Lk <= index_top_k: plain causal attention (exact). Lk > index_top_k:
        prefill: boolean top-k mask (per query) & causal
        decode : gather the top-k keys/values, dense SDPA over them
      indexer: q = wq(x) 16x128, k = LayerNorm(wk(x)), same partial NeoX rotary, per-row FP8-e4m3 round trip,
               score = sum_h relu(q_h.k) * weights_proj(x)_h * 16^-0.5
  * values * attention_value_scale (0.707) before attention; scale head_dim^-0.5
  * MoE: DeepSeek-V3 sigmoid router (fp32, e_score_correction_bias), top-8 normalized, 256 experts, no shared expert.
JANGH (on-disk "jangtq" v2) bundles: when config carries jangtq.version == 2, routed experts are built as
TQSwitchGLU from the per-module entries (fail closed on any missing/inconsistent entry) BEFORE mlx-lm quantizes
the remaining modules; the fused routed path (switch_mlp.routed) is used for the MoE.
Created by Jinho Jang (eric@jangq.ai).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from vmlx_engine.utils.naive_prefill_policy import naive_use_padded_prefill

from .base import BaseModelArgs, create_attention_mask
from .cache import CacheList, KVCache, RotatingKVCache
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "naive_n05_flash"
    vocab_size: int = 152576
    hidden_size: int = 4096
    num_hidden_layers: int = 48
    intermediate_size: int = 16384
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 256
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = True
    routed_scaling_factor: Optional[float] = 1.0
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    head_dim: int = 192
    v_head_dim: int = 128
    rope_theta: float = 1e7
    swa_num_attention_heads: int = 64
    swa_num_key_value_heads: int = 8
    swa_head_dim: int = 192
    swa_v_head_dim: int = 128
    swa_rope_theta: float = 1e4
    partial_rotary_factor: float = 0.334
    sliding_window: int = 128
    attention_value_scale: Optional[float] = 0.707
    add_swa_attention_sink_bias: bool = True
    add_full_attention_sink_bias: bool = False
    layernorm_epsilon: float = 1e-5
    index_top_k: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 16
    indexer_activation_dtype: str = "fp8_e4m3"
    max_position_embeddings: int = 1048576
    tie_word_embeddings: bool = False
    hybrid_layer_pattern: List[int] = field(default_factory=list)
    moe_layer_freq: List[int] = field(default_factory=list)
    jangtq: Optional[dict] = None
    quantization: Optional[dict] = None



# ---------------------------------------------------------------------------------------------------------------
# MLX 0.32.2 BUG (found 2026-09-27): mx.fast.scaled_dot_product_attention with `sinks=` is WRONG and NON-DETERMINISTIC
# once the score matrix exceeds 8 GiB (heads * L * L_keys * 4 bytes), i.e. when MLX takes its blocked SDPA path:
# 4-9% relative error, most query rows affected, results differ call to call. Without sinks the blocked path is
# correct. Every sliding-window layer of this model uses the sink bias, so the sliding-window attention is computed
# explicitly here (reference semantics: bf16 logits, fp32 softmax with the sink as an extra column, bf16 p @ v).
SDPA_SINKS_SAFE_BYTES = 2 * 2**30


def _sink_attention(q, k, v, scale, sinks, q_pos0: int, k_pos0: int, window: int | None, blk: int = 1024):
    """q (B,H,L,D) at absolute positions q_pos0.., k/v (B,KV,Lk,.) at absolute positions k_pos0..  Causal; with
    `window` a query at position p sees keys in (p - window, p]. sinks (H,) or None. Returns (B,H,L,Dv) in v.dtype."""
    B, Hh, L, D = q.shape
    KV, Lk = k.shape[1], k.shape[2]
    g = Hh // KV
    qg = q.reshape(B, KV, g, L, D)
    outs = []
    for s in range(0, L, blk):
        e = min(L, s + blk)
        lo = 0 if window is None else max(0, (q_pos0 + s) - (window - 1) - k_pos0)
        hi = min(Lk, (q_pos0 + e - 1) - k_pos0 + 1)
        ks = k[:, :, None, lo:hi]                                             # (B,KV,1,n,D)
        lg = (qg[:, :, :, s:e] @ ks.swapaxes(-1, -2)) * scale                 # (B,KV,g,b,n) in q.dtype
        qp = mx.arange(q_pos0 + s, q_pos0 + e)[:, None]; kp = mx.arange(k_pos0 + lo, k_pos0 + hi)[None, :]
        ok = qp >= kp
        if window is not None:
            ok = ok & (qp - kp < window)
        lg = mx.where(ok, lg.astype(mx.float32), -mx.inf)
        if sinks is not None:
            sk = mx.broadcast_to(sinks.astype(mx.float32).reshape(1, KV, g, 1, 1), lg.shape[:-1] + (1,))
            p = mx.softmax(mx.concatenate([lg, sk], axis=-1), axis=-1)[..., :-1]
        else:
            p = mx.softmax(lg, axis=-1)
        o = p.astype(v.dtype) @ v[:, :, None, lo:hi]                           # (B,KV,g,b,Dv)
        outs.append(o)
    o = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=3)
    return o.reshape(B, Hh, L, -1)


def _fp8_round(t: mx.array) -> mx.array:
    t = t.astype(mx.float32)
    s = mx.maximum(mx.max(mx.abs(t), axis=-1, keepdims=True), 1e-4) / 448.0
    return mx.from_fp8(mx.to_fp8(mx.clip(t / s, -448.0, 448.0)), dtype=mx.float32) * s


class Indexer(nn.Module):
    def __init__(self, a: ModelArgs):
        super().__init__()
        self.n_heads, self.head_dim = a.index_n_heads, a.index_head_dim
        self.fp8 = a.indexer_activation_dtype == "fp8_e4m3"
        self.wq = nn.Linear(a.hidden_size, a.index_n_heads * a.index_head_dim, bias=False)
        self.wk = nn.Linear(a.hidden_size, a.index_head_dim, bias=False)
        self.k_norm = nn.LayerNorm(a.index_head_dim, eps=1e-5)
        self.weights_proj = nn.Linear(a.hidden_size, a.index_n_heads, bias=False)

    def scores(self, x, rope, offset, cache):
        B, L, _ = x.shape
        q = rope(self.wq(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3), offset=offset)
        k = rope(self.k_norm(self.wk(x))[:, None], offset=offset)             # (B,1,L,128)
        # The FP8 round trip is per key vector, so it is applied ONCE, before the key enters the cache (as the
        # reference does), not to the whole history on every decode step. Cached keys are fp32: an e4m3 value times
        # its fp32 scale is not representable in bf16.
        if self.fp8:
            q, k = _fp8_round(q), _fp8_round(k)
        else:
            q, k = q.astype(mx.float32), k.astype(mx.float32)
        if cache is not None:
            k, _ = cache.update_and_fetch(k, mx.zeros((B, 1, L, 0), dtype=k.dtype))
        w = (self.weights_proj(x) * (self.n_heads ** -0.5)).astype(mx.float32)
        s = mx.maximum(q @ k.swapaxes(-1, -2), 0.0)                             # (B,H,L,Lk)
        return mx.sum(s * w.swapaxes(-1, -2)[..., None], axis=1)                # (B,L,Lk)


class Attention(nn.Module):
    def __init__(self, a: ModelArgs, is_swa: bool):
        super().__init__()
        self.a, self.is_swa = a, is_swa
        p = "swa_" if is_swa else ""
        self.n_heads = getattr(a, p + "num_attention_heads")
        self.n_kv = getattr(a, p + "num_key_value_heads")
        hd = getattr(a, p + "head_dim"); vd = getattr(a, p + "v_head_dim")
        self.scale = hd ** -0.5
        self.q_proj = nn.Linear(a.hidden_size, self.n_heads * hd, bias=False)
        self.k_proj = nn.Linear(a.hidden_size, self.n_kv * hd, bias=False)
        self.v_proj = nn.Linear(a.hidden_size, self.n_kv * vd, bias=False)
        self.o_proj = nn.Linear(self.n_heads * vd, a.hidden_size, bias=False)
        self.rope = nn.RoPE(int(hd * a.partial_rotary_factor), traditional=False,
                            base=a.swa_rope_theta if is_swa else a.rope_theta)
        sink = a.add_swa_attention_sink_bias if is_swa else a.add_full_attention_sink_bias
        self.attention_sink_bias = mx.zeros((self.n_heads,)) if sink else None
        self.indexer = None if is_swa else Indexer(a)

    def _full_sdpa(self, q, k, v, mask, sinks):
        # MLX's full fused kernel requires equal Q/V head widths. Native
        # 192/128 attention otherwise materializes a history-sized score
        # tensor. Zero-padding V preserves QK, scale, mask and softmax;
        # discard only the added output coordinates. The fused reduction is
        # numerically different, so the policy is part of the cache identity.
        # Default "auto": stock while the score tensor fits 4 GiB, padded above
        # (stock materialized ~11 GB at 22.5k history and the admission guard
        # refused every prompt > ~20k). utils/naive_prefill_policy.py.
        if (
            q.shape[2] > 8
            and naive_use_padded_prefill(q.shape[1], q.shape[2], k.shape[2])
            and q.shape[-1] == k.shape[-1] == 192
            and v.shape[-1] == 128
            and q.dtype in (mx.bfloat16, mx.float16)
            and sinks is None
        ):
            padded_v = mx.pad(v, [(0, 0), (0, 0), (0, 0), (0, 64)])
            return mx.fast.scaled_dot_product_attention(
                q, k, padded_v, scale=self.scale, mask=mask, force_fused=True
            )[..., :128]
        return mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask, sinks=sinks
        )

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        kv_cache, idx_cache = (cache[0], cache[1]) if (cache is not None and not self.is_swa) else (cache, None)
        off = kv_cache.offset if kv_cache is not None else 0
        q = self.rope(self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3), offset=off)
        k = self.rope(self.k_proj(x).reshape(B, L, self.n_kv, -1).transpose(0, 2, 1, 3), offset=off)
        v = self.v_proj(x).reshape(B, L, self.n_kv, -1).transpose(0, 2, 1, 3)
        if self.a.attention_value_scale is not None:
            v = v * self.a.attention_value_scale
        if kv_cache is not None:
            k, v = kv_cache.update_and_fetch(k, v)
        sinks = self.attention_sink_bias.astype(q.dtype) if self.attention_sink_bias is not None else None
        if self.is_swa:
            Lk = k.shape[2]
            if self.n_heads * L * Lk * 4 <= SDPA_SINKS_SAFE_BYTES:
                o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask, sinks=sinks)
            else:
                # large prefill chunk: MLX's blocked SDPA mishandles sinks (see note above). The cached keys of a
                # RotatingKVCache are in temporal order after update_and_fetch; they end at position off + L - 1.
                o = _sink_attention(q, k, v, self.scale, sinks, off, off + L - Lk, self.a.sliding_window)
        else:
            Lk = k.shape[2]
            sc = self.indexer.scores(x, self.rope, off, idx_cache)             # indexer cache always advances
            topk = self.a.index_top_k
            if Lk <= topk:
                o = self._full_sdpa(q, k, v, mask, sinks)
            elif L == 1:
                sel = mx.argpartition(-sc, kth=topk - 1, axis=-1)[..., :topk]    # (B,1,topk)
                gi = sel[:, :, :, None]                                         # (B,1,topk,1)
                ks = mx.take_along_axis(k, mx.broadcast_to(gi, (B, k.shape[1], topk, k.shape[3])), axis=2)
                vs = mx.take_along_axis(v, mx.broadcast_to(gi, (B, v.shape[1], topk, v.shape[3])), axis=2)
                o = mx.fast.scaled_dot_product_attention(q, ks, vs, scale=self.scale, sinks=sinks)
            else:
                qpos = mx.arange(off, off + L)[:, None]; kpos = mx.arange(Lk)[None, :]
                causal = qpos >= kpos
                sc = mx.where(causal[None], sc, -mx.inf)
                sel = mx.argpartition(-sc, kth=topk - 1, axis=-1)[..., :topk]
                keep = mx.put_along_axis(mx.zeros(sc.shape, dtype=mx.bool_), sel, mx.array(True), axis=-1)
                o = self._full_sdpa(q, k, v, (keep & causal[None])[:, None], sinks)
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, d, i):
        super().__init__()
        self.gate_proj = nn.Linear(d, i, bias=False)
        self.up_proj = nn.Linear(d, i, bias=False)
        self.down_proj = nn.Linear(i, d, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEGate(nn.Module):
    def __init__(self, a: ModelArgs):
        super().__init__()
        self.weight = mx.zeros((a.n_routed_experts, a.hidden_size), dtype=mx.float32)
        self.e_score_correction_bias = mx.zeros((a.n_routed_experts,), dtype=mx.float32)


class MoE(nn.Module):
    def __init__(self, a: ModelArgs):
        super().__init__()
        self.k, self.norm = a.num_experts_per_tok, a.norm_topk_prob
        self.scaling = a.routed_scaling_factor or 1.0
        self.gate = MoEGate(a)
        self.switch_mlp = SwitchGLU(a.hidden_size, a.moe_intermediate_size, a.n_routed_experts)

    def __call__(self, x):
        logits = x.astype(mx.float32) @ self.gate.weight.astype(mx.float32).T
        scores = mx.sigmoid(logits)
        choice = scores + self.gate.e_score_correction_bias.astype(mx.float32)
        idx = mx.argpartition(-choice, kth=self.k - 1, axis=-1)[..., : self.k]
        w = mx.take_along_axis(scores, idx, axis=-1)
        if self.norm:
            w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20)
        w = w * self.scaling
        if getattr(self.switch_mlp, "is_jangtq2", False):
            return self.switch_mlp.routed(x, idx, w).astype(x.dtype)
        y = self.switch_mlp(x, idx)
        return mx.sum(y * w[..., None].astype(y.dtype), axis=-2).astype(x.dtype)


class DecoderLayer(nn.Module):
    def __init__(self, a: ModelArgs, i: int):
        super().__init__()
        self.is_swa = bool(a.hybrid_layer_pattern[i])
        self.self_attn = Attention(a, self.is_swa)
        self.mlp = MoE(a) if a.moe_layer_freq[i] else MLP(a.hidden_size, a.intermediate_size)
        self.input_layernorm = nn.RMSNorm(a.hidden_size, eps=a.layernorm_epsilon)
        self.post_attention_layernorm = nn.RMSNorm(a.hidden_size, eps=a.layernorm_epsilon)

    def __call__(self, x, mask=None, cache=None):
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class NaiveModel(nn.Module):
    def __init__(self, a: ModelArgs):
        super().__init__()
        self.a = a
        self.embed_tokens = nn.Embedding(a.vocab_size, a.hidden_size)
        self.layers = [DecoderLayer(a, i) for i in range(a.num_hidden_layers)]
        self.norm = nn.RMSNorm(a.hidden_size, eps=a.layernorm_epsilon)
        self.swa_idx = a.hybrid_layer_pattern.index(1)
        self.dsa_idx = a.hybrid_layer_pattern.index(0)

    def __call__(self, x, cache=None, input_embeddings=None):
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(x)
        if cache is None:
            cache = [None] * len(self.layers)
        dsa_c = cache[self.dsa_idx][0] if cache[self.dsa_idx] is not None else None
        full_mask = create_attention_mask(h, dsa_c)
        swa_mask = create_attention_mask(h, cache[self.swa_idx], window_size=self.a.sliding_window)
        for layer, c in zip(self.layers, cache):
            h = layer(h, swa_mask if layer.is_swa else full_mask, c)
        return self.norm(h)


def _install_jangh_experts(model: "Model", a: ModelArgs) -> int:
    """Use the shared validated JANGH contract and preserve existing kernels."""
    from vmlx_engine.jangh.install import install_jangh

    return install_jangh(model, {"jangtq": a.jangtq, "quantization": a.quantization})


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = NaiveModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        jt = args.jangtq or {}
        if jt:
            if int(jt.get("version", 0)) != 2 or jt.get("codebook_family") != "odd-cubic":
                raise ValueError(f"JANGH: unsupported jangtq block {jt}")
            self.jangh_modules = _install_jangh_experts(self, args)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        return self.lm_head(self.model(inputs, cache, input_embeddings))

    def sanitize(self, weights):
        n = self.args.n_routed_experts
        for i in range(self.args.num_hidden_layers):
            pre = f"model.layers.{i}.mlp"
            for p in ("gate_proj", "up_proj", "down_proj"):
                if f"{pre}.experts.0.{p}.weight" in weights:
                    weights[f"{pre}.switch_mlp.{p}.weight"] = mx.stack(
                        [weights.pop(f"{pre}.experts.{e}.{p}.weight") for e in range(n)])
        return weights

    @property
    def layers(self):
        return self.model.layers

    @property
    def cast_predicate(self):
        return lambda k: "e_score_correction_bias" not in k and not k.endswith("mlp.gate.weight")

    @property
    def cache_list_head_counts(self):
        # DSA's indexer stores one shared key head, not index_n_heads queries.
        return tuple(None if layer.is_swa else (layer.self_attn.n_kv, 1)
                     for layer in self.layers)

    def make_cache(self):
        return [RotatingKVCache(max_size=self.args.sliding_window) if l.is_swa else CacheList(KVCache(), KVCache())
                for l in self.layers]
