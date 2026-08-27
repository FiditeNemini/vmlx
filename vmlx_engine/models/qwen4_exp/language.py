"""Qwen4-Exp (Qwen3.8-Flash-Next) — MLX runtime.

Mirrors transformers-main modular_qwen4_exp.py. Reuses mlx_lm's battle-tested
GDN kernel, SwitchGLU experts and caches; adds:
  - mHC hyper-connections (GatedResidual, 4x2560 residual stream)
  - PLE n-gram embedding injection (layer id 2, exact int64 hashing)
  - QSA (DSA-style block-sparse indexer on full-attention layers)
  - sigmoid output gate on the GDN gated norm

Norm conventions (VERIFIED against HF source, per component):
  - Qwen4ExpTextRMSNorm family (hc_norm, ple norms, q/k norms, indexer
    layernorms): checkpoint stores weight-1 → sanitize adds +1, module is
    plain RMSNorm.
  - GDN gated norm (Qwen3NextRMSNormGated lineage): plain weight, NO shift.

The language core is shared by the text, image, and video lanes.
"""

from dataclasses import dataclass, field, replace
import logging
import os
import time
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.gated_delta import gated_delta_update
from mlx_lm.models.qwen3_5 import GatedDeltaNet as _Qwen35GatedDeltaNet
from mlx_lm.models.qwen3_5 import TextModelArgs as _Qwen35TextArgs
from mlx_lm.models.switch_layers import SwitchGLU
from mlx_vlm.models.base import LanguageModelOutput
from mlx_vlm.models.qwen3_5.language import (
    LanguageModel as _Qwen35VlmLanguageModel,
)
from mlx_vlm.models.qwen3_5.language import (
    Qwen3_5RotaryEmbedding,
    apply_multimodal_rotary_pos_emb,
)

from vmlx_engine.models.minimax_m3.cache import (
    MiniMaxM3SparseCache as _SparseIndexerKVCache,
)
from vmlx_engine.metal.qwen4_affine_moe_decode import qwen4_affine_switchglu

from .ngram import NGramHasher


logger = logging.getLogger(__name__)

_HYPER_SPLIT_INDICES = {}


def _layer_profile_enabled(input_ids: Optional[mx.array]) -> bool:
    """Enable an explicit one-token fence after each Qwen4 phase.

    This is deliberately opt-in because the fences destroy normal graph fusion.
    It separates SSD PLE lookup, GDN/QSA, routed MoE, and residual mixing wall
    time instead of attributing the whole lazy graph to a later ``mx.eval``.
    """
    if os.environ.get("VMLINUX_QWEN4_PROFILE_LAYERS", "").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False
    return input_ids is not None and input_ids.ndim == 2 and input_ids.shape[-1] == 1


def _profile_eval(*values: mx.array) -> float:
    started = time.perf_counter()
    mx.eval(*values)
    return (time.perf_counter() - started) * 1000.0


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class Qwen4ExpTextArgs:
    model_type: str = "qwen4_exp_text"
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    vocab_size: int = 248320
    layer_types: Optional[List[str]] = None
    full_attention_interval: int = 4
    # GDN
    linear_num_value_heads: int = 48
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    output_gate_type: str = "sigmoid"
    # MoE
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    norm_topk_prob: bool = True
    # mHC
    hc_count: int = 4
    hc_lowrank: int = 320
    # PLE / n-gram
    ple_layer_ids: List[int] = field(default_factory=lambda: [2])
    ple_embed_dim: int = 2560
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    seed: int = 1234
    split_ngram_parts: int = 128
    # QSA
    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4
    # rope
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    mrope_section: List[int] = field(default_factory=lambda: [11, 11, 10])
    max_position_embeddings: int = 262144
    eos_token_id: int = 248044
    tie_word_embeddings: bool = False
    mtp_num_hidden_layers: int = 0
    mtp_use_dedicated_embeddings: bool = False

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "Qwen4ExpTextArgs":
        text = cfg.get("text_config", cfg)
        rp = text.get("rope_parameters") or {}
        qsa = text.get("sparse_attention_config") or text.get("qsa_config") or {}
        ple = text.get("ple_config") or text.get("ngram_config") or {}
        eos = text.get("eos_token_id", 248044)
        if isinstance(eos, list):
            eos = eos[0]
        kwargs = {}
        for name in cls.__dataclass_fields__:
            if name in text:
                kwargs[name] = text[name]
        kwargs["rope_theta"] = rp.get(
            "rope_theta", text.get("rope_theta", 10_000_000.0)
        )
        kwargs["partial_rotary_factor"] = rp.get(
            "partial_rotary_factor", text.get("partial_rotary_factor", 0.25)
        )
        kwargs["mrope_section"] = rp.get(
            "mrope_section", text.get("mrope_section", [11, 11, 10])
        )
        aliases = {
            "indexer_n_heads": ("indexer_n_heads", "num_attention_heads"),
            "indexer_kv_heads": ("indexer_kv_heads", "num_key_value_heads"),
            "indexer_head_dim": ("indexer_head_dim", "head_dim"),
            "indexer_budget": ("indexer_budget", "budget"),
            "indexer_compress_ratio": (
                "indexer_compress_ratio",
                "block_size",
                "compress_ratio",
            ),
        }
        for target, names in aliases.items():
            if target in text:
                continue
            for name in names:
                if name in qsa:
                    kwargs[target] = qsa[name]
                    break
        for name in (
            "ple_layer_ids",
            "ple_embed_dim",
            "ple_conv_kernel_size",
            "ngram_size",
            "heads_per_ngram",
            "ngram_vocab_size_base",
            "make_ngram_vocab_size_divisible_by",
            "seed",
            "split_ngram_parts",
        ):
            if name not in text and name in ple:
                kwargs[name] = ple[name]
        if "mtp_num_hidden_layers" not in text:
            kwargs["mtp_num_hidden_layers"] = text.get(
                "num_nextn_predict_layers",
                cfg.get("num_nextn_predict_layers", 0),
            )
        kwargs["eos_token_id"] = eos
        return cls(**kwargs)

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "Qwen4ExpTextArgs":
        return cls.from_config(cfg)

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention"
                if (i + 1) % self.full_attention_interval
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        self.layer_types = [
            "full_attention" if t == "qwen_sparse_attention" else t
            for t in self.layer_types
        ]
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if sum(self.mrope_section) * 2 != self.rotary_dim:
            raise ValueError(
                "qwen4_exp mrope_section must cover rotary_dim/2 frequencies: "
                f"section={self.mrope_section}, rotary_dim={self.rotary_dim}"
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("qwen4_exp Q heads must divide evenly over KV heads")
        if self.indexer_budget <= 0 or self.indexer_compress_ratio <= 0:
            raise ValueError("qwen4_exp QSA budget and block size must be positive")
        if self.indexer_budget % self.indexer_compress_ratio:
            raise ValueError("qwen4_exp QSA token budget must contain whole blocks")
        if not (0 < self.num_experts_per_tok <= self.num_experts):
            raise ValueError("qwen4_exp routed expert count is invalid")
        ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if ngram_heads <= 0 or self.ple_embed_dim % ngram_heads:
            raise ValueError(
                "qwen4_exp PLE embedding width must divide across n-gram heads"
            )
        for layer_id in self.ple_layer_ids:
            if layer_id <= 0 or layer_id > self.num_hidden_layers:
                raise ValueError(f"qwen4_exp PLE layer id is invalid: {layer_id}")
            if self.layer_types[layer_id - 1] != "linear_attention":
                raise ValueError(
                    "qwen4_exp PLE must share a GDN ArraysCache slot; "
                    f"layer {layer_id} is {self.layer_types[layer_id - 1]}"
                )


# --------------------------------------------------------------------------- #
# Rope helper (manual cos/sin so arbitrary positions work — indexer needs
# block-start positions). Non-traditional rotate-half, matches HF.
# --------------------------------------------------------------------------- #
class RopeTable:
    def __init__(self, dims: int, base: float):
        self.dims = dims
        self.inv_freq = mx.power(base, -mx.arange(0, dims, 2, dtype=mx.float32) / dims)

    def cos_sin(self, positions: mx.array):
        """positions: [S] int → cos,sin [S, dims/2] fp32"""
        freqs = positions.astype(mx.float32)[:, None] * self.inv_freq[None, :]
        return mx.cos(freqs), mx.sin(freqs)

    def apply(self, x: mx.array, cos: mx.array, sin: mx.array, seq_axis: int = 2):
        """x: [..., S, ..., D] with rotary on first self.dims of D.
        cos/sin: [S, dims/2]; seq_axis tells where S lives (default [B,H,S,D])."""
        half = self.dims // 2
        x_rope, x_pass = x[..., : self.dims], x[..., self.dims :]
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        shape = [1] * x.ndim
        shape[seq_axis] = cos.shape[0]
        shape[-1] = half
        c = cos.reshape(shape)
        s = sin.reshape(shape)
        xt = x1.astype(mx.float32)
        yt = x2.astype(mx.float32)
        out1 = xt * c - yt * s
        out2 = yt * c + xt * s
        return mx.concatenate(
            [out1.astype(x.dtype), out2.astype(x.dtype), x_pass], axis=-1
        )


# --------------------------------------------------------------------------- #
# Norms
# --------------------------------------------------------------------------- #
class GroupedRMSNorm(nn.Module):
    """RMSNorm over groups of `group_size` along the last axis, full-width weight.
    Checkpoint weight is stored -1; sanitize shifts +1."""

    def __init__(self, dims: int, group_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.group_size = group_size
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        x = x.reshape(*shape[:-1], -1, self.group_size)
        x = mx.fast.rms_norm(x, None, self.eps)
        return x.reshape(shape) * self.weight


class ZeroCenteredRMSNorm(nn.Module):
    """Gemma-style RMSNorm for weights stored as a delta from one.

    Qwen4-Exp uses this convention only for the MTP input-fusion pre-norms.
    Unlike the backbone's shifted norms, these checkpoint tensors must remain
    zero-centered and are applied as ``1 + weight`` at runtime.
    """

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        weight = (
            self.weight
            if getattr(self, "_offset_folded", False)
            else 1.0 + self.weight
        )
        return mx.fast.rms_norm(x, weight, self.eps)


class RMSNormGatedSigmoid(nn.Module):
    """GDN output norm: per-head RMSNorm followed by sigmoid(gate) product.
    Plain weight convention (no +1 shift)."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        normed = mx.fast.rms_norm(x, self.weight, self.eps)
        return (normed.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))).astype(
            x.dtype
        )


# --------------------------------------------------------------------------- #
# mHC GatedResidual
# --------------------------------------------------------------------------- #
class GatedResidual(nn.Module):
    def __init__(self, args: Qwen4ExpTextArgs, use_combine: bool = True):
        super().__init__()
        self.hc_count = args.hc_count
        self.hidden_size = args.hidden_size
        self.hc_lowrank = args.hc_lowrank
        hc_hidden = self.hc_count * self.hidden_size
        self.hc_norm = GroupedRMSNorm(
            hc_hidden, self.hidden_size, eps=args.rms_norm_eps
        )
        self.input_mix_weight_down = nn.Linear(hc_hidden, args.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(args.hc_lowrank, hc_hidden, bias=False)
        if use_combine:
            self.block_inject_weight = nn.Linear(hc_hidden, self.hc_count, bias=False)
        self.use_combine = use_combine

    def __call__(self, hyper_input: mx.array):
        expected = self.hc_count * self.hidden_size
        if hyper_input.shape[-1] != expected:
            raise ValueError(
                f"expected {expected} hyper-connection features, "
                f"got {hyper_input.shape[-1]}"
            )
        compiled_forward = getattr(self, "_compiled_forward", None)
        if compiled_forward is not None and hyper_input.shape[-2] == 1:
            return compiled_forward(hyper_input)
        return self._forward(hyper_input)

    def _forward(self, hyper_input: mx.array):
        normed = self.hc_norm(hyper_input)
        input_inject_weight = getattr(self, "input_inject_weight", None)
        if input_inject_weight is None:
            mix = self.input_mix_weight_down(normed)
            block_injection = (
                self.block_inject_weight(normed)
                if self.use_combine
                else None
            )
        else:
            combined = input_inject_weight(normed)
            mix_indices, injection_indices = _hyper_split_indices(
                self.hc_lowrank, self.hc_count
            )
            mix = mx.take(combined, mix_indices, axis=-1)
            block_injection = mx.take(combined, injection_indices, axis=-1)
        mix = nn.silu(mix / self.hc_count)
        mix = mx.sigmoid(self.input_mix_weight_up(mix))
        mix = mix.astype(normed.dtype)
        mix = mix.reshape(*mix.shape[:-1], self.hc_count, self.hidden_size)
        mixed = (
            mix * normed.reshape(*normed.shape[:-1], self.hc_count, self.hidden_size)
        ).mean(-2).astype(hyper_input.dtype)
        if block_injection is None:
            return mixed
        inject_w = (
            2.0 * mx.sigmoid(block_injection / self.hc_count)
        ).astype(hyper_input.dtype)
        return mixed, hyper_input, inject_w

    def combine(
        self, hyper_input: mx.array, block_out: mx.array, inject_w: mx.array
    ) -> mx.array:
        inj = block_out[..., None, :] * inject_w[..., :, None]
        return (hyper_input + inj.reshape(
            *inj.shape[:-2], self.hc_count * self.hidden_size
        )).astype(hyper_input.dtype)


def _hyper_split_indices(lowrank: int, hc_count: int) -> tuple[mx.array, mx.array]:
    key = (lowrank, hc_count)
    indices = _HYPER_SPLIT_INDICES.get(key)
    if indices is None:
        indices = (
            mx.arange(lowrank, dtype=mx.int32),
            mx.arange(lowrank, lowrank + hc_count, dtype=mx.int32),
        )
        _HYPER_SPLIT_INDICES[key] = indices
    return indices


def _can_fuse_hyper_connection(module: GatedResidual) -> bool:
    if hasattr(module, "input_inject_weight"):
        return False
    injection = getattr(module, "block_inject_weight", None)
    down = getattr(module, "input_mix_weight_down", None)
    if injection is None or down is None or type(down) is not type(injection):
        return False
    if down.weight.shape[1:] != injection.weight.shape[1:]:
        return False
    if down.weight.dtype != injection.weight.dtype:
        return False
    for attribute in ("group_size", "bits", "mode"):
        if getattr(down, attribute, None) != getattr(injection, attribute, None):
            return False
    for tensor_name in ("scales", "biases", "bias"):
        if hasattr(down, tensor_name) != hasattr(injection, tensor_name):
            return False
    return True


def fuse_hyper_connection_projections(model: nn.Module) -> int:
    """Fuse each mix-down/injection pair into one row-wise projection."""
    modules = [model]
    modules.extend(module for _, module in model.named_modules() if module is not model)
    targets = []
    seen = set()
    for module in modules:
        if id(module) in seen:
            continue
        seen.add(id(module))
        if isinstance(module, GatedResidual) and _can_fuse_hyper_connection(module):
            targets.append(module)

    for module in targets:
        down = module.input_mix_weight_down
        injection = module.block_inject_weight
        fused = {"weight": mx.concatenate([down.weight, injection.weight], axis=0)}
        for tensor_name in ("scales", "biases", "bias"):
            if hasattr(down, tensor_name):
                fused[tensor_name] = mx.concatenate(
                    [getattr(down, tensor_name), getattr(injection, tensor_name)],
                    axis=0,
                )
        mx.eval(*fused.values())
        for tensor_name, value in fused.items():
            setattr(down, tensor_name, value)
        module.input_inject_weight = down
        del module.input_mix_weight_down
        del module.block_inject_weight
    return len(targets)


def compile_hyper_connections(model: nn.Module) -> int:
    """Compile the fixed-shape single-token hyper-connection decode path."""
    modules = [model]
    modules.extend(module for _, module in model.named_modules() if module is not model)
    compiled = 0
    seen = set()
    for module in modules:
        if id(module) in seen:
            continue
        seen.add(id(module))
        if not isinstance(module, GatedResidual) or hasattr(
            module, "_compiled_forward"
        ):
            continue
        module._compiled_forward = mx.compile(module._forward)
        compiled += 1
    return compiled


def fold_zero_centered_norm_offsets(model: nn.Module) -> int:
    """Fold trained ``1 + delta`` MTP norm weights once at load."""

    modules = [model]
    modules.extend(module for _, module in model.named_modules() if module is not model)
    targets = []
    seen = set()
    for module in modules:
        if id(module) in seen:
            continue
        seen.add(id(module))
        if isinstance(module, ZeroCenteredRMSNorm) and not getattr(
            module, "_offset_folded", False
        ):
            module.weight = 1.0 + module.weight
            module._offset_folded = True
            targets.append(module.weight)
    if targets:
        mx.eval(*targets)
    return len(targets)


# --------------------------------------------------------------------------- #
# PLE (n-gram) layer
# --------------------------------------------------------------------------- #
class ShardedNGramEmbedding(nn.Module):
    """The 51B table kept as its checkpoint row-shards so a lookup only pages
    in the gathered rows (mmap-friendly); concatenating would materialize
    ~95 GiB on first use. Shard sizes follow the HF ceil-split rule."""

    def __init__(self, padded_vocab_size: int, head_dim: int, n_shards: int):
        super().__init__()
        per = -(-padded_vocab_size // n_shards)  # ceil
        self.per = per
        self.head_dim = head_dim
        self.output_dtype = None
        # Full Qwen3.8 has ~320M rows. Constructing placeholder Embeddings for
        # those rows allocates the very table that this module is designed to
        # leave on SSD. Tiny parity models retain ordinary embeddings.
        if padded_vocab_size <= 2_000_000:
            self.shards = [
                nn.Embedding(min(per, padded_vocab_size - i * per), head_dim)
                for i in range(n_shards)
                if min(per, padded_vocab_size - i * per) > 0
            ]
        else:
            self.shards = []

    def set_file_backed(self, table, output_dtype=None) -> None:
        """Install a FileBackedNGramTable so lookups read only touched pages
        (np.memmap) instead of materializing 800 MB shard tensors."""
        self._file_backed = table
        self.output_dtype = table.output_dtype if output_dtype is None else output_dtype

    def __call__(self, rows_np: np.ndarray) -> mx.array:
        """rows_np: int64 [B, S, H] row ids into the concatenated table."""
        fb = getattr(self, "_file_backed", None)
        if fb is not None:
            vals = fb.gather_mlx(rows_np.reshape(-1))
            if self.output_dtype is not None and vals.dtype != self.output_dtype:
                vals = vals.astype(self.output_dtype)
            return vals.reshape(*rows_np.shape, self.head_dim)
        if not self.shards:
            raise RuntimeError(
                "qwen4_exp full PLE table requires a file-backed SSD row reader"
            )
        flat = rows_np.reshape(-1)
        shard_idx = flat // self.per
        local = flat % self.per
        head_dim = self.head_dim
        out = mx.zeros((flat.shape[0], head_dim), dtype=self.shards[0].weight.dtype)
        for s in np.unique(shard_idx):
            sel = np.nonzero(shard_idx == s)[0]
            gathered = self.shards[int(s)].weight[
                mx.array(local[sel].astype(np.uint32))
            ]
            out[mx.array(sel.astype(np.uint32))] = gathered
        return out.reshape(*rows_np.shape, head_dim)


class PLELayer(nn.Module):
    """Cache slots (shared ArraysCache with the host GDN layer):
    [2] = previous `context_len` token ids (int32 [B, C])
    [3] = dilated conv state ([B, (K-1)*dilation, hc_hidden])
    """

    def __init__(self, args: Qwen4ExpTextArgs, ple_layer_index: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.hc_count = args.hc_count
        hc_hidden = args.hidden_size * args.hc_count
        self.hasher = NGramHasher(
            vocab_size=args.vocab_size,
            eos_token_id=args.eos_token_id,
            ngram_size=args.ngram_size,
            heads_per_ngram=args.heads_per_ngram,
            ngram_vocab_size_base=args.ngram_vocab_size_base,
            make_divisible_by=args.make_ngram_vocab_size_divisible_by,
            seed=args.seed,
            ple_layer_index=ple_layer_index,
        )
        head_dim = args.ple_embed_dim // self.hasher.ngram_heads
        self.ngram_embedding = ShardedNGramEmbedding(
            self.hasher.padded_vocab_size, head_dim, args.split_ngram_parts
        )
        self.key_proj = nn.Linear(args.ple_embed_dim, hc_hidden, bias=False)
        self.value_proj = nn.Linear(args.ple_embed_dim, args.hidden_size, bias=False)
        self.norm_key = GroupedRMSNorm(
            hc_hidden, args.hidden_size, eps=args.rms_norm_eps
        )
        self.norm_query = GroupedRMSNorm(
            hc_hidden, args.hidden_size, eps=args.rms_norm_eps
        )
        self.norm_conv = GroupedRMSNorm(
            hc_hidden, args.hidden_size, eps=args.rms_norm_eps
        )
        self.conv_kernel_size = args.ple_conv_kernel_size
        self.conv_dilation = args.ngram_size
        self.short_conv_state_len = (self.conv_kernel_size - 1) * self.conv_dilation
        # depthwise dilated conv taps: checkpoint [C,1,K] → sanitized to [C,K]
        self.conv1d_weight = mx.zeros((hc_hidden, self.conv_kernel_size))

    def _embed(self, input_ids: mx.array, cache) -> mx.array:
        ids_np = np.asarray(input_ids, dtype=np.int64)
        if cache is not None and cache[2] is not None:
            prev = np.asarray(cache[2], dtype=np.int64)
        else:
            prev = None
        rows = self.hasher.hash_tokens(ids_np, prev)  # [B, S, heads]
        emb = self.ngram_embedding(rows)  # [B, S, heads, head_dim]
        if cache is not None:
            ctx = np.concatenate(
                [
                    prev
                    if prev is not None
                    else np.full(
                        (ids_np.shape[0], self.hasher.context_len),
                        self.hasher.eos_token_id,
                        dtype=np.int64,
                    ),
                    ids_np,
                ],
                axis=1,
            )[:, -self.hasher.context_len :]
            cache[2] = mx.array(ctx.astype(np.int32))
        return emb.reshape(*emb.shape[:-2], -1)

    def _short_conv(self, x: mx.array, cache) -> mx.array:
        """x: [B, S, C]; state carries the previous (K-1)*dilation positions."""
        B, S, C = x.shape
        if cache is not None and cache[3] is not None:
            state = cache[3]
        else:
            state = mx.zeros((B, self.short_conv_state_len, C), dtype=x.dtype)
        full = mx.concatenate([state, x], axis=1)
        if cache is not None:
            cache[3] = mx.contiguous(full[:, -self.short_conv_state_len :, :])
        # taps: y[t] = sum_j w[:, j] * full[t + j*dilation], j = 0..K-1 (t in padded coords)
        taps = []
        conv_taps = getattr(self, "_conv_taps", None)
        if conv_taps is None:
            conv_taps = tuple(
                self.conv1d_weight[:, j] for j in range(self.conv_kernel_size)
            )
        for j, weight in enumerate(conv_taps):
            start = j * self.conv_dilation
            taps.append(full[:, start : start + S, :] * weight)
        return nn.silu(sum(taps))

    def prepare_runtime(self) -> None:
        """Materialize contiguous depthwise-convolution taps once at load."""

        self._conv_taps = tuple(
            mx.contiguous(self.conv1d_weight[:, j])
            for j in range(self.conv_kernel_size)
        )
        mx.eval(*self._conv_taps)

    def __call__(self, hidden_states: mx.array, input_ids: mx.array, cache) -> mx.array:
        emb = self._embed(input_ids, cache)
        key = self.norm_key(self.key_proj(emb))
        key = key.reshape(*key.shape[:-1], self.hc_count, self.hidden_size)
        value = self.value_proj(emb)
        query = self.norm_query(hidden_states)
        query = query.reshape(*query.shape[:-1], self.hc_count, self.hidden_size)
        gate = (key * query).sum(-1, keepdims=True) / (self.hidden_size**0.5)
        gate = mx.sqrt(mx.maximum(mx.abs(gate), 1e-6)) * mx.sign(gate)
        gated_value = mx.sigmoid(gate) * value[..., None, :]
        gated_value = gated_value.reshape(*gated_value.shape[:-2], -1)
        gated_value_normed = self.norm_conv(gated_value)
        return gated_value + self._short_conv(gated_value_normed, cache)


# --------------------------------------------------------------------------- #
# GDN — qwen3_5 GatedDeltaNet with sigmoid output gate
# --------------------------------------------------------------------------- #
def _decode_quantized_linears_fused(
    linears: tuple[nn.Module, ...], x: mx.array
) -> tuple[mx.array, ...] | None:
    """Run same-input affine projections as one bit-identical decode QMM.

    Affine quantization packs every output row independently. Concatenating
    compatible projections along that row axis therefore preserves every
    output bit while removing one Metal launch per additional projection.
    Qwen3.8 GDN has four such projections in each of its 36 linear-attention
    layers, so the unfused path paid 108 unnecessary launches per token.
    """
    if (
        x.ndim != 3
        or x.shape[1] != 1
        or not linears
        or not all(isinstance(linear, nn.QuantizedLinear) for linear in linears)
    ):
        return None

    first = linears[0]
    if not all(
        linear.bits == first.bits
        and linear.group_size == first.group_size
        and linear.mode == first.mode
        and linear.biases is not None
        and linear.scales.dtype == x.dtype
        and linear.biases.dtype == x.dtype
        and "bias" not in linear
        for linear in linears
    ):
        return None

    cache_key = tuple(
        (id(linear.weight), id(linear.scales), id(linear.biases))
        for linear in linears
    )
    cached = getattr(first, "_qwen4_fused_decode_linears", None)
    if cached is None or cached[0] != cache_key:
        weights = mx.concatenate([linear.weight for linear in linears], axis=0)
        scales = mx.concatenate([linear.scales for linear in linears], axis=0)
        biases = mx.concatenate([linear.biases for linear in linears], axis=0)
        split_indices = []
        offset = 0
        for linear in linears[:-1]:
            offset += linear.weight.shape[0]
            split_indices.append(offset)
        mx.eval(weights, scales, biases)
        cached = (cache_key, weights, scales, biases, split_indices)
        first._qwen4_fused_decode_linears = cached

    _, weights, scales, biases, split_indices = cached
    output = mx.quantized_matmul(
        x,
        weights,
        scales=scales,
        biases=biases,
        transpose=True,
        group_size=first.group_size,
        bits=first.bits,
        mode=first.mode,
    )
    return tuple(mx.split(output, split_indices, axis=-1))


class GatedDeltaNet(_Qwen35GatedDeltaNet):
    def __init__(self, args: Qwen4ExpTextArgs):
        shim = _Qwen35TextArgs(
            model_type="qwen4_exp_text",
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            rms_norm_eps=args.rms_norm_eps,
            vocab_size=args.vocab_size,
            num_key_value_heads=args.num_key_value_heads,
            linear_num_value_heads=args.linear_num_value_heads,
            linear_num_key_heads=args.linear_num_key_heads,
            linear_key_head_dim=args.linear_key_head_dim,
            linear_value_head_dim=args.linear_value_head_dim,
            linear_conv_kernel_dim=args.linear_conv_kernel_dim,
            head_dim=args.head_dim,
        )
        super().__init__(shim)
        if args.output_gate_type == "sigmoid":
            self.norm = RMSNormGatedSigmoid(
                self.head_v_dim, eps=self.layer_norm_epsilon
            )
        # else: keep the inherited silu-gated norm

    def _process_chunk(
        self,
        qkv,
        a,
        b,
        conv_state,
        ssm_state,
        mask=None,
        lengths=None,
    ):
        batch_size, seq_len = qkv.shape[:2]
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        keep = self.conv_kernel_size - 1
        if lengths is not None:
            ends = mx.clip(lengths, 0, seq_len)
            positions = (ends[:, None] + mx.arange(keep))[..., None]
            new_conv_state = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            new_conv_state = mx.contiguous(conv_input[:, -keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))
        q, k, v = [
            tensor.reshape(batch_size, seq_len, heads, dim)
            for tensor, heads, dim in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        out, new_ssm_state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            ssm_state,
            mask,
            use_kernel=not self.training,
        )
        return out, new_conv_state, new_ssm_state

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        n_confirmed: int = 0,
    ) -> mx.array:
        batch_size, seq_len, _ = inputs.shape
        if self.sharding_group is not None:
            if n_confirmed:
                raise NotImplementedError(
                    "qwen4_exp MTP rollback is not implemented for sharded GDN"
                )
            return super().__call__(inputs, mask=mask, cache=cache)

        projections = (
            self.in_proj_qkv,
            self.in_proj_z,
            self.in_proj_b,
            self.in_proj_a,
        )
        fused = _decode_quantized_linears_fused(projections, inputs)
        qkv, z, b, a = fused or tuple(
            projection(inputs) for projection in projections
        )
        z = z.reshape(
            batch_size, seq_len, self.num_v_heads, self.head_v_dim
        )
        conv_state = cache[0] if cache is not None else None
        if conv_state is None or conv_state.shape[0] != batch_size:
            conv_state = mx.zeros(
                (batch_size, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )
        ssm_state = cache[1] if cache is not None else None
        if ssm_state is not None and ssm_state.shape[0] != batch_size:
            ssm_state = None
        if mask is not None:
            if mask.shape[0] != batch_size:
                mask = None
            else:
                qkv = mx.where(mask[..., None], qkv, 0)

        if 0 < n_confirmed < seq_len:
            confirmed_mask = mask[:, :n_confirmed] if mask is not None else None
            draft_mask = mask[:, n_confirmed:] if mask is not None else None
            out_c, conv_c, ssm_c = self._process_chunk(
                qkv[:, :n_confirmed],
                a[:, :n_confirmed],
                b[:, :n_confirmed],
                conv_state,
                ssm_state,
                confirmed_mask,
            )
            if cache is not None:
                cache.rollback_state = (conv_c, ssm_c)
                draft_qkv = qkv[:, n_confirmed:]
                draft_a = a[:, n_confirmed:]
                draft_b = b[:, n_confirmed:]

                def rollback_to(
                    count,
                    _q=draft_qkv,
                    _a=draft_a,
                    _b=draft_b,
                    _c=conv_c,
                    _s=ssm_c,
                    _m=draft_mask,
                    _self=self,
                ):
                    _, conv_k, ssm_k = _self._process_chunk(
                        _q[:, :count],
                        _a[:, :count],
                        _b[:, :count],
                        _c,
                        _s,
                        _m[:, :count] if _m is not None else None,
                    )
                    return conv_k, ssm_k

                cache.rollback_to = rollback_to
            out_d, conv_f, ssm_f = self._process_chunk(
                qkv[:, n_confirmed:],
                a[:, n_confirmed:],
                b[:, n_confirmed:],
                conv_c,
                ssm_c,
                draft_mask,
            )
            out = mx.concatenate([out_c, out_d], axis=1)
        else:
            lengths = getattr(cache, "lengths", None) if cache is not None else None
            out, conv_f, ssm_f = self._process_chunk(
                qkv, a, b, conv_state, ssm_state, mask, lengths=lengths
            )

        if cache is not None:
            cache[0] = conv_f
            cache[1] = ssm_f
            advance = getattr(cache, "advance", None)
            if callable(advance):
                advance(seq_len)
        out = self.norm(out, z)
        return self.out_proj(out.reshape(batch_size, seq_len, -1))


# --------------------------------------------------------------------------- #
# QSA attention
# --------------------------------------------------------------------------- #
# QSA and MiniMax-M3's MSA have the same persistence shape: full K/V plus an
# append-only indexer payload lane. QSA's payload stores each raw index key and
# its three M-RoPE coordinates. Upstream pools raw keys first, normalizes the
# pooled block, then rotates it at the block's first token; rotating token keys
# before pooling is not equivalent. Keeping the coordinates beside the raw key
# makes that exact order restart-safe for image/video prefixes.
# Reuse the engine's already hardened
# three-lane sparse cache transport so RAM cloning, partial-block SSD storage,
# restart restore, logical-offset trimming, and validation preserve all three
# arrays together. The selector algorithm remains QSA-specific; only its typed
# storage protocol is shared.
QSACache = _SparseIndexerKVCache


class QSAIndexer(nn.Module):
    def __init__(self, args: Qwen4ExpTextArgs):
        super().__init__()
        self.n_heads = args.indexer_n_heads
        self.head_dim = args.indexer_head_dim
        self.token_budget = args.indexer_budget
        self.compress_ratio = args.indexer_compress_ratio
        self.block_topk = self.token_budget // self.compress_ratio
        self.index_qk_proj = nn.Linear(
            args.hidden_size, (self.n_heads + 1) * self.head_dim, bias=False
        )
        self.q_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.rotary_emb = Qwen3_5RotaryEmbedding(
            args.rotary_dim,
            max_position_embeddings=args.max_position_embeddings,
            base=args.rope_theta,
            mrope_section=args.mrope_section,
        )

    @staticmethod
    def _position_payload(position_ids: mx.array, batch: int, length: int) -> mx.array:
        """Return exact per-token M-RoPE coordinates as ``[B, S, 3]``."""
        if position_ids.ndim == 2:
            if position_ids.shape == (batch, length):
                position_ids = mx.broadcast_to(
                    position_ids[None, ...], (3, batch, length)
                )
            elif batch == 1 and position_ids.shape == (3, length):
                position_ids = position_ids[:, None, :]
            else:
                raise ValueError(
                    "qwen4_exp QSA position_ids must be [B,S], [3,S] for B=1, "
                    "or [3,B,S]"
                )
        if position_ids.ndim != 3 or position_ids.shape != (3, batch, length):
            raise ValueError(
                "qwen4_exp QSA position_ids must resolve to [3,B,S], got "
                f"{position_ids.shape}"
            )
        return position_ids.transpose(1, 2, 0).astype(mx.float32)

    def __call__(
        self,
        hidden_states: mx.array,
        cache: Optional[QSACache],
        *,
        offset: Optional[int] = None,
        position_ids: Optional[mx.array] = None,
    ) -> Optional[mx.array]:
        """Return an additive float mask [B, 1, S, T], or ``None``.

        At or below ``block_topk`` complete micro-blocks, every block is in the
        native QSA budget. Persist the raw index lane, then bypass pooling,
        scoring, and selection exactly as the upstream Qwen4 implementation
        does. Doing an ``argpartition`` whose result cannot hide anything is
        pure decode overhead.
        """
        B, S, _ = hidden_states.shape
        offset = (
            (cache.offset if cache is not None else 0) if offset is None else offset
        )

        qk = self.index_qk_proj(hidden_states)
        q, raw_k = mx.split(qk, [self.n_heads * self.head_dim], axis=-1)
        q = q.reshape(B, S, self.n_heads, self.head_dim)
        q = self.q_layernorm(q).transpose(0, 2, 1, 3)

        if position_ids is None:
            position_ids = mx.arange(offset, offset + S)[None, :]
        qcos, qsin = self.rotary_emb(q, position_ids)
        q, _ = apply_multimodal_rotary_pos_emb(q, q, qcos, qsin)
        q = q.transpose(0, 2, 1, 3)

        # A single float32 payload preserves the raw projection values and
        # integer coordinates exactly through the existing three-lane cache
        # transport. Native positions are <=1M, well inside exact float32 int
        # representation. The extra three scalars are split before scoring.
        current_positions = self._position_payload(position_ids, B, S)
        payload = mx.concatenate(
            [raw_k.astype(mx.float32), current_positions], axis=-1
        )

        if cache is not None:
            all_payload = cache.update_index(payload[:, None, :, :])[:, 0, :, :]
        else:
            all_payload = payload
        all_keys = all_payload[..., : self.head_dim]
        all_positions = all_payload[..., self.head_dim :]
        T = all_payload.shape[1]

        # pooled block keys (block b = tokens [4b, 4b+4)). QSA selects 512
        # complete blocks. At or below that count every complete block plus
        # the current incomplete tail is visible, so building scores and an
        # argpartition is both redundant and contrary to the upstream Qwen4
        # fast path.
        num_blocks = T // self.compress_ratio
        if num_blocks <= self.block_topk:
            return None
        pooled = (
            all_keys[:, : num_blocks * self.compress_ratio, :]
            .reshape(B, num_blocks, self.compress_ratio, self.head_dim)
            .astype(mx.float32)
            .mean(axis=2)
        )
        pooled = self.k_layernorm(pooled[:, :, None, :]).transpose(0, 2, 1, 3)
        block_positions = all_positions[
            :, : num_blocks * self.compress_ratio : self.compress_ratio, :
        ].transpose(2, 0, 1)
        block_cos, block_sin = self.rotary_emb(pooled, block_positions)
        pooled, _ = apply_multimodal_rotary_pos_emb(
            pooled, pooled, block_cos, block_sin
        )
        pooled = pooled[:, 0, :, :]

        # scores: relu(q·k) summed over heads / sqrt(D) → [B, S, NB]
        scores = mx.einsum(
            "bshd,bnd->bshn", q.astype(mx.float32), pooled.astype(mx.float32)
        )
        scores = mx.maximum(scores, 0.0).sum(axis=2) / (self.head_dim**0.5)

        # per query i (absolute pos p = offset+i): visible tokens 0..p
        #   complete blocks for that query: ncb(p) = (p+1)//ratio
        #   keep: topk(block_topk) among blocks [0, ncb) + tail tokens [ncb*ratio, p]
        abs_pos = np.arange(offset, offset + S)
        ncb = (abs_pos + 1) // self.compress_ratio  # [S]
        ncb_mx = mx.array(ncb)
        block_ids = mx.arange(num_blocks)
        complete = block_ids[None, :] < ncb_mx[:, None]  # [S, NB]
        neg = mx.array(-np.inf, dtype=mx.float32)
        masked_scores = mx.where(complete[None], scores, neg)

        k_sel = min(self.block_topk, num_blocks)
        top_idx = mx.argpartition(-masked_scores, kth=k_sel - 1, axis=-1)[
            ..., :k_sel
        ]  # [B,S,k]
        keep_blocks = mx.zeros((B, S, num_blocks), dtype=mx.bool_)
        keep_blocks = mx.put_along_axis(keep_blocks, top_idx, mx.array(True), axis=-1)
        # queries with fewer complete blocks than k_sel picked -inf entries; drop those
        keep_blocks = keep_blocks & complete[None]

        # expand to tokens
        keep_tokens = mx.repeat(keep_blocks[..., None], self.compress_ratio, axis=-1)
        keep_tokens = keep_tokens.reshape(B, S, num_blocks * self.compress_ratio)
        if T > num_blocks * self.compress_ratio:
            pad = mx.ones((B, S, T - num_blocks * self.compress_ratio), dtype=mx.bool_)
            keep_tokens = mx.concatenate([keep_tokens, pad], axis=-1)
        # tail tokens (incomplete block for THIS query) always visible:
        token_ids = mx.arange(T)
        tail = token_ids[None, :] >= (ncb_mx * self.compress_ratio)[:, None]  # [S, T]
        keep_tokens = keep_tokens | tail[None]

        min_val = mx.array(-np.inf, dtype=mx.float32)
        return mx.where(keep_tokens[:, None], mx.array(0.0, dtype=mx.float32), min_val)


class QSAAttention(nn.Module):
    def __init__(self, args: Qwen4ExpTextArgs):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            args.hidden_size, self.num_heads * self.head_dim * 2, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, args.hidden_size, bias=False
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.indexer = QSAIndexer(args)
        self.rotary_emb = Qwen3_5RotaryEmbedding(
            args.rotary_dim,
            max_position_embeddings=args.max_position_embeddings,
            base=args.rope_theta,
            mrope_section=args.mrope_section,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[QSACache] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        B, S, _ = x.shape
        offset = cache.offset if cache is not None else 0

        qg = self.q_proj(x).reshape(B, S, self.num_heads, 2 * self.head_dim)
        queries, gate = mx.split(qg, 2, axis=-1)
        gate = gate.reshape(B, S, -1)
        keys = self.k_proj(x).reshape(B, S, self.num_kv_heads, self.head_dim)
        values = self.v_proj(x).reshape(B, S, self.num_kv_heads, self.head_dim)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if position_ids is None:
            position_ids = mx.arange(offset, offset + S)[None, :]
        cos, sin = self.rotary_emb(values, position_ids)
        queries, keys = apply_multimodal_rotary_pos_emb(queries, keys, cos, sin)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        T = keys.shape[2]

        # Append K/V before the raw indexer lane. The shared sparse transport
        # validates and slices both at the post-append logical offset; pass the
        # saved pre-append offset for this query chunk's absolute positions.
        index_mask = self.indexer(
            x,
            cache,
            offset=offset,
            position_ids=position_ids,
        )

        # At single-token decode the cache contains no future keys, so the
        # causal mask is identically zero. Avoid allocating that F32 tensor.
        if S == 1:
            full_mask = index_mask
        else:
            q_pos = mx.arange(offset, offset + S)[:, None]
            k_pos = mx.arange(T)[None, :]
            causal = mx.where(k_pos <= q_pos, 0.0, -np.inf).astype(queries.dtype)
            causal = causal[None, None]
            full_mask = causal if index_mask is None else causal + index_mask

        out = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.scale,
            mask=full_mask,
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(out * mx.sigmoid(gate))


# --------------------------------------------------------------------------- #
# MoE
# --------------------------------------------------------------------------- #
class SharedExpertMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoeBlock(nn.Module):
    def __init__(self, args: Qwen4ExpTextArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.shared_expert = SharedExpertMLP(
            args.hidden_size, args.shared_expert_intermediate_size
        )
        self.shared_expert_gate = nn.Linear(args.hidden_size, 1, bias=False)

    def __call__(
        self,
        x: mx.array,
        profile_phases: Optional[Dict[str, float]] = None,
    ) -> mx.array:
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        if profile_phases is not None:
            profile_phases["moe_router"] = _profile_eval(inds, scores)

        routed, fused_reduction = qwen4_affine_switchglu(
            self.switch_mlp, x, inds, scores
        )
        if profile_phases is not None:
            profile_phases["moe_routed"] = _profile_eval(routed)
            profile_phases["moe_reduce"] = 0.0 if fused_reduction else 0.0

        shared_gate = mx.sigmoid(self.shared_expert_gate(x))
        if profile_phases is not None:
            profile_phases["moe_shared_gate"] = _profile_eval(shared_gate)

        shared = self.shared_expert(x)
        if profile_phases is not None:
            profile_phases["moe_shared"] = _profile_eval(shared)

        output = routed + shared_gate * shared
        if profile_phases is not None:
            profile_phases["moe_finalize"] = _profile_eval(output)
        return output


# --------------------------------------------------------------------------- #
# Decoder layer / model
# --------------------------------------------------------------------------- #
class DecoderLayer(nn.Module):
    def __init__(self, args: Qwen4ExpTextArgs, layer_idx: int):
        super().__init__()
        self.layer_type = args.layer_types[layer_idx]
        self.is_linear = self.layer_type == "linear_attention"
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = QSAAttention(args)
        self.mlp = SparseMoeBlock(args)
        if (layer_idx + 1) in args.ple_layer_ids:
            self.ple = PLELayer(args, args.ple_layer_ids.index(layer_idx + 1))
        else:
            self.ple = None
        self.attn_hyper_connection = GatedResidual(args)
        self.mlp_hyper_connection = GatedResidual(args)

    def __call__(
        self,
        h,
        mask=None,
        cache=None,
        input_ids=None,
        position_ids=None,
        profile_layer: Optional[int] = None,
        n_confirmed: int = 0,
    ):
        phase_ms: Dict[str, float] = {}
        if self.ple is not None:
            if cache is not None and 0 < n_confirmed < h.shape[1]:
                confirmed_h = h[:, :n_confirmed]
                draft_h = h[:, n_confirmed:]
                confirmed_ids = input_ids[:, :n_confirmed]
                draft_ids = input_ids[:, n_confirmed:]
                confirmed_h = confirmed_h + self.ple(
                    confirmed_h, confirmed_ids, cache
                )
                ple_context = cache[2]
                ple_conv = cache[3]
                cache.rollback_aux_state = (ple_context, ple_conv)

                def rollback_aux_to(
                    count,
                    _h=draft_h,
                    _ids=draft_ids,
                    _context=ple_context,
                    _conv=ple_conv,
                    _cache=cache,
                    _ple=self.ple,
                ):
                    _cache[2] = _context
                    _cache[3] = _conv
                    if count:
                        _ple(_h[:, :count], _ids[:, :count], _cache)
                    return _cache[2], _cache[3]

                cache.rollback_aux_to = rollback_aux_to
                draft_h = draft_h + self.ple(draft_h, draft_ids, cache)
                h = mx.concatenate([confirmed_h, draft_h], axis=1)
            else:
                h = h + self.ple(h, input_ids, cache)
            if profile_layer is not None:
                phase_ms["ple"] = _profile_eval(h)

        x, hyper, inject = self.attn_hyper_connection(h)
        if profile_layer is not None:
            phase_ms["attn_hc"] = _profile_eval(x)
        if self.is_linear:
            r = self.linear_attn(
                x, mask=mask, cache=cache, n_confirmed=n_confirmed
            )
        else:
            r = self.self_attn(
                x,
                mask=mask,
                cache=cache,
                position_ids=position_ids,
            )
        if profile_layer is not None:
            phase_ms["gdn" if self.is_linear else "qsa"] = _profile_eval(r)
        h = self.attn_hyper_connection.combine(hyper, r, inject)
        if profile_layer is not None:
            phase_ms["attn_combine"] = _profile_eval(h)

        x, hyper, inject = self.mlp_hyper_connection(h)
        if profile_layer is not None:
            phase_ms["mlp_hc"] = _profile_eval(x)
        r = self.mlp(x, phase_ms if profile_layer is not None else None)
        h = self.mlp_hyper_connection.combine(hyper, r, inject)
        if profile_layer is not None:
            phase_ms["mlp_combine"] = _profile_eval(h)
            logger.info(
                "QWEN4_LAYER_PROFILE layer=%d type=%s total_ms=%.3f phases_ms=%s",
                profile_layer,
                "gdn" if self.is_linear else "qsa",
                sum(phase_ms.values()),
                ",".join(f"{name}:{value:.3f}" for name, value in phase_ms.items()),
            )
        return h


class Qwen4ExpTextModel(nn.Module):
    def __init__(self, args: Qwen4ExpTextArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.hyper_connection_mixer = GatedResidual(args, use_combine=False)
        self.fa_idx = next(
            (i for i, layer in enumerate(self.layers) if not layer.is_linear),
            0,
        )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        inputs_embeds=None,
        position_ids=None,
        return_expanded: bool = False,
        n_confirmed: int = 0,
        **_kwargs,
    ):
        h = inputs_embeds if inputs_embeds is not None else self.embed_tokens(inputs)
        h = mx.tile(h, (1, 1, self.args.hc_count))
        profile = _layer_profile_enabled(inputs)
        if profile:
            logger.info("QWEN4_LAYER_PROFILE begin seq_len=1")
        if cache is None:
            cache = [None] * len(self.layers)
        for layer_index, (layer, c) in enumerate(zip(self.layers, cache)):
            h = layer(
                h,
                mask=None,
                cache=c,
                input_ids=inputs,
                position_ids=position_ids,
                profile_layer=layer_index if profile else None,
                n_confirmed=n_confirmed,
            )
        mixed = self.hyper_connection_mixer(h)
        if profile:
            mixer_ms = _profile_eval(mixed)
            logger.info("QWEN4_LAYER_PROFILE final_mixer_ms=%.3f", mixer_ms)
        return (mixed, h) if return_expanded else mixed


class MTPModule(nn.Module):
    """Trained qwen4_exp next-token head with its own full QSA/MoE layer."""

    def __init__(self, args: Qwen4ExpTextArgs):
        super().__init__()
        self.pre_fc_norm_embedding = ZeroCenteredRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.pre_fc_norm_hidden = ZeroCenteredRMSNorm(
            args.hc_count * args.hidden_size, eps=args.rms_norm_eps
        )
        self.fc_embedding = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.fc_hidden = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        mtp_args = replace(
            args,
            num_hidden_layers=args.mtp_num_hidden_layers,
            layer_types=["full_attention"] * args.mtp_num_hidden_layers,
            ple_layer_ids=[],
        )
        self.layers = [
            DecoderLayer(mtp_args, i) for i in range(args.mtp_num_hidden_layers)
        ]
        self.hyper_connection_mixer = GatedResidual(args, use_combine=False)

    def fuse_inputs(self, hidden_states, token_embeddings):
        """Preserve all four trained mHC branches through MTP input fusion."""
        embedded = self.fc_embedding(self.pre_fc_norm_embedding(token_embeddings))
        hidden = self.pre_fc_norm_hidden(hidden_states)
        original_shape = hidden.shape
        hidden = hidden.reshape(
            *hidden.shape[:-1],
            self.hyper_connection_mixer.hc_count,
            self.hyper_connection_mixer.hidden_size,
        )
        hidden = self.fc_hidden(hidden)
        return (embedded[..., None, :] + hidden).reshape(original_shape)

    def __call__(
        self,
        hidden_states,
        next_token_ids,
        embed_tokens,
        cache=None,
        return_expanded: bool = False,
    ):
        fused = self.fuse_inputs(hidden_states, embed_tokens(next_token_ids))
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, layer_cache in zip(self.layers, cache):
            fused = layer(
                fused,
                mask=None,
                cache=layer_cache,
                input_ids=next_token_ids,
            )
        mixed = self.hyper_connection_mixer(fused)
        return (mixed, fused) if return_expanded else mixed


class LanguageModel(nn.Module):
    """mlx-vlm language-model interface for the qwen4_exp text core."""

    def __init__(self, args: Qwen4ExpTextArgs, config=None):
        super().__init__()
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = Qwen4ExpTextModel(args)
        self._position_ids = None
        self._rope_deltas = None
        if args.mtp_num_hidden_layers > 0:
            self.mtp = MTPModule(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        inputs_embeds=None,
        return_hidden: bool = False,
        return_logits: bool = True,
        **kwargs,
    ):
        position_ids = kwargs.get("position_ids")
        cache_offset = 0
        if cache:
            anchor = cache[self.model.fa_idx]
            cache_offset = int(getattr(anchor, "offset", 0) or 0)

        if position_ids is None:
            # The vision wrapper computes exact 3-axis M-RoPE positions before
            # chunked prefill. Slice that retained plan at the logical cache
            # offset; never derive media history from a scalar token count.
            if self._position_ids is not None:
                stop = cache_offset + inputs.shape[1]
                if stop <= self._position_ids.shape[-1]:
                    position_ids = self._position_ids[:, :, cache_offset:stop]

            if position_ids is None:
                batch_size, seq_length = inputs.shape
                if self._rope_deltas is not None and cache_offset > 0:
                    delta = mx.array(cache_offset + self._rope_deltas)
                    if delta.ndim == 0:
                        delta = delta[None]
                    if delta.shape[0] < batch_size:
                        delta = mx.tile(delta, (batch_size, 1))
                    else:
                        delta = delta[:batch_size]
                    text_pos = mx.arange(seq_length)[None, :]
                    text_pos = mx.broadcast_to(text_pos, (batch_size, seq_length))
                    text_pos = text_pos + delta.reshape(batch_size, 1)
                    position_ids = mx.broadcast_to(
                        text_pos[None, ...],
                        (3, batch_size, seq_length),
                    )
                else:
                    position_ids = mx.arange(
                        cache_offset,
                        cache_offset + seq_length,
                    )[None, :]
                    position_ids = mx.broadcast_to(
                        position_ids, (batch_size, seq_length)
                    )

        hidden, expanded_hidden = self.model(
            inputs,
            cache=cache,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            return_expanded=True,
            n_confirmed=int(kwargs.get("n_confirmed", 0) or 0),
        )
        if not return_logits:
            return (hidden, expanded_hidden) if return_hidden else hidden
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(hidden)
        else:
            logits = self.lm_head(hidden)
        if return_hidden:
            return logits, expanded_hidden
        return LanguageModelOutput(logits=logits)

    get_rope_index = _Qwen35VlmLanguageModel.get_rope_index

    def make_cache(self):
        caches = []
        for i, t in enumerate(self.args.layer_types):
            if t == "linear_attention":
                size = 4 if (i + 1) in self.args.ple_layer_ids else 2
                caches.append(ArraysCache(size=size))
            else:
                caches.append(QSACache())
        return caches

    def mtp_forward(
        self, hidden_states, next_token_ids, mtp_cache, return_hidden=False
    ):
        mtp_hidden, expanded_hidden = self.mtp(
            hidden_states,
            next_token_ids,
            self.model.embed_tokens,
            mtp_cache,
            return_expanded=True,
        )
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(mtp_hidden)
        else:
            logits = self.lm_head(mtp_hidden)
        return (logits, expanded_hidden) if return_hidden else logits

    def make_mtp_cache(self):
        if not hasattr(self, "mtp"):
            return []
        return [QSACache() for _layer in self.mtp.layers]

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads

    @property
    def quant_predicate(self):
        def predicate(path, _module):
            # Router scores and recurrent-state coefficients stay in their
            # checkpoint dtype. Quantizing either changes expert selection or
            # destabilizes the GDN state update.
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return False
            if path.endswith("A_log") or path.endswith("dt_bias"):
                return False
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(path: str):
            return not (path.endswith("A_log") or path.endswith("dt_bias"))

        return predicate


# Text-only callers in mlx-lm style still import ``Model``.
Model = LanguageModel
