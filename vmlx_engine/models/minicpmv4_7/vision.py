"""MiniCPM-V-4.7 vision tower, intra-ViT window merger and LLM projector.

Port of ``modeling_minicpmv4_7.py`` (transformers 5.19.0):
  * ``MiniCPMV4_7VisionEmbeddings``  L55-98   (conv patch embed + learned abs pos-emb)
  * ``MiniCPMV4_7VisionAttention``   L153-239 (full attention inside each crop)
  * ``MiniCPMV4_7VisionEncoderLayer`` L242-273
  * ``MiniCPMV4_7ViTWindowAttentionMerger`` L304-398 (2x2 windows, tanh GELU)
  * ``MiniCPMV4_7VisionModel.forward`` L450-496 (merger after ``insert_layer_id``)
  * ``MiniCPMV4_7DownsampleMLP`` / ``MiniCPMV4_7Merger`` L499-575 (erf GELU)

Activations follow torch's op-math: GELU/LayerNorm are evaluated in float32
and rounded back to the activation dtype, as torch does for bf16 inputs.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import VisionConfig

_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_INV_SQRT_2 = 1.0 / math.sqrt(2.0)


def gelu_pytorch_tanh(x: mx.array) -> mx.array:
    """``F.gelu(x, approximate="tanh")`` evaluated in float32 op-math.

    Numerically the same formula as ``mlx.nn.gelu_approx`` (max |diff| 4.8e-7
    in float32, measured on mlx 0.32.3); written out so bf16 activations are
    computed in float32 and rounded once, as torch does.
    """
    y = x.astype(mx.float32)
    out = 0.5 * y * (1.0 + mx.tanh(_SQRT_2_OVER_PI * (y + 0.044715 * (y * y * y))))
    return out.astype(x.dtype)


def gelu_erf(x: mx.array) -> mx.array:
    """``nn.GELU()`` (exact erf) in float32 op-math."""
    y = x.astype(mx.float32)
    return (0.5 * y * (1.0 + mx.erf(y * _INV_SQRT_2))).astype(x.dtype)


def nearest_position_ids(target_sizes: Sequence[Sequence[int]], num_patches_per_side: int) -> np.ndarray:
    """``get_vision_nearest_position_ids`` (vision_utils.py) as exact integers.

    torch buckets ``arange(h)/h`` into ``arange(1/n, 1, 1/n)`` with
    ``right=True``; that equals ``floor(i * n / h)`` (asserted against torch
    in tests/test_minicpmv47_runtime.py for every h, w <= 1024).
    """
    n = int(num_patches_per_side)
    rows = []
    for height, width in target_sizes:
        height, width = int(height), int(width)
        bucket_h = (np.arange(height, dtype=np.int64) * n) // height
        bucket_w = (np.arange(width, dtype=np.int64) * n) // width
        rows.append((bucket_h[:, None] * n + bucket_w[None, :]).reshape(-1))
    if not rows:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate(rows)


def window_permutation(target_sizes: Sequence[Sequence[int]], window: int = 2) -> np.ndarray:
    """Token order of ``get_vision_window_index(spatial_merge_size=1, window_size=w, patch_size=1)``.

    Per crop: windows raster over the (h/w, w/w) grid, tokens inside a window
    raster. For even grids this is the gather used by both the window merger
    (it never un-permutes) and the final 2x2 merger.
    """
    perms = []
    offset = 0
    for height, width in target_sizes:
        height, width = int(height), int(width)
        if height % window or width % window:
            raise ValueError(
                f"MiniCPM-V-4.7 patch grid ({height}, {width}) is not divisible by the {window}x{window} window"
            )
        index = np.arange(height * width, dtype=np.int64).reshape(height, width)
        index = index.reshape(height // window, window, width // window, window)
        perms.append(index.transpose(0, 2, 1, 3).reshape(-1) + offset)
        offset += height * width
    if not perms:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate(perms)


def _equal_length_runs(segments: Sequence[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
    """Group consecutive, contiguous ``(start, length)`` segments of equal length.

    Returns ``(start, count, length)`` runs covering the segments in order.
    """
    runs: List[List[int]] = []
    expected = None
    for start, length in segments:
        start, length = int(start), int(length)
        if expected is not None and start != expected:
            raise ValueError("MiniCPM-V-4.7 attention segments must be contiguous")
        if runs and runs[-1][2] == length:
            runs[-1][1] += 1
        else:
            runs.append([start, 1, length])
        expected = start + length
    return [(s, c, l) for s, c, l in runs]


class PatchEmbedding(nn.Module):
    """``nn.Conv2d(C, D, k=P, s=P, padding="valid")`` on the NaViT strip.

    The processor packs each crop as ``[C, P, L*P]`` (patches side by side in
    raster order). A stride-P/kernel-P conv over that strip is one GEMM per
    patch; it is evaluated as such. The weight keeps MLX conv layout
    ``(D, P, P, C)`` (sanitize transposes torch's ``(D, C, P, P)``).
    """

    def __init__(self, in_channels: int, out_channels: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.weight = mx.zeros((out_channels, patch_size, patch_size, in_channels))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, pixel_values: mx.array) -> mx.array:
        # pixel_values: (1, C, P, L*P) -> (L, P*P*C) in (kh, kw, c) order.
        c, p = self.in_channels, self.patch_size
        if pixel_values.ndim == 4:
            pixel_values = pixel_values[0]
        if pixel_values.shape[0] != c or pixel_values.shape[1] != p or pixel_values.shape[2] % p:
            raise ValueError(
                f"MiniCPM-V-4.7 expects NaViT-packed pixels (C={c}, {p}, L*{p}); got {tuple(pixel_values.shape)}"
            )
        num_patches = pixel_values.shape[2] // p
        x = pixel_values.reshape(c, p, num_patches, p).transpose(2, 1, 3, 0)
        x = x.reshape(num_patches, p * p * c)
        w = self.weight.reshape(self.weight.shape[0], -1)
        return mx.addmm(self.bias, x, w.T)


class VisionEmbeddings(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            config.num_channels, config.hidden_size, config.patch_size
        )
        self.num_patches_per_side = config.image_size // config.patch_size
        self.position_embedding = nn.Embedding(
            self.num_patches_per_side**2, config.hidden_size
        )

    def __call__(self, pixel_values: mx.array, target_sizes) -> mx.array:
        patches = self.patch_embedding(pixel_values)
        pos_ids = mx.array(nearest_position_ids(target_sizes, self.num_patches_per_side))
        if pos_ids.shape[0] != patches.shape[0]:
            raise ValueError(
                f"MiniCPM-V-4.7 target sizes cover {pos_ids.shape[0]} patches but pixels hold {patches.shape[0]}"
            )
        return patches + self.position_embedding(pos_ids)


class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(self.dim, self.dim, bias=True)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=True)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=True)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=True)

    def __call__(self, x: mx.array, segments: Sequence[Tuple[int, int]]) -> mx.array:
        """x: (N, D). Full, non-causal attention inside each ``(start, length)`` segment."""
        n = x.shape[0]
        q = self.q_proj(x).reshape(n, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(n, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(n, self.num_heads, self.head_dim)

        # Consecutive segments of equal length are attended as one batch;
        # every segment is still an independent sequence (no cross-crop
        # attention), exactly the reference's per-crop split.
        pieces: List[mx.array] = []
        for start, count, length in _equal_length_runs(segments):
            span = slice(start, start + count * length)
            qg = q[span].reshape(count, length, self.num_heads, self.head_dim)
            kg = k[span].reshape(count, length, self.num_heads, self.head_dim)
            vg = v[span].reshape(count, length, self.num_heads, self.head_dim)
            out = mx.fast.scaled_dot_product_attention(
                qg.transpose(0, 2, 1, 3),
                kg.transpose(0, 2, 1, 3),
                vg.transpose(0, 2, 1, 3),
                scale=self.scale,
            )
            pieces.append(out.transpose(0, 2, 1, 3).reshape(count * length, self.dim))
        attn = pieces[0] if len(pieces) == 1 else mx.concatenate(pieces, axis=0)
        return self.out_proj(attn)


class VisionMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        if config.hidden_act != "gelu_pytorch_tanh":
            raise ValueError(f"Unsupported MiniCPM-V-4.7 vision activation {config.hidden_act!r}")
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(gelu_pytorch_tanh(self.fc1(x)))


class VisionEncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = VisionAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = VisionMLP(config)

    def __call__(self, x: mx.array, segments) -> mx.array:
        x = x + self.self_attn(self.layer_norm1(x), segments)
        return x + self.mlp(self.layer_norm2(x))


class VisionEncoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layers = [VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]


class ViTWindowAttentionMerger(nn.Module):
    """2x2 window attention + 4-to-1 MLP merge after encoder layer ``insert_layer_id``."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        wh, ww = tuple(config.window_kernel_size)
        if wh != ww:
            raise ValueError(f"window_kernel_size must be square; got {(wh, ww)}")
        self.window = int(wh)
        self.embed_dim = config.hidden_size
        self.self_attn = VisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pre_norm = nn.LayerNorm(config.window_hidden_size, eps=config.layer_norm_eps)
        self.linear_1 = nn.Linear(config.window_hidden_size, config.window_intermediate_size, bias=True)
        self.linear_2 = nn.Linear(config.window_intermediate_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array, target_sizes) -> mx.array:
        window_size = self.window * self.window
        perm = mx.array(window_permutation(target_sizes, self.window))
        residual = x[perm]
        h = self.layer_norm1(x)[perm]
        num_windows = h.shape[0] // window_size
        segments = [(i * window_size, window_size) for i in range(num_windows)]
        h = residual + self.self_attn(h, segments)
        patch = h.reshape(num_windows, window_size, self.embed_dim)
        flat = patch.reshape(num_windows, window_size * self.embed_dim)
        patch_residual = mx.mean(patch.astype(mx.float32), axis=1).astype(h.dtype)
        out = self.linear_2(gelu_pytorch_tanh(self.linear_1(self.pre_norm(flat))))
        return out + patch_residual


def _segments(target_sizes) -> List[Tuple[int, int]]:
    segments = []
    start = 0
    for height, width in target_sizes:
        length = int(height) * int(width)
        segments.append((start, length))
        start += length
    return segments


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.embeddings = VisionEmbeddings(config)
        self.encoder = VisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.vit_merger = ViTWindowAttentionMerger(config)

    def __call__(
        self,
        pixel_values: mx.array,
        target_sizes,
        use_vit_merger: bool = True,
        capture: dict | None = None,
    ) -> Tuple[mx.array, List[List[int]]]:
        """Returns ``(hidden (N', D), target_sizes after the window merger)``.

        ``capture`` (optional dict) receives intermediate tensors for parity
        harnesses: ``embeddings``, ``layer_<i>`` and ``vit_merger``.
        """
        sizes = [[int(h), int(w)] for h, w in np.asarray(target_sizes).reshape(-1, 2).tolist()]
        h = self.embeddings(pixel_values, sizes)
        if capture is not None:
            capture["embeddings"] = h
        segments = _segments(sizes)
        insert_layer_id = int(self.config.insert_layer_id) if use_vit_merger else -1
        for layer_index, layer in enumerate(self.encoder.layers):
            h = layer(h, segments)
            if capture is not None:
                capture[f"layer_{layer_index}"] = h
            if use_vit_merger and layer_index == insert_layer_id:
                h = self.vit_merger(h, sizes)
                sizes = [[hh // 2, ww // 2] for hh, ww in sizes]
                segments = _segments(sizes)
                if capture is not None:
                    capture["vit_merger"] = h
        return self.post_layernorm(h), sizes


class DownsampleMLP(nn.Module):
    def __init__(self, hidden_size: int, llm_embed_dim: int):
        super().__init__()
        merged = hidden_size * 4
        # eps is hard-coded to 1e-6 in the reference (L505).
        self.pre_norm = nn.LayerNorm(merged, eps=1e-6)
        self.linear_1 = nn.Linear(merged, merged, bias=True)
        self.linear_2 = nn.Linear(merged, llm_embed_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_2(gelu_erf(self.linear_1(self.pre_norm(x))))


class Merger(nn.Module):
    """Per crop: raster 2x2 grouping -> DownsampleMLP (``merger_times`` rounds)."""

    def __init__(self, vision_hidden: int, llm_hidden: int, merger_times: int = 1, merge_kernel_size=(2, 2)):
        super().__init__()
        mh, mw = tuple(merge_kernel_size)
        if mh != mw:
            raise ValueError(f"merge_kernel_size must be square; got {(mh, mw)}")
        self.merge = int(mh)
        self.merger_times = int(merger_times)
        self.mlp = [DownsampleMLP(vision_hidden, vision_hidden) for _ in range(self.merger_times - 1)]
        self.mlp.append(DownsampleMLP(vision_hidden, llm_hidden))

    def __call__(self, x: mx.array, target_sizes) -> mx.array:
        sizes = [[int(h), int(w)] for h, w in target_sizes]
        hidden = x
        for round_index, mlp in enumerate(self.mlp):
            group = self.merge * self.merge
            perm = mx.array(window_permutation(sizes, self.merge))
            hidden = hidden[perm].reshape(-1, group * hidden.shape[-1])
            hidden = mlp(hidden)
            if round_index + 1 < len(self.mlp):
                sizes = [[hh // self.merge, ww // self.merge] for hh, ww in sizes]
        return hidden
