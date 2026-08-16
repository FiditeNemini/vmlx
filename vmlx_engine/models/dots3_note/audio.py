# SPDX-License-Identifier: Apache-2.0
"""dots3_note audio tower (whisper-shaped dots encoder) as an nn.Module tree.

Math is a verbatim port of the verified functional reference
(``dots3-ref/towers.py`` AudioTower: EXACT vs torch at capture), re-hosted on
mlx.nn modules so the parameter paths match the checkpoint when this tower is
mounted as ``audio_encoder`` by the outer model.

Traps that shaped this file:

- The conv stem masks the time axis BEFORE every conv and updates
  ``valid = (valid + 1) // 2`` after each; skipping the inter-conv masks bleeds
  the zero-padded tail through the 3x3 kernels into real frames.
- Rope here is ROTATE-HALF with partial_rotary_factor 0.5 (first rot_dim dims
  only, theta 1e4, applied in fp32). The LM uses GPT-J INTERLEAVED rope, and
  the vision tower a 2D block-major variant — three towers, three rope forms;
  reusing one across towers is silent garbage.
- ``k_proj`` has NO bias while q/v/out do; ``conv_out`` has NO bias. A default
  ``bias=True`` there makes the strict weight map fail loud — keep it that way.
- ``fc1`` projects to 2*ffn and splits (gate, value) for swiglu; norms are
  RMSNorm eps 1e-6, but the adapter's ``proj.0`` is a LayerNorm at eps 1e-5
  run in fp32.
- ``downsample_hidden_size`` (480) and ``hop_length`` (160) are NOT fields of
  the AudioConfig dataclass (BaseModelConfig.from_dict drops unknown keys), so
  they are read via getattr with the PR defaults.
"""

import math
from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import AudioConfig


def _gelu_erf(x: mx.array) -> mx.array:
    return 0.5 * x * (1 + mx.erf(x / math.sqrt(2.0)))


def _rot_half(x: mx.array) -> mx.array:
    a, b = mx.split(x, 2, axis=-1)
    return mx.concatenate([-b, a], -1)


def _apply_partial_rope(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array, rot_dim: int
) -> Tuple[mx.array, mx.array]:
    cs, sn = cos[None, None], sin[None, None]  # [1, 1, T, rot_dim]
    qr, qp = q[..., :rot_dim], q[..., rot_dim:]
    kr, kp = k[..., :rot_dim], k[..., rot_dim:]
    qr = qr * cs + _rot_half(qr) * sn
    kr = kr * cs + _rot_half(kr) * sn
    return mx.concatenate([qr, qp], -1), mx.concatenate([kr, kp], -1)


class AudioAttention(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        d = config.d_model
        self.q_proj = nn.Linear(d, d, bias=True)
        self.k_proj = nn.Linear(d, d, bias=False)  # checkpoint has no k bias
        self.v_proj = nn.Linear(d, d, bias=True)
        self.out_proj = nn.Linear(d, d, bias=True)


class AudioEncoderLayer(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        d = config.d_model
        ffn = config.encoder_ffn_dim
        self.heads = config.encoder_attention_heads
        self.head_dim = d // self.heads
        self.d_model = d
        self.self_attn = AudioAttention(config)
        self.self_attn_layer_norm = nn.RMSNorm(d, eps=1e-6)
        self.final_layer_norm = nn.RMSNorm(d, eps=1e-6)
        self.fc1 = nn.Linear(d, 2 * ffn, bias=True)  # (gate, value) fused
        self.fc2 = nn.Linear(ffn, d, bias=True)

    def __call__(
        self,
        x: mx.array,
        amask: mx.array,
        cos: mx.array,
        sin: mx.array,
        rot_dim: int,
    ) -> mx.array:
        B, T, _ = x.shape
        h = self.self_attn_layer_norm(x)
        q = self.self_attn.q_proj(h)
        k = self.self_attn.k_proj(h)
        v = self.self_attn.v_proj(h)
        q = q.reshape(B, T, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        # Rope + SDPA in fp32, mask additive -inf beyond chunk_token_lens.
        qf, kf = q.astype(mx.float32), k.astype(mx.float32)
        qf, kf = _apply_partial_rope(qf, kf, cos, sin, rot_dim)
        o = mx.fast.scaled_dot_product_attention(
            qf,
            kf,
            v.astype(mx.float32),
            scale=self.head_dim ** -0.5,
            mask=amask,
        )
        o = o.transpose(0, 2, 1, 3).reshape(B, T, self.d_model).astype(x.dtype)
        x = x + self.self_attn.out_proj(o)
        h = self.final_layer_norm(x)
        gu = self.fc1(h)
        g, u = mx.split(gu, 2, axis=-1)
        return x + self.fc2((g * mx.sigmoid(g)) * u)


class SpeechEncoder(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        d = config.d_model
        # Not dataclass fields — see module docstring.
        down = getattr(config, "downsample_hidden_size", 480)
        self.hop = int(getattr(config, "hop_length", 160))
        self.conv2d1 = nn.Conv2d(1, down, 3, stride=2, padding=1, bias=True)
        self.conv2d2 = nn.Conv2d(down, down, 3, stride=2, padding=1, bias=True)
        self.conv2d3 = nn.Conv2d(down, down, 3, stride=2, padding=1, bias=True)
        bins = config.num_mel_bins
        for _ in range(3):
            bins = (bins + 1) // 2
        self.conv_out = nn.Linear(down * bins, d, bias=False)
        self.layers = [AudioEncoderLayer(config) for _ in range(config.encoder_layers)]
        self.layer_norm = nn.RMSNorm(d, eps=1e-6)
        heads = config.encoder_attention_heads
        head_dim = d // heads
        self.rot_dim = (int(head_dim * config.partial_rotary_factor) // 2) * 2
        theta = float(config.rope_theta)
        self._inv_freq = (
            1.0
            / theta
            ** (np.arange(0, self.rot_dim, 2, dtype=np.float64) / self.rot_dim)
        ).astype(np.float32)

    def _conv_stem(self, mel, sample_lens: np.ndarray) -> mx.array:
        """mel [B, n_mels, T]; masks the time axis between convs."""
        x = (mel if isinstance(mel, mx.array) else mx.array(np.asarray(mel)))[:, None]
        valid = mx.array((sample_lens // self.hop).astype(np.int32))

        def mask(x, v):
            t = mx.arange(x.shape[-1])[None]
            m = (t < v[:, None]).astype(x.dtype)
            return x * m[:, None, None, :]

        for conv in (self.conv2d1, self.conv2d2, self.conv2d3):
            x = mask(x, valid)
            # nn.Conv2d is NHWC; keep the logical layout [B, C, F, T].
            x = conv(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
            x = _gelu_erf(x)
            valid = (valid + 1) // 2
        x = mask(x, valid)
        B, C, F, T = x.shape
        x = x.transpose(0, 3, 1, 2).reshape(B, T, C * F)
        return self.conv_out(x)

    def _rope(self, T: int) -> Tuple[mx.array, mx.array]:
        pos = np.arange(T, dtype=np.float32)
        f = pos[:, None] * self._inv_freq[None]
        emb = np.concatenate([f, f], -1)
        return mx.array(np.cos(emb)), mx.array(np.sin(emb))

    def __call__(
        self,
        input_features,
        chunk_sample_lens: np.ndarray,
        chunk_token_lens: np.ndarray,
    ) -> mx.array:
        x = self._conv_stem(input_features, chunk_sample_lens)
        Tmax = int(chunk_token_lens.max())
        x = x[:, :Tmax]
        cos, sin = self._rope(Tmax)
        pos_mask = mx.array(np.arange(Tmax)[None] < chunk_token_lens[:, None])
        amask = mx.where(pos_mask[:, None, None, :], 0.0, -mx.inf).astype(mx.float32)
        for layer in self.layers:
            x = layer(x, amask, cos, sin, self.rot_dim)
        return self.layer_norm(x)


class AudioEncoder(nn.Module):
    """Wrapper matching the checkpoint's ``dots_encoder.speech_encoder`` path."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.speech_encoder = SpeechEncoder(config)


class AudioAdapter(nn.Module):
    """LN -> Linear -> GELU -> Linear.

    ``proj`` is a plain 4-item list so checkpoint keys ``proj.0/proj.1/proj.3``
    resolve; index 2 is the parameterless GELU placeholder. ``proj.0`` is a
    LayerNorm at eps 1e-5 run in fp32 — NOT the towers' RMSNorm.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = [
            nn.LayerNorm(in_dim, eps=1e-5, bias=True),
            nn.Linear(in_dim, out_dim, bias=True),
            nn.GELU(),  # exact erf form; keeps index 3 aligned
            nn.Linear(out_dim, out_dim, bias=True),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        h = mx.fast.layer_norm(
            x.astype(mx.float32),
            self.proj[0].weight.astype(mx.float32),
            self.proj[0].bias.astype(mx.float32),
            1e-5,
        )
        h = self.proj[1](h)
        h = _gelu_erf(h)
        return self.proj[3](h)


class AudioModel(nn.Module):
    """dots3_note audio tower: conv stem /8 + 32 rope layers + adapter.

    ``__call__(input_features, chunk_sample_lens, chunk_token_lens,
    audio_chunk_counts)``:
      input_features [n_chunks, n_mels, frames] log-mel (60 s chunks),
      chunk_sample_lens raw sample counts per chunk, chunk_token_lens
      ceil(samples / (hop * 8 * merge_factor)) per chunk, audio_chunk_counts
      chunks per original audio. Returns (embeddings
      [total_audio_tokens, adapter_out_dim], per-audio token lengths).
    """

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        self.dots_encoder = AudioEncoder(config)
        self.audio_adapter = AudioAdapter(config.adapter_in_dim, config.adapter_out_dim)

    def __call__(
        self,
        input_features,
        chunk_sample_lens=None,
        chunk_token_lens=None,
        audio_chunk_counts=None,
    ) -> Tuple[mx.array, np.ndarray]:
        # Chunk metadata may not survive every serving-plumbing hop. For the
        # common all-valid case the lens are fully derivable from the mel
        # shape: samples = frames x hop, tokens = ceil-div by the three
        # stride-2 convs, one audio per chunk.
        n_chunks = int(getattr(input_features, "shape", [1])[0])
        frames = int(getattr(input_features, "shape", [1, 1, 0])[-1])
        if chunk_sample_lens is None:
            hop = int(getattr(self.dots_encoder.speech_encoder, "hop", 160))
            chunk_sample_lens = np.full((n_chunks,), frames * hop, dtype=np.int64)
        if chunk_token_lens is None:
            t = frames
            for _ in range(3):
                t = (t + 1) // 2
            chunk_token_lens = np.full((n_chunks,), t, dtype=np.int64)
        if audio_chunk_counts is None:
            audio_chunk_counts = np.ones((n_chunks,), dtype=np.int64)
        sample_lens = np.asarray(chunk_sample_lens)
        token_lens = np.asarray(chunk_token_lens)
        counts = np.asarray(audio_chunk_counts)
        x = self.dots_encoder.speech_encoder(input_features, sample_lens, token_lens)
        chunks = [x[i, : int(token_lens[i])] for i in range(x.shape[0])]
        embeds: List[mx.array] = []
        lens: List[int] = []
        off = 0
        # Chunks of one audio are CONCATENATED before the adapter, not padded.
        for cnt in counts.tolist():
            e = mx.concatenate(chunks[off : off + cnt], 0)
            h = self.audio_adapter(e)
            embeds.append(h)
            lens.append(h.shape[0])
            off += cnt
        return mx.concatenate(embeds, 0), np.array(lens)
