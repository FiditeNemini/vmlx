# SPDX-License-Identifier: Apache-2.0
"""Muse Glimmer config.

Every field here was read out of a real bundle's ``config.json`` and
cross-checked against the safetensors shapes; nothing is inherited from a
lookalike family. Muse borrows Gemma's four-norm sandwich and little else — it
has a gated attention output, no q_norm/k_norm, and a vision tower that uses
LayerNorm with bias plus a learned position table rather than RMSNorm + RoPE.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..base import BaseModelConfig


def _rope_theta(params: Any, default: float) -> float:
    """Pull rope_theta out of the nested ``rope_parameters`` block."""
    if isinstance(params, dict):
        rope = params.get("rope_parameters")
        if isinstance(rope, dict) and rope.get("rope_theta") is not None:
            try:
                return float(rope["rope_theta"])
            except (TypeError, ValueError):
                pass
    return default


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "muse_glimmer_text"
    hidden_size: int = 6656
    num_hidden_layers: int = 52
    intermediate_size: int = 19968
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 128
    vocab_size: int = 202048
    max_position_embeddings: int = 131072
    sliding_window: int = 2048
    hidden_activation: str = "silu"
    tie_word_embeddings: bool = False
    attention_bias: bool = False

    rms_norm_eps: float = 1e-5
    # Distinct from rms_norm_eps on purpose — the bundle declares 1e-08 for the
    # post-norm family. Collapsing the two changes the numerics.
    post_norm_eps: float = 1e-8

    # Muse does NOT use 1/sqrt(head_dim). It declares its own QK multiplier;
    # substituting the conventional scale silently degrades quality.
    qk_scale_factor: float = 3.87
    # Applied to the hidden state before the LM head.
    output_multiplier: float = 0.19611613513818404
    final_logit_softcapping: float = 20.0

    rope_theta: float = 500000.0
    # Per-layer theta. The bundle ships 500000 on sliding layers and 0 on
    # full-attention layers; see ``layer_uses_rope``.
    layer_rope_theta: List[float] = field(default_factory=list)
    # "sliding_attention" / "full_attention", one per layer.
    layer_types: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, params):
        config = super().from_dict(params)
        config.rope_theta = _rope_theta(params, config.rope_theta)
        return config

    def layer_is_sliding(self, index: int) -> bool:
        if self.layer_types and index < len(self.layer_types):
            return str(self.layer_types[index]) == "sliding_attention"
        # Fall back to the declared 3:1 cadence: every 4th layer is global.
        return (index + 1) % 4 != 0

    def layer_uses_rope(self, index: int) -> bool:
        """Whether layer ``index`` applies rotary embeddings.

        A per-layer theta of 0 means NO positional encoding on that layer, not
        "inherit the global theta" — the bundle pairs it exclusively with the
        full-attention layers, the interleaved-NoPE arrangement where local
        layers carry position and global layers are left position-free. Treating
        0 as "inherit" would apply rope where the checkpoint expects none.
        """
        if self.layer_rope_theta and index < len(self.layer_rope_theta):
            try:
                return float(self.layer_rope_theta[index]) > 0.0
            except (TypeError, ValueError):
                return True
        return True

    def layer_rope_base(self, index: int) -> float:
        if self.layer_rope_theta and index < len(self.layer_rope_theta):
            try:
                theta = float(self.layer_rope_theta[index])
            except (TypeError, ValueError):
                theta = 0.0
            if theta > 0.0:
                return theta
        return self.rope_theta


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "muse_glimmer_vision"
    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_hidden_layers: int = 50
    num_attention_heads: int = 16
    patch_size: int = 14
    # Frames consumed per temporal patch — 2 is what makes this tower a video
    # encoder rather than a still-image one.
    patch_temporal: int = 2
    # 2x2 spatial merge in the adapter.
    merge_size: int = 2
    pos_emb_height: int = 32
    pos_emb_width: int = 32
    max_position_embeddings: int = 1024
    layer_norm_eps: float = 1e-5
    hidden_act: str = "gelu"
    rope_theta: float = 10000.0
    layer_types: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, params):
        config = super().from_dict(params)
        config.rope_theta = _rope_theta(params, config.rope_theta)
        return config

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def position_table_size(self) -> int:
        return int(self.pos_emb_height) * int(self.pos_emb_width)

    def layer_is_windowed(self, index: int) -> bool:
        if self.layer_types and index < len(self.layer_types):
            return str(self.layer_types[index]) == "window_attention"
        return (index + 1) % 4 != 0


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "muse_glimmer"
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: Optional[VisionConfig] = None

    image_token_id: int = 200092
    video_token_id: int = 200091
    # Adapter input width: vision hidden * merge_size**2.
    out_hidden_size: int = 6144
    projector_hidden_size: int = 4096
    projector_hidden_act: str = "gelu"

    eos_token_id: Optional[List[int]] = None
    quantization: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, params):
        params = dict(params or {})
        text = params.get("text_config")
        vision = params.get("vision_config")
        config = super().from_dict(
            {k: v for k, v in params.items() if k not in ("text_config", "vision_config")}
        )
        config.text_config = TextConfig.from_dict(text or {})
        config.vision_config = (
            VisionConfig.from_dict(vision) if isinstance(vision, dict) else None
        )
        return config
