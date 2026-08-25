# SPDX-License-Identifier: Apache-2.0
"""dots3_note config.

Every constant here was read out of the real bundle's ``config.json`` and
cross-checked against the safetensors shapes and the conversion handoff
(04-RUNTIME-HANDOFF.md). Two traps this file owns:

1. ``config.json`` is INCOMPLETE — 17 architectural constants are absent
   upstream and come from ``Dots3NoteConfig`` dataclass defaults. The
   defaults below are those values; deleting one produces a wrong-but-
   runnable model, not an exception.
2. ``layer_types`` says ``full_attention`` in the file, but the reference
   config's ``__post_init__`` rewrites those entries to
   ``deepseek_sparse_attention`` when DSA is on. Layer routing here matches
   on "not sliding_attention", never on the literal ``full_attention``.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    # When installed under the mlx_vlm namespace (the serving path) the
    # relative import resolves to mlx_vlm.models.base; direct imports as
    # vmlx_engine.models.dots3_note (tests, tooling) need the absolute form.
    from ..base import BaseModelConfig  # type: ignore[import]
except ImportError:  # pragma: no cover - direct-import lane
    from mlx_vlm.models.base import BaseModelConfig


@dataclass
class AttnGeom:
    """One MLA attention geometry (the model carries TWO — full and SWA)."""

    num_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rope_theta: float
    gate_type: str = "headwise"
    sliding_window: Optional[int] = None  # None => full attention

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def scale(self) -> float:
        return self.qk_head_dim ** -0.5


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "dots3_note"
    hidden_size: int = 5120
    num_hidden_layers: int = 46
    vocab_size: int = 152064
    rms_norm_eps: float = 1e-5
    intermediate_size: int = 13824  # dense layer 0 (and the MTP layer FFN)
    first_k_dense_replace: int = 1
    n_routed_experts: int = 256
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 1536
    n_shared_experts: int = 1
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    # ABSENT from config.json — dataclass defaults (handoff §1).
    # n_group=1 is the load-bearing one: it collapses noaux_tc group routing
    # to a plain top-k over sigmoid(logits)+bias. Do NOT implement
    # DeepSeek-style group-limited routing for this model.
    n_group: int = 1
    topk_group: int = 1
    # Router logits run in the ACTIVATION dtype (bf16 in deployment), not
    # fp32 — fp32 gating shifts top-8 selection on near-ties.
    moe_gating_fp32: bool = False
    use_dsa: bool = True
    use_sliding_window: bool = True
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 524288

    # Full/DSA MLA geometry.
    num_attention_heads: int = 128
    q_lora_rank: int = 1024
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    rope_theta: float = 80000000.0
    attention_gate_type: str = "headwise"

    # SWA MLA geometry. sliding_window=513 is odd on purpose: self + 512
    # past (HF semantics q_idx - kv_idx < window).
    swa_num_attention_heads: int = 64
    swa_q_lora_rank: int = 1024
    swa_kv_lora_rank: int = 1024
    swa_qk_nope_head_dim: int = 192
    swa_qk_rope_head_dim: int = 64
    swa_v_head_dim: int = 128
    swa_rope_theta: float = 50000.0
    swa_attention_gate_type: str = "headwise"
    sliding_window_size: int = 513

    apply_mla_qkv_lora_rescale: bool = True

    # DSA indexer (full layers only; dense-equivalent at seq <= index_topk).
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 2048

    layer_types: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.layer_types:
            # Full/DSA at {0, 1, then every 4th}; all others sliding.
            full = {0, 1} | set(range(5, self.num_hidden_layers, 4))
            self.layer_types = [
                "full_attention" if i in full else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]

    def is_sliding(self, layer_idx: int) -> bool:
        if layer_idx >= len(self.layer_types):
            return False  # MTP layer 46 uses the full-attention geometry
        return self.layer_types[layer_idx] == "sliding_attention"

    def is_moe(self, layer_idx: int) -> bool:
        if layer_idx >= self.num_hidden_layers:
            return False  # MTP layer FFN is dense
        return layer_idx >= self.first_k_dense_replace

    def swa_geom(self) -> AttnGeom:
        return AttnGeom(
            num_heads=self.swa_num_attention_heads,
            q_lora_rank=self.swa_q_lora_rank,
            kv_lora_rank=self.swa_kv_lora_rank,
            qk_nope_head_dim=self.swa_qk_nope_head_dim,
            qk_rope_head_dim=self.swa_qk_rope_head_dim,
            v_head_dim=self.swa_v_head_dim,
            rope_theta=self.swa_rope_theta,
            gate_type=self.swa_attention_gate_type,
            sliding_window=self.sliding_window_size,
        )

    def full_geom(self) -> AttnGeom:
        return AttnGeom(
            num_heads=self.num_attention_heads,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            rope_theta=self.rope_theta,
            gate_type=self.attention_gate_type,
            sliding_window=None,
        )

    def geom(self, layer_idx: int) -> AttnGeom:
        return self.swa_geom() if self.is_sliding(layer_idx) else self.full_geom()


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "dots3_note_vision"
    embed_dim: int = 1536
    hidden_size: int = 5120
    intermediate_size: int = 4224
    moe_intermediate_size: int = 2112
    num_hidden_layers: int = 42
    num_attention_heads: int = 24
    num_channels: int = 3
    patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 1
    rms_norm_eps: float = 1e-5
    use_bias: bool = False
    use_qk_norm: bool = True
    is_causal: bool = False
    post_norm: bool = True
    pre_pixel_shuffle: bool = True
    # Blocks 0-24 dense (-1); 25-41 pyramid MoE with this many routed
    # experts each (top-2, sigmoid router with a persistent f32 bias buffer).
    pyramid_num_routed: List[int] = field(default_factory=list)
    capacity_factor: float = 2.0
    router_scoring_func: str = "sigmoid"
    router_scale: float = 1.0
    adapter_type: str = "patch_merger"
    adapter_in_dim: int = 1536
    adapter_out_dim: int = 5120
    adapter_merge_size: int = 2

    def __post_init__(self):
        if not self.pyramid_num_routed:
            self.pyramid_num_routed = [-1] * 25 + [
                4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 64
            ]


@dataclass
class AudioConfig(BaseModelConfig):
    model_type: str = "dots3_note_audio"
    encoder_type: str = "dots"
    d_model: int = 1280
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    encoder_layers: int = 32
    num_mel_bins: int = 128
    max_source_positions: int = 6000
    activation_function: str = "swiglu"
    use_conv2d_stem: bool = True
    use_rope: bool = True
    use_rms_norm: bool = True
    use_causal: bool = False
    partial_rotary_factor: float = 0.5
    rope_theta: float = 10000.0
    merge_factor: int = 1
    chunk_seconds: int = 60
    sampling_rate: int = 16000
    adapter_in_dim: int = 1280
    adapter_out_dim: int = 5120

    @classmethod
    def from_dict(cls, params):
        params = dict(params or {})
        whisper = params.pop("whisper_config", None)
        if isinstance(whisper, dict):
            for key in (
                "d_model",
                "encoder_attention_heads",
                "encoder_ffn_dim",
                "encoder_layers",
                "num_mel_bins",
                "max_source_positions",
                "activation_function",
            ):
                if key in whisper:
                    params.setdefault(key, whisper[key])
        rope = params.pop("rope_parameters", None)
        if isinstance(rope, dict):
            if "partial_rotary_factor" in rope:
                params.setdefault(
                    "partial_rotary_factor", rope["partial_rotary_factor"]
                )
            if "rope_theta" in rope:
                params.setdefault("rope_theta", rope["rope_theta"])
        params.setdefault(
            "adapter_in_dim", params.pop("whisper_adapter_in_dim", 1280)
        )
        params.setdefault(
            "adapter_out_dim", params.pop("whisper_adapter_out_dim", 5120)
        )
        return super().from_dict(params)


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: Optional[VisionConfig] = None
    audio_config: Optional[AudioConfig] = None
    model_type: str = "dots3_note"
    # Media placeholder ids. Image and video share one prompt-ordered visual
    # feature buffer, but retain native IDs so mixed media remains identifiable
    # to the language model. Audio uses its own feature path and ID.
    image_token_id: int = 151660
    video_token_id: int = 151680
    audio_token_id: int = 151720
    bos_token_id: int = 151643
    eos_token_id: List[int] = field(default_factory=lambda: [151643, 151668])
    pad_token_id: int = 151659
    vocab_size: int = 152064

    @classmethod
    def from_dict(cls, params):
        params = dict(params or {})
        # The bundle stores the LM fields at TOP level (no text_config
        # wrapper) with vision_config/audio_config nested beside them.
        text = TextConfig.from_dict(params)
        vision = params.get("vision_config")
        audio = params.get("audio_config")
        eos = params.get("eos_token_id", [151643, 151668])
        if isinstance(eos, int):
            eos = [eos]
        config = cls(
            text_config=text,
            vision_config=(
                VisionConfig.from_dict(vision)
                if isinstance(vision, dict)
                else None
            ),
            audio_config=(
                AudioConfig.from_dict(audio) if isinstance(audio, dict) else None
            ),
            model_type=params.get("model_type", "dots3_note"),
            bos_token_id=params.get("bos_token_id", 151643),
            eos_token_id=list(eos),
            vocab_size=params.get("vocab_size", 152064),
        )
        return config
