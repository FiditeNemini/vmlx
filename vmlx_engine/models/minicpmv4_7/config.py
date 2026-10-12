"""Configuration for MiniCPM-V-4.7 (``model_type: minicpmv4_7``).

Mirrors ``transformers.models.minicpmv4_7.configuration_minicpmv4_7`` (5.18+).
The text backbone is an unmodified Qwen3.5 text model: ``qwen3_5_text`` (1B,
dense) or ``qwen3_5_moe_text`` (35B-A3B). Its config class is chosen from the
nested ``text_config.model_type`` and reuses vMLX's vendored Qwen3.5 configs.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from mlx_vlm.models.base import BaseModelConfig


def text_model_is_moe(text_config: Any) -> bool:
    """True for the MoE backbone, read from config metadata (not the name)."""
    if isinstance(text_config, dict):
        model_type = str(text_config.get("model_type") or "")
        experts = text_config.get("num_experts") or 0
    else:
        model_type = str(getattr(text_config, "model_type", "") or "")
        experts = getattr(text_config, "num_experts", 0) or 0
    if model_type in ("qwen3_5_moe_text", "qwen3_5_moe"):
        return True
    if model_type in ("qwen3_5_text", "qwen3_5"):
        return False
    return int(experts) > 0


class TextConfig:
    """Dispatching text config: returns the vendored Qwen3.5 dense/MoE config.

    ``mlx_vlm.utils.load_model`` calls ``TextConfig.from_dict(config["text_config"])``
    on the model module; the concrete dataclass depends on the backbone.
    """

    @classmethod
    def from_dict(cls, params: Dict[str, Any]):
        if text_model_is_moe(params):
            from mlx_vlm.models.qwen3_5_moe.config import TextConfig as MoeTextConfig

            return MoeTextConfig.from_dict(params)
        from mlx_vlm.models.qwen3_5.config import TextConfig as DenseTextConfig

        return DenseTextConfig.from_dict(params)


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "minicpmv4_7_vision"
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 980
    patch_size: int = 14
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    insert_layer_id: int = 6
    window_kernel_size: Tuple[int, int] = (2, 2)

    @property
    def window_hidden_size(self) -> int:
        return self.hidden_size * self.window_kernel_size[0] * self.window_kernel_size[1]

    @property
    def window_intermediate_size(self) -> int:
        return (
            self.intermediate_size
            * self.window_kernel_size[0]
            * self.window_kernel_size[1]
        )

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                key: value
                for key, value in params.items()
                if key in inspect.signature(cls).parameters
            }
        )


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: Any
    vision_config: Any = None
    model_type: str = "minicpmv4_7"
    insert_layer_id: int = 6
    downsample_mode: str = "16x"
    merge_kernel_size: Tuple[int, int] = (2, 2)
    merger_times: int = 1
    image_token_id: int = 248056
    video_token_id: int = 248057
    image_start_id: int = 248078
    image_end_id: int = 248079
    slice_start_id: int = 248088
    slice_end_id: int = 248089
    newline_id: int = 198
    image_token_index: Optional[int] = None
    video_token_index: Optional[int] = None
    tie_word_embeddings: bool = False
    mrope_mode: str = "canvas"
    eos_token_id: Optional[List[int]] = None
    vocab_size: Optional[int] = None
    ignore_index: int = -100
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id
        if self.video_token_index is None:
            self.video_token_index = self.video_token_id
        if isinstance(self.vision_config, dict):
            self.vision_config = VisionConfig.from_dict(self.vision_config)
        elif self.vision_config is None:
            self.vision_config = VisionConfig()
        # HF MiniCPMV4_7Config.__post_init__ copies the top-level value into
        # the vision config (configuration_minicpmv4_7.py, __post_init__).
        self.vision_config.insert_layer_id = int(self.insert_layer_id)
        if self.downsample_mode not in ("16x", "4x"):
            raise ValueError(
                f"MiniCPM-V-4.7 downsample_mode must be '16x' or '4x', got {self.downsample_mode!r}"
            )

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                key: value
                for key, value in params.items()
                if key in inspect.signature(cls).parameters
            }
        )
