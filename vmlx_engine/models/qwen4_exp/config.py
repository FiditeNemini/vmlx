import inspect
from dataclasses import dataclass
from typing import List, Optional

from mlx_vlm.models.base import BaseModelConfig
from mlx_vlm.models.qwen3_vl.config import VisionConfig as Qwen3VLVisionConfig

from .language import Qwen4ExpTextArgs as TextConfig


@dataclass
class VisionConfig(Qwen3VLVisionConfig):
    model_type: str = "qwen4_exp"


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str = "qwen4_exp"
    ignore_index: int = -100
    image_token_id: int = 248056
    video_token_id: int = 248057
    image_token_index: Optional[int] = None
    video_token_index: Optional[int] = None
    vision_start_token_id: int = 248045
    vision_end_token_id: int = 248046
    vocab_size: int = 248320
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id
        if self.video_token_index is None:
            self.video_token_index = self.video_token_id

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                key: value
                for key, value in params.items()
                if key in inspect.signature(cls).parameters
            }
        )
