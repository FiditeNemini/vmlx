from mlx_vlm.models.base import install_auto_processor_patch
from mlx_vlm.models.qwen3_vl import processing_qwen3_vl  # noqa: F401
from .processing import Qwen4Processor

from .config import ModelConfig, TextConfig, VisionConfig
from .language import LanguageModel
from .qwen4_exp import Model
from .vision import VisionModel

install_auto_processor_patch("qwen4_exp", Qwen4Processor)

__all__ = [
    "LanguageModel",
    "Model",
    "ModelConfig",
    "TextConfig",
    "VisionConfig",
    "VisionModel",
]
