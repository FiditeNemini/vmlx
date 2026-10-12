"""MiniCPM-V-4.7 MLX runtime (registered as ``mlx_vlm.models.minicpmv4_7``)."""

from mlx_vlm.models.base import install_auto_processor_patch

from .config import ModelConfig, TextConfig, VisionConfig
from .language import language_model_class
from .minicpmv4_7 import Model
from .processing import MiniCPMV47ProcessorLoader
from .vision import VisionModel

install_auto_processor_patch("minicpmv4_7", MiniCPMV47ProcessorLoader)

__all__ = [
    "Model",
    "ModelConfig",
    "TextConfig",
    "VisionConfig",
    "VisionModel",
    "language_model_class",
]
