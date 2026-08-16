# SPDX-License-Identifier: Apache-2.0
"""Vendored dots3_note runtime, registered under ``mlx_vlm.models``."""

from .config import AudioConfig, ModelConfig, TextConfig, VisionConfig
from .dots3_note import Model
from .language import LanguageModel

__all__ = [
    "AudioConfig",
    "LanguageModel",
    "Model",
    "ModelConfig",
    "TextConfig",
    "VisionConfig",
]
