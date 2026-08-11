# SPDX-License-Identifier: Apache-2.0
"""Vendored Muse Glimmer runtime, registered under ``mlx_vlm.models``."""

from .config import ModelConfig, TextConfig, VisionConfig
from .language import LanguageModel
from .muse_glimmer import Model
from .processor import (
    MuseGlimmerProcessor,
    MuseGlimmerImageProcessor,
    MuseGlimmerVideoProcessor,
    expand_media_placeholders,
    merged_token_count,
)
from .vision import MuseVisionAdapter, VisionModel

# mlx_vlm's load_image_processor looks for `ImageProcessor` on the model module
# and instantiates it directly. Without this alias the bundle's declared
# MuseGlimmerImageProcessor resolves to nothing, AutoProcessor degrades to a
# text-only processor, and images are dropped with no error anywhere.
ImageProcessor = MuseGlimmerImageProcessor

__all__ = [
    "ImageProcessor",
    "LanguageModel",
    "Model",
    "ModelConfig",
    "MuseGlimmerImageProcessor",
    "MuseGlimmerVideoProcessor",
    "MuseVisionAdapter",
    "TextConfig",
    "VisionConfig",
    "VisionModel",
    "MuseGlimmerProcessor",
    "expand_media_placeholders",
    "merged_token_count",
]
