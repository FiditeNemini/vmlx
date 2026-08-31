# SPDX-License-Identifier: Apache-2.0
"""Register the source-owned GLM-5.3-Flash (glm5_next) text runtime.

The JANG bundle declares ``model_type=glm5_next``. mlx-lm has no native
package for it, so vMLX vendors the MLX runtime (glm5_next.py + kda.py,
ported from the parity-proven jang_tools glm5_next reference) and installs it
under ``mlx_lm.models.glm5_next`` at load time so the standard loader path
finds it. Idempotent; defers to upstream if mlx-lm ever ships native support.

Created by Jinho Jang (eric@jangq.ai).
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import logging
import sys
import types
from pathlib import Path

logger = logging.getLogger("vmlx_engine")

_REGISTERED = False
_PACKAGE = "mlx_lm.models.glm5_next"
_VLM_PACKAGE = "mlx_vlm.models.glm5_next"
_VENDORED = Path(__file__).resolve().parent / "glm5_next.py"


def _register_prompt_cache_classes() -> None:
    """Expose GLM's typed caches to the generic prompt-disk loader."""

    import mlx_lm.models.cache as mlx_cache

    mod = importlib.import_module(_PACKAGE)
    for name in ("Glm5KDACache", "Glm5MLACache"):
        setattr(mlx_cache, name, getattr(mod, name))


def glm5_next_runtime_available() -> bool:
    if _PACKAGE in sys.modules:
        return True
    if importlib.util.find_spec(_PACKAGE) is not None:
        return True
    return _VENDORED.is_file()


def register_glm5_next_runtime() -> bool:
    """Install the vendored glm5_next module under the mlx-lm namespace."""
    global _REGISTERED
    if _REGISTERED:
        _register_prompt_cache_classes()
        return True
    try:
        importlib.import_module(_PACKAGE)
        _register_prompt_cache_classes()
        _REGISTERED = True
        logger.debug("glm5_next runtime already provided by mlx-lm")
        return False
    except ModuleNotFoundError:
        pass
    if not _VENDORED.is_file():
        return False
    spec = importlib.util.spec_from_file_location(_PACKAGE, _VENDORED)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_PACKAGE] = mod
    spec.loader.exec_module(mod)
    _register_prompt_cache_classes()
    _REGISTERED = True
    logger.info("Registered vendored glm5_next runtime (%s)", _VENDORED)
    return True


def glm5_next_vlm_runtime_available() -> bool:
    try:
        return importlib.util.find_spec("mlx_vlm.models") is not None
    except Exception:
        return False


def register_glm5_next_vlm_runtime() -> bool:
    """Expose the source-owned vision wrapper to mlx-vlm's normal dispatcher."""

    register_glm5_next_runtime()
    if _VLM_PACKAGE in sys.modules:
        return True
    try:
        parent = importlib.import_module("mlx_vlm.models")
    except ImportError:
        return False

    from .config import ModelConfig, TextConfig, VisionConfig
    from .processing import Glm5NextImageProcessor, Glm5NextProcessor
    from .vision import VisionModel
    from .vlm import LanguageModel, Model

    module = types.ModuleType(_VLM_PACKAGE)
    exports = {
        "Model": Model,
        "ModelConfig": ModelConfig,
        "TextConfig": TextConfig,
        "VisionConfig": VisionConfig,
        "LanguageModel": LanguageModel,
        "VisionModel": VisionModel,
        "ImageProcessor": Glm5NextImageProcessor,
        "Processor": Glm5NextProcessor,
    }
    for name, value in exports.items():
        setattr(module, name, value)
    module.__all__ = sorted(exports)
    module.__file__ = str(Path(__file__).resolve())
    module.__package__ = _VLM_PACKAGE
    module.__path__ = []
    module.__spec__ = importlib.machinery.ModuleSpec(
        _VLM_PACKAGE,
        loader=None,
        origin=module.__file__,
        is_package=True,
    )
    sys.modules[_VLM_PACKAGE] = module
    parent.glm5_next = module
    logger.info("Registered source-owned glm5_next MLX-VLM runtime")
    return True
