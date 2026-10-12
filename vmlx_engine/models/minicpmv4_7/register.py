"""Install vMLX's MiniCPM-V-4.7 runtime as ``mlx_vlm.models.minicpmv4_7``.

mlx-vlm 0.5 has no ``minicpmv4_7`` module; ``mlx_vlm.utils.get_model_and_args``
imports ``mlx_vlm.models.<model_type>``. The vendored Qwen3.5 family is
registered first because the text backbone is imported from it.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger("vmlx_engine")
_REGISTERED = False
_PACKAGE_DIR = Path(__file__).resolve().parent
PACKAGE_NAME = "mlx_vlm.models.minicpmv4_7"


def register_minicpmv4_7_runtime() -> bool:
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        parent = importlib.import_module("mlx_vlm.models")
    except ImportError as exc:
        logger.debug("minicpmv4_7: mlx_vlm.models unavailable: %s", exc)
        return False

    from ..qwen3_5_family import register_qwen3_5_family_runtime

    if not register_qwen3_5_family_runtime():
        raise RuntimeError("minicpmv4_7 needs the vMLX Qwen3.5 family runtime")

    # The package only uses absolute ``mlx_vlm.models.*`` imports, so it is
    # aliased rather than executed a second time under the mlx_vlm name: one
    # module object, one set of classes (isinstance-safe), one processor hook.
    module = importlib.import_module("vmlx_engine.models.minicpmv4_7")
    sys.modules[PACKAGE_NAME] = module
    setattr(parent, "minicpmv4_7", module)
    _REGISTERED = True
    logger.info("Registered vMLX-owned minicpmv4_7 MLX-VLM runtime")
    return True


def minicpmv4_7_runtime_available() -> bool:
    if _REGISTERED:
        return True
    if not (_PACKAGE_DIR / "__init__.py").is_file():
        return False
    try:
        return importlib.util.find_spec("mlx_vlm.models") is not None
    except Exception:
        return False
