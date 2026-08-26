from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger("vmlx_engine")
_REGISTERED = False
_PACKAGE_DIR = Path(__file__).resolve().parent


def register_qwen4_exp_runtime() -> bool:
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        parent = importlib.import_module("mlx_vlm.models")
    except ImportError as exc:
        logger.debug("qwen4_exp: mlx_vlm.models unavailable: %s", exc)
        return False

    package_name = "mlx_vlm.models.qwen4_exp"
    init_path = _PACKAGE_DIR / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        package_name,
        init_path,
        submodule_search_locations=[str(_PACKAGE_DIR)],
    )
    if spec is None or spec.loader is None:
        return False
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    setattr(parent, "qwen4_exp", module)
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(package_name, None)
        try:
            delattr(parent, "qwen4_exp")
        except AttributeError:
            pass
        raise
    _REGISTERED = True
    logger.info("Registered vMLX-owned qwen4_exp MLX-VLM runtime")
    return True


def qwen4_exp_runtime_available() -> bool:
    if _REGISTERED:
        return True
    if not (_PACKAGE_DIR / "__init__.py").is_file():
        return False
    try:
        return importlib.util.find_spec("mlx_vlm.models") is not None
    except Exception:
        return False
