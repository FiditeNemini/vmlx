# SPDX-License-Identifier: Apache-2.0
"""Register the source-owned Muse Glimmer runtime.

Muse Glimmer bundles use ``model_type=muse_glimmer`` with a vision+video tower.
Neither mlx-vlm nor mlx-lm ships this architecture, and the bundles carry no
remote code, so vMLX vendors the runtime and installs it under the upstream
``mlx_vlm.models.muse_glimmer`` namespace at VLM-load time.

Idempotent, and defers to upstream if mlx-vlm ever ships native support.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger("vmlx_engine")

_REGISTERED = False
_PACKAGE_NAME = "mlx_vlm.models.muse_glimmer"
_VENDORED_DIR = Path(__file__).resolve().parent / "muse_glimmer"


def muse_glimmer_runtime_available() -> bool:
    """Return True when an upstream or vendored Muse Glimmer runtime exists."""
    if _PACKAGE_NAME in sys.modules:
        return True
    if importlib.util.find_spec(_PACKAGE_NAME) is not None:
        return True
    return (_VENDORED_DIR / "__init__.py").is_file()


def register_muse_glimmer_runtime() -> bool:
    """Install the vendored Muse Glimmer modules into ``sys.modules`` if needed."""
    global _REGISTERED
    if _REGISTERED:
        return True

    try:
        importlib.import_module(_PACKAGE_NAME)
        _REGISTERED = True
        logger.debug("Muse Glimmer runtime already provided by mlx-vlm")
        return False
    except ModuleNotFoundError:
        pass

    init_path = _VENDORED_DIR / "__init__.py"
    if not init_path.is_file():
        return False

    spec = importlib.util.spec_from_file_location(
        _PACKAGE_NAME,
        init_path,
        submodule_search_locations=[str(_VENDORED_DIR)],
    )
    if spec is None or spec.loader is None:
        return False

    module = importlib.util.module_from_spec(spec)
    sys.modules[_PACKAGE_NAME] = module
    try:
        parent = importlib.import_module("mlx_vlm.models")
        setattr(parent, "muse_glimmer", module)
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_PACKAGE_NAME, None)
        raise

    _REGISTERED = True
    logger.info(
        "Registered source-owned Muse Glimmer runtime under "
        "mlx_vlm.models.muse_glimmer"
    )
    return True


_register_muse_glimmer_runtime = register_muse_glimmer_runtime
