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
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger("vmlx_engine")

_REGISTERED = False
_PACKAGE = "mlx_lm.models.glm5_next"
_VENDORED = Path(__file__).resolve().parent / "glm5_next.py"


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
        return True
    try:
        importlib.import_module(_PACKAGE)
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
    _REGISTERED = True
    logger.info("Registered vendored glm5_next runtime (%s)", _VENDORED)
    return True
