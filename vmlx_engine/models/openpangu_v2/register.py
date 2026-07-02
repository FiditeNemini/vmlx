# SPDX-License-Identifier: Apache-2.0
"""Register the source-owned openPangu-2.0-Flash (openpangu_v2) text runtime.

The JANG bundle declares ``model_type=openpangu_v2``. mlx-lm has no native
package for it (HF ships config/tokenizer only; the real forward lives in the
Ascend omni-npu repo), so vMLX vendors the MLX runtime (openpangu_v2.py +
cache.py, ported from the proven vmlx-swift feat/openpangu-v2 implementation)
and installs it under ``mlx_lm.models.openpangu_v2`` at load time so the
standard loader path finds it.

Idempotent; defers to upstream if mlx-lm ever ships native support.

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
_PACKAGE = "mlx_lm.models.openpangu_v2"
_VENDORED = Path(__file__).resolve().parent / "openpangu_v2.py"


def openpangu_v2_runtime_available() -> bool:
    if _PACKAGE in sys.modules:
        return True
    if importlib.util.find_spec(_PACKAGE) is not None:
        return True
    return _VENDORED.is_file()


def register_openpangu_v2_runtime() -> bool:
    """Install the vendored openpangu_v2 module under the mlx-lm namespace."""
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        importlib.import_module(_PACKAGE)
        _REGISTERED = True
        logger.debug("openpangu_v2 runtime already provided by mlx-lm")
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
    logger.info("Registered vendored openpangu_v2 runtime (%s)", _VENDORED)
    return True
