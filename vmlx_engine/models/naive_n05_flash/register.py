# SPDX-License-Identifier: Apache-2.0
"""Register the vMLX-owned Naive-N0.5-Flash (naive_n05_flash) text runtime under mlx_lm.models.naive_n05_flash.

mlx-lm ships no naive_n05_flash package (NaiveAI publishes transformers remote code only). Idempotent; defers to
upstream if mlx-lm ever ships native support. Created by Jinho Jang (eric@jangq.ai).
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger("vmlx_engine")
_REGISTERED = False
_PACKAGE = "mlx_lm.models.naive_n05_flash"
_VENDORED = Path(__file__).resolve().parent / "naive_n05_flash.py"


def naive_n05_flash_runtime_available() -> bool:
    return _PACKAGE in sys.modules or importlib.util.find_spec(_PACKAGE) is not None or _VENDORED.is_file()


def register_naive_n05_flash_runtime() -> bool:
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        importlib.import_module(_PACKAGE)
        _REGISTERED = True
        return False
    except ModuleNotFoundError as exc:
        if exc.name != _PACKAGE:
            raise
    if not _VENDORED.is_file():
        return False
    spec = importlib.util.spec_from_file_location(_PACKAGE, _VENDORED)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create Naive runtime spec: {_VENDORED}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_PACKAGE] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(_PACKAGE, None)
        raise
    _REGISTERED = True
    logger.info("Registered vendored naive_n05_flash runtime (%s)", _VENDORED)
    return True
