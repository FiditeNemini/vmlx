# SPDX-License-Identifier: Apache-2.0
"""Register the source-owned dots3_note runtime.

dots3_note bundles use ``model_type=dots3_note`` (280B/16B-active omni MoE,
hybrid dual-geometry MLA + SWA, DSA indexer, MTP layer 46, vision + audio
towers). Neither mlx-vlm nor mlx-lm nor the installed transformers ships
this architecture (transformers PR #47844 is unreleased at 5.7.0), so vMLX
vendors the runtime and installs it under ``mlx_vlm.models.dots3_note`` at
load time.

Idempotent. Unlike Muse there is no upstream package to defer to — if a
future mlx-vlm ships one, revisit deliberately rather than auto-preferring
it: this port carries the dual-geometry MLA, the literal (non-centered)
norm contract, and the video-rides-image-tokens scatter, none of which an
upstream lookalike is guaranteed to have.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger("vmlx_engine")

_REGISTERED = False
_PACKAGE_NAME = "mlx_vlm.models.dots3_note"
_VENDORED_DIR = Path(__file__).resolve().parent / "dots3_note"


def dots3_note_runtime_available() -> bool:
    if _PACKAGE_NAME in sys.modules:
        return True
    return (_VENDORED_DIR / "__init__.py").is_file()


def register_dots3_note_runtime() -> bool:
    """Install the vendored dots3_note modules into ``sys.modules``."""
    global _REGISTERED
    if _REGISTERED:
        return True

    init_path = _VENDORED_DIR / "__init__.py"
    if not init_path.is_file():
        return False

    for name in [
        n
        for n in sys.modules
        if n == _PACKAGE_NAME or n.startswith(_PACKAGE_NAME + ".")
    ]:
        sys.modules.pop(name, None)

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
        setattr(parent, "dots3_note", module)
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_PACKAGE_NAME, None)
        raise

    _REGISTERED = True
    logger.info(
        "Registered source-owned dots3_note runtime under mlx_vlm.models.dots3_note"
    )
    return True
