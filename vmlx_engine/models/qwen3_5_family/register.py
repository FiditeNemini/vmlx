# SPDX-License-Identifier: Apache-2.0
"""Register the vMLX-owned Qwen 3.5 / Qwen 3.5 MoE VLM runtime.

Replaces upstream ``mlx_vlm.models.qwen3_5`` and ``mlx_vlm.models.qwen3_5_moe``
under ``sys.modules`` so both physical mlx-vlm loads AND JANG-VL-bundle loads
(via ``jang_tools.loader`` / ``smelt_loader``) pick up the vendored classes
whose ``quant_predicate`` keeps router gates as float ``nn.Linear``.

The vendored files use relative imports (``from ..base``, ``from ..cache``,
``from ..qwen3_vl``) — those continue to resolve to the upstream mlx-vlm
package because we insert the vendored module UNDER ``mlx_vlm.models``.

Idempotent: safe to call from every loader init path; only patches once.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger("vmlx_engine")

_REGISTERED = False
_VENDORED_DIR = Path(__file__).resolve().parent


def _install_vendored_submodule(pkg_name: str, subdir_name: str) -> bool:
    """Overlay vendored ``subdir_name`` on top of upstream ``pkg_name``.

    Loads the vendored package from disk and puts it in ``sys.modules`` under
    the upstream name. Subsequent ``from mlx_vlm.models.<name> import ...``
    (including the ones inside jang_tools + mlx_vlm itself) will get vMLX.
    """
    init_path = _VENDORED_DIR / subdir_name / "__init__.py"
    if not init_path.is_file():
        logger.debug("qwen3_5_family: vendored %s missing at %s", pkg_name, init_path)
        return False

    # Ensure the mlx_vlm.models parent is import-resolved so relative imports
    # inside the vendored file (from ..base, from ..cache) work at exec_module.
    try:
        parent = importlib.import_module("mlx_vlm.models")
    except ImportError as exc:
        logger.debug("qwen3_5_family: mlx_vlm.models not importable (%s)", exc)
        return False

    spec = importlib.util.spec_from_file_location(
        pkg_name,
        init_path,
        submodule_search_locations=[str(init_path.parent)],
    )
    if spec is None or spec.loader is None:
        return False

    module = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so intra-package `from .language import ...` in the
    # vendored __init__.py resolves via sys.modules.
    sys.modules[pkg_name] = module
    setattr(parent, subdir_name, module)
    try:
        spec.loader.exec_module(module)
    except Exception:
        # Roll back on failure so a later attempt or upstream fallback works.
        sys.modules.pop(pkg_name, None)
        try:
            delattr(parent, subdir_name)
        except AttributeError:
            pass
        raise

    logger.info(
        "Registered vMLX-owned Qwen 3.5 VLM runtime under %s (gate-quant fix)",
        pkg_name,
    )
    return True


def register_qwen3_5_family_runtime() -> bool:
    """Install vendored qwen3_5 + qwen3_5_moe over the upstream namespace.

    Returns True if the vMLX vendor is now active (either newly installed or
    already active from a prior call). False if mlx-vlm itself is missing.
    """
    global _REGISTERED
    if _REGISTERED:
        return True

    # Order matters: qwen3_5 first, because qwen3_5_moe's language.py imports
    # from ..qwen3_5 (which will be looked up as mlx_vlm.models.qwen3_5).
    ok_base = _install_vendored_submodule("mlx_vlm.models.qwen3_5", "qwen3_5")
    if not ok_base:
        return False
    ok_moe = _install_vendored_submodule("mlx_vlm.models.qwen3_5_moe", "qwen3_5_moe")
    if not ok_moe:
        return False

    _REGISTERED = True
    return True


def qwen3_5_family_runtime_available() -> bool:
    """Return True when the vMLX-owned Qwen 3.5 family VLM runtime can load.

    Light check used by routing gates (``api/utils.is_mllm_model`` and the
    model config registry) BEFORE the heavy VLM load path runs. Mirrors
    ``gemma4_unified_register.gemma4_unified_runtime_available``: no vendored
    module execution, just "are the vendored files present and is mlx-vlm
    importable so registration will succeed at load time".
    """
    if _REGISTERED:
        return True
    if not (
        (_VENDORED_DIR / "qwen3_5" / "__init__.py").is_file()
        and (_VENDORED_DIR / "qwen3_5_moe" / "__init__.py").is_file()
    ):
        return False
    try:
        return importlib.util.find_spec("mlx_vlm.models") is not None
    except Exception:
        return False
