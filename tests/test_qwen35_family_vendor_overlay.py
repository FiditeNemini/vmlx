"""The vendored qwen3_5 overlay must survive upstream-first import order.

qwen4_exp's language.py imports mlx_vlm.models.qwen3_5.language (for
GatedDeltaNet). When that happened before register_qwen3_5_family_runtime(),
the stale upstream SUBMODULE entries in sys.modules made the vendored
__init__'s `from .language import ...` bind the UPSTREAM classes — a
half-vendored runtime with the gate-quant fix silently absent. Live
signature: chunked-prefill bitwise equivalence failed ("layer N keys
differ") in any process that imported qwen4_exp first.
"""

import sys
from pathlib import Path

import pytest


VENDORED_DIR = (
    Path(__file__).resolve().parents[1] / "vmlx_engine" / "models" / "qwen3_5_family"
)


def _snapshot(prefixes):
    return {
        key: module
        for key, module in sys.modules.items()
        if any(key == p or key.startswith(p + ".") for p in prefixes)
    }


def _restore(prefixes, saved):
    for key in [
        k for k in sys.modules
        if any(k == p or k.startswith(p + ".") for p in prefixes)
    ]:
        sys.modules.pop(key, None)
    sys.modules.update(saved)


def test_upstream_first_import_still_yields_fully_vendored_runtime():
    pytest.importorskip("mlx_vlm")
    from vmlx_engine.models.qwen3_5_family import register as reg

    prefixes = ["mlx_vlm.models.qwen3_5", "mlx_vlm.models.qwen3_5_moe"]
    saved = _snapshot(prefixes)
    saved_flag = reg._REGISTERED
    try:
        # Simulate qwen4_exp importing upstream first.
        _restore(prefixes, {})
        import importlib

        upstream_lang = importlib.import_module("mlx_vlm.models.qwen3_5.language")
        assert "site-packages" in str(getattr(upstream_lang, "__file__", ""))

        reg._REGISTERED = False
        assert reg.register_qwen3_5_family_runtime() is True

        pkg = sys.modules["mlx_vlm.models.qwen3_5"]
        assert str(VENDORED_DIR) in str(pkg.__file__), pkg.__file__
        # The class bound by the vendored __init__ must come from the vendored
        # language module, not the stale upstream one.
        lang_mod = sys.modules.get("mlx_vlm.models.qwen3_5.language")
        assert lang_mod is not None
        assert str(VENDORED_DIR) in str(lang_mod.__file__), lang_mod.__file__
        assert pkg.LanguageModel.__module__.startswith("mlx_vlm.models.qwen3_5")
        assert sys.modules[pkg.LanguageModel.__module__] is lang_mod
    finally:
        reg._REGISTERED = saved_flag
        _restore(prefixes, saved)
