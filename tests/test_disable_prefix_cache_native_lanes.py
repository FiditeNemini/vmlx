"""--disable-prefix-cache reaches the DFlash2 lanes' own prefix reuse (ledger T-NAX-d)."""
import logging
from types import SimpleNamespace

from vmlx_engine.cli import _apply_disable_prefix_cache_to_native_lanes


def test_flag_turns_off_glm_store_and_qwen_session_reuse(monkeypatch):
    for name in ("VMLX_GLM_DFLASH2_PREFIX_CACHE", "VMLX_DFLASH2_PREFIX_REUSE"):
        monkeypatch.delenv(name, raising=False)
    _apply_disable_prefix_cache_to_native_lanes(SimpleNamespace(disable_prefix_cache=True), logging.getLogger("t"))
    import os
    assert os.environ["VMLX_GLM_DFLASH2_PREFIX_CACHE"] == "0"
    assert os.environ["VMLX_DFLASH2_PREFIX_REUSE"] == "0"
    from vmlx_engine.dflash2_runtime import _prefix_reuse_enabled
    assert _prefix_reuse_enabled() is False


def test_no_flag_or_explicit_env_is_left_alone(monkeypatch):
    import os
    monkeypatch.delenv("VMLX_GLM_DFLASH2_PREFIX_CACHE", raising=False)
    monkeypatch.setenv("VMLX_DFLASH2_PREFIX_REUSE", "1")
    _apply_disable_prefix_cache_to_native_lanes(SimpleNamespace(disable_prefix_cache=False), logging.getLogger("t"))
    assert "VMLX_GLM_DFLASH2_PREFIX_CACHE" not in os.environ
    _apply_disable_prefix_cache_to_native_lanes(SimpleNamespace(disable_prefix_cache=True), logging.getLogger("t"))
    assert os.environ["VMLX_DFLASH2_PREFIX_REUSE"] == "1"            # explicit setting wins
    assert os.environ["VMLX_GLM_DFLASH2_PREFIX_CACHE"] == "0"
