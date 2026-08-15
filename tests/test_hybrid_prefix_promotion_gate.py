# SPDX-License-Identifier: Apache-2.0
"""The hybrid restored-prefix promotion switch must stay off unless asked.

Promotion makes a repeated prompt reuse 99.9% instead of 49.4%, but it was
measured to break the model on the first extended turn: 3/3 runs of a 12-turn
script died at turn 2 with a reasoning-only reply and no answer, while the same
script with promotion off completed 12/12. So the default matters more than the
speed, and both env spellings must reach the one helper -- a second name is how
VMLX_NATIVE_MTP once became a silent no-op.
"""

import pytest

from vmlx_engine.mllm_scheduler import _hybrid_prefix_promotion_enabled

_NAMES = ("VMLX_HYBRID_PREFIX_PROMOTION", "VMLINUX_HYBRID_PREFIX_PROMOTION")


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    for name in _NAMES:
        monkeypatch.delenv(name, raising=False)


def test_off_by_default():
    assert _hybrid_prefix_promotion_enabled() is False


@pytest.mark.parametrize("name", _NAMES)
@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "ON"])
def test_either_spelling_opts_in(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    assert _hybrid_prefix_promotion_enabled() is True


@pytest.mark.parametrize("name", _NAMES)
@pytest.mark.parametrize("value", ["0", "false", "off", "", "no"])
def test_falsey_values_stay_off(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    assert _hybrid_prefix_promotion_enabled() is False


def test_the_guard_consults_both_switches():
    """Pin that the skip is gated by BOTH routes, not unconditional.

    Clean store (re-prefill the N-1 key) is checked first because it is the
    safe way to extend a path-dependent chain; promotion (write back the
    restored cache) is the measured-harmful fallback and must stay behind its
    own switch.
    """
    from pathlib import Path

    source = Path("vmlx_engine/mllm_scheduler.py").read_text()
    marker = '"hybrid restored-prefix promotion disabled"'
    window = source[max(0, source.index(marker) - 1400):source.index(marker)]
    assert "_hybrid_clean_store_enabled()" in window
    assert "_hybrid_prefix_promotion_enabled()" in window
    assert 'getattr(self, "_is_hybrid", False)' in window
    # clean store is the preferred route, so it must be tested first
    assert window.index("_hybrid_clean_store_enabled()") < window.index(
        "_hybrid_prefix_promotion_enabled()"
    )


def test_hybrid_joins_the_path_dependent_clean_store_branch():
    """Hybrid must reach the same clean re-prefill route as ZAYA/mixed-SWA."""
    from pathlib import Path

    source = Path("vmlx_engine/mllm_scheduler.py").read_text()
    assert "_uses_hybrid_clean_store" in source
    branch = source.index("or _uses_hybrid_clean_store")
    window = source[max(0, branch - 300):branch]
    assert "_uses_zaya_cache" in window
    assert "_uses_mixed_attention_cache" in window


class TestHybridCleanStoreDefaultsOn:
    """The clean re-prefill route is the DEFAULT for hybrid models.

    It was gated off while the route still crashed mid-store; the half-written
    state that produced is what collapsed the model on the first extended turn.
    With the crash fixed the route is byte-exact on Qwen3.8, so leaving it off
    would silently cost every hybrid conversation its prefix reuse past turn one.
    """

    def _enabled(self, monkeypatch, value):
        import importlib

        module = importlib.import_module("vmlx_engine.mllm_scheduler")
        for name in ("VMLX_HYBRID_CLEAN_STORE", "VMLINUX_HYBRID_CLEAN_STORE"):
            monkeypatch.delenv(name, raising=False)
        if value is not None:
            monkeypatch.setenv("VMLX_HYBRID_CLEAN_STORE", value)
        return module._hybrid_clean_store_enabled()

    def test_on_when_unset(self, monkeypatch):
        assert self._enabled(monkeypatch, None) is True

    def test_explicit_opt_out_is_honoured(self, monkeypatch):
        for value in ("0", "false", "FALSE", "no", "off", "OFF"):
            assert self._enabled(monkeypatch, value) is False, value

    def test_explicit_opt_in_still_works(self, monkeypatch):
        for value in ("1", "true", "yes", "on"):
            assert self._enabled(monkeypatch, value) is True, value

    def test_unrecognised_value_does_not_silently_disable(self, monkeypatch):
        """A typo must not quietly turn the route off and freeze reuse again."""
        assert self._enabled(monkeypatch, "maybe") is True

    def test_alternate_spelling_can_disable(self, monkeypatch):
        import importlib

        module = importlib.import_module("vmlx_engine.mllm_scheduler")
        monkeypatch.delenv("VMLX_HYBRID_CLEAN_STORE", raising=False)
        monkeypatch.setenv("VMLINUX_HYBRID_CLEAN_STORE", "0")
        assert module._hybrid_clean_store_enabled() is False

    def test_promotion_stays_off_by_default(self, monkeypatch):
        """The RESTORED-prefix writeback is a different thing and stays off."""
        import importlib

        module = importlib.import_module("vmlx_engine.mllm_scheduler")
        for name in ("VMLX_HYBRID_PREFIX_PROMOTION", "VMLINUX_HYBRID_PREFIX_PROMOTION"):
            monkeypatch.delenv(name, raising=False)
        assert module._hybrid_prefix_promotion_enabled() is False


def test_hybrid_clean_store_excludes_media_turns():
    """A media prompt must never take the text-only re-prefill route.

    The clean route rebuilds the prefix from token ids alone. On a media turn
    that silently drops the vision embeddings and stores a text-only prefix
    under the media key, so a later hit answers about an image it never saw.
    """
    from pathlib import Path

    source = Path("vmlx_engine/mllm_scheduler.py").read_text()
    marker = "_uses_hybrid_clean_store = bool("
    window = source[source.index(marker):source.index(marker) + 320]
    assert "not media_context" in window, (
        "hybrid clean store must be gated off for media-bearing prompts"
    )
