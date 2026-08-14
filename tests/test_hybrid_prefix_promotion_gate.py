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


def test_the_guard_still_consults_the_switch():
    """Pin that the skip is gated, not unconditional -- and vice versa."""
    from pathlib import Path

    source = Path("vmlx_engine/mllm_scheduler.py").read_text()
    guard = source.index('_skip_cache_store_reason = (\n                        "hybrid restored-prefix promotion disabled"')
    window = source[max(0, guard - 900):guard]
    assert "_hybrid_prefix_promotion_enabled()" in window
    assert 'getattr(self, "_is_hybrid", False)' in window
