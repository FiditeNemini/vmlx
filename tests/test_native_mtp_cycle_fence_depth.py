"""Per-cycle device fence defaults by draft depth.

Measured live in the app on Qwen3.8-27B-JANG_4D-CRACK, 64 prompt, fresh chat,
IDENTICAL 4139-token output both arms: fenced 26.5 t/s vs unfenced 37.9/37.5
t/s. A depth-1 cycle issues one draft forward and one 2-token verify, so there
is virtually no lazy work for the barrier to bound - only its cost lands, every
cycle. At depth 2 the fence is a measured win (35B MXFP8, 1.45x -> 1.68x), so
the default must split on depth rather than flip globally.
"""

import importlib

import pytest


def _reload_with(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("VMLX_MTP_CYCLE_FENCE", raising=False)
    else:
        monkeypatch.setenv("VMLX_MTP_CYCLE_FENCE", value)
    import vmlx_engine.mllm_batch_generator as gen

    return importlib.reload(gen)


@pytest.fixture
def unset_env(monkeypatch):
    return _reload_with(monkeypatch, None)


class TestDepthDefault:
    def test_depth_one_is_fenced_by_default(self, unset_env):
        """Default ON at depth 1 too: the exemption re-exposed the 2026-08-15
        lazy-accumulation stall on dots3 (43.5 -> 11.6 t/s at identical 89%
        acceptance, avg_cycle 40 -> 162ms)."""
        assert unset_env._native_mtp_cycle_fence_enabled(1) is True

    def test_depth_two_is_fenced_by_default(self, unset_env):
        assert unset_env._native_mtp_cycle_fence_enabled(2) is True

    def test_depth_three_is_fenced_by_default(self, unset_env):
        assert unset_env._native_mtp_cycle_fence_enabled(3) is True

    def test_zero_and_none_depth_are_treated_as_one(self, unset_env):
        assert unset_env._native_mtp_cycle_fence_enabled(0) is True
        assert unset_env._native_mtp_cycle_fence_enabled(None) is True


class TestExplicitOverrideWins:
    def test_explicit_off_disables_even_at_depth_two(self, monkeypatch):
        gen = _reload_with(monkeypatch, "0")
        assert gen._native_mtp_cycle_fence_enabled(2) is False
        assert gen._native_mtp_cycle_fence_enabled(1) is False

    def test_explicit_on_enables_even_at_depth_one(self, monkeypatch):
        gen = _reload_with(monkeypatch, "1")
        assert gen._native_mtp_cycle_fence_enabled(1) is True

    @pytest.mark.parametrize("value", ["off", "false", "no", "OFF", "False"])
    def test_falsey_spellings_disable(self, monkeypatch, value):
        gen = _reload_with(monkeypatch, value)
        assert gen._native_mtp_cycle_fence_enabled(2) is False

    @pytest.mark.parametrize("value", ["on", "true", "yes", "ON", "True"])
    def test_truthy_spellings_enable(self, monkeypatch, value):
        gen = _reload_with(monkeypatch, value)
        assert gen._native_mtp_cycle_fence_enabled(1) is True

    def test_unrecognised_value_falls_back_to_the_depth_default(self, monkeypatch):
        """A typo must not silently pin the fence one way."""
        gen = _reload_with(monkeypatch, "maybe")
        assert gen._native_mtp_cycle_fence_enabled(1) is True
        assert gen._native_mtp_cycle_fence_enabled(2) is True


def test_module_reloads_clean(monkeypatch):
    """Leave the module in its unset-env state for the rest of the suite."""
    gen = _reload_with(monkeypatch, None)
    assert gen._native_mtp_cycle_fence_enabled(1) is True
