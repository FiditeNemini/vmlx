"""Sample size required before MTP demotes a request to autoregressive.

Demotion is irreversible for the remainder of the request, so it must not be
decided on the handful of cold cycles that open a request.  Measured on
Qwen3.8-27B-JANG_4D-CRACK, same bundle and same prompt: the first request after
load read 7/12 = 58.3% with a cold MTP cache and demoted the whole request,
while the next request scored 2033/2104 = 96.6% and held MTP for 34.1 t/s.
"""

import mlx.core as mx  # noqa: F401  (import parity with the generator module)
import pytest

from vmlx_engine import mllm_batch_generator as gen


class _Stats:
    def __init__(self, drafted, accepted):
        self.cycles = drafted
        self.drafted_by_depth = [drafted, 0, 0]
        self.accepted_by_depth = [accepted, 0, 0]
        self.mtp_forwards = drafted
        self.accepted_tokens = accepted


class _State:
    def __init__(self, drafted, accepted, depth=1):
        self.depth = depth
        self.stats = _Stats(drafted, accepted)
        self.ar_fallback_pending = False
        self.ar_fallback_reason = None


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for name in (
        "VMLINUX_NATIVE_MTP_ADAPTIVE_WARMUP_CYCLES",
        "VMLX_NATIVE_MTP_ADAPTIVE_WARMUP_CYCLES",
        "VMLINUX_NATIVE_MTP_AR_FALLBACK_MIN_SAMPLE",
        "VMLX_NATIVE_MTP_AR_FALLBACK_MIN_SAMPLE",
        "VMLINUX_NATIVE_MTP_D1_MIN_ACCEPT",
        "VMLX_NATIVE_MTP_D1_MIN_ACCEPT",
        "VMLINUX_NATIVE_MTP_COST_FALLBACK",
        "VMLX_NATIVE_MTP_COST_FALLBACK",
    ):
        monkeypatch.delenv(name, raising=False)


def test_cold_twelve_cycle_window_no_longer_demotes():
    """The exact live case: 7 of 12 accepted must NOT kill the request."""
    state = _State(drafted=12, accepted=7)
    gen._native_mtp_maybe_adapt_depth("req-cold", state)
    assert state.ar_fallback_pending is False


def test_midrange_head_is_kept_under_skip_replay_economics():
    """58.6% stays on MTP: with the replay skipped (default), a rejected
    cycle costs the same as an accepted one, so the floor is ~0.35."""
    state = _State(drafted=483, accepted=283)
    gen._native_mtp_maybe_adapt_depth("req-mid", state)
    assert state.ar_fallback_pending is False


def test_sustained_sub_breakeven_head_still_demotes():
    """A head below even the skip-replay floor must still fall back."""
    state = _State(drafted=483, accepted=140)  # 29.0%
    gen._native_mtp_maybe_adapt_depth("req-weak", state)
    assert state.ar_fallback_pending is True
    assert "d1_acceptance" in (state.ar_fallback_reason or "")


def test_healthy_head_is_never_demoted():
    state = _State(drafted=2104, accepted=2033)  # 96.6%, the warm live run
    gen._native_mtp_maybe_adapt_depth("req-healthy", state)
    assert state.ar_fallback_pending is False


def test_demotion_needs_the_full_sample_not_just_the_warmup():
    """Just past warmup but short of the sample floor must not demote."""
    state = _State(drafted=20, accepted=6)  # 30%, far below the 0.65 floor
    gen._native_mtp_maybe_adapt_depth("req-small", state)
    assert state.ar_fallback_pending is False


def test_sample_floor_is_tunable(monkeypatch):
    monkeypatch.setenv("VMLX_NATIVE_MTP_AR_FALLBACK_MIN_SAMPLE", "16")
    state = _State(drafted=20, accepted=6)
    gen._native_mtp_maybe_adapt_depth("req-tuned", state)
    assert state.ar_fallback_pending is True


def test_boundary_at_the_sample_floor_demotes():
    state = _State(drafted=64, accepted=20)  # 31.3% at exactly the floor
    gen._native_mtp_maybe_adapt_depth("req-boundary", state)
    assert state.ar_fallback_pending is True


def test_one_short_of_the_floor_does_not_demote():
    state = _State(drafted=63, accepted=20)
    gen._native_mtp_maybe_adapt_depth("req-just-under", state)
    assert state.ar_fallback_pending is False


class TestRuntimeCostGate:
    """Wall-clock cost gate: demote when MTP is measurably slower than AR.

    Live case: a dots3 prefix restored from block-disk keeps healthy
    acceptance but MTP decodes at ~12 t/s while plain AR on the same restored
    cache does 35.1. Acceptance gates cannot see that; the cost gate can.
    """

    def _state(self, ar_ms, span_ago_s, cycles, accepted):
        import time

        state = _State(drafted=cycles, accepted=accepted)
        state.ar_step_ms = ar_ms
        state.cycle_span_start = time.perf_counter() - span_ago_s
        return state

    def test_expensive_cycles_fall_back(self):
        # 60 cycles + 40 accepted = 100 tokens over 8s = 80ms/token vs AR 28ms
        state = self._state(ar_ms=28.0, span_ago_s=8.0, cycles=60, accepted=40)
        gen._native_mtp_maybe_adapt_depth("req-cost", state)
        assert state.ar_fallback_pending is True
        assert "runtime_cost" in (state.ar_fallback_reason or "")

    def test_profitable_cycles_stay(self):
        # 100 tokens over 2.5s = 25ms/token vs AR 40ms -> keep MTP
        state = self._state(ar_ms=40.0, span_ago_s=2.5, cycles=60, accepted=40)
        gen._native_mtp_maybe_adapt_depth("req-fast", state)
        assert state.ar_fallback_pending is False

    def test_needs_a_real_sample(self):
        state = self._state(ar_ms=28.0, span_ago_s=8.0, cycles=20, accepted=10)
        gen._native_mtp_maybe_adapt_depth("req-small", state)
        assert state.ar_fallback_pending is False

    def test_no_baseline_no_gate(self):
        """Old states without the seed timing must never trip the gate."""
        state = self._state(ar_ms=0.0, span_ago_s=8.0, cycles=60, accepted=40)
        gen._native_mtp_maybe_adapt_depth("req-nobase", state)
        assert state.ar_fallback_pending is False

    def test_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv("VMLX_NATIVE_MTP_RUNTIME_COST_GATE", "0")
        state = self._state(ar_ms=28.0, span_ago_s=8.0, cycles=60, accepted=40)
        gen._native_mtp_maybe_adapt_depth("req-off", state)
        assert state.ar_fallback_pending is False
