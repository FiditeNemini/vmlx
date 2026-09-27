"""Opt-in confirmation of a small *total* D1 window loss."""
from dataclasses import replace

import pytest

from vmlx_engine.native_mtp_ar_safety import ArSafetyTrip


def _trip(cost=20.125, cycles=40):
    return ArSafetyTrip(
        cycles=cycles, mtp_ms_per_tok=cost, ar_baseline=20.0,
        seed_ar_ms=20.0, margin=1.0, window=8,
        cycle_median_ms_per_tok=cost, cycle_max_ms_per_tok=cost,
        anchor_cycle_ms=20.0, cur_cycle_ms=cost,
        anchor_context_tokens=100, context_now=140,
    )


def _ring(cost=20.125, start=32, tokens=1):
    return [(start + i, (start + i) * tokens, i * cost * tokens / 1000)
            for i in range(9)]


@pytest.mark.parametrize("cost,tokens,expected", [
    (19.0, 1, False), (20.0, 1, False), (20.125, 1, True),
    (22.0, 1, True), (22.5, 1, False), (23.0, 1, False),
    (21.0, 2, True), (21.5, 2, False), (40.0, 1, False),
])
def test_observed_excess_is_bounded_by_one_ar_step(cost, tokens, expected):
    assert _trip(cost).marginal_loss_needs_confirmation(
        _ring(cost, tokens=tokens), 20.0
    ) is expected


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 0.0, None])
def test_invalid_baseline_cannot_delay_exit(bad):
    assert not _trip().marginal_loss_needs_confirmation(_ring(), bad)


@pytest.mark.parametrize("kind", [
    "empty", "short", "extra", "nan", "backward_time", "same_tokens",
    "same_cycle", "ragged", "none", "string", "infinite_trip",
])
def test_invalid_window_retains_immediate_path(kind):
    ring, trip = _ring(), _trip()
    if kind == "empty": ring = []
    elif kind == "short": ring = ring[:-1]
    elif kind == "extra": ring = ring + [(41, 41, .2)]
    elif kind == "nan": ring[4] = (36, 36, float("nan"))
    elif kind == "backward_time": ring[4] = (36, 36, -1)
    elif kind == "same_tokens": ring[4] = (36, 35, .08)
    elif kind == "same_cycle": ring[4] = (35, 36, .08)
    elif kind == "ragged": ring[4] = (36, 36)
    elif kind == "none": ring = None
    elif kind == "string": ring[4] = (36, 36, "bad")
    else: trip = replace(trip, mtp_ms_per_tok=float("inf"))
    assert not trip.marginal_loss_needs_confirmation(ring, 20.0)


def _state(m, ceiling):
    state = m.MLLMNativeMTPState(depth=1)
    state.depth_ceiling = state.ladder_depth = ceiling
    state.ar_step_ms = 20.0
    state.stats.cycles = 40
    state.last_ar_measure_emitted = 0
    state.ar_tier = m.NativeMTPArTier(depth=ceiling)
    state.ar_tier.step_walls_ms = [20.0] * 16
    state.ar_safety.ring = _ring()
    return state


@pytest.mark.parametrize("ceiling", [1, 2, 3])
@pytest.mark.parametrize("enabled", [False, True])
def test_opt_in_only_and_second_loss_exits(monkeypatch, ceiling, enabled):
    from vmlx_engine import mllm_batch_generator as m
    monkeypatch.setenv("VMLX_MTP_CONFIRM_MARGINAL_LOSS", str(int(enabled)))
    monkeypatch.setattr(m, "_native_mtp_calibration_enabled", lambda: False)
    state = _state(m, ceiling)
    monkeypatch.setattr(m, "ar_safety_step", lambda *a, **k: _trip())
    assert m._native_mtp_maybe_ar_safety_fallback("marginal", state) is not enabled
    assert state.ar_fallback_pending is not enabled
    if enabled:
        assert state.ar_trip_pending_cycle == 40
        assert state.ar_safety.ring == []
        state.stats.cycles = 50
        state.ar_safety.ring = _ring(start=42)
        monkeypatch.setattr(m, "ar_safety_step", lambda *a, **k: _trip(cycles=50))
        assert m._native_mtp_maybe_ar_safety_fallback("second-loss", state)
        assert state.ar_fallback_pending and state.ar_trip_pending_cycle == 0


@pytest.mark.parametrize("cost", [22.5, 40.0])
def test_large_loss_never_waits(monkeypatch, cost):
    from vmlx_engine import mllm_batch_generator as m
    monkeypatch.setenv("VMLX_MTP_CONFIRM_MARGINAL_LOSS", "1")
    monkeypatch.setattr(m, "_native_mtp_calibration_enabled", lambda: False)
    state = _state(m, 1)
    state.ar_safety.ring = _ring(cost)
    monkeypatch.setattr(m, "ar_safety_step", lambda *a, **k: _trip(cost))
    assert m._native_mtp_maybe_ar_safety_fallback("large-loss", state)


def test_recovery_clears_confirmation(monkeypatch):
    from vmlx_engine import mllm_batch_generator as m
    monkeypatch.setenv("VMLX_MTP_CONFIRM_MARGINAL_LOSS", "1")
    monkeypatch.setattr(m, "_native_mtp_calibration_enabled", lambda: False)
    state = _state(m, 1)
    monkeypatch.setattr(m, "ar_safety_step", lambda *a, **k: _trip())
    assert not m._native_mtp_maybe_ar_safety_fallback("marginal", state)
    state.stats.cycles = 50
    state.ar_safety.ring = _ring(18.0, start=42)
    monkeypatch.setattr(m, "ar_safety_step", lambda *a, **k: None)
    assert not m._native_mtp_maybe_ar_safety_fallback("recovered", state)
    assert state.ar_trip_pending_cycle == 0 and not state.ar_fallback_pending
