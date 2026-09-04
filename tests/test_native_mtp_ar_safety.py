"""Windowed, context-scaled AR-safety governor — the pure decision math.

The governor drops a request to AR when a WINDOWED MTP ms/token exceeds a
context-scaled AR baseline, for BOTH fixed and adaptive depth. These tests pin
the arithmetic (unit-testable without a model): fast MTP holds, slow MTP trips,
long-context growth does NOT false-trip, and the div-by-small/empty guards.
"""

from vmlx_engine.mllm_batch_generator import _native_mtp_windowed_ar_verdict


def test_fast_mtp_holds():
    # MTP at 5ms/tok, AR seed 10ms, no context growth -> MTP is 2x faster.
    v = _native_mtp_windowed_ar_verdict(
        ar_step_ms=10.0,
        first_verify_ms=12.0,
        window_cycles=16,
        delta_emitted=32,          # 2 tok/cycle
        delta_wall_ms=160.0,       # 5 ms/tok
        delta_verify_ms=12.0 * 16, # verify flat -> scale 1.0
        margin=1.25,
    )
    assert v is None


def test_slow_mtp_trips():
    # MTP at 20ms/tok vs AR 10ms (flat context) -> 2x slower, must trip.
    v = _native_mtp_windowed_ar_verdict(
        ar_step_ms=10.0,
        first_verify_ms=12.0,
        window_cycles=16,
        delta_emitted=16,          # 1 tok/cycle (rejections)
        delta_wall_ms=320.0,       # 20 ms/tok
        delta_verify_ms=12.0 * 16, # flat -> scale 1.0, baseline 10.0
        margin=1.25,
    )
    assert v is not None
    mtp_ms_per_tok, ar_baseline = v
    assert abs(mtp_ms_per_tok - 20.0) < 1e-6
    assert abs(ar_baseline - 10.0) < 1e-6


def test_long_context_does_not_false_trip():
    # Long context: BOTH MTP and AR slowed. MTP now 18ms/tok, but AR has
    # grown too (verify avg 24ms vs first 12ms => scale 2.0 => baseline 20ms).
    # 18 < 20*1.25 -> MTP is still worth it; must NOT trip. This is the
    # context-fairness fix: a fixed short-context AR baseline (10ms) WOULD
    # have wrongly tripped here (18 > 10*1.25).
    v = _native_mtp_windowed_ar_verdict(
        ar_step_ms=10.0,
        first_verify_ms=12.0,
        window_cycles=16,
        delta_emitted=16,
        delta_wall_ms=288.0,       # 18 ms/tok
        delta_verify_ms=24.0 * 16, # verify doubled -> scale 2.0 -> baseline 20
        margin=1.25,
    )
    assert v is None
    # Sanity: against a STALE short-context baseline it would have tripped.
    stale = _native_mtp_windowed_ar_verdict(
        ar_step_ms=10.0,
        first_verify_ms=0.0,       # no scaling -> baseline stays 10
        window_cycles=16,
        delta_emitted=16,
        delta_wall_ms=288.0,
        delta_verify_ms=24.0 * 16,
        margin=1.25,
    )
    assert stale is not None


def test_context_scaled_slow_still_trips():
    # Even with context growth, if MTP is genuinely slower than the scaled
    # baseline it must trip: MTP 30ms/tok vs scaled baseline 20ms -> trip.
    v = _native_mtp_windowed_ar_verdict(
        ar_step_ms=10.0,
        first_verify_ms=12.0,
        window_cycles=16,
        delta_emitted=16,
        delta_wall_ms=480.0,       # 30 ms/tok
        delta_verify_ms=24.0 * 16, # scale 2.0 -> baseline 20
        margin=1.25,
    )
    assert v is not None


def test_baseline_never_below_seed_floor():
    # If verify SHRANK (impossible in practice, but guard), scale clamps to 1
    # so the baseline never drops below the seed AR floor.
    v = _native_mtp_windowed_ar_verdict(
        ar_step_ms=10.0,
        first_verify_ms=100.0,
        window_cycles=16,
        delta_emitted=16,
        delta_wall_ms=200.0,       # 12.5 ms/tok
        delta_verify_ms=10.0 * 16, # verify avg 10 < first 100 -> scale clamps 1
        margin=1.25,
    )
    # baseline = 10 (floored), 12.5 < 10*1.25=12.5 -> NOT strictly greater
    assert v is None
    v2 = _native_mtp_windowed_ar_verdict(
        ar_step_ms=10.0, first_verify_ms=100.0, window_cycles=16,
        delta_emitted=16, delta_wall_ms=201.0, delta_verify_ms=10.0 * 16,
        margin=1.25,
    )
    assert v2 is not None  # 12.5625 > 12.5


def test_guards_return_none():
    base = dict(
        ar_step_ms=10.0, first_verify_ms=12.0, window_cycles=16,
        delta_emitted=16, delta_wall_ms=320.0, delta_verify_ms=192.0,
        margin=1.25,
    )
    assert _native_mtp_windowed_ar_verdict(**{**base, "ar_step_ms": 0.0}) is None
    assert _native_mtp_windowed_ar_verdict(**{**base, "delta_emitted": 0}) is None
    assert _native_mtp_windowed_ar_verdict(**{**base, "delta_wall_ms": 0.0}) is None
    assert _native_mtp_windowed_ar_verdict(**{**base, "window_cycles": 0}) is None


def test_safety_runs_and_trips_under_fixed_policy(monkeypatch):
    # The whole point: this valve fires regardless of depth policy. Simulate a
    # fixed-depth state whose windowed cost is 2x AR and confirm it demotes to
    # AR (fixed policy disables depth adaptation, NOT this safety valve).
    import time

    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.setenv("VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")  # fixed
    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)

    state = m.MLLMNativeMTPState(
        mtp_cache=[], next_main=None, drafts=[], draft_lps=[], draft_ids=[],
        depth=3,
    )
    state.ar_step_ms = 10.0
    state.first_verify_ms = 12.0
    # Prefill the ring so the window is already full and past warmup, with a
    # slow recent window (20 ms/tok, flat verify => baseline 10ms).
    base_t = time.perf_counter() - 1.0
    state.stats.cycles = 40
    state.stats.accepted_tokens = 0
    state.stats.verify_ms = 12.0 * 40
    win = 16
    state.cost_window = [
        (24 + i, 24 + i, 12.0 * (24 + i), base_t + i * 0.020)
        for i in range(win + 1)
    ]
    tripped = m._native_mtp_maybe_ar_safety_fallback("req-fixed", state)
    assert tripped is True
    assert state.ar_fallback_pending is True
    assert state.depth == 1
    assert "windowed_ar_safety d3" in (state.ar_fallback_reason or "")
