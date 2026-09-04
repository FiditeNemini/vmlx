"""Windowed AR-safety governor — shared decision math + both lanes.

The governor drops a request to AR when a WINDOWED MTP ms/token exceeds a
scaled AR baseline, for BOTH fixed and adaptive depth.  The scale is the
cycle-wall growth since the warmup anchor, in both directions, so batching,
thermal and context move the baseline with the measurement.
"""

import time

from vmlx_engine.native_mtp_ar_safety import (
    ArSafetyState,
    ar_safety_step,
    windowed_ar_verdict as _native_mtp_windowed_ar_verdict,
)

BASE = dict(
    ar_step_ms=10.0,
    anchor_cycle_ms=30.0,
    cur_cycle_ms=30.0,   # flat -> scale 1.0
    context_ratio=1.0,
    delta_emitted=32,
    delta_wall_ms=160.0,  # 5 ms/tok
    margin=1.25,
)


def verdict(**over):
    return _native_mtp_windowed_ar_verdict(**{**BASE, **over})


def test_fast_mtp_holds():
    assert verdict() is None


def test_slow_mtp_trips():
    v = verdict(delta_emitted=16, delta_wall_ms=320.0)  # 20 ms/tok vs 10
    assert v is not None
    mtp_ms_per_tok, ar_baseline = v
    assert abs(mtp_ms_per_tok - 20.0) < 1e-6
    assert abs(ar_baseline - 10.0) < 1e-6


def test_acceptance_collapse_trips_without_cycle_wall_change():
    # Cycle wall unchanged (30ms), tokens/cycle fell from 3.2 to 1.0 ->
    # 30 ms/tok vs baseline 10 x 1.25 -> trips. Never masked by the scale.
    v = verdict(delta_emitted=16, delta_wall_ms=480.0, cur_cycle_ms=30.0)
    assert v is not None


def test_batched_join_does_not_false_trip():
    # A second request joined: wall-clock per cycle doubled (30 -> 60) and
    # so did ms/tok (5 -> 10). AR for this request would also have doubled;
    # the scale follows (baseline 20) -> hold.
    v = verdict(cur_cycle_ms=60.0, delta_emitted=32, delta_wall_ms=320.0)
    assert v is None
    # ...but a losing request under batching still trips: 30 ms/tok vs 25.
    v2 = verdict(cur_cycle_ms=60.0, delta_emitted=16, delta_wall_ms=480.0)
    assert v2 is not None


def test_batched_leave_scales_baseline_down():
    # Anchor taken under batch-2 (60ms cycles); the other request left, the
    # cycle wall halved (30). A losing solo request (15 ms/tok) must trip
    # against the scaled-down baseline (10 x 0.5 = 5 -> 6.25 threshold).
    v = verdict(anchor_cycle_ms=60.0, cur_cycle_ms=30.0,
                delta_emitted=16, delta_wall_ms=240.0)
    assert v is not None
    _, ar_baseline = v
    assert abs(ar_baseline - 5.0) < 1e-6


def test_long_context_is_conservative_not_lax():
    # Context grew: AR went 10 -> 20 (absolute +10), cycle wall 30 -> 40
    # (same absolute +10, smaller ratio). Scale = 1.33 -> baseline 13.3,
    # below true AR (20): the valve can demote EARLY (MTP at 18 trips even
    # though it beats true AR) but never LATE (MTP at 30 > 20 trips too).
    early = verdict(cur_cycle_ms=40.0, delta_emitted=16, delta_wall_ms=288.0)
    assert early is not None
    late = verdict(cur_cycle_ms=40.0, delta_emitted=16, delta_wall_ms=480.0)
    assert late is not None
    # And genuinely fast MTP at long context holds.
    fast = verdict(cur_cycle_ms=40.0, delta_emitted=32, delta_wall_ms=320.0)
    assert fast is None


def test_single_stall_cycle_does_not_trip():
    per_cycle = [5.0] * 15 + [400.0]
    v = verdict(delta_emitted=16, delta_wall_ms=sum(per_cycle),
                per_cycle_ms_per_tok=per_cycle)
    assert v is None
    sustained = [sum(per_cycle) / 16] * 16
    v2 = verdict(delta_emitted=16, delta_wall_ms=sum(sustained),
                 per_cycle_ms_per_tok=sustained)
    assert v2 is not None


def test_guards_return_none():
    assert verdict(ar_step_ms=0.0, delta_emitted=16, delta_wall_ms=320.0) is None
    assert verdict(delta_emitted=0, delta_wall_ms=320.0) is None
    assert verdict(delta_emitted=16, delta_wall_ms=0.0) is None


def _drive(st, *, cycle_ms, tok_per_cycle, seed_ar_ms=10.0, primed=True,
           start_cycle=1, end_cycle=60, t0=0.0, emitted0=0):
    """Feed the governor a synthetic run; returns (trip_cycle, trip)."""
    t = t0
    emitted = emitted0
    for cyc in range(start_cycle, end_cycle):
        t += cycle_ms(cyc) / 1000.0
        emitted += tok_per_cycle(cyc)
        trip = ar_safety_step(st, cycles=cyc, emitted=emitted, now=t,
                              seed_ar_ms=seed_ar_ms, primed=primed)
        if trip is not None:
            return cyc, trip, t, emitted
    return None, None, t, emitted


def test_first_judgment_cycle_primed_vs_unprimed(monkeypatch):
    for name in ("VMLX_NATIVE_MTP_AR_SAFETY_WARMUP", "VMLX_NATIVE_MTP_AR_SAFETY_WINDOW"):
        monkeypatch.delenv(name, raising=False)
    # 20 ms/cycle, 1 tok/cycle = 20 ms/tok vs AR 10 -> losing from the start.
    c, *_ = _drive(ArSafetyState(prompt_tokens=50), cycle_ms=lambda c: 20.0,
                   tok_per_cycle=lambda c: 1, primed=True)
    assert c == 16  # warmup 8 + window 8
    c2, *_ = _drive(ArSafetyState(prompt_tokens=50), cycle_ms=lambda c: 20.0,
                    tok_per_cycle=lambda c: 1, primed=False)
    assert c2 == 24  # warmup 16 + window 8


def test_anchor_is_warmup_median_and_join_is_absorbed(monkeypatch):
    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY_WARMUP", raising=False)
    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY_WINDOW", raising=False)
    st = ArSafetyState(prompt_tokens=50)
    # Healthy solo: 30ms cycles, 3 tok/cycle (10 ms/tok = AR) for 30 cycles,
    # then a second request joins: 60ms cycles, still 3 tok/cycle.
    c, trip, *_ = _drive(st, cycle_ms=lambda c: 30.0 if c <= 30 else 60.0,
                         tok_per_cycle=lambda c: 3, end_cycle=80)
    assert abs(st.anchor_cycle_ms - 30.0) < 1e-6
    assert c is None, (c, trip and trip.log_text(3))


def test_acceptance_collapse_under_batch_still_trips(monkeypatch):
    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY_WARMUP", raising=False)
    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY_WINDOW", raising=False)
    st = ArSafetyState(prompt_tokens=50)
    # Batched from the start (60ms cycles, seed also measured batched = 20ms):
    # healthy 3 tok/cycle, then acceptance collapses to 1 tok/cycle at 40.
    c, trip, *_ = _drive(st, cycle_ms=lambda c: 60.0,
                         tok_per_cycle=lambda c: 3 if c < 40 else 1,
                         seed_ar_ms=20.0, end_cycle=80)
    assert c is not None and 40 < c <= 50


def _vlm_state(m, depth=3):
    state = m.MLLMNativeMTPState(
        mtp_cache=[], next_main=None, drafts=[], draft_lps=[], draft_ids=[],
        depth=depth,
    )
    state.ar_step_ms = 10.0
    state.ar_safety.prompt_tokens = 100
    state.ar_safety.anchor_cycle_ms = 20.0
    state.ar_safety.anchor_context_tokens = 120
    return state


def test_safety_runs_and_trips_under_fixed_policy(monkeypatch):
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.setenv("VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")  # fixed
    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    state = _vlm_state(m)
    base_t = time.perf_counter() - 1.0
    state.stats.cycles = 40
    state.stats.accepted_tokens = 0
    # Full ring, 1 tok/cycle at 20ms/cycle = 20 ms/tok, flat cycle wall.
    state.ar_safety.ring = [(31 + i, 31 + i, base_t + i * 0.020) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req-fixed", state) is True
    assert state.ar_fallback_pending is True
    assert state.depth == 1
    assert "windowed_ar_safety d3" in (state.ar_fallback_reason or "")


def test_disabled_by_kill_switch(monkeypatch):
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.setenv("VMLX_NATIVE_MTP_AR_SAFETY", "0")
    state = _vlm_state(m)
    base_t = time.perf_counter() - 1.0
    state.stats.cycles = 40
    state.ar_safety.ring = [(31 + i, 31 + i, base_t + i * 0.020) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.ar_fallback_pending is False


def test_text_lane_trips_under_fixed_policy(monkeypatch):
    from vmlx_engine.patches.mlx_lm_mtp import batch_generator as tl

    monkeypatch.setenv("VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    state = tl._MtpState()
    state.adaptive_enabled = False
    state.depth = 3
    state.ar_step_ms = 10.0
    state.ar_safety.prompt_tokens = 100
    state.ar_safety.anchor_cycle_ms = 20.0
    state.ar_safety.anchor_context_tokens = 120
    base_t = time.perf_counter() - 1.0
    state.stats.cycles = 40
    state.stats.draft_tokens_accepted = 0
    state.ar_safety.ring = [(31 + i, 31 + i, base_t + i * 0.020) for i in range(9)]
    assert tl._text_mtp_maybe_cost_fallback("req", state, now=time.perf_counter()) is False
    assert tl._text_mtp_maybe_ar_safety_fallback("req", state) is True
    assert state.ar_fallback_pending is True
    assert "windowed_ar_safety d3" in (state.ar_fallback_reason or "")
    assert state.stats.fallback_reason == state.ar_fallback_reason


def test_text_lane_fast_window_holds(monkeypatch):
    from vmlx_engine.patches.mlx_lm_mtp import batch_generator as tl

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    state = tl._MtpState()
    state.depth = 3
    state.ar_step_ms = 10.0
    state.ar_safety.prompt_tokens = 100
    state.ar_safety.anchor_cycle_ms = 20.0
    base_t = time.perf_counter() - 1.0
    state.stats.cycles = 40
    state.stats.draft_tokens_accepted = 40
    state.ar_safety.ring = [(31 + i, 2 * (31 + i), base_t + i * 0.010) for i in range(9)]
    assert tl._text_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.ar_fallback_pending is False
