"""Windowed, context-scaled AR-safety governor — the pure decision math.

The governor drops a request to AR when a WINDOWED MTP ms/token exceeds a
context-scaled AR baseline, for BOTH fixed and adaptive depth. These tests pin
the arithmetic (unit-testable without a model): fast MTP holds, slow MTP trips,
long-context growth does NOT false-trip, an abrupt MTP-only slowdown cannot
hide behind the context scale, a single stalled cycle cannot trip, and the
div-by-small/empty guards.
"""

import time

from vmlx_engine.native_mtp_ar_safety import windowed_ar_verdict as _native_mtp_windowed_ar_verdict

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
    # MTP at 5ms/tok vs AR 10ms -> 2x faster.
    assert verdict() is None


def test_slow_mtp_trips():
    # MTP at 20ms/tok vs AR 10ms (flat context) -> 2x slower, must trip.
    v = verdict(delta_emitted=16, delta_wall_ms=320.0)
    assert v is not None
    mtp_ms_per_tok, ar_baseline = v
    assert abs(mtp_ms_per_tok - 20.0) < 1e-6
    assert abs(ar_baseline - 10.0) < 1e-6


def test_long_context_does_not_false_trip():
    # Long context: cycle wall doubled (30 -> 60ms) AND context doubled, so
    # AR would have doubled too: baseline 20ms. MTP at 18ms/tok is still worth
    # it -> must NOT trip. A stale short-context baseline (10ms) WOULD have.
    v = verdict(cur_cycle_ms=60.0, context_ratio=2.0,
                delta_emitted=16, delta_wall_ms=288.0)
    assert v is None
    stale = verdict(anchor_cycle_ms=0.0, cur_cycle_ms=60.0, context_ratio=2.0,
                    delta_emitted=16, delta_wall_ms=288.0)
    assert stale is not None


def test_context_scaled_slow_still_trips():
    # Even with context growth, MTP genuinely slower than the scaled baseline
    # must trip: 30ms/tok vs baseline 20 x 1.25 = 25.
    v = verdict(cur_cycle_ms=60.0, context_ratio=2.0,
                delta_emitted=16, delta_wall_ms=480.0)
    assert v is not None


def test_abrupt_mtp_only_slowdown_cannot_hide_behind_scale():
    # Cycle wall doubled but context grew only 10%: a forward cannot grow
    # faster than context, so the scale is capped at 1.1 -> baseline 11ms.
    # MTP at 18ms/tok > 13.75 -> trips (the slowdown is MTP's, not context's).
    v = verdict(cur_cycle_ms=60.0, context_ratio=1.1,
                delta_emitted=16, delta_wall_ms=288.0)
    assert v is not None
    _, ar_baseline = v
    assert abs(ar_baseline - 11.0) < 1e-6


def test_baseline_never_below_seed_floor():
    # Cycle wall SHRANK vs the anchor: scale clamps to 1, baseline = seed.
    v = verdict(anchor_cycle_ms=100.0, cur_cycle_ms=30.0, context_ratio=3.0,
                delta_emitted=16, delta_wall_ms=200.0)  # 12.5 <= 12.5 holds
    assert v is None
    v2 = verdict(anchor_cycle_ms=100.0, cur_cycle_ms=30.0, context_ratio=3.0,
                 delta_emitted=16, delta_wall_ms=201.0)  # 12.5625 > 12.5
    assert v2 is not None


def test_single_stall_cycle_does_not_trip():
    # 15 healthy cycles at 5 ms/tok and ONE 400ms stall: the window MEAN is
    # over threshold but the per-cycle MEDIAN is not -> hold (demotion is
    # irreversible; a background store / page fault must not cause it).
    per_cycle = [5.0] * 15 + [400.0]
    v = verdict(delta_emitted=16, delta_wall_ms=sum(per_cycle),
                per_cycle_ms_per_tok=per_cycle)
    assert v is None
    # Same mean but SUSTAINED (every cycle slow) -> trips.
    sustained = [sum(per_cycle) / 16] * 16
    v2 = verdict(delta_emitted=16, delta_wall_ms=sum(sustained),
                 per_cycle_ms_per_tok=sustained)
    assert v2 is not None


def test_guards_return_none():
    assert verdict(ar_step_ms=0.0, delta_emitted=16, delta_wall_ms=320.0) is None
    assert verdict(delta_emitted=0, delta_wall_ms=320.0) is None
    assert verdict(delta_emitted=16, delta_wall_ms=0.0) is None


def _state(m, depth=3):
    state = m.MLLMNativeMTPState(
        mtp_cache=[], next_main=None, drafts=[], draft_lps=[], draft_ids=[],
        depth=depth,
    )
    state.ar_step_ms = 10.0
    state.ar_safety.prompt_tokens = 100
    return state


def test_safety_runs_and_trips_under_fixed_policy(monkeypatch):
    # The whole point: this valve fires regardless of depth policy. Simulate a
    # fixed-depth state whose windowed cost is 2x AR and confirm it demotes to
    # AR (fixed policy disables depth adaptation, NOT this safety valve).
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.setenv("VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")  # fixed
    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)

    state = _state(m)
    state.ar_safety.anchor_cycle_ms = 20.0
    state.ar_safety.anchor_context_tokens = 120
    base_t = time.perf_counter() - 1.0
    state.stats.cycles = 40
    state.stats.accepted_tokens = 0
    # Full ring, 1 tok/cycle at 20ms/cycle = 20 ms/tok, flat cycle wall.
    state.ar_safety.ring = [(24 + i, 24 + i, base_t + i * 0.020) for i in range(17)]
    tripped = m._native_mtp_maybe_ar_safety_fallback("req-fixed", state)
    assert tripped is True
    assert state.ar_fallback_pending is True
    assert state.depth == 1
    assert "windowed_ar_safety d3" in (state.ar_fallback_reason or "")


def test_anchor_is_first_full_window_not_cycle_one(monkeypatch):
    # The context reference is taken from the first FULL post-warmup window
    # (median cycle wall + context length there), and a fast window holds.
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    state = _state(m)
    assert state.ar_safety.anchor_cycle_ms == 0.0
    base_t = time.perf_counter() - 1.0
    state.stats.cycles = 40
    state.stats.accepted_tokens = 40
    # 2 tok/cycle at 10ms/cycle = 5 ms/tok (fast).
    state.ar_safety.ring = [(24 + i, 2 * (24 + i), base_t + i * 0.010) for i in range(17)]
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert abs(state.ar_safety.anchor_cycle_ms - 10.0) < 1e-6
    assert state.ar_safety.anchor_context_tokens == 100 + 80
    assert state.ar_fallback_pending is False


def test_disabled_by_kill_switch(monkeypatch):
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.setenv("VMLX_NATIVE_MTP_AR_SAFETY", "0")
    state = _state(m)
    state.ar_safety.anchor_cycle_ms = 20.0
    state.ar_safety.anchor_context_tokens = 120
    base_t = time.perf_counter() - 1.0
    state.stats.cycles = 40
    state.ar_safety.ring = [(24 + i, 24 + i, base_t + i * 0.020) for i in range(17)]
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.ar_fallback_pending is False


def test_text_lane_trips_under_fixed_policy(monkeypatch):
    # The TEXT lane (GLM / DSV4 / non-VL bundles) had the same hole: its
    # runtime cost gate is chained to adaptive_enabled. The shared valve must
    # fire there under FIXED depth too — same scenario as the VLM test.
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
    state.ar_safety.ring = [(24 + i, 24 + i, base_t + i * 0.020) for i in range(17)]
    # The adaptive-only gate stays silent under fixed policy...
    assert tl._text_mtp_maybe_cost_fallback("req", state, now=time.perf_counter()) is False
    # ...the policy-independent valve does not.
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
    base_t = time.perf_counter() - 1.0
    state.stats.cycles = 40
    state.stats.draft_tokens_accepted = 40
    state.ar_safety.ring = [(24 + i, 2 * (24 + i), base_t + i * 0.010) for i in range(17)]
    assert tl._text_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.ar_fallback_pending is False
