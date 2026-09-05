"""Windowed AR-safety governor — shared decision math + both lanes.

The governor drops a request to AR when a WINDOWED wall ms/token exceeds the
seed AR step scaled by the growth of the request's OWN cycle span (context
and thermal, never concurrency), for BOTH fixed and adaptive depth.
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


def test_context_growth_scales_baseline_up_and_seed_is_a_floor():
    # Cycle wall 30 -> 45 (context/thermal): baseline 15 -> threshold 18.75;
    # MTP at 17 ms/tok holds.
    assert verdict(cur_cycle_ms=45.0, delta_emitted=16, delta_wall_ms=272.0) is None
    # Cycle wall BELOW the anchor (slow early cycles after a big prefill
    # inflated the anchor): the scale clamps at 1, baseline stays at the
    # seed (10 -> threshold 12.5); MTP at 12 ms/tok must NOT trip. Measured
    # 2026-09-04: a scale below 1 falsely demoted healthy MTP 3x in 22 turns.
    v = verdict(cur_cycle_ms=24.0, delta_emitted=16, delta_wall_ms=192.0)
    assert v is None
    v2 = verdict(cur_cycle_ms=24.0, delta_emitted=16, delta_wall_ms=272.0)
    assert v2 is not None and abs(v2[1] - 10.0) < 1e-6


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


def _drive(st, *, cycle_ms, tok_per_cycle, seed_ar_ms=10.0,
           primed=True, start_cycle=1, end_cycle=60, t0=0.0, emitted0=0):
    """Feed the governor a synthetic run; returns (trip_cycle, trip, t, emitted).
    ``cycle_ms`` is the wall between cycles."""
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


def test_anchor_is_warmup_median_of_own_span(monkeypatch):
    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY_WARMUP", raising=False)
    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY_WINDOW", raising=False)
    st = ArSafetyState(prompt_tokens=50)
    # Solo healthy: 30ms cycles, 4 tok/cycle (7.5 ms/tok vs AR 10) -> holds,
    # anchor = 30 (warmup own-span median; cycle 1 is 200ms cold, ignored by
    # the median).
    c, trip, *_ = _drive(st, cycle_ms=lambda c: 200.0 if c == 1 else 30.0,
                         tok_per_cycle=lambda c: 4, end_cycle=80)
    assert abs(st.anchor_cycle_ms - 30.0) < 1e-6
    assert c is None, (c, trip and trip.log_text(3))


def test_acceptance_collapse_trips_solo(monkeypatch):
    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY_WARMUP", raising=False)
    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY_WINDOW", raising=False)
    st = ArSafetyState(prompt_tokens=50)
    c, trip, *_ = _drive(st, cycle_ms=lambda c: 30.0,
                         tok_per_cycle=lambda c: 3 if c < 40 else 1, end_cycle=80)
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
    state.depth_ceiling = 3
    base_t = time.perf_counter() - 1.0
    state.stats.cycles = 40
    state.stats.accepted_tokens = 0
    # Full ring, 1 tok/cycle at 20ms/cycle = 20 ms/tok, flat cycle wall.
    state.ar_safety.ring = [(31 + i, 31 + i, base_t + i * 0.020) for i in range(9)]
    # Rung 1: D3 loses -> D1 with a fresh window (not AR yet).
    assert m._native_mtp_maybe_ar_safety_fallback("req-fixed", state) is False
    assert state.depth == 1 and state.ar_fallback_pending is False
    assert state.ar_safety.ring == [] and state.ar_safety.cycle_base == 40
    assert state.promote_at_cycle > 40
    # Rung 2: D1 loses too -> AR.
    state.stats.cycles = 80
    state.ar_safety.anchor_cycle_ms = 20.0
    state.ar_safety.ring = [(71 + i, 71 + i, base_t + i * 0.020) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req-fixed", state) is True
    assert state.ar_fallback_pending is True
    assert "windowed_ar_safety d1" in (state.ar_fallback_reason or "")


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
    # Rung 1: D3 -> D1.
    assert tl._text_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.depth == 1 and state.ar_fallback_pending is False
    # Rung 2: D1 -> AR.
    state.stats.cycles = 80
    state.ar_safety.anchor_cycle_ms = 20.0
    state.ar_safety.ring = [(71 + i, 71 + i, base_t + i * 0.020) for i in range(9)]
    assert tl._text_mtp_maybe_ar_safety_fallback("req", state) is True
    assert state.ar_fallback_pending is True
    assert "windowed_ar_safety d1" in (state.ar_fallback_reason or "")
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


def test_probe_margin_requires_beating_measured_ar():
    # A re-entry probe is judged with margin 1/1.10: MTP at 9.5 ms/tok vs
    # measured AR 10 does NOT beat it by the hysteresis -> trip (probe lost);
    # at 8.5 it does -> hold (probe kept).
    st = ArSafetyState(prompt_tokens=50)
    c, *_ = _drive(st, cycle_ms=lambda c: 19.0, tok_per_cycle=lambda c: 2,
                   seed_ar_ms=10.0, primed=False)
    assert c is None  # 9.5 <= 12.5 at the default margin: a settled request holds
    st2 = ArSafetyState(prompt_tokens=50)
    t = 0.0; emitted = 0; tripped = None
    for cyc in range(1, 60):
        t += 0.019; emitted += 2
        trip = ar_safety_step(st2, cycles=cyc, emitted=emitted, now=t,
                              seed_ar_ms=10.0, primed=False, margin=1 / 1.10)
        if trip is not None:
            tripped = cyc; break
    assert tripped == 24  # unprimed warmup 16 + window 8
    st3 = ArSafetyState(prompt_tokens=50)
    t = 0.0; emitted = 0; tripped = None
    for cyc in range(1, 60):
        t += 0.017; emitted += 2
        trip = ar_safety_step(st3, cycles=cyc, emitted=emitted, now=t,
                              seed_ar_ms=10.0, primed=False, margin=1 / 1.10)
        if trip is not None:
            tripped = cyc; break
    assert tripped is None


def test_ar_tier_backoff_and_measurement():
    from vmlx_engine.mllm_batch_generator import NativeMTPArTier

    tier = NativeMTPArTier(depth=3)
    t = 100.0
    for _ in range(15):
        tier.record_step(t); t += 0.025
    assert tier.probe_due() is False  # 15 < 16 tokens
    tier.record_step(t); t += 0.025
    assert tier.probe_due() is True
    assert abs(tier.measured_ar_ms_per_tok() - 25.0) < 1e-6
    tier.probe_failed()
    assert tier.next_probe_tokens == 32 and tier.backoff == 1 and tier.fallbacks == 2
    assert tier.probe_due() is False
    tier.probe_failed()
    assert tier.next_probe_tokens == 64


def test_probe_kept_switches_baseline_to_measured_ar(monkeypatch):
    # A probe that beats measured AR is kept: probe flag clears, reentries++.
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    state = _vlm_state(m)
    tier = m.NativeMTPArTier(depth=3)
    t0 = time.perf_counter() - 2.0
    for i in range(16):
        tier.record_step(t0 + i * 0.025)  # measured AR 25 ms/tok
    state.ar_tier = tier
    state.probe = True
    state.depth = 1
    state.stats.prompt_prime_source = "unprimed"
    base_t = time.perf_counter() - 1.0
    state.stats.cycles = 40
    state.stats.accepted_tokens = 80  # 3 tok/cycle
    state.ar_safety.anchor_cycle_ms = 40.0
    # 40ms cycles, 3 tok/cycle = 13.3 ms/tok < 25/1.10 -> kept
    state.ar_safety.ring = [(31 + i, 3 * (31 + i), base_t + i * 0.040) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.probe is False
    assert tier.reentries == 1
    assert state.promote_at_cycle == 40 + m._NATIVE_MTP_PROMOTE_FIRST_CYCLES
    # A probe that LOSES: 1 tok/cycle at 40ms = 40 ms/tok -> falls back, backoff.
    state2 = _vlm_state(m)
    tier2 = m.NativeMTPArTier(depth=3)
    for i in range(16):
        tier2.record_step(t0 + i * 0.025)
    state2.ar_tier = tier2
    state2.probe = True
    state2.stats.prompt_prime_source = "unprimed"
    state2.stats.cycles = 40
    state2.stats.accepted_tokens = 0
    state2.ar_safety.anchor_cycle_ms = 40.0
    state2.ar_safety.ring = [(31 + i, 31 + i, base_t + i * 0.040) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req2", state2) is True
    assert state2.ar_fallback_pending is True
    assert tier2.backoff == 1 and tier2.next_probe_tokens == 32


def test_settled_reentry_that_trips_again_backs_off(monkeypatch):
    # After a kept probe, a later trip must count as a fallback, restart the
    # AR window, and back off further (no immediate re-probe).
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    state = _vlm_state(m)
    tier = m.NativeMTPArTier(depth=3)
    t0 = time.perf_counter() - 2.0
    for i in range(20):
        tier.record_step(t0 + i * 0.025)
    tier.reentries = 1
    state.ar_tier = tier
    state.probe = False  # settled at D1 after a kept probe
    state.depth = 1
    base_t = time.perf_counter() - 1.0
    state.stats.cycles = 40
    state.stats.accepted_tokens = 0
    state.ar_safety.anchor_cycle_ms = 40.0
    state.ar_safety.ring = [(31 + i, 31 + i, base_t + i * 0.040) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is True
    assert tier.fallbacks == 2
    assert tier.tokens_since_fallback == 0
    assert tier.next_probe_tokens == 32
    assert tier.probe_due() is False


def test_promotion_probe_wins_and_loses(monkeypatch):
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    state = _vlm_state(m)
    state.depth = 1
    state.ladder_depth = 3
    state.promote_at_cycle = 40
    base_t = time.perf_counter() - 1.0
    state.stats.cycles = 40
    state.stats.accepted_tokens = 40
    # D1 has been running at 2 tok/cycle, 20ms/cycle = 10 ms/tok.
    state.ar_safety.ring = [(31 + i, 2 * (31 + i), base_t + i * 0.020) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.promote_probe is True and state.depth == 3
    assert abs(state.d1_ms_per_tok - 10.0) < 1e-6
    # Promotion window: D3 at 3.5 tok/cycle, 28ms/cycle = 8 ms/tok < 10/1.1 -> promoted.
    state.stats.cycles = 60
    state.ar_safety.anchor_cycle_ms = 28.0
    state.ar_safety.ring = [(51 + i, int(3.5 * (51 + i)), base_t + i * 0.028) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.promote_probe is False and state.depth == 3
    # Later D3 loses to AR (20 ms/tok vs 10) -> back to D1 with backoff.
    state.stats.cycles = 100
    state.ar_safety.anchor_cycle_ms = 20.0
    state.ar_safety.ring = [(91 + i, 91 + i, base_t + i * 0.020) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.depth == 1 and state.ar_fallback_pending is False
    assert state.promote_at_cycle > 100


def test_probe_early_abort_on_clear_loser(monkeypatch):
    from vmlx_engine.native_mtp_ar_safety import ArSafetyState, ar_safety_step

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    st = ArSafetyState(prompt_tokens=50)
    st.reset(100)
    t = 0.0; emitted = 0; tripped = None
    for cyc in range(101, 140):
        t += 0.040; emitted += 1  # 40 ms/tok vs AR 10: a clear loser
        trip = ar_safety_step(st, cycles=cyc, emitted=emitted, now=t,
                              seed_ar_ms=10.0, primed=False, margin=1 / 1.10, probe=True)
        if trip is not None:
            tripped = cyc; break
    # warmup 4 + 5 ring samples -> aborted by cycle ~110, not after 24 cycles
    assert tripped is not None and tripped <= 112


def test_retry_budget_caps_probes_and_promotions():
    from vmlx_engine import mllm_batch_generator as m

    tier = m.NativeMTPArTier(depth=3)
    t = 100.0
    for _ in range(20):
        tier.record_step(t); t += 0.025
    assert tier.probe_due() is True
    tier.probes = m._NATIVE_MTP_MAX_REENTRY_PROBES
    assert tier.probe_due() is False  # budget spent: stay AR for the request
    assert m._NATIVE_MTP_MAX_PROMOTIONS >= 1


# ---- seed-uncertainty margin + sticky-start evidence (2026-09-05) ----------

def _drive_uniform(st, *, ms_per_tok, cycles=40, seed=10.0, **kw):
    """Feed uniform cycles (1 token each) at ms_per_tok; return the first trip."""
    t = 100.0
    for c in range(1, cycles + 1):
        t += ms_per_tok / 1000.0
        trip = ar_safety_step(st, cycles=c, emitted=c, now=t, seed_ar_ms=seed, **kw)
        if trip is not None:
            return trip
    return None


def test_seed_baseline_uses_seed_margin_measured_uses_exact(monkeypatch):
    monkeypatch.delenv("VMLX_NATIVE_MTP_SEED_COST_MARGIN", raising=False)
    monkeypatch.delenv("VMLX_NATIVE_MTP_RUNTIME_COST_MARGIN", raising=False)
    # 5% slower than the one-step seed: inside the seed's error band -> hold.
    assert _drive_uniform(ArSafetyState(), ms_per_tok=10.5) is None
    # Same cost against a MEASURED baseline: a real loss -> trip.
    trip = _drive_uniform(ArSafetyState(), ms_per_tok=10.5, baseline_measured=True)
    assert trip is not None and abs(trip.margin - 1.0) < 1e-9
    # 15% slower than the seed is outside the band -> trip with the seed margin.
    trip = _drive_uniform(ArSafetyState(), ms_per_tok=11.5)
    assert trip is not None and abs(trip.margin - 1.10) < 1e-9


def test_seed_margin_never_below_runtime_margin(monkeypatch):
    from vmlx_engine.native_mtp_ar_safety import ar_safety_seed_margin

    monkeypatch.setenv("VMLX_NATIVE_MTP_RUNTIME_COST_MARGIN", "1.25")
    monkeypatch.setenv("VMLX_NATIVE_MTP_SEED_COST_MARGIN", "1.05")
    assert abs(ar_safety_seed_margin() - 1.25) < 1e-9


def test_sticky_start_evidence_rules():
    from vmlx_engine.mllm_batch_generator import (
        _native_mtp_first_promotion_cycle,
        _native_mtp_should_record_last_tier,
        _NATIVE_MTP_PROMOTE_FIRST_CYCLES,
    )

    # first request after load never seeds the next start rung
    assert not _native_mtp_should_record_last_tier(0, 200, 8)
    # too few judged cycles never seeds it
    assert not _native_mtp_should_record_last_tier(3, 15, 8)
    assert _native_mtp_should_record_last_tier(1, 16, 8)
    # a sticky D1 start is re-examined after one window, own demotion waits
    assert _native_mtp_first_promotion_cycle(True) == 8
    assert _native_mtp_first_promotion_cycle(False) == _NATIVE_MTP_PROMOTE_FIRST_CYCLES


def test_depth_probe_keeps_d1_when_it_beats_configured_depth(monkeypatch):
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    state = _vlm_state(m)
    state.depth = 3
    state.ladder_depth = 3
    state.depth_probe_at_cycle = 16
    base_t = time.perf_counter() - 1.0
    state.stats.cycles = 20
    state.stats.accepted_tokens = 40
    # D3 window: 3 tok/cycle at 30 ms/cycle = 10 ms/tok (AR seed 12 -> healthy)
    state.ar_step_ms = 12.0
    state.ar_safety.ring = [(11 + i, 3 * (11 + i), base_t + i * 0.030) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.depth_probe is True and state.depth == 1 and state.depth_probes == 1
    assert abs(state.dcfg_ms_per_tok - 10.0) < 1e-6
    # D1 window: 1.9 tok/cycle at 15 ms/cycle = 7.9 ms/tok < 10/1.1 -> keep D1
    state.stats.cycles = 40
    state.ar_safety.anchor_cycle_ms = 15.0
    state.ar_safety.ring = [(31 + i, int(1.9 * (31 + i)), base_t + i * 0.015) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.depth_probe is False and state.depth == 1
    assert state.promote_at_cycle > 40 and state.d1_ms_per_tok > 0


def test_depth_probe_returns_to_configured_depth_when_d1_loses(monkeypatch):
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    state = _vlm_state(m)
    state.depth = 3
    state.ladder_depth = 3
    state.depth_probe_at_cycle = 16
    base_t = time.perf_counter() - 1.0
    state.stats.cycles = 20
    state.stats.accepted_tokens = 40
    state.ar_step_ms = 12.0
    state.ar_safety.ring = [(11 + i, 3 * (11 + i), base_t + i * 0.030) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.depth == 1 and state.depth_probe is True
    # D1 window at 9.5 ms/tok: not 10% better than D3's 10 -> probe lost
    state.stats.cycles = 40
    state.ar_safety.anchor_cycle_ms = 19.0
    state.ar_safety.ring = [(31 + i, 2 * (31 + i), base_t + i * 0.019) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.depth_probe is False and state.depth == 3
    assert state.depth_probe_at_cycle > 40 and state.depth_probe_backoff == 1
    # budget: a second probe is allowed, a third is not
    state.stats.cycles = state.depth_probe_at_cycle + 1
    state.ar_safety.ring = [(state.stats.cycles - 9 + i, 3 * (state.stats.cycles - 9 + i), base_t + i * 0.030) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.depth_probes == 2 and state.depth == 1
    state.depth_probe = False; state.depth = 3; state.depth_probe_at_cycle = state.stats.cycles
    state.ar_safety.ring = [(state.stats.cycles - 9 + i, 3 * (state.stats.cycles - 9 + i), base_t + i * 0.030) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.depth == 3 and state.depth_probes == 2


def test_depth_probe_follows_policy_and_explicit_switch(monkeypatch):
    """Contract: a FIXED policy request that beats AR at its configured depth
    never tries D1 (only the AR-safety valve may step it down); an ADAPTIVE
    request tries D1 once; VMLX_NATIVE_MTP_DEPTH_PROBE overrides either way."""
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    monkeypatch.delenv("VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH", raising=False)
    for policy_env, probe_env, expect_depth in (
        ("0", None, 3),   # fixed: no probe
        ("1", None, 1),   # adaptive: probe
        ("0", "1", 1),    # fixed + explicit probe on
        ("1", "0", 3),    # adaptive + explicit probe off
    ):
        monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH", policy_env)
        if probe_env is None:
            monkeypatch.delenv("VMLX_NATIVE_MTP_DEPTH_PROBE", raising=False)
        else:
            monkeypatch.setenv("VMLX_NATIVE_MTP_DEPTH_PROBE", probe_env)
        state = _vlm_state(m)
        state.depth = 3
        state.ladder_depth = 3
        state.depth_probe_at_cycle = 16
        state.ar_step_ms = 12.0
        base_t = time.perf_counter() - 1.0
        state.stats.cycles = 20
        state.stats.accepted_tokens = 40
        state.ar_safety.anchor_cycle_ms = 30.0  # steady: ring cycles match the anchor (no drift)
        state.ar_safety.ring = [(11 + i, 3 * (11 + i), base_t + i * 0.030) for i in range(9)]
        assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
        assert state.depth == expect_depth, f"policy={policy_env} probe_env={probe_env}"
    monkeypatch.delenv("VMLX_NATIVE_MTP_DEPTH_PROBE", raising=False)
    monkeypatch.delenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH", raising=False)


def test_fixed_policy_still_steps_down_only_through_ar_safety(monkeypatch):
    """Under fixed, a window slower than plain decoding still trips the valve
    (D3 -> D1 with the logged reason); a healthy window never moves."""
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
    state = _vlm_state(m)
    state.depth = 3; state.ladder_depth = 3; state.depth_probe_at_cycle = 16
    state.ar_step_ms = 10.0
    base_t = time.perf_counter() - 1.0
    state.stats.cycles = 20; state.stats.accepted_tokens = 20
    # 1 tok/cycle at 20 ms/cycle = 20 ms/tok vs AR 10 -> valve trips D3 -> D1
    state.ar_safety.ring = [(11 + i, 11 + i, base_t + i * 0.020) for i in range(9)]
    state.ar_safety.anchor_cycle_ms = 20.0
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.depth == 1 and state.depth_probe is False
    monkeypatch.delenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH")


def test_finish_line_fields_policy_configured_and_throughput(caplog):
    import logging
    from vmlx_engine.mllm_batch_generator import MLLMNativeMTPStats, _native_mtp_log_stats

    stats = MLLMNativeMTPStats()
    stats.cycles = 10; stats.accepted_tokens = 12; stats.drafted_tokens = 30
    stats.init_emits = 2; stats.draft_emits = 12; stats.bonus_emits = 4; stats.verify_emits = 6
    stats.cycles_by_depth[3] = 7; stats.cycles_by_depth[1] = 3
    stats.configured_depth = 3; stats.depth_policy = "fixed"; stats.span_seconds = 0.5
    with caplog.at_level(logging.INFO, logger="vmlx_engine.mllm_batch_generator"):
        _native_mtp_log_stats("req", stats, "length", None)
    line = next(r.getMessage() for r in caplog.records if "finish=length" in r.getMessage())
    assert "cycles_by_depth[d1=3,d3=7]" in line
    assert "policy=fixed configured=D3" in line
    assert "confirmed_tok_s=48.0 span_s=0.50" in line


def test_finish_line_reports_cycles_by_depth():
    from vmlx_engine.mllm_batch_generator import MLLMNativeMTPStats

    stats = MLLMNativeMTPStats()
    for depth in (3, 3, 1, 3, 1, 1, 1):
        stats.cycles_by_depth[depth] += 1
    assert stats.cycles_by_depth[3] == 3 and stats.cycles_by_depth[1] == 4
    occupancy = ",".join(f"d{d}={n}" for d, n in enumerate(stats.cycles_by_depth) if d > 0 and n > 0)
    assert occupancy == "d1=4,d3=3"


# ---- bounded AR calibration (context-matched baseline refresh) -----------------

def _running_state(m, *, depth=3, anchor_ms=30.0, cycle_ms=30.0, cycles=40, emitted_per_cycle=3):
    state = _vlm_state(m)
    state.depth = depth; state.ladder_depth = depth
    state.ar_step_ms = 12.0
    state.stats.cycles = cycles
    state.stats.accepted_tokens = cycles * (emitted_per_cycle - 1)
    base_t = time.perf_counter() - 1.0
    state.ar_safety.anchor_cycle_ms = anchor_ms
    state.ar_safety.ring = [(cycles - 8 + i, emitted_per_cycle * (cycles - 8 + i), base_t + i * cycle_ms / 1000.0) for i in range(9)]
    return state


def test_calibration_triggers_on_cycle_wall_drift_and_reenters_at_same_depth(monkeypatch):
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
    # cycle wall 2x the anchor (drift +100%), MTP still cheap per token (10 ms/tok)
    state = _running_state(m, anchor_ms=30.0, cycle_ms=60.0, emitted_per_cycle=6)
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is True
    assert state.ar_fallback_pending and "ar_calibration" in state.ar_fallback_reason
    assert "drift=+1.00" in state.ar_fallback_reason
    tier = state.ar_tier
    assert tier is not None and tier.calibration and tier.reenter_depth == 3
    assert tier.next_probe_tokens == m._NATIVE_MTP_CALIBRATION_TOKENS
    assert tier.backoff == 0 and state.calibrations == 1
    monkeypatch.delenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH")


def test_calibration_triggers_on_stale_interval_not_on_steady_short_runs(monkeypatch):
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
    steady = _running_state(m, anchor_ms=30.0, cycle_ms=30.0, cycles=40)
    assert m._native_mtp_maybe_ar_safety_fallback("req", steady) is False
    assert not steady.ar_fallback_pending and steady.calibrations == 0
    stale = _running_state(m, anchor_ms=30.0, cycle_ms=30.0, cycles=400, emitted_per_cycle=3)  # ~1,200 emitted
    assert m._native_mtp_maybe_ar_safety_fallback("req", stale) is True
    assert "stale=True" in stale.ar_fallback_reason and stale.calibrations == 1
    monkeypatch.setenv("VMLX_NATIVE_MTP_AR_CALIBRATION", "0")
    off = _running_state(m, anchor_ms=30.0, cycle_ms=60.0, emitted_per_cycle=6)
    assert m._native_mtp_maybe_ar_safety_fallback("req", off) is False
    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_CALIBRATION")
    monkeypatch.delenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH")


def test_calibration_budget_is_bounded(monkeypatch):
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
    state = _running_state(m, anchor_ms=30.0, cycle_ms=60.0, emitted_per_cycle=6)
    state.calibrations = m._NATIVE_MTP_MAX_CALIBRATIONS
    # budget exhausted: no calibration; the ordinary valve judges the window instead
    m._native_mtp_maybe_ar_safety_fallback("req", state)
    assert "ar_calibration" not in (state.ar_fallback_reason or "")
    monkeypatch.delenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH")


def test_tier_records_ar_wall_and_calibration_counters():
    from vmlx_engine.mllm_batch_generator import NativeMTPArTier

    t = NativeMTPArTier(depth=3)
    now = 100.0
    for _ in range(6):
        t.record_step(now); now += 0.025
    assert t.total_ar_tokens == 6 and abs(t.total_ar_ms - 125.0) < 1e-6
    assert abs(t.measured_ar_ms_per_tok() - 25.0) < 1e-6
