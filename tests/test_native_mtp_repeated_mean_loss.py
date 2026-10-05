"""Measured AR safety must bound repeated losses without judging one stall."""
import pytest
from vmlx_engine.native_mtp_ar_safety import ArSafetyState, ar_safety_step


def feed(st, durations, *, start=0, now=1.0, baseline=25.0):
    trips = []
    for cycle, duration in enumerate(durations, start + 1):
        now += duration / 1000
        trip = ar_safety_step(st, cycles=cycle, emitted=cycle, now=now,
                              seed_ar_ms=baseline, baseline_measured=True,
                              scale_context=False)
        trips.append(trip)
    return trips, now


def test_repeated_minority_stalls_need_two_disjoint_windows():
    st = ArSafetyState()
    pattern = [20] * 6 + [80] * 2
    trips, now = feed(st, [20] * 8 + pattern)
    assert all(t is None for t in trips)
    trips, _ = feed(st, pattern, start=16, now=now)
    assert all(t is None for t in trips[:-1])
    assert trips[-1].confirmed_mean_loss
    assert "confirmed_mean_loss=true" in trips[-1].reason(1)
    assert "confirmed_mean_loss=true" in trips[-1].log_text(1)
    assert trips[-1].mtp_ms_per_tok == pytest.approx(35)
    assert trips[-1].cycle_median_ms_per_tok == pytest.approx(20)


def test_one_off_stall_then_winning_windows_does_not_trip():
    st = ArSafetyState()
    trips, _ = feed(st, [20] * 8 + [20] * 7 + [160] + [20] * 32)
    assert all(t is None for t in trips)
    assert st.mean_loss_end_cycle is None


@pytest.mark.parametrize('reset_kind', ['depth', 'baseline'])
def test_loss_confirmation_does_not_cross_reference_change(reset_kind):
    st = ArSafetyState()
    pattern = [20] * 6 + [80] * 2
    _, now = feed(st, [20] * 8 + pattern)
    assert st.mean_loss_end_cycle == 16
    if reset_kind == 'depth':
        st.reset(16)
        durations = [20] * 8 + pattern
        baseline = 25
    else:
        durations = pattern
        baseline = 30
    trips, _ = feed(st, durations, start=16, now=now, baseline=baseline)
    assert all(t is None for t in trips)


def test_uniform_measured_loss_keeps_existing_immediate_verdict():
    trips, _ = feed(ArSafetyState(), [30] * 16)
    assert trips[-1] is not None
    assert not trips[-1].confirmed_mean_loss


def test_equal_cost_does_not_trip():
    trips, _ = feed(ArSafetyState(), [25] * 40)
    assert all(t is None for t in trips)


def test_cost_is_token_weighted_not_mean_of_cycle_rates():
    st = ArSafetyState()
    _, now = feed(st, [20] * 8)
    emitted = 8
    # Five 30ms single-token cycles and three 100ms ten-token cycles:
    # median per-cycle rate30 > AR25, but450ms/35tokens wins comfortably.
    for cycle in range(9, 41):
        long = (cycle - 9) % 8 >= 5
        now += .100 if long else .030
        emitted += 10 if long else 1
        assert ar_safety_step(st, cycles=cycle, emitted=emitted, now=now,
            seed_ar_ms=25, baseline_measured=True, scale_context=False) is None
    assert st.mean_loss_end_cycle is None


def test_confirmed_loss_uses_existing_adjacent_depth_and_direct_ar_owner():
    # Execute the actual post-verdict owner, not a rewritten policy. No MLX
    # import is needed for this wall/counter-only transition.
    import ast
    import logging
    from pathlib import Path
    from types import SimpleNamespace
    source = Path('vmlx_engine/mllm_batch_generator.py').read_text()
    owner = next(n for n in ast.parse(source).body
                 if isinstance(n, ast.FunctionDef) and n.name == '_native_mtp_maybe_ar_safety_fallback')
    first = next(i for i,n in enumerate(owner.body)
                 if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'prior_depth' for t in n.targets))
    fn = ast.FunctionDef(name='transition', args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=x) for x in ['state','trip','depth_now','cycles','measured_ar']], kwonlyargs=[],kw_defaults=[],defaults=[]), body=owner.body[first:],decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    import os
    ns = dict(ar_safety_window_cycles=lambda:8, _NATIVE_MTP_PROMOTE_FIRST_CYCLES=32,
              _NATIVE_MTP_CALIBRATION_MIN_SPACING_TOKENS=512,
              _NATIVE_MTP_D1_TRIP_CONFIRM_WINDOWS=2, os=os,
              logger=logging.getLogger(__name__), request_id='unit', tier=None)
    exec(compile(module,'actual-ar-safety-transition','exec'),ns)
    trips,_=feed(ArSafetyState(),[20]*8+([20]*6+[80]*2)*2)
    trip=trips[-1]
    for depth in [1,2,3]:
        safety=ArSafetyState(mean_loss_end_cycle=16)
        state=SimpleNamespace(depth=depth,promote_backoff=0,ar_safety=safety,
                              ar_trip_pending_cycle=0,stats=SimpleNamespace(accepted_tokens=0),
                              last_ar_measure_emitted=0,ar_fallback_pending=False)
        result=ns['transition'](state,trip,depth,24,25)
        if depth==1:
            assert result and state.ar_fallback_pending
            assert state.ar_trip_pending_cycle==0
        else:
            assert not result and state.depth==depth-1
            assert safety.mean_loss_end_cycle is None and safety.ring==[]


def test_alternating_loss_and_small_win_eventually_confirms_independent_loss():
    st = ArSafetyState()
    pattern = [20] * 6 + [80] * 2 + [20] * 8
    now = 1.0
    first_trip = None
    for cycle, duration in enumerate([20] * 8 + pattern * 4, 1):
        prior_pending = st.mean_loss_end_cycle
        now += duration / 1000
        trip = ar_safety_step(st, cycles=cycle, emitted=cycle, now=now,
                             seed_ar_ms=25, baseline_measured=True,
                             scale_context=False)
        if trip is not None:
            assert trip.confirmed_mean_loss
            assert prior_pending is not None
            assert st.ring[0][0] >= prior_pending
            first_trip = cycle
            break
    assert first_trip == 39
