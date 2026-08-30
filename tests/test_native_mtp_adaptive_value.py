"""Rolling wall-value selection for request-local native MTP depth."""

import pytest

from vmlx_engine.native_mtp_adaptive import (
    NativeMTPAdaptiveValueState,
    adaptive_value_snapshot,
    add_depth_cycle_sample,
    arm_depth_cycle,
    choose_depth_by_value,
    depth_conditional_acceptance,
    depth_value_tps,
    finish_armed_depth_cycle,
    note_forced_depth_change,
)


def _samples(
    state,
    depth,
    *,
    count=8,
    elapsed_ms=10.0,
    accepted=None,
    first_cycle=1,
):
    accepted = depth if accepted is None else accepted
    for offset in range(count):
        assert add_depth_cycle_sample(
            state,
            depth=depth,
            accepted_drafts=accepted,
            elapsed_ms=elapsed_ms,
            cycle=first_cycle + offset,
            window=16,
        )


def _choose(state, current, cycle, **overrides):
    kwargs = {
        "current_depth": current,
        "depth_ceiling": 3,
        "cycle": cycle,
        "minimum_samples": 8,
        "cooldown_cycles": 8,
        "probe_interval_cycles": 48,
        "hysteresis": 0.05,
        "raise_min_acceptance": 0.88,
    }
    kwargs.update(overrides)
    return choose_depth_by_value(state, **kwargs)


def test_value_is_confirmed_tokens_per_real_wall_interval():
    state = NativeMTPAdaptiveValueState()
    _samples(state, 2, count=4, elapsed_ms=6.0, accepted=2)

    assert depth_value_tps(state, 2, minimum_samples=4) == pytest.approx(500.0)


def test_deepest_acceptance_is_conditional_on_reaching_that_depth():
    state = NativeMTPAdaptiveValueState()
    for cycle, accepted in enumerate((2, 2, 1, 0), start=1):
        add_depth_cycle_sample(
            state,
            depth=2,
            accepted_drafts=accepted,
            elapsed_ms=10.0,
            cycle=cycle,
            window=16,
        )

    # Three cycles reached D2 and two accepted it.
    assert depth_conditional_acceptance(
        state, 2, minimum_samples=4
    ) == pytest.approx(2 / 3)


def test_armed_interval_rejects_transition_contamination():
    state = NativeMTPAdaptiveValueState()
    arm_depth_cycle(state, depth=2, now=10.0)
    assert not finish_armed_depth_cycle(
        state,
        depth=3,
        accepted_drafts=3,
        cycle=1,
        now=10.01,
        window=16,
    )
    assert state.samples_by_depth == [[], [], []]

    arm_depth_cycle(state, depth=2, now=20.0)
    assert finish_armed_depth_cycle(
        state,
        depth=2,
        accepted_drafts=2,
        cycle=2,
        now=20.006,
        window=16,
    )
    assert depth_value_tps(state, 2) == pytest.approx(500.0)


def test_strong_d2_opens_a_bounded_d3_probe():
    state = NativeMTPAdaptiveValueState()
    _samples(state, 2, elapsed_ms=6.0, accepted=2)

    decision = _choose(state, 2, 8)

    assert decision is not None
    assert decision.target_depth == 3
    assert decision.event == "probe_start"
    assert state.active_probe_origin == 2
    assert state.active_probe_target == 3


def test_slow_d3_probe_returns_to_d2_even_with_perfect_acceptance():
    state = NativeMTPAdaptiveValueState()
    _samples(state, 2, elapsed_ms=6.0, accepted=2)
    assert _choose(state, 2, 8).target_depth == 3
    # Perfect D3 acceptance still loses: 4 / 15ms < D2's 3 / 6ms.
    _samples(state, 3, elapsed_ms=15.0, accepted=3, first_cycle=9)

    decision = _choose(state, 3, 16)

    assert decision is not None
    assert decision.target_depth == 2
    assert decision.event == "probe_revert"
    assert state.probe_revert_counts == [0, 0, 1]


def test_reverted_neighbor_uses_exponential_reprobe_backoff():
    state = NativeMTPAdaptiveValueState()
    _samples(state, 3, elapsed_ms=6.0, accepted=3)
    assert _choose(state, 3, 8).target_depth == 2
    _samples(state, 2, elapsed_ms=15.0, accepted=2, first_cycle=9)

    decision = _choose(state, 2, 16)
    assert decision is not None
    assert decision.target_depth == 3
    assert decision.event == "probe_revert"
    assert state.probe_revert_counts[1] == 1

    # The old fixed interval would probe D2 again at cycle 56 (48 cycles
    # after cycle 8). One loss doubles that target's interval to 96 cycles.
    assert _choose(state, 3, 56) is None
    decision = _choose(state, 3, 104)
    assert decision is not None
    assert decision.target_depth == 2
    assert decision.event == "probe_start"


def test_fast_d3_probe_is_kept_only_after_beating_hysteresis():
    state = NativeMTPAdaptiveValueState()
    _samples(state, 2, elapsed_ms=6.0, accepted=2)
    assert _choose(state, 2, 8).target_depth == 3
    _samples(state, 3, elapsed_ms=6.0, accepted=3, first_cycle=9)

    decision = _choose(state, 3, 16)

    assert decision is not None
    assert decision.target_depth == 3
    assert decision.event == "probe_keep"
    assert state.active_probe_target == 0
    assert state.last_change_cycle == 16
    assert _choose(state, 3, 17) is None


def test_small_apparent_gain_reverts_instead_of_flapping():
    state = NativeMTPAdaptiveValueState()
    _samples(state, 2, elapsed_ms=6.0, accepted=2)
    assert _choose(state, 2, 8).target_depth == 3
    # 515 tok/s is only 3% over D2's 500, below the 5% keep margin.
    _samples(
        state,
        3,
        elapsed_ms=4.0 / 515.0 * 1000.0,
        accepted=3,
        first_cycle=9,
    )

    decision = _choose(state, 3, 16)

    assert decision.target_depth == 2
    assert decision.event == "probe_revert"


def test_cooldown_blocks_an_immediate_second_probe():
    state = NativeMTPAdaptiveValueState(last_change_cycle=7)
    _samples(state, 2, elapsed_ms=6.0, accepted=2)

    assert _choose(state, 2, 8) is None


def test_after_d3_probe_d2_can_probe_d1_in_the_same_request():
    state = NativeMTPAdaptiveValueState(last_change_cycle=40)
    state.last_probe_cycle[2] = 50  # D3 was just tested; do not retry it.
    _samples(state, 2, elapsed_ms=6.0, accepted=2, first_cycle=41)

    decision = _choose(state, 2, 56)

    assert decision is not None
    assert decision.target_depth == 1
    assert decision.event == "probe_start"


def test_old_bad_depth_is_reprobed_after_the_phase_interval():
    state = NativeMTPAdaptiveValueState(last_change_cycle=1)
    state.last_probe_cycle[2] = 1
    _samples(state, 2, elapsed_ms=6.0, accepted=2, first_cycle=41)

    decision = _choose(state, 2, 50)

    assert decision is not None
    assert decision.target_depth == 3


def test_acceptance_safety_change_resets_probe_without_lowering_capability():
    state = NativeMTPAdaptiveValueState(
        active_probe_origin=2,
        active_probe_target=3,
        probe_revert_counts=[0, 2, 1],
    )
    note_forced_depth_change(
        state,
        origin=3,
        target=2,
        cycle=20,
        reason="acceptance_gate",
    )

    assert state.active_probe_origin == 0
    assert state.active_probe_target == 0
    assert state.probe_revert_counts == [0, 0, 0]
    assert state.last_change_cycle == 20
    assert state.transitions[-1]["event"] == "safety_change"


def test_snapshot_exposes_value_acceptance_probe_and_transition_history():
    state = NativeMTPAdaptiveValueState()
    _samples(state, 2, elapsed_ms=6.0, accepted=2)
    decision = _choose(state, 2, 8)
    assert decision is not None

    snapshot = adaptive_value_snapshot(state, minimum_samples=8)

    assert snapshot["basis"] == "rolling_wall_confirmed_tokens_per_second"
    assert snapshot["values_tok_s"]["d2"] == 500.0
    assert snapshot["conditional_acceptance"]["d2"] == 1.0
    assert snapshot["active_probe"] == {"origin": 2, "target": 3}
    assert snapshot["probe_revert_counts"] == {"d1": 0, "d2": 0, "d3": 0}
    assert snapshot["transitions"][-1]["event"] == "probe_start"
