"""Request-local wall-value controller for native MTP draft depth.

Acceptance is an admission signal, not a throughput result.  A deeper chain
can accept more draft tokens while taking sufficiently longer to verify that
it emits fewer confirmed tokens per wall second.  This module keeps a small
rolling window for each depth, probes one neighbouring depth at a time, and
keeps the probe only when its measured confirmed-token rate wins by a
hysteresis margin.

The controller never changes sampling or verifier semantics.  Draft tokens
still become output only after the main model accepts them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NativeMTPDepthCycleSample:
    elapsed_ms: float
    confirmed_tokens: int
    accepted_drafts: int
    cycle: int


@dataclass(frozen=True)
class NativeMTPDepthDecision:
    target_depth: int
    reason: str
    event: str


@dataclass
class NativeMTPAdaptiveValueState:
    """Mutable, request-local measurements and probe lifecycle."""

    samples_by_depth: list[list[NativeMTPDepthCycleSample]] = field(
        default_factory=lambda: [[], [], []]
    )
    last_sample_cycle: list[int] = field(default_factory=lambda: [0, 0, 0])
    last_probe_cycle: list[int] = field(default_factory=lambda: [0, 0, 0])
    # Consecutive wall-value losses by probe TARGET. A stable winning depth
    # must not pay the same known-loser experiment every fixed interval for
    # the rest of a long response. Each loss doubles that target's re-probe
    # interval (capped below); a later win or a safety-driven phase change
    # resets the debt.
    probe_revert_counts: list[int] = field(default_factory=lambda: [0, 0, 0])
    active_probe_origin: int = 0
    active_probe_target: int = 0
    last_change_cycle: int = 0
    armed_at: float = 0.0
    armed_depth: int = 0
    transitions: list[dict[str, object]] = field(default_factory=list)


def _bounded_depth(depth: int) -> int:
    return max(1, min(3, int(depth or 1)))


def add_depth_cycle_sample(
    state: NativeMTPAdaptiveValueState,
    *,
    depth: int,
    accepted_drafts: int,
    elapsed_ms: float,
    cycle: int,
    window: int,
) -> bool:
    """Record one complete stable-depth interval, rejecting invalid samples."""

    depth = _bounded_depth(depth)
    accepted_drafts = max(0, min(depth, int(accepted_drafts)))
    elapsed_ms = float(elapsed_ms)
    if elapsed_ms <= 0.0:
        return False
    sample = NativeMTPDepthCycleSample(
        elapsed_ms=elapsed_ms,
        confirmed_tokens=1 + accepted_drafts,
        accepted_drafts=accepted_drafts,
        cycle=max(0, int(cycle)),
    )
    bucket = state.samples_by_depth[depth - 1]
    bucket.append(sample)
    del bucket[: max(0, len(bucket) - max(1, int(window)))]
    state.last_sample_cycle[depth - 1] = sample.cycle
    return True


def finish_armed_depth_cycle(
    state: NativeMTPAdaptiveValueState,
    *,
    depth: int,
    accepted_drafts: int,
    cycle: int,
    now: float,
    window: int,
) -> bool:
    """Finish the interval armed after the preceding controller decision.

    The interval includes the real prefetch/queue-drain overlap between two
    completed verify cycles.  A transition sample is discarded unless the
    armed depth is exactly the depth that just completed.
    """

    armed_at = float(state.armed_at or 0.0)
    armed_depth = int(state.armed_depth or 0)
    state.armed_at = 0.0
    state.armed_depth = 0
    if armed_at <= 0.0 or armed_depth != _bounded_depth(depth):
        return False
    return add_depth_cycle_sample(
        state,
        depth=depth,
        accepted_drafts=accepted_drafts,
        elapsed_ms=(float(now) - armed_at) * 1000.0,
        cycle=cycle,
        window=window,
    )


def arm_depth_cycle(
    state: NativeMTPAdaptiveValueState,
    *,
    depth: int,
    now: float,
) -> None:
    state.armed_depth = _bounded_depth(depth)
    state.armed_at = float(now)


def depth_value_tps(
    state: NativeMTPAdaptiveValueState,
    depth: int,
    *,
    minimum_samples: int = 1,
) -> float | None:
    bucket = state.samples_by_depth[_bounded_depth(depth) - 1]
    if len(bucket) < max(1, int(minimum_samples)):
        return None
    elapsed_ms = sum(sample.elapsed_ms for sample in bucket)
    if elapsed_ms <= 0.0:
        return None
    confirmed = sum(sample.confirmed_tokens for sample in bucket)
    return 1000.0 * confirmed / elapsed_ms


def depth_conditional_acceptance(
    state: NativeMTPAdaptiveValueState,
    depth: int,
    *,
    minimum_samples: int = 1,
) -> float | None:
    """Return acceptance of the deepest draft conditional on reaching it."""

    depth = _bounded_depth(depth)
    bucket = state.samples_by_depth[depth - 1]
    if len(bucket) < max(1, int(minimum_samples)):
        return None
    reached = sum(sample.accepted_drafts >= depth - 1 for sample in bucket)
    if reached <= 0:
        return 0.0
    accepted = sum(sample.accepted_drafts >= depth for sample in bucket)
    return accepted / reached


def _record_transition(
    state: NativeMTPAdaptiveValueState,
    *,
    cycle: int,
    origin: int,
    target: int,
    event: str,
    reason: str,
) -> None:
    state.transitions.append(
        {
            "cycle": int(cycle),
            "from": _bounded_depth(origin),
            "to": _bounded_depth(target),
            "event": str(event),
            "reason": str(reason),
        }
    )
    del state.transitions[: max(0, len(state.transitions) - 24)]


def note_forced_depth_change(
    state: NativeMTPAdaptiveValueState,
    *,
    origin: int,
    target: int,
    cycle: int,
    reason: str,
) -> None:
    """Synchronize the rolling controller with an acceptance safety gate."""

    origin = _bounded_depth(origin)
    target = _bounded_depth(target)
    state.active_probe_origin = 0
    state.active_probe_target = 0
    # A safety gate is evidence that the workload phase changed. Do not carry
    # old wall-value loser backoff into the new phase.
    state.probe_revert_counts[:] = [0, 0, 0]
    if target == origin:
        return
    state.last_change_cycle = int(cycle)
    _record_transition(
        state,
        cycle=cycle,
        origin=origin,
        target=target,
        event="safety_change",
        reason=reason,
    )


def choose_depth_by_value(
    state: NativeMTPAdaptiveValueState,
    *,
    current_depth: int,
    depth_ceiling: int,
    cycle: int,
    minimum_samples: int,
    cooldown_cycles: int,
    probe_interval_cycles: int,
    hysteresis: float,
    raise_min_acceptance: float,
    initial_probe_cycles: int = 48,
) -> NativeMTPDepthDecision | None:
    """Choose a neighbouring depth using rolling confirmed tokens per second.

    A probe is a bounded experiment, not a permanent promotion.  Its target
    window is cleared before entry so an old workload phase cannot vote in the
    new phase.  After ``minimum_samples`` the target must beat the origin by
    ``hysteresis`` or the controller returns to the origin.
    """

    current = _bounded_depth(current_depth)
    ceiling = max(1, min(3, int(depth_ceiling or 1)))
    cycle = max(0, int(cycle))
    minimum_samples = max(2, int(minimum_samples))
    cooldown_cycles = max(minimum_samples, int(cooldown_cycles))
    probe_interval_cycles = max(cooldown_cycles, int(probe_interval_cycles))
    initial_probe_cycles = max(minimum_samples, int(initial_probe_cycles))
    hysteresis = max(0.0, float(hysteresis))

    probe_origin = int(state.active_probe_origin or 0)
    probe_target = int(state.active_probe_target or 0)
    if probe_target:
        if probe_target != current or abs(probe_target - probe_origin) != 1:
            state.active_probe_origin = 0
            state.active_probe_target = 0
        else:
            target_value = depth_value_tps(
                state, current, minimum_samples=minimum_samples
            )
            origin_value = depth_value_tps(
                state, probe_origin, minimum_samples=minimum_samples
            )
            if target_value is None or origin_value is None:
                return None
            state.active_probe_origin = 0
            state.active_probe_target = 0
            if target_value >= origin_value * (1.0 + hysteresis):
                state.probe_revert_counts[current - 1] = 0
                reason = (
                    f"probe_value D{current}={target_value:.2f}>=D{probe_origin}="
                    f"{origin_value:.2f}x{1.0 + hysteresis:.3f}"
                )
                state.last_change_cycle = cycle
                _record_transition(
                    state,
                    cycle=cycle,
                    origin=probe_origin,
                    target=current,
                    event="probe_keep",
                    reason=reason,
                )
                return NativeMTPDepthDecision(current, reason, "probe_keep")
            reason = (
                f"probe_value D{current}={target_value:.2f}<D{probe_origin}="
                f"{origin_value:.2f}x{1.0 + hysteresis:.3f}"
            )
            state.last_change_cycle = cycle
            state.probe_revert_counts[current - 1] = min(
                31, int(state.probe_revert_counts[current - 1] or 0) + 1
            )
            _record_transition(
                state,
                cycle=cycle,
                origin=current,
                target=probe_origin,
                event="probe_revert",
                reason=reason,
            )
            return NativeMTPDepthDecision(
                probe_origin, reason, "probe_revert"
            )

    current_value = depth_value_tps(
        state, current, minimum_samples=minimum_samples
    )
    if current_value is None:
        return None
    # The preserved MTP head cache starts cold after prompt prefill. The first
    # few cycles systematically undervalue the configured/profile seed and
    # previously made a validated D3 request fall to D2 at cycle 12, only to
    # climb back to D3 at cycle 36. Accumulate one stable initial window before
    # the first neighbor experiment; subsequent phase probes retain their
    # normal cooldown/backoff behavior.
    if not state.transitions and cycle < initial_probe_cycles:
        return None
    if cycle - int(state.last_change_cycle or 0) < cooldown_cycles:
        return None

    candidates: list[int] = []
    conditional_acceptance = depth_conditional_acceptance(
        state, current, minimum_samples=minimum_samples
    )
    if (
        current < ceiling
        and conditional_acceptance is not None
        and conditional_acceptance >= float(raise_min_acceptance)
    ):
        candidates.append(current + 1)
    if current > 1:
        candidates.append(current - 1)

    # Prefer an adjacent depth whose recent measured value already beats the
    # current phase, then perform a fresh probe before committing to it.
    for candidate in candidates:
        candidate_value = depth_value_tps(
            state, candidate, minimum_samples=minimum_samples
        )
        if (
            candidate_value is not None
            and candidate_value > current_value * (1.0 + hysteresis)
        ):
            break
    else:
        candidate = 0
        for possible in candidates:
            last_probe = int(state.last_probe_cycle[possible - 1] or 0)
            never_probed = last_probe <= 0
            losses = max(0, int(state.probe_revert_counts[possible - 1] or 0))
            # Cap at 4x: enough to stop long stable generations repeatedly
            # paying for the same loser, while still allowing a genuine phase
            # change to re-evaluate the neighbor within the same request.
            effective_interval = probe_interval_cycles * (1 << min(losses, 2))
            if never_probed or cycle - last_probe >= effective_interval:
                candidate = possible
                break
        if candidate <= 0:
            return None

    state.samples_by_depth[candidate - 1].clear()
    state.active_probe_origin = current
    state.active_probe_target = candidate
    state.last_probe_cycle[candidate - 1] = cycle
    state.last_change_cycle = cycle
    reason = (
        f"neighbor_probe D{current}->{candidate} current_value={current_value:.2f} "
        f"conditional_accept={conditional_acceptance if conditional_acceptance is not None else 'n/a'}"
    )
    _record_transition(
        state,
        cycle=cycle,
        origin=current,
        target=candidate,
        event="probe_start",
        reason=reason,
    )
    return NativeMTPDepthDecision(candidate, reason, "probe_start")


def adaptive_value_snapshot(
    state: NativeMTPAdaptiveValueState,
    *,
    minimum_samples: int = 1,
) -> dict[str, object]:
    values: dict[str, float | None] = {}
    conditional: dict[str, float | None] = {}
    sample_counts: dict[str, int] = {}
    for depth in (1, 2, 3):
        label = f"d{depth}"
        value = depth_value_tps(
            state, depth, minimum_samples=minimum_samples
        )
        rate = depth_conditional_acceptance(
            state, depth, minimum_samples=minimum_samples
        )
        values[label] = round(value, 3) if value is not None else None
        conditional[label] = round(rate, 6) if rate is not None else None
        sample_counts[label] = len(state.samples_by_depth[depth - 1])
    return {
        "basis": "rolling_wall_confirmed_tokens_per_second",
        "values_tok_s": values,
        "conditional_acceptance": conditional,
        "sample_counts": sample_counts,
        "active_probe": {
            "origin": int(state.active_probe_origin or 0),
            "target": int(state.active_probe_target or 0),
        },
        "probe_revert_counts": {
            f"d{depth}": int(state.probe_revert_counts[depth - 1] or 0)
            for depth in (1, 2, 3)
        },
        "transitions": list(state.transitions),
    }
