"""Adaptive depth must be able to climb, not only fall.

The controller could only ever LOWER draft depth, so a bundle whose tuning
sidecar says depth 1 stayed at depth 1 no matter how well its head performed.
That caps throughput at (1 + acceptance) tokens per cycle. MTPLX runs the same
model family at depth 3 with 0.95/0.88/0.80 acceptance = 3.46 tokens per cycle,
which is the whole difference between a 1.5x and a 2.5x speedup.

Raising is timid on purpose: near-perfect shallow acceptance, over a real
sample, one step at a time, under a ceiling that every demotion lowers.
"""

import pytest

from vmlx_engine import mllm_batch_generator as gen


class _Stats:
    def __init__(self, drafted_by_depth, accepted_by_depth, cycles=None):
        self.drafted_by_depth = list(drafted_by_depth)
        self.accepted_by_depth = list(accepted_by_depth)
        self.cycles = cycles if cycles is not None else max(drafted_by_depth)
        self.mtp_forwards = self.cycles


class _State:
    def __init__(self, depth, drafted_by_depth, accepted_by_depth, ceiling=3):
        self.depth = depth
        self.depth_ceiling = ceiling
        self.stats = _Stats(drafted_by_depth, accepted_by_depth)
        self.ar_fallback_pending = False
        self.ar_fallback_reason = None


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "VMLINUX_NATIVE_MTP_ADAPTIVE_RAISE",
        "VMLX_NATIVE_MTP_ADAPTIVE_RAISE",
        "VMLINUX_NATIVE_MTP_RAISE_MIN_ACCEPT",
        "VMLX_NATIVE_MTP_RAISE_MIN_ACCEPT",
        "VMLINUX_NATIVE_MTP_RAISE_MIN_SAMPLE",
        "VMLX_NATIVE_MTP_RAISE_MIN_SAMPLE",
        "VMLINUX_NATIVE_MTP_ADAPTIVE_WARMUP_CYCLES",
        "VMLX_NATIVE_MTP_ADAPTIVE_WARMUP_CYCLES",
        "VMLINUX_NATIVE_MTP_COST_FALLBACK",
        "VMLX_NATIVE_MTP_COST_FALLBACK",
    ):
        monkeypatch.delenv(name, raising=False)


class TestRaises:
    def test_excellent_d1_climbs_to_depth_two(self):
        """0.95 at depth 1 — the MTPLX-grade head we are building toward."""
        state = _State(1, [200, 0, 0], [190, 0, 0])
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 2

    def test_climbs_one_step_at_a_time(self):
        """Never jump 1 -> 3 on a single observation."""
        state = _State(1, [200, 0, 0], [199, 0, 0])
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 2

    def test_depth_two_can_climb_to_three(self):
        state = _State(2, [200, 200, 0], [195, 190, 0])
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 3

    def test_never_exceeds_three(self):
        state = _State(3, [200, 200, 200], [199, 199, 199])
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 3


class TestDoesNotRaise:
    def test_current_bundle_acceptance_never_triggers_a_raise(self):
        """53-65% is what the 6-bit-head bundle actually measures live.

        This is the safety property: shipping the raise path must not change
        behaviour for the bundles we serve today.
        """
        for accepted in (34, 40, 42):  # 53.1%, 62.5%, 65.6% of 64
            state = _State(1, [64, 0, 0], [accepted, 0, 0])
            gen._native_mtp_maybe_adapt_depth("req", state)
            assert state.depth == 1

    def test_small_sample_does_not_trigger(self):
        """A lucky opening streak must not promote the whole request."""
        state = _State(1, [12, 0, 0], [12, 0, 0])
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 1

    def test_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_RAISE", "0")
        state = _State(1, [200, 0, 0], [195, 0, 0])
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 1

    def test_threshold_is_tunable(self, monkeypatch):
        monkeypatch.setenv("VMLX_NATIVE_MTP_RAISE_MIN_ACCEPT", "0.60")
        state = _State(1, [200, 0, 0], [130, 0, 0])  # 65%
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 2


class TestHysteresis:
    def test_ceiling_blocks_returning_to_a_failed_depth(self):
        """A depth demoted for poor acceptance is never retried."""
        state = _State(1, [200, 0, 0], [195, 0, 0], ceiling=1)
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 1

    def test_demotion_lowers_the_ceiling(self):
        """Depth 2 with bad d2 must demote AND record the ceiling."""
        state = _State(2, [200, 200, 0], [195, 40, 0])  # d2 = 20%
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 1
        assert state.depth_ceiling == 1

    def test_no_oscillation_after_a_demotion(self):
        """Demote on bad d2, then excellent d1 must NOT climb back."""
        state = _State(2, [200, 200, 0], [195, 40, 0])
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 1
        # Now d1 looks superb; the ceiling must still hold it down.
        state.stats = _Stats([400, 200, 0], [395, 40, 0])
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 1
