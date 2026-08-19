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
    def test_depth_three_gate_compares_joint_rate_to_joint_floor(self):
        """A healthy conditional d3 rate must not be compared to 0.85 raw."""
        # Joint d2=85%, joint d3=75%, so conditional d3=88.2% and passes.
        state = _State(3, [200, 200, 200], [190, 170, 150])
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 3

    def test_depth_three_gate_still_demotes_bad_conditional_rate(self):
        # Joint d2=85%, joint d3=65%, so conditional d3=76.5% and fails.
        state = _State(3, [200, 200, 200], [190, 170, 130])
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 2
        assert state.depth_ceiling == 2

    def test_accelerated_depth_three_waits_for_a_real_sample(self, monkeypatch):
        from vmlx_engine.metal import native_mtp_verify_qmm

        monkeypatch.setattr(
            native_mtp_verify_qmm,
            "native_mtp_verify_qmm_active",
            lambda: True,
        )
        # The exact live cold window was 22/48 joint d3. The accelerated lane
        # waits for 128 drafts because the completed run recovered to 582/767.
        state = _State(3, [48, 48, 48], [45, 30, 22])
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 3

    def test_accelerated_depth_three_uses_profitable_floor(self, monkeypatch):
        from vmlx_engine.metal import native_mtp_verify_qmm

        monkeypatch.setattr(
            native_mtp_verify_qmm,
            "native_mtp_verify_qmm_active",
            lambda: True,
        )
        # Representative 128-cycle window: joint d2=64.8%, joint d3=47.7%,
        # therefore conditional d3=73.5%. This remains profitable with the
        # four-row verifier while independently satisfying the D2 gate.
        state = _State(3, [128, 128, 128], [118, 83, 61])
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 3

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


class TestRestoredPrefixGates:
    """Restored-prefix requests start with a COLD head cache (backbone
    hiddens are not stored), so early gate windows measure a context-starved
    head, not the bundle. Live A/B: run 3 of a warm conversation demoted
    D3->D1 at cycle 129 on d2=0.574 that recovers to ~0.85 warm, and the
    lowered ceiling made 17.4 t/s permanent (cold run 1 = 40.2 t/s)."""

    def _state(self, depth, drafted, accepted, restored=True, ceiling=3):
        state = _State(depth, drafted, accepted, ceiling=ceiling)
        state.restored_prefix = restored
        return state

    def test_cold_window_sample_does_not_demote_restored_request(self):
        # 129 drafted at joint d2=0.574 — the exact live demotion window.
        # Fresh requests demote here; restored ones must wait for 4x sample.
        state = self._state(3, [129, 129, 129], [110, 74, 50])
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 3

    def test_fresh_request_still_demotes_on_the_same_window(self):
        state = self._state(3, [129, 129, 129], [110, 74, 50], restored=False)
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth < 3

    def test_restored_demote_at_full_sample_keeps_ceiling(self):
        # Even when a restored request eventually demotes (sustained bad d2
        # over the stretched sample), the ceiling stays put so the raise
        # path can climb back once the head cache is warm.
        state = self._state(2, [800, 800, 0], [780, 160, 0])  # d2 20%
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 1
        assert state.depth_ceiling == 3

    def test_fresh_demote_still_lowers_ceiling(self):
        state = self._state(2, [800, 800, 0], [780, 160, 0], restored=False)
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 1
        assert state.depth_ceiling == 1
