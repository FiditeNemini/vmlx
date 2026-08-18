"""Draft acceptance for native MTP verify cycles.

Exact-match acceptance is only correct under greedy decode.  These tests pin
both arms: the default exact-match behaviour must stay byte-identical to the
loop it replaced, and the gated stochastic arm must implement the standard
min(1, p_target/p_draft) rule so drafts survive at temperature > 0.
"""

import math

import mlx.core as mx
import pytest

from vmlx_engine import mllm_batch_generator as gen


def _lp(pairs, vocab=8):
    """Build a log-prob row from {token_id: probability}."""
    probs = [1e-9] * vocab
    for token_id, prob in pairs.items():
        probs[token_id] = prob
    total = sum(probs)
    return mx.array([math.log(p / total) for p in probs])


def _accepted(draft_ids, target_ids, draft_lps=None, target_lps=None):
    return gen._native_mtp_accepted_count(
        draft_ids,
        target_ids,
        draft_lps if draft_lps is not None else [None] * len(draft_ids),
        target_lps if target_lps is not None else [None] * len(draft_ids),
    )


class TestExactMatchArm:
    """Default arm: identical to the original inline loop."""

    def test_all_matching_drafts_accepted(self):
        assert _accepted([3, 4, 5], [3, 4, 5, 9]) == 3

    def test_stops_at_first_mismatch(self):
        assert _accepted([3, 4, 5], [3, 7, 5, 9]) == 1

    def test_leading_mismatch_accepts_nothing(self):
        assert _accepted([3, 4], [7, 4, 9]) == 0

    def test_no_drafts_accepts_nothing(self):
        assert _accepted([], [9]) == 0

    def test_mismatch_is_not_rescued_when_gate_is_off(self, monkeypatch):
        """Even with distributions available, the default arm must break."""
        monkeypatch.setattr(gen, "_NATIVE_MTP_STOCHASTIC_ACCEPT", False)
        # p_target(draft) far exceeds p_draft(draft): the stochastic arm would
        # accept this outright, so it proves the gate is what decides.
        assert (
            _accepted(
                [1],
                [2],
                draft_lps=[_lp({1: 0.05, 2: 0.95})],
                target_lps=[_lp({1: 0.95, 2: 0.05})],
            )
            == 0
        )


class TestStochasticArm:
    @pytest.fixture(autouse=True)
    def _enable(self, monkeypatch):
        monkeypatch.setattr(gen, "_NATIVE_MTP_STOCHASTIC_ACCEPT", True)

    def test_matching_draft_still_accepted(self):
        assert _accepted([3], [3, 9], draft_lps=[None], target_lps=[None]) == 1

    def test_mismatch_accepted_when_target_prefers_the_draft(self):
        """p_target/p_draft >= 1 accepts deterministically, no draw needed."""
        assert (
            _accepted(
                [1],
                [2],
                draft_lps=[_lp({1: 0.05, 2: 0.95})],
                target_lps=[_lp({1: 0.95, 2: 0.05})],
            )
            == 1
        )

    def test_mismatch_rejected_when_target_finds_draft_near_impossible(self):
        """p_target/p_draft ~ 0 must reject for any draw in (0, 1]."""
        assert (
            _accepted(
                [1],
                [2],
                draft_lps=[_lp({1: 0.999, 2: 0.001})],
                target_lps=[_lp({1: 1e-9, 2: 0.999})],
            )
            == 0
        )

    def test_falls_back_to_exact_match_without_distributions(self):
        """Greedy/compact samplers expose no rows; exact match is correct."""
        assert _accepted([1, 2], [9, 2], draft_lps=[None, None], target_lps=[None, None]) == 0

    def test_acceptance_is_never_negative_or_past_the_draft_count(self):
        accepted = _accepted(
            [1, 2],
            [1, 2, 3],
            draft_lps=[_lp({1: 0.9}), _lp({2: 0.9})],
            target_lps=[_lp({1: 0.9}), _lp({2: 0.9})],
        )
        assert 0 <= accepted <= 2

    def test_ratio_is_read_at_the_draft_token_not_the_target_token(self):
        """Regression: using p at the target's own token inverts the test.

        Target overwhelmingly prefers token 2 (what it sampled) and considers
        the draft token 1 nearly impossible, so the draft must be rejected.
        Reading the ratio at token 2 instead would accept it every time.
        """
        rejected = 0
        for _ in range(12):
            rejected += (
                _accepted(
                    [1],
                    [2],
                    draft_lps=[_lp({1: 0.98, 2: 0.02})],
                    target_lps=[_lp({1: 1e-9, 2: 0.999})],
                )
                == 0
            )
        assert rejected == 12

    def test_malformed_distribution_does_not_raise(self):
        """A short/ragged row must end the cycle, never propagate an error."""
        assert _accepted([5], [6], draft_lps=[mx.array([0.0])], target_lps=[mx.array([0.0])]) == 0
