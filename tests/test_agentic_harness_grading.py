"""Offline regressions for the agentic scenario harness grading core.

Each test encodes a defect Codex's audit found in the first harness cut:
the city fixture blessed whichever city the model called, repeated
continuation calls were only notes, growth targeting drifted 2.4x past
its milestone, reuse was graded against the previous turn's cached count,
and deep sleep was an arbitrary fixed delay.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

from api_agentic_scenarios import (  # noqa: E402
    _grade_cache_match,
    best_cache_match,
    classify_cache_candidate,
    expected_block_reuse,
    fit_filler_to_target,
    grade_tool_call,
)


class TestToolCallGrading:
    def test_wrong_city_fails_even_when_valid_json(self):
        calls = [{"id": "call_1", "name": "get_weather", "arguments": '{"city": "Paris"}'}]
        call, errors = grade_tool_call(calls, "Berlin")
        assert call is None
        assert any("manifest requested 'Berlin'" in error for error in errors)

    def test_manifest_city_match_is_case_insensitive(self):
        calls = [{"id": "call_1", "name": "get_weather", "arguments": '{"city": "berlin"}'}]
        call, errors = grade_tool_call(calls, "Berlin")
        assert errors == []
        assert call is not None

    def test_zero_or_multiple_calls_fail(self):
        assert grade_tool_call([], "Paris")[0] is None
        two = [
            {"id": "a", "name": "get_weather", "arguments": '{"city": "Paris"}'},
            {"id": "b", "name": "get_weather", "arguments": '{"city": "Paris"}'},
        ]
        call, errors = grade_tool_call(two, "Paris")
        assert call is None and errors

    def test_invalid_json_and_missing_id_fail(self):
        bad_json = [{"id": "a", "name": "get_weather", "arguments": "{city: Paris"}]
        assert grade_tool_call(bad_json, "Paris")[0] is None
        no_id = [{"id": None, "name": "get_weather", "arguments": '{"city": "Paris"}'}]
        call, errors = grade_tool_call(no_id, "Paris")
        assert call is None
        assert any("no id" in error for error in errors)


class TestBlockReuseOracle:
    def test_expected_reuse_is_block_floor_of_lcp(self):
        # Divergence well before the predecessor's terminal token: only the
        # chain-hash block-floor branch applies.
        previous = list(range(1000))
        current = list(range(500)) + [-1] * 500
        assert expected_block_reuse(previous, current, 64) == (500 // 64) * 64

    def test_disjoint_prefixes_expect_zero(self):
        assert expected_block_reuse([1, 2, 3], [4, 5, 6], 64) == 0

    def test_identical_prefix_uses_exact_n_minus_1_boundary_not_block_floor(self):
        # Content identical through the predecessor's own last token: the
        # N-1 partial-terminal-block index applies exactly, NOT block_floor.
        # A 1000-token predecessor is not a multiple of 64 -- this is the
        # exact shape that caught the block_floor(lcp - 1) composition bug
        # (block_floor(999) == 960, but the true safe extent is 999).
        previous = list(range(1000))
        current = list(range(999)) + [12345]
        assert expected_block_reuse(previous, current, 64) == 999
        assert expected_block_reuse(previous, current, 64) != (999 // 64) * 64

    def test_identical_prefix_full_blocks(self):
        tokens = list(range(640))
        assert expected_block_reuse(tokens, tokens + [1], 64) == 639


class TestGrowthFitting:
    def test_filler_converges_to_target(self):
        # ~1 token per 4 characters, a crude but monotonic fake tokenizer.
        def count(text: str) -> int:
            return max(1, len(text) // 4)

        filler = fit_filler_to_target(
            count, render_overhead_tokens=64, current_tokens=1000,
            target_tokens=8192, seed=7, tolerance=512,
        )
        measured = count(filler)
        needed = 8192 - 1000 - 64
        assert abs(measured - needed) <= 512, (measured, needed)


class TestDeepSleepConfirmation:
    def test_polls_health_until_confirmed(self, monkeypatch):
        import api_agentic_scenarios as harness

        states = iter([
            {"status": "healthy", "load_progress": {"model_loaded": True}},
            {"status": "standby_deep", "load_progress": {"model_loaded": True}},
            {"status": "standby_deep", "load_progress": {"model_loaded": False}},
        ])

        class _FakeResponse:
            def read(self):
                return b'{"status": "deep_sleep"}'

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        monkeypatch.setattr(harness.urllib.request, "urlopen", lambda *a, **k: _FakeResponse())
        errors = harness.confirm_deep_sleep(
            "http://x", None, poll=lambda: next(states), timeout_s=5.0, interval_s=0.0
        )
        assert errors == []

    def test_times_out_when_health_never_confirms(self, monkeypatch):
        import api_agentic_scenarios as harness

        class _FakeResponse:
            def read(self):
                return b'{"status": "deep_sleep"}'

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        monkeypatch.setattr(harness.urllib.request, "urlopen", lambda *a, **k: _FakeResponse())
        errors = harness.confirm_deep_sleep(
            "http://x", None,
            poll=lambda: {"status": "healthy", "load_progress": {"model_loaded": True}},
            timeout_s=0.2, interval_s=0.05,
        )
        assert errors and "never confirmed" in errors[0]


class TestCacheMatchKindsAndBestMatch:
    """Six scenarios named by Codex's third re-audit: chain-block,
    exact-partial (N-1), and best-match-across-history must be modeled
    separately, never collapsed into one universal formula."""

    def test_tools_off_to_on_diverges_at_the_prompt_head(self):
        # A tool schema injected before the conversation shifts every
        # downstream token position -- only the shared system preamble (well
        # under one block) survives, never a chain-block or N-1 match.
        no_tools = [1, 2, 3, 4] + list(range(100, 200))
        tools_on = [1, 2, 3, 4] + list(range(900, 1000))  # diverges at index 4
        match = classify_cache_candidate(no_tools, tools_on, 64)
        assert match["match_kind"] == "none"
        assert match["safe_expected"] == 0

    def test_tools_on_continuation_uses_exact_n_minus_1(self):
        # The continuation appends assistant tool_call + tool result after
        # an UNCHANGED tools-on selection prompt -- content is identical
        # through the selection prompt's own last token, so the N-1 exact
        # boundary applies (not floored to a block).
        selection_prompt = [7] * 6432  # not a multiple of 64
        continuation = selection_prompt[:-1] + [8, 9, 10]
        match = classify_cache_candidate(selection_prompt, continuation, 64)
        assert match["match_kind"] == "n_minus_1_exact"
        assert match["safe_expected"] == 6431

    def test_tools_on_to_off_selects_the_older_compatible_chain(self):
        # History: an early no-tools chain, then a tools-on turn that
        # diverges from it near the start. A LATER no-tools turn that shares
        # no_tools_v1's PREFIX (but then diverges before v1's own end, e.g.
        # a shorter follow-up question) must match turn 0 via chain-block,
        # not the intervening tools-on turn 1 it shares almost nothing with.
        no_tools_v1 = [1, 2, 3] + list(range(100, 100 + 200))  # len 203
        tools_on = [1, 2, 3] + list(range(900, 900 + 200))
        no_tools_v2 = no_tools_v1[:150] + [-1, -2, -3]  # LCP=150 with v1, len<v1
        best = best_cache_match([no_tools_v1, tools_on], no_tools_v2, 64)
        assert best["source_index"] == 0
        assert best["match_kind"] == "chain_block"
        assert best["safe_expected"] == 128  # floor(150/64)*64

    def test_off_boundary_exact_partial_floors_to_the_block(self):
        # Divergence lands mid-block (not on a 64-token boundary) and well
        # before the predecessor's own end -- only full chain-hash blocks
        # are safe, the partial final block is NOT credited.
        previous = list(range(1000))
        current = list(range(150)) + [-1] * 850  # LCP=150, floor(150/64)*64=128
        match = classify_cache_candidate(previous, current, 64)
        assert match["match_kind"] == "chain_block"
        assert match["safe_expected"] == 128
        assert match["safe_expected"] < match["lcp"]

    def test_n_minus_1_wins_over_an_earlier_chain_block_candidate(self):
        # best_cache_match must pick the candidate with the HIGHEST safe
        # extent, not the first or the most recent -- here an exact N-1
        # match against turn 1 beats a larger raw source_len chain-block
        # partial match against turn 0.
        turn0 = list(range(2000))  # current diverges from turn0 early
        turn1 = list(range(5000, 5999))  # current matches turn1 through N-1
        current = turn1[:-1] + [42]
        best = best_cache_match([turn0, turn1], current, 64)
        assert best["source_index"] == 1
        assert best["match_kind"] == "n_minus_1_exact"
        assert best["safe_expected"] == 998

    def test_hybrid_companion_over_restore_is_flagged_not_silently_cleared(self):
        # A hybrid SSM/GDN-companion family's separate short key CAN
        # legitimately explain extra credit this LCP oracle doesn't model --
        # but that must still surface as a distinguishable note, never
        # silently pass as if nothing happened.
        match = {"safe_expected": 0, "match_kind": "none", "lcp": 40,
                 "source_len": 6141, "source_index": 0}
        errors = _grade_cache_match(256, match, "server-verified", hybrid_companion_family=True)
        assert len(errors) == 1
        assert "OVER-restore" in errors[0] and "hybrid-companion" in errors[0]

    def test_non_hybrid_over_restore_has_no_exemption(self):
        match = {"safe_expected": 0, "match_kind": "none", "lcp": 40,
                 "source_len": 6141, "source_index": 0}
        errors = _grade_cache_match(256, match, "server-verified", hybrid_companion_family=False)
        assert len(errors) == 1
        assert "OVER-restore" in errors[0] and "hybrid-companion" not in errors[0]

    def test_under_restore_always_fails_regardless_of_hybrid_flag(self):
        match = {"safe_expected": 6400, "match_kind": "chain_block", "lcp": 6432,
                 "source_len": 6433, "source_index": 0}
        for hybrid in (True, False):
            errors = _grade_cache_match(128, match, "server-verified", hybrid_companion_family=hybrid)
            assert len(errors) == 1 and "UNDER-restore" in errors[0]

    def test_no_slack_exact_boundary_passes_clean(self):
        match = {"safe_expected": 6400, "match_kind": "chain_block", "lcp": 6432,
                 "source_len": 6433, "source_index": 0}
        assert _grade_cache_match(6400, match, "server-verified", False) == []

    def test_responses_approximate_render_never_hard_fails(self):
        # The Responses wire's client reconstruction was proven wrong
        # against server telemetry repeatedly this campaign -- any
        # discrepancy downgrades to an UNVERIFIED note, never UNDER/OVER.
        match = {"safe_expected": 6400, "match_kind": "chain_block", "lcp": 6432,
                 "source_len": 6433, "source_index": 0}
        errors = _grade_cache_match(0, match, "approximate-unverified", False)
        assert len(errors) == 1
        assert errors[0].startswith("UNVERIFIED")
        assert "UNDER-restore" not in errors[0] and "OVER-restore" not in errors[0]


class TestRenderConfidenceNeverTrustsWireType:
    """Codex re-audit, 2026-08-28: 'Never label Chat "server-verified" from
    wire type.' No wire is trusted by default -- confirmed server-exact
    rendering is the only thing that can ever upgrade this."""

    def test_chat_wire_no_tool_history_is_still_unverified(self):
        from api_agentic_scenarios import render_confidence_for

        assert render_confidence_for("chat", False) == "approximate-unverified"

    def test_chat_wire_with_tool_history_is_unverified(self):
        from api_agentic_scenarios import render_confidence_for

        assert render_confidence_for("chat", True) == "approximate-unverified"

    def test_responses_wire_is_unverified_both_states(self):
        from api_agentic_scenarios import render_confidence_for

        assert render_confidence_for("responses", False) == "approximate-unverified"
        assert render_confidence_for("responses", True) == "approximate-unverified"
