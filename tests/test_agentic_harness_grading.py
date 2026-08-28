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
        previous = list(range(1000))
        current = list(range(990)) + [7, 8, 9]
        assert expected_block_reuse(previous, current, 64) == (990 // 64) * 64

    def test_disjoint_prefixes_expect_zero(self):
        assert expected_block_reuse([1, 2, 3], [4, 5, 6], 64) == 0

    def test_identical_prefix_full_blocks(self):
        tokens = list(range(640))
        assert expected_block_reuse(tokens, tokens + [1], 64) == 640


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
