"""Parity fixes: controls that existed on one surface and silently died on another.

Each of these was traced to both ends — a value the user sets, and the place
that never receives it. None of them error; they all silently do nothing,
which is why they survived.
"""

import pytest

from vmlx_engine.api.anthropic_adapter import AnthropicRequest, to_chat_completion
from vmlx_engine.mllm_scheduler import (
    MLLMSchedulerConfig,
    _resolve_prefix_cache_byte_budget,
)


class TestAnthropicDialectDropsControls:
    """/v1/messages never ACCEPTED these, so pydantic dropped them and the
    server resolved None — while chat, responses and ollama all honour them."""

    def _req(self, **kw):
        return AnthropicRequest(
            model="m", messages=[{"role": "user", "content": "hi"}],
            max_tokens=8, **kw
        )

    def test_min_p_survives_the_conversion(self):
        assert to_chat_completion(self._req(min_p=0.05)).min_p == 0.05

    def test_repetition_penalty_survives(self):
        assert to_chat_completion(self._req(repetition_penalty=1.1)).repetition_penalty == 1.1

    def test_cache_salt_survives(self):
        assert to_chat_completion(self._req(cache_salt="tenant-a")).cache_salt == "tenant-a"

    def test_skip_prefix_cache_survives(self):
        assert to_chat_completion(self._req(skip_prefix_cache=True)).skip_prefix_cache is True

    def test_omitted_controls_stay_none(self):
        """Absence must remain absence — not a fabricated default that would
        override the bundle's own config."""
        c = to_chat_completion(self._req())
        assert c.min_p is None
        assert c.repetition_penalty is None
        assert c.cache_salt is None
        assert c.skip_prefix_cache is None


class TestMLLMPrefixCacheByteBudget:
    """--prefix-cache-max-bytes reached argv, but MLLMSchedulerConfig never
    declared the field, so every VL session ignored it and fell back to the
    RAM-percent default. One-of-two-schedulers."""

    def test_explicit_budget_is_honoured_on_the_mllm_lane(self):
        cfg = MLLMSchedulerConfig(prefix_cache_max_bytes=777_000_000)
        assert _resolve_prefix_cache_byte_budget(cfg) == 777_000_000

    def test_explicit_budget_wins_over_the_ram_fallback(self):
        cfg = MLLMSchedulerConfig(
            prefix_cache_max_bytes=123_456_789, cache_memory_mb=8192
        )
        assert _resolve_prefix_cache_byte_budget(cfg) == 123_456_789

    def test_without_an_explicit_budget_the_fallback_still_applies(self):
        cfg = MLLMSchedulerConfig(cache_memory_mb=2048)
        assert _resolve_prefix_cache_byte_budget(cfg) == 2048 * 1024 * 1024


class TestHy3MinimalEffort:
    """Asking for the LOWEST reasoning effort turned reasoning OFF: hy3's
    template only opens the rail for low/high, and "minimal" fell through
    every branch to a silent no_think — while every stamped family clamps the
    same value to "low"."""

    def _map(self, effort, enable_thinking=None):
        from vmlx_engine.server import _normalize_hy3_reasoning_effort

        return _normalize_hy3_reasoning_effort(effort, enable_thinking=enable_thinking)

    def test_minimal_clamps_to_low_not_off(self):
        assert self._map("minimal") == "low", (
            "minimal fell through to no_think — the lowest effort silently "
            "disabled reasoning entirely"
        )

    @pytest.mark.parametrize("effort,expected", [
        ("low", "low"), ("high", "high"),
        ("medium", "high"), ("xhigh", "high"), ("max", "high"),
        ("none", "no_think"), ("off", "no_think"), ("", "no_think"),
    ])
    def test_the_rest_of_the_ladder_is_unchanged(self, effort, expected):
        assert self._map(effort) == expected

    def test_explicit_thinking_off_still_wins(self):
        assert self._map("minimal", enable_thinking=False) == "no_think"
