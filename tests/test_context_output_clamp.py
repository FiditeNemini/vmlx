"""Context-exhaustion clamp — output budget bounded by declared context.

Field-failure class (2026-08-15 directive): prompt + generation silently hit
the model's positional ceiling and the truncation was undiagnosable. The
clamp bounds max_tokens to (declared_context − prompt) and LOGS a
context-exhaustion notice whenever it binds.
"""

from __future__ import annotations

import logging

import pytest

from vmlx_engine import context_limits


@pytest.fixture(autouse=True)
def _reset():
    context_limits.set_declared_context_tokens(0)
    yield
    context_limits.set_declared_context_tokens(0)


def test_unknown_ceiling_passes_through():
    assert (
        context_limits.clamp_output_to_declared_context(5000, 32768) == 32768
    )


def test_within_budget_untouched():
    context_limits.set_declared_context_tokens(32768)
    assert (
        context_limits.clamp_output_to_declared_context(5875, 20000) == 20000
    )


def test_binding_clamp_logs_context_exhaustion_notice(caplog):
    context_limits.set_declared_context_tokens(32768)
    with caplog.at_level(logging.WARNING):
        clamped = context_limits.clamp_output_to_declared_context(
            5875, 32768, request_id="req-1"
        )
    assert clamped == 32768 - 5875
    assert any("CONTEXT EXHAUSTION" in rec.message for rec in caplog.records)


def test_prompt_at_or_over_ceiling_defers_to_prompt_guard():
    context_limits.set_declared_context_tokens(1000)
    # never returns a <1 budget on its own — the prompt-limit guard owns
    # over-ceiling prompts
    assert context_limits.clamp_output_to_declared_context(1200, 512) == 512


def test_env_toggle_disables(monkeypatch, caplog):
    context_limits.set_declared_context_tokens(32768)
    monkeypatch.setenv("VMLX_CONTEXT_OUTPUT_CLAMP", "0")
    with caplog.at_level(logging.WARNING):
        assert (
            context_limits.clamp_output_to_declared_context(5875, 32768)
            == 32768
        )
    assert not caplog.records


def test_none_max_tokens_passes_through():
    context_limits.set_declared_context_tokens(32768)
    assert context_limits.clamp_output_to_declared_context(5875, None) is None


def test_binding_clamp_records_registry_entry_and_pop_clears():
    context_limits.set_declared_context_tokens(32768)
    clamped = context_limits.clamp_output_to_declared_context(
        5875, 32768, request_id="req-registry"
    )
    assert clamped == 32768 - 5875
    record = context_limits.pop_context_clamp("req-registry")
    assert record == {
        "prompt_tokens": 5875,
        "requested_max_tokens": 32768,
        "clamped_max_tokens": 32768 - 5875,
        "declared_context_tokens": 32768,
    }
    assert context_limits.pop_context_clamp("req-registry") is None


def test_non_binding_requests_record_nothing():
    context_limits.set_declared_context_tokens(32768)
    context_limits.clamp_output_to_declared_context(
        100, 200, request_id="req-clean"
    )
    assert context_limits.pop_context_clamp("req-clean") is None


def test_registry_is_bounded():
    context_limits.set_declared_context_tokens(1000)
    for i in range(300):
        context_limits.clamp_output_to_declared_context(
            100, 5000, request_id=f"req-{i}"
        )
    # oldest entries evicted; newest retained
    assert context_limits.pop_context_clamp("req-0") is None
    assert context_limits.pop_context_clamp("req-299") is not None
