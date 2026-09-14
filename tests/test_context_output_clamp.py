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
from vmlx_engine.errors import PromptTooLongError


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


@pytest.mark.parametrize("declared", [1000, 262144, 1048576])
@pytest.mark.parametrize("excess", [0, 1, 200])
def test_prompt_without_output_room_is_typed_rejection(declared, excess):
    context_limits.set_declared_context_tokens(declared)
    request_id = f"no-room-{declared}-{excess}"
    with pytest.raises(PromptTooLongError) as caught:
        context_limits.clamp_output_to_declared_context(
            declared + excess, 512, request_id=request_id
        )
    assert caught.value.prompt_tokens == declared + excess
    assert caught.value.max_prompt_tokens == declared - 1
    assert caught.value.request_id == request_id
    assert caught.value.source == PromptTooLongError.DECLARED_CONTEXT_SOURCE
    assert f"declared context of {declared} tokens" in str(caught.value)
    assert "one output token" in str(caught.value)
    assert context_limits.pop_context_clamp(request_id) is None


def test_one_output_slot_remains_valid_and_larger_output_is_reported():
    context_limits.set_declared_context_tokens(262144)
    assert context_limits.clamp_output_to_declared_context(262143, 1) == 1
    assert context_limits.clamp_output_to_declared_context(
        262143, 512, request_id="last-slot"
    ) == 1
    assert context_limits.pop_context_clamp("last-slot") == {
        "prompt_tokens": 262143,
        "requested_max_tokens": 512,
        "clamped_max_tokens": 1,
        "declared_context_tokens": 262144,
    }


def test_one_million_declared_context_is_not_capped_at_256k():
    context_limits.set_declared_context_tokens(1048576)
    assert context_limits.clamp_output_to_declared_context(300000, 4096) == 4096


def test_declared_boundary_preserves_diagnostic_opt_out(monkeypatch):
    context_limits.set_declared_context_tokens(262144)
    monkeypatch.setenv("VMLX_CONTEXT_OUTPUT_CLAMP", "0")
    assert context_limits.clamp_output_to_declared_context(262144, 512) == 512


def test_no_room_error_round_trip_keeps_declared_limit_and_correct_remedy():
    import json
    from vmlx_engine.engine.batched import _raise_prompt_too_long_from_output
    from vmlx_engine.request import RequestOutput
    from vmlx_engine.server import _prompt_too_long_response_from_error

    output = RequestOutput(
        request_id="no-room-wire",
        finished=True,
        finish_reason="error",
        error_code="prompt_too_long",
        error_prompt_tokens=262144,
        error_max_prompt_tokens=262143,
        error_source=PromptTooLongError.DECLARED_CONTEXT_SOURCE,
    )
    with pytest.raises(PromptTooLongError) as caught:
        _raise_prompt_too_long_from_output(output)
    response = _prompt_too_long_response_from_error(caught.value)
    assert response.status_code == 413
    error = json.loads(response.body)["error"]
    assert error["code"] == "prompt_too_long"
    assert "declared context of 262144 tokens" in error["message"]
    assert "one output token" in error["message"]
    assert "--max-prompt-tokens" not in error["message"]


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


def test_responses_length_terminal_attaches_clamp_record():
    import vmlx_engine.server as server

    context_limits.set_declared_context_tokens(128000)
    context_limits.clamp_output_to_declared_context(
        22, 127990, request_id="resp-term-1"
    )
    terminal = server._responses_terminal_state(
        "length", request_id="resp-term-1"
    )
    details = terminal.incomplete_details
    assert details["reason"] == "max_output_tokens"
    assert details["context_exhaustion"]["clamped_max_tokens"] == 127978
    # peek is non-destructive: a second derivation still sees the record
    again = server._responses_terminal_state(
        "length", request_id="resp-term-1"
    )
    assert "context_exhaustion" in again.incomplete_details
    context_limits.pop_context_clamp("resp-term-1")


def test_responses_length_terminal_without_record_is_spec_shaped():
    import vmlx_engine.server as server

    terminal = server._responses_terminal_state(
        "length", request_id="resp-term-none"
    )
    assert terminal.incomplete_details == {"reason": "max_output_tokens"}
