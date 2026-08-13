"""The UI-vs-API parity probe must observe the endpoint the app actually uses.

`_log_inbound_request_fields` was written against the chat-completions shape,
but the vMLX app talks to `/v1/responses`. There the token cap is
`max_output_tokens`, and reasoning may arrive NESTED as
`reasoning={"effort": ...}` rather than as a flat `reasoning_effort`.

Because the probe only inspected the flat names, a live capture of a real UI
turn logged `model=... stream=True` and nothing else, which reads as "the UI
sent no reasoning setting at all". A parity probe that under-reports on the
endpoint it exists to observe is worse than no probe: its silence looks like
evidence.
"""

import os
from types import SimpleNamespace

import pytest

from vmlx_engine.server import _log_inbound_request_fields


class _Req(SimpleNamespace):
    """Stand-in for a parsed request; the probe uses getattr throughout."""


def _capture(tmp_path, monkeypatch, **fields):
    sink = tmp_path / "fields.log"
    monkeypatch.setenv("VMLX_LOG_REQUEST_FIELDS", str(sink))
    request = _Req(messages=None, tools=None, **fields)
    _log_inbound_request_fields("responses", request)
    return sink.read_text(encoding="utf-8") if sink.exists() else ""


def test_probe_is_off_without_the_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv("VMLX_LOG_REQUEST_FIELDS", raising=False)
    sink = tmp_path / "fields.log"
    _log_inbound_request_fields("responses", _Req(model="m", messages=None, tools=None))
    assert not sink.exists()


def test_nested_responses_reasoning_is_reported(tmp_path, monkeypatch):
    # The exact shape the Responses API uses. Previously invisible.
    text = _capture(tmp_path, monkeypatch, model="m", reasoning={"effort": "max"})
    assert "reasoning=" in text
    assert "max" in text


def test_flat_reasoning_effort_still_reported(tmp_path, monkeypatch):
    text = _capture(tmp_path, monkeypatch, model="m", reasoning_effort="high")
    assert "reasoning_effort='high'" in text


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_output_tokens", 512),   # Responses spelling of the cap
        ("max_thinking_tokens", 128),
        ("thinking_mode", "auto"),
        ("temperature", 0.6),
        ("top_p", 0.95),
        ("tool_choice", "auto"),
    ],
)
def test_settings_bearing_fields_are_reported(tmp_path, monkeypatch, field, value):
    text = _capture(tmp_path, monkeypatch, model="m", **{field: value})
    assert field in text


def test_message_content_is_never_logged(tmp_path, monkeypatch):
    # The probe is meant to be safe to switch on against a real session, so it
    # reports counts and settings only — prompts can carry private data.
    text = _capture(
        tmp_path,
        monkeypatch,
        model="m",
        temperature=0.6,
    )
    assert "SECRET" not in text
    text2 = _capture(
        tmp_path,
        monkeypatch,
        model="m",
        instructions="SECRET system prompt",
        input="SECRET user text",
    )
    assert "SECRET" not in text2


def test_a_bad_sink_path_does_not_raise(tmp_path, monkeypatch):
    # Diagnostics must never take a request down.
    monkeypatch.setenv("VMLX_LOG_REQUEST_FIELDS", str(tmp_path / "no" / "such" / "d.log"))
    _log_inbound_request_fields("responses", _Req(model="m", messages=None, tools=None))
