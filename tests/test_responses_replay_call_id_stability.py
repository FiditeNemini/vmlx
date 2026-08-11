# SPDX-License-Identifier: Apache-2.0
"""A replayed tool call must keep the same id every request.

Responses replay rebuilds assistant turns from client-supplied `function_call`
items. When the client omits `call_id` (allowed), the fallback used to be a fresh
uuid4 — so the SAME conversation produced a DIFFERENT id on every request. Chat
templates that render the id (gemma4's does) then emit different prompt bytes each
turn, and the prefix cache goes cold from the first historical tool call onward for
the rest of the conversation.
"""

from vmlx_engine.server import (
    _responses_input_to_messages,
    _responses_output_to_assistant_messages,
)


def _ids(messages):
    return [
        tc.get("id")
        for m in messages
        for tc in (m.get("tool_calls") or [])
    ]


HISTORY = [
    {"type": "message", "role": "user", "content": "weather in Oslo?"},
    {"type": "function_call", "name": "get_weather", "arguments": '{"city": "Oslo"}'},
    {"type": "function_call_output", "output": "12C"},
]


def test_input_replay_ids_are_stable_across_requests():
    first = _ids(_responses_input_to_messages(HISTORY))
    second = _ids(_responses_input_to_messages(HISTORY))
    assert first and all(first)
    assert first == second, "replaying identical history must yield identical ids"


def test_output_replay_ids_are_stable_across_requests():
    items = [
        {"type": "function_call", "name": "f", "arguments": '{"a": 1}'},
        {"type": "function_call", "name": "g", "arguments": '{"b": 2}'},
    ]
    assert _ids(_responses_output_to_assistant_messages(items)) == _ids(
        _responses_output_to_assistant_messages(items)
    )


def test_two_identical_calls_in_one_turn_stay_distinct():
    """Stability must not collapse repeated calls onto one id."""
    items = [
        {"type": "function_call", "name": "f", "arguments": '{"a": 1}'},
        {"type": "function_call", "name": "f", "arguments": '{"a": 1}'},
    ]
    ids = _ids(_responses_output_to_assistant_messages(items))
    assert len(ids) == 2 and len(set(ids)) == 2


def test_an_explicit_call_id_is_always_preserved():
    items = [{"type": "function_call", "name": "f", "arguments": "{}", "call_id": "call_mine"}]
    assert _ids(_responses_output_to_assistant_messages(items)) == ["call_mine"]
