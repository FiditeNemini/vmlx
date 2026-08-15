# SPDX-License-Identifier: Apache-2.0
"""Responses API conversation-history regression tests."""

from __future__ import annotations

import json
import inspect
from types import SimpleNamespace

import pytest


def _sse_payloads(events: list[str], event_type: str) -> list[dict]:
    payloads: list[dict] = []
    prefix = f"event: {event_type}\n"
    for event in events:
        if not event.startswith(prefix):
            continue
        for line in event.splitlines():
            if line.startswith("data: "):
                payloads.append(json.loads(line[6:]))
                break
    return payloads


def test_responses_output_history_converter_accepts_pydantic_and_dict_items():
    from vmlx_engine.api.models import ResponsesOutputMessage, ResponsesOutputText
    from vmlx_engine.server import _responses_output_to_assistant_messages

    pydantic_items = [
        ResponsesOutputMessage(
            role="assistant",
            content=[ResponsesOutputText(type="output_text", text="visible answer")],
        )
    ]
    dict_items = [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "streamed answer"}],
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": '{"q":"teal"}',
        },
    ]

    assert _responses_output_to_assistant_messages(pydantic_items) == [
        {"role": "assistant", "content": "visible answer"}
    ]
    assert _responses_output_to_assistant_messages(dict_items) == [
        {
            "role": "assistant",
            "content": "streamed answer",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": {"q": "teal"}},
                }
            ],
        },
    ]


def test_responses_reasoning_replays_before_tool_call_and_visible_answer():
    """Responses reasoning items must survive conversion into native history.

    DeepSeek-V4 and other local reasoning templates consume
    ``reasoning_content`` on historical assistant turns.  Dropping it changes
    the token transcript even though the UI still displayed the reasoning.
    """
    from vmlx_engine.server import _responses_input_to_messages

    converted = _responses_input_to_messages(
        [
            {
                "type": "reasoning",
                "content": [
                    {"type": "reasoning", "text": "inspect the package"}
                ],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "file_info",
                "arguments": '{"path":"panel/package.json"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "Size: 5.2 KB",
            },
            {
                "type": "reasoning",
                "content": [
                    {"type": "reasoning", "text": "report the observed size"}
                ],
            },
            {"type": "output_text", "text": "5.2 KB"},
        ]
    )

    assert converted == [
        {
            "role": "user",
            "content": "Previous assistant tool-call history follows.",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "file_info",
                        "arguments": {"path": "panel/package.json"},
                    },
                }
            ],
            "reasoning_content": "inspect the package",
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "Size: 5.2 KB"},
        {
            "role": "assistant",
            "content": "5.2 KB",
            "reasoning_content": "report the observed size",
        },
    ]


def test_responses_reasoning_visible_text_and_tool_call_share_one_assistant_turn():
    """Electron may emit visible text before a function call in one response.

    The native chat transcript must retain that as one assistant generation;
    otherwise replay-reasoning templates see a completed answer turn followed
    by an empty-thinking tool-call turn.
    """
    from vmlx_engine.server import _responses_input_to_messages

    converted = _responses_input_to_messages(
        [
            {
                "type": "reasoning",
                "content": [{"type": "reasoning", "text": "inspect first"}],
            },
            {"type": "output_text", "text": "I will inspect it."},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "file_info",
                "arguments": '{"path":"panel/package.json"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "Size: 5.2 KB",
            },
        ]
    )

    assert converted == [
        {
            "role": "user",
            "content": "Previous assistant tool-call history follows.",
        },
        {
            "role": "assistant",
            "content": "I will inspect it.",
            "reasoning_content": "inspect first",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "file_info",
                        "arguments": {"path": "panel/package.json"},
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "Size: 5.2 KB"},
    ]


def test_responses_role_assistant_preserves_reasoning_content():
    from vmlx_engine.server import _responses_input_to_messages

    assert _responses_input_to_messages(
        [
            {
                "role": "assistant",
                "content": "visible",
                "reasoning_content": "native prior thought",
            }
        ]
    ) == [
        {
            "role": "assistant",
            "content": "visible",
            "reasoning_content": "native prior thought",
        }
    ]


def test_responses_output_merges_visible_text_with_tool_calls_for_tool_adjacency():
    """A Responses item stream may contain visible assistant text plus a
    function_call. Store both on the same assistant message so the next
    function_call_output is adjacent to the assistant.tool_calls turn."""
    from vmlx_engine.server import (
        _responses_input_to_messages,
        _responses_output_to_assistant_messages,
    )

    user_messages = [{"role": "user", "content": "review cache path"}]
    output_items = [
        {
            "type": "message",
            "content": [{"type": "output_text", "text": "I will inspect it."}],
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "inspect_symbol",
            "arguments": '{"path":"vmlx_engine/server.py","symbol":"cache_stats"}',
        },
    ]
    stored = user_messages + _responses_output_to_assistant_messages(output_items)
    current = _responses_input_to_messages(
        [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "cache_stats source excerpt",
            },
            {"role": "user", "content": "continue"},
        ]
    )
    combined = stored + current

    tool_index = next(i for i, msg in enumerate(combined) if msg.get("role") == "tool")
    previous = combined[tool_index - 1]
    assert previous["role"] == "assistant"
    assert previous.get("tool_calls")
    assert previous["content"] == "I will inspect it."
    assert previous["tool_calls"][0]["function"]["arguments"] == {
        "path": "vmlx_engine/server.py",
        "symbol": "cache_stats",
    }


def test_responses_object_exposes_output_text_convenience_field():
    from vmlx_engine.api.models import (
        ResponsesFunctionCall,
        ResponsesObject,
        ResponsesOutputMessage,
        ResponsesOutputText,
        ResponsesUsage,
    )

    response = ResponsesObject(
        model="unit-test",
        output=[
            ResponsesOutputMessage(
                role="assistant",
                content=[ResponsesOutputText(type="reasoning", text="hidden")],
            ),
            ResponsesOutputMessage(
                role="assistant",
                content=[ResponsesOutputText(type="output_text", text="Paris")],
            ),
            ResponsesFunctionCall(name="lookup", arguments="{}"),
        ],
        usage=ResponsesUsage(input_tokens=1, output_tokens=1, total_tokens=2),
    )

    dumped = response.model_dump()

    assert response.output_text == "Paris"
    assert dumped["output_text"] == "Paris"


def test_dsv4_responses_tool_choice_none_does_not_synthesize_tools():
    """Historical tool schemas must not override explicit tool_choice=none."""
    import inspect
    from vmlx_engine import server

    src = inspect.getsource(server.create_response)

    assert "elif not _suppress_tools and _is_dsv4_resp_msgs" in src
    assert "_synthesize_tools_from_message_tool_calls(messages)" in src


def test_responses_nonstream_suppression_never_revives_raw_tool_markup():
    from vmlx_engine.server import _select_responses_visible_text

    raw = '<tool_call>\n<function=file_info>{"path":"x"}</function>\n</tool_call>'

    assert _select_responses_visible_text(
        cleaned_text="",
        raw_text=raw,
        tool_calls=None,
        suppress_tools=True,
    ) == ""
    assert _select_responses_visible_text(
        cleaned_text="visible answer",
        raw_text=raw,
        tool_calls=None,
        suppress_tools=True,
    ) == "visible answer"
    assert _select_responses_visible_text(
        cleaned_text="",
        raw_text="ordinary answer",
        tool_calls=None,
        suppress_tools=False,
    ) == "ordinary answer"
    assert _select_responses_visible_text(
        cleaned_text="",
        raw_text=(
            '<｜DSML｜tool_calls>\n'
            '<｜DSML｜invoke name="list_directory">\n'
            '<｜DSML｜parameter name="path">.</｜DSML｜parameter>\n'
            '</｜DSML｜invoke>\n'
            '</｜DSML｜tool_calls>'
        ),
        tool_calls=None,
        suppress_tools=False,
    ) == ""


def test_responses_previous_history_keeps_instructions_as_leading_system():
    from vmlx_engine.server import (
        _normalize_leading_system_messages,
        _responses_input_to_messages,
    )

    previous_messages = [
        {"role": "system", "content": "base instructions"},
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "first answer"},
    ]
    current_messages = _responses_input_to_messages(
        "second turn",
        instructions="new response instructions",
        preserve_multimodal=False,
    )

    normalized = _normalize_leading_system_messages(
        previous_messages + current_messages
    )

    assert [m["role"] for m in normalized] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert normalized[0]["content"] == (
        "base instructions\n\nnew response instructions"
    )
    assert normalized[-1] == {"role": "user", "content": "second turn"}


def test_responses_previous_history_dedupes_identical_instructions_for_prefix_stability():
    from vmlx_engine.server import (
        _normalize_leading_system_messages,
        _responses_input_to_messages,
    )

    previous_messages = [
        {"role": "system", "content": "stable coding-review instructions"},
        {"role": "user", "content": "first turn"},
    ]
    current_messages = _responses_input_to_messages(
        "second turn",
        instructions="stable coding-review instructions",
        preserve_multimodal=False,
    )

    normalized = _normalize_leading_system_messages(
        previous_messages + current_messages
    )

    assert [m["role"] for m in normalized] == ["system", "user", "user"]
    assert normalized[0]["content"] == "stable coding-review instructions"
    assert normalized[1:] == [
        {"role": "user", "content": "first turn"},
        {"role": "user", "content": "second turn"},
    ]


def test_normalize_leading_system_messages_moves_mid_conversation_system():
    from vmlx_engine.server import _normalize_leading_system_messages

    normalized = _normalize_leading_system_messages(
        [
            {"role": "user", "content": "hello"},
            {"role": "developer", "content": "developer policy"},
            {"role": "assistant", "content": "hi"},
            {"role": "system", "content": "json only"},
        ]
    )

    assert [m["role"] for m in normalized] == ["system", "user", "assistant"]
    assert normalized[0]["content"] == "developer policy\n\njson only"
    assert normalized[1:] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_dsv4_preserves_native_system_and_latest_reminder_order():
    from vmlx_engine.server import _normalize_leading_system_messages

    messages = [
        {"role": "system", "content": "base policy"},
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "first answer"},
        {"role": "system", "content": "tail system"},
        {"role": "latest_reminder", "content": "2026-08-02,US,App,en"},
        {"role": "user", "content": "second turn"},
    ]

    normalized = _normalize_leading_system_messages(
        messages,
        preserve_native_order=True,
    )

    assert normalized == messages
    assert normalized is not messages
    assert normalized[3]["role"] == "system"
    assert normalized[4]["role"] == "latest_reminder"


def test_qwen_generic_path_still_hoists_mid_conversation_system():
    from vmlx_engine.server import _normalize_leading_system_messages

    normalized = _normalize_leading_system_messages(
        [
            {"role": "system", "content": "base policy"},
            {"role": "user", "content": "first turn"},
            {"role": "assistant", "content": "first answer"},
            {"role": "system", "content": "tail system"},
        ],
        preserve_native_order=False,
    )

    assert normalized == [
        {"role": "system", "content": "base policy\n\ntail system"},
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "first answer"},
    ]


def test_responses_history_json_instruction_stays_in_leading_system():
    from vmlx_engine.server import (
        _inject_json_instruction,
        _normalize_leading_system_messages,
        _responses_input_to_messages,
    )

    previous_messages = [
        {"role": "system", "content": "base instructions"},
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "first answer"},
    ]
    current_messages = _responses_input_to_messages(
        "second turn",
        instructions="new response instructions",
        preserve_multimodal=False,
    )
    messages = _inject_json_instruction(
        previous_messages + current_messages,
        "Return valid JSON only.",
    )
    normalized = _normalize_leading_system_messages(messages)

    assert [m["role"] for m in normalized] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert normalized[0]["content"] == (
        "base instructions\n\nReturn valid JSON only.\n\n"
        "new response instructions"
    )


def test_responses_and_chat_call_sites_normalize_before_template_rendering():
    from vmlx_engine import server

    responses_src = inspect.getsource(server.create_response)
    chat_src = inspect.getsource(server.create_chat_completion)
    cache_contract_src = inspect.getsource(server._cache_contract_render_and_tokenize)

    assert "messages = _normalize_leading_system_messages(" in responses_src
    assert "preserve_native_order=_is_dsv4_resp_msgs" in responses_src
    assert "messages = _normalize_leading_system_messages(" in chat_src
    assert (
        "preserve_native_order=_is_loaded_dsv4_model(request.model)" in chat_src
    )
    assert "preserve_native_order=_is_loaded_dsv4_model(model)" in cache_contract_src
    assert responses_src.index(
        "messages = _normalize_leading_system_messages("
    ) > responses_src.index("_inject_json_instruction(messages, json_instruction)")
    assert chat_src.index(
        "messages = _normalize_leading_system_messages("
    ) > chat_src.index("_inject_json_instruction(messages, json_instruction)")


@pytest.mark.asyncio
async def test_responses_streaming_stores_history_for_previous_response_id():
    from vmlx_engine.api.models import ResponsesRequest
    from vmlx_engine import server

    class FakeEngine:
        tokenizer = SimpleNamespace(has_thinking=False)

        async def stream_chat(self, messages, **kwargs):
            yield SimpleNamespace(
                new_text="stored answer",
                prompt_tokens=3,
                completion_tokens=2,
                finish_reason="stop",
                finished=True,
            )

    server._responses_history.clear()
    server._reasoning_parser = None

    request = ResponsesRequest(model="unit-test-model", input="hello", stream=True)
    messages = [{"role": "user", "content": "hello"}]

    events = [
        event async for event in server.stream_responses_api(
            FakeEngine(), messages, request
        )
    ]
    completed = _sse_payloads(events, "response.completed")
    assert completed, "stream must emit response.completed"
    response_id = completed[-1]["response"]["id"]
    assert completed[-1]["response"]["output_text"] == "stored answer"

    history = server._responses_get_history(response_id)

    assert history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "stored answer"},
    ]


@pytest.mark.asyncio
async def test_responses_streaming_does_not_store_request_scoped_instructions():
    from vmlx_engine.api.models import ResponsesRequest
    from vmlx_engine import server

    class FakeEngine:
        tokenizer = SimpleNamespace(has_thinking=False)

        async def stream_chat(self, messages, **kwargs):
            assert messages[0] == {
                "role": "system",
                "content": "Call file_info exactly once.",
            }
            yield SimpleNamespace(
                new_text="stored answer",
                prompt_tokens=3,
                completion_tokens=2,
                finish_reason="stop",
                finished=True,
            )

    server._responses_history.clear()
    server._reasoning_parser = None
    request = ResponsesRequest(
        model="unit-test-model",
        input="hello",
        instructions="Call file_info exactly once.",
        stream=True,
    )
    generation_messages = [
        {"role": "system", "content": request.instructions},
        {"role": "user", "content": "hello"},
    ]
    persistent_messages = [
        {"role": "system", "content": "Explicit input policy."},
        {"role": "user", "content": "hello"},
    ]

    try:
        events = [
            event async for event in server.stream_responses_api(
                FakeEngine(),
                generation_messages,
                request,
                history_messages=persistent_messages,
            )
        ]
        completed = _sse_payloads(events, "response.completed")
        response_id = completed[-1]["response"]["id"]
        history = server._responses_get_history(response_id)

        assert history == [
            {"role": "system", "content": "Explicit input policy."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "stored answer"},
        ]
        assert request.instructions not in json.dumps(history)
    finally:
        server._responses_history.clear()


@pytest.mark.asyncio
async def test_responses_streaming_length_finish_emits_incomplete_terminal_event():
    """A length-capped stream must not masquerade as response.completed."""
    from vmlx_engine.api.models import ResponsesRequest
    from vmlx_engine import server

    class FakeEngine:
        tokenizer = SimpleNamespace(has_thinking=False)

        async def stream_chat(self, messages, **kwargs):
            yield SimpleNamespace(
                new_text="partial answer",
                prompt_tokens=3,
                completion_tokens=4,
                finish_reason="length",
                finished=True,
            )

    original_parser = server._reasoning_parser
    server._reasoning_parser = None
    try:
        request = ResponsesRequest(
            model="unit-test-model",
            input="hello",
            stream=True,
            max_output_tokens=4,
        )
        events = [
            event
            async for event in server.stream_responses_api(
                FakeEngine(),
                [{"role": "user", "content": "hello"}],
                request,
            )
        ]

        assert _sse_payloads(events, "response.completed") == []
        incomplete = _sse_payloads(events, "response.incomplete")
        assert len(incomplete) == 1
        response = incomplete[0]["response"]
        assert response["status"] == "incomplete"
        assert response["incomplete_details"] == {"reason": "max_output_tokens"}
        assert response["output"]
        assert {item["status"] for item in response["output"]} == {"incomplete"}
    finally:
        server._reasoning_parser = original_parser


@pytest.mark.asyncio
async def test_responses_streaming_reasoning_only_stores_placeholder_and_marker():
    from vmlx_engine.api.models import ResponsesRequest
    from vmlx_engine.reasoning.base import DeltaMessage
    from vmlx_engine import server

    class FakeReasoningParser:
        def reset_state(self, **_kwargs):
            pass

        def extract_reasoning_streaming(self, previous_text, current_text, delta_text):
            return DeltaMessage(reasoning=delta_text)

        def extract_reasoning(self, model_output):
            return model_output, None

    class FakeEngine:
        tokenizer = SimpleNamespace(has_thinking=True)

        async def stream_chat(self, messages, **kwargs):
            yield SimpleNamespace(
                new_text="thinking only",
                prompt_tokens=3,
                completion_tokens=2,
                finish_reason="stop",
                finished=True,
                cached_tokens=0,
            )

    server._responses_history.clear()
    server._responses_was_reasoning_only.clear()
    original_parser = server._reasoning_parser
    server._reasoning_parser = FakeReasoningParser()
    try:
        request = ResponsesRequest(
            model="unit-test-model",
            input="hello",
            stream=True,
            enable_thinking=True,
        )
        messages = [{"role": "user", "content": "hello"}]

        events = [
            event async for event in server.stream_responses_api(
                FakeEngine(), messages, request
            )
        ]
        incomplete = _sse_payloads(events, "response.incomplete")
        assert incomplete, "stream must emit response.incomplete"
        assert _sse_payloads(events, "response.completed") == []
        response_id = incomplete[-1]["response"]["id"]

        assert server._responses_history[response_id] == [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "thinking only",
            },
        ], "the STORE must keep the reasoning-only placeholder"
        assert server._responses_get_history(response_id) == [
            {"role": "user", "content": "hello"},
        ], "replay must drop the contentless placeholder (dialect F2)"
        assert server._chain_warnings_for_previous_response_id(response_id)
        warnings = incomplete[-1]["response"].get("warnings") or []
        assert any("produced reasoning only" in w for w in warnings)
        assert any("ended normally" in w for w in warnings)
        assert all("max_output_tokens" not in w for w in warnings)
        assert incomplete[-1]["response"]["incomplete_details"] == {
            "reason": "reasoning_only_no_content"
        }
        assert {
            item["status"] for item in incomplete[-1]["response"]["output"]
        } == {"incomplete"}
    finally:
        server._reasoning_parser = original_parser
        server._responses_history.clear()
        server._responses_was_reasoning_only.clear()


# ─── Reasoning-only history alignment fix (claude 04:50 audit Option A) ────
# When the model returns ONLY reasoning items (no output_text, no function_call),
# `_responses_output_to_assistant_messages` historically returned [], leaving the
# stored history without an assistant turn. The next `previous_response_id` chain
# then loses cache prefix alignment AND loses the model's record that turn N
# happened. Fix: inject an empty-content placeholder assistant turn so the chain
# template still has a valid user→assistant→user sequence.
#
# See docs/internal/AUDIT_responses_reasoning_history_alignment_2026_05_09.md
# for the full root-cause trace and fix-design discussion.


def test_reasoning_only_output_emits_assistant_placeholder():
    """When output is reasoning-only, an empty-content assistant turn must be
    emitted so the next-turn cache prefix has a proper anchor and the model
    sees that the prior turn completed."""
    from vmlx_engine.server import _responses_output_to_assistant_messages

    items = [
        {
            "type": "reasoning",
            "content": [{"type": "reasoning_text", "text": "thinking..."}],
        }
    ]
    msgs = _responses_output_to_assistant_messages(items)
    assert msgs == [
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "thinking...",
        }
    ], "reasoning-only output must preserve the assistant reasoning turn"


def test_message_item_with_only_reasoning_content_emits_assistant_placeholder():
    """Some Responses-compatible payloads represent reasoning as content parts
    on a message item rather than a top-level `type=reasoning` item. That is
    still reasoning-only output and must preserve the assistant history anchor.
    """
    from vmlx_engine.server import _responses_output_to_assistant_messages

    items = [
        {
            "type": "message",
            "content": [{"type": "reasoning", "text": "thinking..."}],
        }
    ]

    assert _responses_output_to_assistant_messages(items) == [
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "thinking...",
        }
    ]


def test_reasoning_only_detection_ignores_empty_output_text_message():
    """Streaming Responses emits an output_text message shell before it knows
    whether the model will produce visible content. An empty shell must not
    mask a reasoning-only completion.
    """
    from vmlx_engine.server import _responses_output_is_reasoning_only

    assert _responses_output_is_reasoning_only(
        [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": ""}],
            },
            {
                "type": "reasoning",
                "content": [{"type": "reasoning", "text": "thinking..."}],
            },
        ]
    )


def test_current_response_warning_for_reasoning_only_output():
    from vmlx_engine.server import _current_response_warnings_for_reasoning_only

    stop_warnings = _current_response_warnings_for_reasoning_only(True, "stop")
    length_warnings = _current_response_warnings_for_reasoning_only(True, "length")
    cancelled_warnings = _current_response_warnings_for_reasoning_only(
        True, "aborted"
    )
    unknown_warnings = _current_response_warnings_for_reasoning_only(True, None)

    assert stop_warnings is not None
    assert any("produced reasoning only" in w for w in stop_warnings)
    assert any("ended normally" in w for w in stop_warnings)
    assert all("max_output_tokens" not in w for w in stop_warnings)
    assert length_warnings is not None
    assert any("max_output_tokens" in w for w in length_warnings)
    assert cancelled_warnings is not None
    assert any("cancelled or aborted" in w for w in cancelled_warnings)
    assert all("max_output_tokens" not in w for w in cancelled_warnings)
    assert unknown_warnings is not None
    assert any("did not report a terminal cause" in w for w in unknown_warnings)
    assert all("max_output_tokens" not in w for w in unknown_warnings)
    assert _current_response_warnings_for_reasoning_only(False) is None


@pytest.mark.parametrize(
    ("finish_reason", "expected_finish", "status", "item_status", "event_type", "details"),
    [
        ("stop", "stop", "completed", "completed", "response.completed", None),
        (
            "length",
            "length",
            "incomplete",
            "incomplete",
            "response.incomplete",
            {"reason": "max_output_tokens"},
        ),
        (
            "aborted",
            "cancelled",
            "incomplete",
            "incomplete",
            "response.incomplete",
            {"reason": "cancelled"},
        ),
        (
            "abort",
            "cancelled",
            "incomplete",
            "incomplete",
            "response.incomplete",
            {"reason": "cancelled"},
        ),
        (
            "cancelled",
            "cancelled",
            "incomplete",
            "incomplete",
            "response.incomplete",
            {"reason": "cancelled"},
        ),
        (
            "canceled",
            "cancelled",
            "incomplete",
            "incomplete",
            "response.incomplete",
            {"reason": "cancelled"},
        ),
        ("error", "error", "failed", "incomplete", "response.failed", None),
    ],
)
def test_responses_terminal_state_normalizes_every_supported_finish_alias(
    finish_reason,
    expected_finish,
    status,
    item_status,
    event_type,
    details,
):
    from vmlx_engine.server import _responses_terminal_state

    terminal = _responses_terminal_state(finish_reason)

    assert terminal.finish_reason == expected_finish
    assert terminal.response_status == status
    assert terminal.item_status == item_status
    assert terminal.event_type == event_type
    assert terminal.incomplete_details == details


def test_responses_terminal_state_marks_reasoning_only_stop_incomplete():
    from vmlx_engine.server import _responses_terminal_state

    terminal = _responses_terminal_state("stop", reasoning_only_no_content=True)

    assert terminal.finish_reason == "stop"
    assert terminal.response_status == "incomplete"
    assert terminal.item_status == "incomplete"
    assert terminal.event_type == "response.incomplete"
    assert terminal.incomplete_details == {"reason": "reasoning_only_no_content"}


def test_chat_completion_warning_for_reasoning_only_message():
    from vmlx_engine.server import _chat_completion_warnings_for_reasoning_only

    warnings = _chat_completion_warnings_for_reasoning_only(
        content=None,
        reasoning="internal analysis",
        tool_calls=None,
        finish_reason="stop",
    )

    assert warnings is not None
    assert any("produced reasoning only" in w for w in warnings)
    assert any("ended normally" in w for w in warnings)
    assert all("max_output_tokens" not in w for w in warnings)
    assert _chat_completion_warnings_for_reasoning_only(
        content="visible answer",
        reasoning="internal analysis",
        tool_calls=None,
        finish_reason="stop",
    ) is None
    assert _chat_completion_warnings_for_reasoning_only(
        content=None,
        reasoning="internal analysis",
        tool_calls=[{"id": "call_1"}],
        finish_reason="stop",
    ) is None


def test_empty_output_does_NOT_emit_placeholder():
    """An empty output_items list means the model truly produced nothing.
    Do not invent an assistant turn — that's different from reasoning-only."""
    from vmlx_engine.server import _responses_output_to_assistant_messages
    assert _responses_output_to_assistant_messages([]) == []


def test_visible_text_output_does_NOT_get_placeholder():
    """When output_text is present, replay normally (no placeholder injection)."""
    from vmlx_engine.server import _responses_output_to_assistant_messages

    items = [
        {
            "type": "reasoning",
            "content": [{"type": "reasoning_text", "text": "thinking..."}],
        },
        {
            "type": "message",
            "content": [{"type": "output_text", "text": "the answer is 42"}],
        },
    ]
    msgs = _responses_output_to_assistant_messages(items)
    assert msgs == [
        {
            "role": "assistant",
            "content": "the answer is 42",
            "reasoning_content": "thinking...",
        }
    ], "output_text should replay with its native reasoning context"


def test_function_call_only_does_NOT_get_placeholder():
    """function_call items already produce an assistant.tool_calls turn — no
    extra placeholder needed."""
    from vmlx_engine.server import _responses_output_to_assistant_messages

    items = [
        {
            "type": "function_call",
            "name": "get_weather",
            "arguments": '{"city": "SF"}',
            "call_id": "call_abc",
        }
    ]
    msgs = _responses_output_to_assistant_messages(items)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "assistant"
    assert msgs[0].get("tool_calls"), "function_call must produce tool_calls turn"


def test_reasoning_plus_function_call_does_NOT_get_extra_placeholder():
    """Mixed reasoning + function_call: only the function_call assistant turn."""
    from vmlx_engine.server import _responses_output_to_assistant_messages

    items = [
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "..."}]},
        {
            "type": "function_call",
            "name": "x",
            "arguments": "{}",
            "call_id": "c1",
        },
    ]
    msgs = _responses_output_to_assistant_messages(items)
    assert len(msgs) == 1
    assert msgs[0].get("tool_calls"), "function_call assistant turn already present"


def test_reasoning_only_history_stored_but_dropped_at_replay():
    """Integration: the STORE preserves the reasoning-only placeholder (the
    chain warning and introspection read it), while `_responses_get_history`
    — the replay boundary — drops it. Replaying it handed strict templates a
    contentless assistant turn: the exact 500 every dialect's input drop
    fixes, reintroduced by the server's own store (dialect F2)."""
    from vmlx_engine.server import (
        _responses_output_to_assistant_messages,
        _responses_store_history,
        _responses_get_history,
        _responses_history,
    )

    _responses_history.clear()
    user_messages = [{"role": "user", "content": "what is 2+2?"}]
    output_items = [
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "..."}]}
    ]
    full = user_messages + _responses_output_to_assistant_messages(output_items)
    _responses_store_history("resp_test_reasoning_only", full)

    assert _responses_history["resp_test_reasoning_only"] == [
        {"role": "user", "content": "what is 2+2?"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "...",
        },
    ], "the STORE must preserve the assistant reasoning placeholder"

    loaded = _responses_get_history("resp_test_reasoning_only")
    assert loaded == [
        {"role": "user", "content": "what is 2+2?"},
    ], "replay must not hand templates a contentless assistant turn"

    _responses_history.clear()


# ─── Option C: warning surface for reasoning-only chains ──────────────────
# When a `previous_response_id` chain references a response that produced only
# reasoning (no visible output, no tool calls), the chained response should
# expose a `warnings` array so clients can surface the coherence/cache-prefix
# risk without breaking the API contract.


def test_responses_object_supports_warnings_field():
    """ResponsesObject must allow a `warnings: list[str] | None` field."""
    from vmlx_engine.api.models import ResponsesObject
    obj = ResponsesObject(model="m", warnings=["test warning"])
    assert obj.warnings == ["test warning"]


def test_responses_object_warnings_default_none():
    from vmlx_engine.api.models import ResponsesObject
    obj = ResponsesObject(model="m")
    assert obj.warnings is None
    assert obj.incomplete_details is None


def test_responses_object_supports_incomplete_details():
    from vmlx_engine.api.models import ResponsesObject

    obj = ResponsesObject(
        model="m",
        status="incomplete",
        incomplete_details={"reason": "max_output_tokens"},
    )

    assert obj.incomplete_details == {"reason": "max_output_tokens"}


def test_reasoning_only_marker_stored_per_response_id():
    """Storing a reasoning-only response must mark it so chained turns can warn."""
    from vmlx_engine.server import (
        _responses_output_to_assistant_messages,
        _responses_store_history,
        _responses_history,
        _responses_was_reasoning_only,
    )
    _responses_history.clear()
    _responses_was_reasoning_only.clear()

    user_messages = [{"role": "user", "content": "hi"}]
    items = [
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "..."}]}
    ]
    full = user_messages + _responses_output_to_assistant_messages(items)
    _responses_store_history("resp_ro_marker", full, reasoning_only=True)

    assert "resp_ro_marker" in _responses_was_reasoning_only

    _responses_history.clear()
    _responses_was_reasoning_only.clear()


def test_visible_response_does_not_set_reasoning_only_marker():
    from vmlx_engine.server import (
        _responses_output_to_assistant_messages,
        _responses_store_history,
        _responses_history,
        _responses_was_reasoning_only,
    )
    _responses_history.clear()
    _responses_was_reasoning_only.clear()

    user_messages = [{"role": "user", "content": "hi"}]
    items = [{"type": "message", "content": [{"type": "output_text", "text": "yes"}]}]
    full = user_messages + _responses_output_to_assistant_messages(items)
    _responses_store_history("resp_normal", full, reasoning_only=False)

    assert "resp_normal" not in _responses_was_reasoning_only

    _responses_history.clear()
    _responses_was_reasoning_only.clear()


def test_chained_response_helper_emits_warning_for_reasoning_only_predecessor():
    """`_chain_warnings_for_previous_response_id` returns a warning when the
    predecessor was reasoning-only, else None."""
    from vmlx_engine.server import (
        _responses_history,
        _responses_store_history,
        _responses_was_reasoning_only,
        _chain_warnings_for_previous_response_id,
    )
    _responses_history.clear()
    _responses_was_reasoning_only.clear()
    _responses_store_history(
        "resp_ro_predecessor",
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": ""}],
        reasoning_only=True,
    )

    warnings = _chain_warnings_for_previous_response_id("resp_ro_predecessor")
    assert warnings is not None
    assert any("reasoning" in w.lower() for w in warnings)

    assert _chain_warnings_for_previous_response_id("resp_normal") is None
    assert _chain_warnings_for_previous_response_id(None) is None
    assert _chain_warnings_for_previous_response_id("") is None

    _responses_history.clear()
    _responses_was_reasoning_only.clear()


def test_chained_response_helper_ignores_stale_marker_without_history_slot():
    """A marker without a matching history slot must not warn.

    The runtime store path evicts both in lockstep, but tests and future admin
    cleanup paths may clear `_responses_history` directly. The warning helper
    should defend the invariant at read time too, otherwise a stale marker can
    create a false-positive warning for a missing predecessor.
    """
    from vmlx_engine.server import (
        _responses_history,
        _responses_was_reasoning_only,
        _chain_warnings_for_previous_response_id,
    )

    _responses_history.clear()
    _responses_was_reasoning_only.clear()
    _responses_was_reasoning_only.add("resp_stale_marker")

    assert _chain_warnings_for_previous_response_id("resp_stale_marker") is None

    _responses_history.clear()
    _responses_was_reasoning_only.clear()


def test_reasoning_only_marker_eviction_with_history_capacity():
    """When history slots are evicted by capacity, marker must also be evicted
    to avoid stale-warning false positives on response_id collisions."""
    from vmlx_engine.server import (
        _responses_store_history,
        _responses_history,
        _responses_was_reasoning_only,
    )
    import vmlx_engine.server as server
    _responses_history.clear()
    _responses_was_reasoning_only.clear()

    original_max = server._RESPONSES_HISTORY_MAX
    try:
        server._RESPONSES_HISTORY_MAX = 2
        _responses_store_history("resp_a", [{"role": "user", "content": "a"}], reasoning_only=True)
        _responses_store_history("resp_b", [{"role": "user", "content": "b"}], reasoning_only=False)
        _responses_store_history("resp_c", [{"role": "user", "content": "c"}], reasoning_only=False)
        # resp_a should be evicted
        assert "resp_a" not in _responses_history
        assert "resp_a" not in _responses_was_reasoning_only, (
            "evicted history slot must also drop its reasoning_only marker"
        )
    finally:
        server._RESPONSES_HISTORY_MAX = original_max
        _responses_history.clear()
        _responses_was_reasoning_only.clear()


# --- dialect F3: parallel tool calls must reconstruct identically on both
# replay modes. Explicit replay built one assistant turn PER function_call
# item while previous_response_id chain replay coalesced all of them into
# one turn -> different prompt bytes (prefix-cache split on every chained
# multi-tool turn) and consecutive assistant turns that strict-alternation
# templates reject.


def test_parallel_tool_calls_coalesce_on_explicit_replay():
    from vmlx_engine.server import _responses_input_to_messages

    msgs = _responses_input_to_messages(
        [
            {"type": "message", "role": "user", "content": "check both files"},
            {"type": "function_call", "call_id": "c1", "name": "read_file",
             "arguments": '{"path": "a.txt"}'},
            {"type": "function_call", "call_id": "c2", "name": "read_file",
             "arguments": '{"path": "b.txt"}'},
            {"type": "function_call_output", "call_id": "c1", "output": "A"},
            {"type": "function_call_output", "call_id": "c2", "output": "B"},
            {"type": "message", "role": "user", "content": "and now?"},
        ],
        None,
    )
    assistants = [m for m in msgs if m.get("role") == "assistant"]
    assert len(assistants) == 1, assistants
    assert len(assistants[0]["tool_calls"]) == 2
    tools = [m for m in msgs if m.get("role") == "tool"]
    assert [t["tool_call_id"] for t in tools] == ["c1", "c2"]


def test_parallel_tool_calls_explicit_and_chain_replay_share_one_shape():
    from vmlx_engine.server import (
        _responses_input_to_messages,
        _responses_output_to_assistant_messages,
    )

    output_items = [
        {"type": "function_call", "call_id": "c1", "name": "read_file",
         "arguments": '{"path": "a.txt"}'},
        {"type": "function_call", "call_id": "c2", "name": "read_file",
         "arguments": '{"path": "b.txt"}'},
    ]
    chain = _responses_output_to_assistant_messages(output_items)
    explicit = _responses_input_to_messages(
        [{"type": "message", "role": "user", "content": "go"}] + output_items,
        None,
    )
    explicit_assistants = [m for m in explicit if m.get("role") == "assistant"]
    assert len(chain) == 1 and len(explicit_assistants) == 1
    key = lambda turn: [
        (t["function"]["name"], t["function"]["arguments"])
        for t in turn["tool_calls"]
    ]
    assert key(chain[0]) == key(explicit_assistants[0])


def test_sequential_and_reasoning_separated_calls_stay_split():
    from vmlx_engine.server import _responses_input_to_messages

    sequential = _responses_input_to_messages(
        [
            {"type": "message", "role": "user", "content": "step by step"},
            {"type": "function_call", "call_id": "s1", "name": "f", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "s1", "output": "r1"},
            {"type": "function_call", "call_id": "s2", "name": "g", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "s2", "output": "r2"},
        ],
        None,
    )
    seq_assistants = [m for m in sequential if m.get("role") == "assistant"]
    assert len(seq_assistants) == 2
    assert all(len(m["tool_calls"]) == 1 for m in seq_assistants)

    with_reasoning = _responses_input_to_messages(
        [
            {"type": "message", "role": "user", "content": "hi"},
            {"type": "function_call", "call_id": "r1", "name": "f", "arguments": "{}"},
            {"type": "reasoning",
             "content": [{"type": "reasoning_text", "text": "thinking"}]},
            {"type": "function_call", "call_id": "r2", "name": "g", "arguments": "{}"},
        ],
        None,
    )
    r_assistants = [m for m in with_reasoning if m.get("role") == "assistant"]
    assert len(r_assistants) == 2
    assert r_assistants[1].get("reasoning_content") == "thinking"
