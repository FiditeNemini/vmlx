"""Anthropic /v1/messages streaming adapter coding-harness fixes (2026-06-22).

Covers: (1) stop_reason=tool_use authoritative on finish_reason==tool_calls,
(2) signature_delta emitted for thinking blocks, (3) mid-stream error -> error event.
"""
import json
from types import SimpleNamespace

import pytest

from vmlx_engine.api.anthropic_adapter import AnthropicStreamAdapter


def _chunk(delta, finish=None):
    return "data: " + json.dumps({
        "id": "chatcmpl-x", "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }) + "\n\n"


def _run(chunks):
    a = AnthropicStreamAdapter("test-model", "msg_test")
    out = []
    for c in chunks:
        out += a.process_chunk(c)
    out += a.finalize()
    return out, a


def _types(events):
    types = []
    for e in events:
        for line in e.splitlines():
            if line.startswith("data: "):
                try:
                    d = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                t = d.get("delta", {}).get("type") if d.get("type") == "content_block_delta" else d.get("type")
                types.append(t)
    return types


def test_stop_reason_tool_use_when_finish_is_tool_calls():
    # reasoning, then text, then a tool call with finish_reason=tool_calls
    events, _ = _run([
        _chunk({"role": "assistant"}),
        _chunk({"reasoning_content": "let me think "}),
        _chunk({"reasoning_content": "about it"}),
        _chunk({"content": "I'll call a tool."}),
        _chunk({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                "function": {"name": "get_weather", "arguments": "{\"city\":\"NYC\"}"}}]},
               finish="tool_calls"),
    ])
    # find message_delta stop_reason
    stop = None
    for e in events:
        for line in e.splitlines():
            if line.startswith("data: "):
                d = json.loads(line[6:])
                if d.get("type") == "message_delta":
                    stop = d["delta"]["stop_reason"]
    assert stop == "tool_use", f"expected tool_use, got {stop}"


def test_signature_delta_emitted_for_thinking():
    events, _ = _run([
        _chunk({"role": "assistant"}),
        _chunk({"reasoning_content": "deep thought"}),
        _chunk({"content": "answer"}),
    ])
    ts = _types(events)
    assert "signature_delta" in ts, f"no signature_delta in {ts}"
    # signature must precede the thinking block's content_block_stop
    assert "thinking_delta" in ts


def test_signature_delta_when_thinking_closed_at_finalize():
    # thinking only, no following text -> closed at finalize, must still sign
    events, _ = _run([
        _chunk({"role": "assistant"}),
        _chunk({"reasoning_content": "only thinking"}),
    ], )
    ts = _types(events)
    assert "signature_delta" in ts, f"no signature_delta in {ts}"


def test_mid_stream_error_emits_error_event():
    a = AnthropicStreamAdapter("test-model", "msg_test")
    out = []
    out += a.process_chunk(_chunk({"role": "assistant"}))
    out += a.process_chunk("data: " + json.dumps({"error": {"type": "overloaded_error", "message": "boom"}}) + "\n\n")
    out += a.finalize()
    saw_error = any('"type": "error"' in e and "boom" in e for e in out)
    assert saw_error, f"no error event in {out}"
    # finalize must not append a normal message_delta after error
    assert not any("message_delta" in e for e in out), "message_delta emitted after error"


def test_split_tool_id_then_name_never_opens_empty_anthropic_name():
    """Buffer vMLX's id-first tool delta until the required name arrives."""
    adapter = AnthropicStreamAdapter("test-model", "msg_test")

    first = adapter.process_chunk(_chunk({
        "tool_calls": [{
            "index": 0,
            "id": "call_split",
            "type": "function",
            "function": {"name": "", "arguments": ""},
        }]
    }))
    assert not any('"type": "tool_use"' in event for event in first)

    second = adapter.process_chunk(_chunk({
        "tool_calls": [{
            "index": 0,
            "function": {
                "name": "file_info",
                "arguments": '{"path":"panel/package.json"}',
            },
        }]
    }, finish="tool_calls"))
    final = adapter.finalize()
    wire = "".join(first + second + final)

    assert '"type": "tool_use"' in wire
    assert '"id": "call_split"' in wire
    assert '"name": "file_info"' in wire
    assert '"name": ""' not in wire
    assert '"partial_json": "{\\"path\\":\\"panel/package.json\\"}"' in wire
    assert '"stop_reason": "tool_use"' in wire


@pytest.mark.asyncio
async def test_non_stream_messages_merges_split_tool_id_and_name(monkeypatch):
    """The non-stream /v1/messages adapter must mirror streaming tool merging."""
    import vmlx_engine.server as server

    async def fake_stream_chat_completion(*args, **kwargs):
        first = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_split",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        }
        second = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "name": "file_info",
                                    "arguments": '{"path":"panel/package.json"}',
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7},
        }
        yield "data: " + json.dumps(first) + "\n\n"
        yield "data: " + json.dumps(second) + "\n\n"
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(server, "stream_chat_completion", fake_stream_chat_completion)
    monkeypatch.setattr(server, "_model_name", "split-tool-test")
    monkeypatch.setattr(server, "_model_path", None)
    monkeypatch.setattr(server, "_reasoning_parser", None)

    monkeypatch.setattr(
        server,
        "get_engine",
        lambda: SimpleNamespace(
            tokenizer=SimpleNamespace(has_thinking=False),
            is_mllm=False,
        ),
    )

    class _Request:
        async def json(self):
            return {
                "model": "split-tool-test",
                "messages": [{"role": "user", "content": "call file_info"}],
                "stream": False,
                "tools": [
                    {
                        "name": "file_info",
                        "input_schema": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                        },
                    }
                ],
            }

    response = await server.create_anthropic_message(_Request())

    tool_blocks = [item for item in response["content"] if item.get("type") == "tool_use"]
    assert tool_blocks == [
        {
            "type": "tool_use",
            "id": "call_split",
            "name": "file_info",
            "input": {"path": "panel/package.json"},
        }
    ]
    assert response["stop_reason"] == "tool_use"
