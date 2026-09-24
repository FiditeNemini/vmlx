# SPDX-License-Identifier: Apache-2.0
"""Terminal finish_reason contract guard for streaming chat completions.

Natural completion previously ended `…content chunk → [DONE]` with no chunk
carrying a non-null finish_reason (docs/STREAMING-FINISH-REASON-FINDINGS-
2026-07-02.md; GitHub #226 item 4). `_terminal_finish_guard` — applied at
the SSE route call sites around stream_chat_completion — must inject
exactly one terminal `finish_reason:"stop"` chunk in that case, and must
NOT double-emit when any branch already finished the stream.
"""

import asyncio
import json

import pytest

from vmlx_engine import server as server_mod


def _sse(payload) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=True)}\n\n"


def _chunk(delta=None, finish_reason="__omit__", raw_null=False):
    c = {
        "id": "chatcmpl-test1234",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "test-model",
        "choices": [{"index": 0, "delta": delta or {}}],
    }
    if raw_null:
        c["choices"][0]["finish_reason"] = None
    elif finish_reason != "__omit__":
        c["choices"][0]["finish_reason"] = finish_reason
    return c


def _collect(frames, *, required_tool_call=False):
    """Run the terminal-finish guard over a fake SSE stream."""

    async def fake_stream():
        for f in frames:
            yield f

    async def run():
        out = []
        async for s in server_mod._terminal_finish_guard(
            fake_stream(), required_tool_call=required_tool_call
        ):
            out.append(s)
        return out

    return asyncio.run(run())


def _finish_reasons(frames):
    frs = []
    for s in frames:
        s = s.strip()
        if not s.startswith("data: ") or s == "data: [DONE]":
            continue
        p = json.loads(s[6:])
        for ch in p.get("choices") or []:
            if ch.get("finish_reason"):
                frs.append(ch["finish_reason"])
    return frs


class TestTerminalFinishGuard:
    def test_natural_completion_injects_terminal_stop(self):
        frames = [
            _sse(_chunk(delta={"role": "assistant"}, raw_null=True)),
            _sse(_chunk(delta={"content": "ban"})),
            _sse(_chunk(delta={"content": "ana"})),
            "data: [DONE]\n\n",
        ]
        out = _collect(frames)
        assert _finish_reasons(out) == ["stop"]
        # terminal chunk must precede [DONE]
        assert out[-1] == "data: [DONE]\n\n"
        term = json.loads(out[-2].strip()[6:])
        assert term["choices"][0]["finish_reason"] == "stop"
        assert term["id"] == "chatcmpl-test1234"
        assert term["model"] == "test-model"

    def test_existing_finish_is_not_duplicated(self):
        frames = [
            _sse(_chunk(delta={"content": "hi"})),
            _sse(_chunk(delta={}, finish_reason="length")),
            "data: [DONE]\n\n",
        ]
        out = _collect(frames)
        assert _finish_reasons(out) == ["length"]

    def test_tool_calls_finish_is_not_duplicated(self):
        frames = [
            _sse(_chunk(delta={"tool_calls": [{"index": 0}]}, raw_null=True)),
            _sse(_chunk(delta={}, finish_reason="tool_calls")),
            "data: [DONE]\n\n",
        ]
        out = _collect(frames)
        assert _finish_reasons(out) == ["tool_calls"]

    def test_raw_null_finish_reason_does_not_count_as_seen(self):
        frames = [
            _sse(_chunk(delta={"content": "x"}, raw_null=True)),
            "data: [DONE]\n\n",
        ]
        out = _collect(frames)
        assert _finish_reasons(out) == ["stop"]

    def test_usage_only_tail_chunk_still_gets_terminal(self):
        usage_chunk = {
            "id": "chatcmpl-test1234",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "test-model",
            "choices": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }
        frames = [
            _sse(_chunk(delta={"content": "hi"})),
            _sse(usage_chunk),
            "data: [DONE]\n\n",
        ]
        out = _collect(frames)
        assert _finish_reasons(out) == ["stop"]
        assert out[-1] == "data: [DONE]\n\n"
        terminal = json.loads(out[-3].strip()[6:])
        final_usage = json.loads(out[-2].strip()[6:])
        assert terminal["choices"][0]["finish_reason"] == "stop"
        assert terminal["usage"] is None
        assert final_usage == usage_chunk

    def test_structured_error_is_not_followed_by_synthetic_stop(self):
        error = {
            "id": "chatcmpl-test1234",
            "object": "chat.completion.chunk",
            "error": {
                "message": "VLM prefill rejected before Metal forward",
                "type": "invalid_request_error",
                "code": "vlm_image_prefill_budget_exceeded",
            },
        }
        frames = [
            _sse(_chunk(delta={"role": "assistant"}, raw_null=True)),
            _sse(error),
            "data: [DONE]\n\n",
        ]
        out = _collect(frames)
        assert _finish_reasons(out) == []
        assert json.loads(out[-2].strip()[6:])["error"]["code"] == (
            "vlm_image_prefill_budget_exceeded"
        )
        assert out[-1] == "data: [DONE]\n\n"

    def test_required_tool_suppresses_provisional_stop_and_length_before_error(self):
        error = {
            "id": "chatcmpl-test1234",
            "object": "chat.completion.chunk",
            "error": {
                "message": "required tool missing",
                "type": "invalid_request_error",
                "code": "tool_calls_required",
            },
        }
        frames = [
            _sse(_chunk(delta={"reasoning_content": "checking"}, raw_null=True)),
            _sse(_chunk(delta={}, finish_reason="stop")),
            _sse(_chunk(delta={}, finish_reason="length")),
            _sse(error),
            "data: [DONE]\n\n",
        ]
        out = _collect(frames, required_tool_call=True)
        assert _finish_reasons(out) == []
        errors = [
            json.loads(frame.strip()[6:])["error"]
            for frame in out
            if frame.startswith("data: ")
            and frame.strip() != "data: [DONE]"
            and "error" in json.loads(frame.strip()[6:])
        ]
        assert [error["code"] for error in errors] == ["tool_calls_required"]

    def test_required_tool_preserves_valid_tool_calls_terminal(self):
        frames = [
            _sse(
                _chunk(
                    delta={
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "file_info",
                                    "arguments": '{"path":"panel/package.json"}',
                                },
                            }
                        ]
                    },
                    raw_null=True,
                )
            ),
            _sse(_chunk(delta={}, finish_reason="tool_calls")),
            "data: [DONE]\n\n",
        ]
        out = _collect(frames, required_tool_call=True)
        assert _finish_reasons(out) == ["tool_calls"]

    def test_required_tool_fails_closed_if_generator_omits_error(self):
        frames = [
            _sse(_chunk(delta={"reasoning_content": "checking"}, raw_null=True)),
            _sse(_chunk(delta={}, finish_reason="length")),
            "data: [DONE]\n\n",
        ]
        out = _collect(frames, required_tool_call=True)
        assert _finish_reasons(out) == []
        payload = json.loads(out[-2].strip()[6:])
        assert payload["error"]["code"] == "tool_calls_required"
        assert out[-1] == "data: [DONE]\n\n"


@pytest.mark.parametrize("arguments", ['{"path":"x"}', '{"path":', '[]', ''])
@pytest.mark.parametrize("late_error", [False, True])
def test_required_length_waits_for_final_call_and_errors(arguments, late_error):
    frames = [
        _sse(_chunk(delta={"tool_calls": [{"index": 0, "id": "call_x",
            "function": {"name": "read_file", "arguments": arguments[:4]}}]})),
        _sse(_chunk(delta={"tool_calls": [{"index": 0,
            "function": {"arguments": arguments[4:]}}]})),
        _sse(_chunk(finish_reason="length")),
    ]
    if late_error:
        frames.append(_sse({"error": {"code": "tool_calls_required"}}))
    usage = {"id": "chatcmpl-test1234", "choices": [],
             "usage": {"prompt_tokens": 10, "completion_tokens": 128}}
    frames.extend([_sse(usage), "data: [DONE]\n\n"])
    out = _collect(frames, required_tool_call=True)
    success = arguments == '{"path":"x"}' and not late_error
    assert _finish_reasons(out) == (["length"] if success else [])
    errors = [json.loads(f[6:])["error"] for f in out
              if f.startswith("data: ") and f.strip() != "data: [DONE]"
              and "error" in json.loads(f[6:])]
    assert len(errors) == (0 if success else 1)
    assert json.loads(out[-2][6:]) == usage
    assert out[-1] == "data: [DONE]\n\n"
