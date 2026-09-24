"""Native Qwen tool examples in reasoning must not become executable calls."""

import json
from types import SimpleNamespace

import pytest

from vmlx_engine import server
from vmlx_engine.engine.base import GenerationOutput
from vmlx_engine.reasoning.qwen3_parser import Qwen3ReasoningParser


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["chat", "chat_nonstream", "responses", "responses_nonstream"])
@pytest.mark.parametrize("implicit", [False, True])
@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("exact_once", [False, True])
async def test_qwen_tool_example_stays_in_reasoning(monkeypatch, api, implicit, closed, exact_once):
    import vmlx_engine.model_config_registry as registry

    def call(value):
        return (
            '<tool_call><function=record_payload>'
            f'<parameter=content>{value}</parameter>'
            '</function></tool_call>'
        )

    reasoning = "Consider this example: " + call("example") + " Still checking."
    text = ("" if implicit else "<think>") + reasoning
    if closed:
        text += "</think>" + call("actual")

    class Engine:
        tokenizer = SimpleNamespace(has_thinking=implicit)
        is_mllm = True
        preserve_native_tool_format = True

        async def chat(self, **kwargs):
            return GenerationOutput(
                text=text, raw_text=text, prompt_tokens=10, completion_tokens=128,
                finished=True, finish_reason="stop" if closed else "length",
            )

        async def stream_chat(self, **kwargs):
            for start in range(0, len(text), 7):
                end = min(start + 7, len(text))
                final = end == len(text)
                yield GenerationOutput(
                    text=text[:end], raw_text=text[:end], new_text=text[start:end],
                    tokens=[], prompt_tokens=10, completion_tokens=128 if final else 1,
                    finished=final,
                    finish_reason=("stop" if closed else "length") if final else None,
                )

    cfg = SimpleNamespace(
        family_name="qwen3_5", supports_thinking=True,
        supports_instruct_mode=True, reasoning_parser="qwen3",
        tool_parser="qwen", think_in_template=implicit, architecture_hints={},
    )
    engine = Engine()
    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setattr(server, "_model_name", "reasoning-isolation-test")
    monkeypatch.setattr(server, "_served_model_name", "reasoning-isolation-test")
    monkeypatch.setattr(server, "_model_path", None)
    monkeypatch.setattr(server, "_model_type", "mllm")
    monkeypatch.setattr(server, "_mcp_manager", None)
    monkeypatch.setattr(server, "_default_enable_thinking", None)
    monkeypatch.setattr(server, "_reasoning_parser", Qwen3ReasoningParser())
    monkeypatch.setattr(server, "_tool_call_parser", "qwen")
    monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", False)
    monkeypatch.setattr(registry, "get_model_config_registry", lambda: SimpleNamespace(lookup=lambda _: cfg))
    server._begin_tool_call_drop_capture()
    fn = {"name": "record_payload", "parameters": {
        "type": "object", "properties": {"content": {"type": "string"}},
        "required": ["content"], "additionalProperties": False,
    }}
    messages = [{"role": "user", "content": (
        "Call record_payload exactly once when ready." if exact_once
        else "Record the value when ready."
    )}]
    if api in {"chat", "chat_nonstream"}:
        request = server.ChatCompletionRequest(
            model="reasoning-isolation-test", messages=messages,
            tools=[{"type": "function", "function": fn}], enable_thinking=True,
            stream=True, max_tokens=128,
        )
        if api == "chat_nonstream":
            request.stream = False
            response = await server.create_chat_completion(request, fastapi_request=None)
            body = response.model_dump() if hasattr(response, "model_dump") else json.loads(response.body)
            choice = body["choices"][0]
            message = choice["message"]
            assert not message.get("content")
            calls = message.get("tool_calls") or []
            assert len(calls) == int(closed)
            if closed:
                assert json.loads(calls[0]["function"]["arguments"]) == {"content": "actual"}
            assert message["reasoning_content"] == reasoning
            assert choice["finish_reason"] == ("tool_calls" if closed else "length")
            return
        iterator = server.stream_chat_completion(
            engine, messages, request, tools=request.tools, enable_thinking=True,
            max_tokens=128,
        )
    else:
        request = server.ResponsesRequest(
            model="reasoning-isolation-test", input=messages,
            tools=[{"type": "function", **fn}], enable_thinking=True,
            stream=True, max_output_tokens=128,
        )
        if api == "responses_nonstream":
            request.stream = False
            response = await server.create_response(request, fastapi_request=None)
            assert not response.output_text
            calls = [item for item in response.output if item.type == "function_call"]
            assert len(calls) == int(closed)
            if closed:
                assert json.loads(calls[0].arguments) == {"content": "actual"}
            reason, = [item for item in response.output if item.type == "reasoning"]
            assert reason.summary[0].text == reasoning
            assert response.status == ("completed" if closed else "incomplete")
            return
        iterator = server.stream_responses_api(engine, messages, request)
    frames = [frame async for frame in iterator]
    events = [json.loads(line[6:]) for frame in frames for line in frame.splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    if api == "chat":
        choices = [c for e in events for c in e.get("choices", [])]
        calls = [c for choice in choices for c in choice.get("delta", {}).get("tool_calls", [])
                 if c.get("function", {}).get("arguments")]
        assert len(calls) == int(closed)
        if closed:
            assert json.loads(calls[0]["function"]["arguments"]) == {"content": "actual"}
        assert not any(c.get("delta", {}).get("content") for c in choices)
        reasoning_result = "".join(c.get("delta", {}).get("reasoning_content") or "" for c in choices)
    else:
        terminal, = [e for e in events if e.get("type") in
                     {"response.completed", "response.incomplete", "response.failed"}]
        assert terminal["type"] == ("response.completed" if closed else "response.incomplete")
        output = terminal["response"]["output"]
        calls = [item for item in output if item["type"] == "function_call"]
        assert len(calls) == int(closed)
        if closed:
            assert json.loads(calls[0]["arguments"]) == {"content": "actual"}
        assert not terminal["response"].get("output_text")
        assert not any(e.get("type") == "response.output_text.delta" for e in events)
        reasoning_result = "".join(e["delta"] for e in events
                                   if e.get("type") == "response.reasoning_summary_text.delta")
    assert reasoning_result == reasoning
