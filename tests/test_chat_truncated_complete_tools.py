"""Complete parsed calls must not hide an exhausted generation budget."""

import json
from types import SimpleNamespace

import pytest

from vmlx_engine import server
from vmlx_engine.engine.base import GenerationOutput


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["stop", "length"])
@pytest.mark.parametrize("required", [False, True])
async def test_chat_parsed_calls_preserve_budget_terminal(monkeypatch, finish, required):
    call = (
        '<tool_call><function=record_payload>'
        '<parameter=content>{"n":1}</parameter>'
        '</function></tool_call>'
    )
    text = call + "\n" + call + "\nChecking the requested values again"

    class Engine:
        tokenizer = SimpleNamespace(has_thinking=False)
        is_mllm = False
        preserve_native_tool_format = True

        async def stream_chat(self, **kwargs):
            for start in range(0, len(text), 7):
                end = min(start + 7, len(text))
                final = end == len(text)
                yield GenerationOutput(
                    text=text[:end], new_text=text[start:end], tokens=[],
                    prompt_tokens=10, completion_tokens=128 if final else 1,
                    finished=final, finish_reason=finish if final else None,
                )

    engine = Engine()
    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setattr(server, "_model_name", "tool-budget-test")
    monkeypatch.setattr(server, "_served_model_name", "tool-budget-test")
    monkeypatch.setattr(server, "_model_path", None)
    monkeypatch.setattr(server, "_reasoning_parser", None)
    monkeypatch.setattr(server, "_tool_call_parser", "xml_function")
    monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", False)
    monkeypatch.setattr(server, "_stream_tool_call_early_stop_parser", lambda _: None)
    server._begin_tool_call_drop_capture()
    tools = [{"type": "function", "function": {
        "name": "record_payload", "parameters": {
            "type": "object", "properties": {"content": {"type": "string"}},
        },
    }}]
    messages = [{"role": "user", "content": "Record the values."}]
    request = server.ChatCompletionRequest(
        model="tool-budget-test", messages=messages, tools=tools,
        stream=True, max_tokens=128, tool_choice="required" if required else "auto",
    )
    iterator = server.stream_chat_completion(
        engine, messages, request, fastapi_request=None, tools=tools, max_tokens=128,
    )
    chunks = [chunk async for chunk in server._terminal_finish_guard(iterator, required_tool_call=required)]
    events = [json.loads(line[6:]) for chunk in chunks for line in chunk.splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    choices = [choice for event in events for choice in event.get("choices", [])]
    calls = [call for choice in choices
             for call in choice.get("delta", {}).get("tool_calls", [])
             if call.get("function", {}).get("arguments")]
    assert len(calls) == 2
    assert all(json.loads(call["function"]["arguments"]) == {"content": '{"n":1}'}
               for call in calls)
    terminals = [choice["finish_reason"] for choice in choices if choice.get("finish_reason")]
    assert terminals == ["length" if finish == "length" else "tool_calls"]
    assert not any(event.get("error") for event in events)
