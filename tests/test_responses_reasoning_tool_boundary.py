# SPDX-License-Identifier: Apache-2.0
"""Reasoning deltas must not lose the tail that shares a chunk with a call."""
import json
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("dialect", ["spark25", "qwen-json", "qwen-xml"])
@pytest.mark.parametrize("boundary", ["coalesced", "single_chunk", "separate"])
async def test_reasoning_tail_precedes_tool_without_delta_loss(monkeypatch, dialect, boundary):
    import vmlx_engine.model_config_registry as registry
    import vmlx_engine.server as server
    from vmlx_engine.api.models import ResponsesRequest
    from vmlx_engine.engine.base import GenerationOutput
    from vmlx_engine.reasoning.qwen3_parser import Qwen3ReasoningParser

    args = {"fields": ["delivery"], "max_fields": 1, "pretty": False, "filter": None}
    if dialect == "spark25":
        block = "<tool_call>read_json_fields" + "".join(
            f"<arg_key>{key}</arg_key><arg_value>{json.dumps(value)}</arg_value>"
            for key, value in args.items()
        ) + "</tool_call>"
    elif dialect == "qwen-json":
        block = "<tool_call>" + json.dumps({"name": "read_json_fields", "arguments": args}) + "</tool_call>"
    else:
        block = "<tool_call><function=read_json_fields>" + "".join(
            f"<parameter={key}>{json.dumps(value)}</parameter>"
            for key, value in args.items()
        ) + "</function></tool_call>"
    reasoning = "Let me make the call."
    if boundary == "coalesced":
        chunks = ["<think>Let me make the", " call.</think>" + block[:11], block[11:]]
    elif boundary == "single_chunk":
        chunks = ["<think>" + reasoning + "</think>" + block]
    else:
        chunks = ["<think>" + reasoning, "</think>", block]

    class Engine:
        tokenizer = SimpleNamespace(has_thinking=False)

        async def stream_chat(self, **kwargs):
            text = ""
            for i, delta in enumerate(chunks):
                text += delta
                terminal = i == len(chunks) - 1
                yield GenerationOutput(
                    text=text, raw_text=text, new_text=delta,
                    prompt_tokens=20, completion_tokens=i+1,
                    finished=terminal, finish_reason="stop" if terminal else None,
                )

    cfg = SimpleNamespace(
        family_name="spark2_5" if dialect == "spark25" else "qwen3_5",
        supports_thinking=True, supports_instruct_mode=True,
        reasoning_parser="qwen3", tool_parser="spark25" if dialect == "spark25" else "qwen",
        think_in_template=False, architecture_hints={},
    )
    monkeypatch.setattr(server, "_model_name", "stream-boundary-test")
    monkeypatch.setattr(server, "_model_path", None)
    monkeypatch.setattr(server, "_default_timeout", 5.0)
    monkeypatch.setattr(server, "_default_enable_thinking", None)
    monkeypatch.setattr(server, "_reasoning_parser", Qwen3ReasoningParser())
    monkeypatch.setattr(server, "_tool_call_parser", cfg.tool_parser)
    monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", False)
    monkeypatch.setattr(registry, "get_model_config_registry", lambda: SimpleNamespace(lookup=lambda key: cfg))
    request = ResponsesRequest(
        model="stream-boundary-test", input="Read the delivery fields.",
        enable_thinking=True, max_output_tokens=128, stream=True,
        tools=[{"type": "function", "name": "read_json_fields", "parameters": {
            "type": "object", "properties": {
                "fields": {"type": "array", "items": {"type": "string"}},
                "max_fields": {"type": "integer"}, "pretty": {"type": "boolean"},
                "filter": {"type": ["object", "null"]},
            }, "required": list(args), "additionalProperties": False,
        }}],
    )
    events = []
    async for frame in server.stream_responses_api(
        Engine(), [{"role": "user", "content": request.input}], request,
    ):
        for line in frame.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    terminals = [e for e in events if e["type"] in ("response.completed", "response.failed", "response.incomplete")]
    assert len(terminals) == 1 and terminals[0]["type"] == "response.completed"
    output = terminals[0]["response"]["output"]
    calls = [item for item in output if item["type"] == "function_call"]
    assert len(calls) == 1 and json.loads(calls[0]["arguments"]) == args
    reason_item, = [item for item in output if item["type"] == "reasoning"]
    assert reason_item["summary"][0]["text"] == reasoning
    deltas = [e for e in events if e["type"] == "response.reasoning_summary_text.delta"]
    assert "".join(e["delta"] for e in deltas) == reasoning
    assert all(e["item_id"] == reason_item["id"] for e in deltas)
    done_index = next(i for i,e in enumerate(events) if e["type"] == "response.reasoning_summary_text.done")
    tool_index = next(i for i,e in enumerate(events) if e["type"] == "response.function_call_arguments.delta")
    assert all(events.index(e) < done_index < tool_index for e in deltas)
    assert not any(e["type"] == "response.output_text.delta" for e in events)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prefix", "cleaned_override", "expected"),
    [
        ("I'll read the JSON file with the", None, "I'll read the JSON file with the"),
        ("I'll read", "I'll read the file.", "I'll read the file."),
        ("I'll read the file.", "I'll read the file.", "I'll read the file."),
        ("", "I'll read the file.", "I'll read the file."),
        ("", None, ""),
        ("I'll read", "Rewritten parser prose.", "I'll read"),
        ("I'll read ", "I'll read the file.", "I'll read the file."),
    ],
    ids=["native-none", "cleaned-suffix", "no-duplicate", "empty-prefix", "empty-both", "nonmonotonic-cleanup", "prefix-whitespace"],
)
async def test_responses_tool_finalization_keeps_streamed_content(
    monkeypatch, prefix, cleaned_override, expected,
):
    """A valid glm47 call must not erase prose already emitted before it."""
    import vmlx_engine.model_config_registry as registry
    import vmlx_engine.server as server
    from vmlx_engine.api.models import ResponsesRequest
    from vmlx_engine.engine.base import GenerationOutput
    from vmlx_engine.reasoning.qwen3_parser import Qwen3ReasoningParser

    reasoning = "Check the fixture."
    block = '<tool_call>read_file<arg_key>path</arg_key><arg_value>/fixture.json</arg_value></tool_call>'
    chunks = ["<think>" + reasoning, "</think>" + prefix, block]

    class Engine:
        tokenizer = SimpleNamespace(has_thinking=False)

        async def stream_chat(self, **kwargs):
            text = ""
            for i, delta in enumerate(chunks):
                text += delta
                terminal = i == len(chunks) - 1
                yield GenerationOutput(
                    text=text, raw_text=text, new_text=delta,
                    prompt_tokens=20, completion_tokens=i + 1,
                    finished=terminal, finish_reason="stop" if terminal else None,
                )

    cfg = SimpleNamespace(
        family_name="qwen3_5", supports_thinking=True,
        supports_instruct_mode=True, reasoning_parser="qwen3",
        tool_parser="glm47", think_in_template=False, architecture_hints={},
    )
    monkeypatch.setattr(server, "_model_name", "tool-finalization-test")
    monkeypatch.setattr(server, "_model_path", None)
    monkeypatch.setattr(server, "_default_timeout", 5.0)
    monkeypatch.setattr(server, "_default_enable_thinking", None)
    monkeypatch.setattr(server, "_reasoning_parser", Qwen3ReasoningParser())
    monkeypatch.setattr(server, "_tool_call_parser", "glm47")
    monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", False)
    monkeypatch.setattr(registry, "get_model_config_registry", lambda: SimpleNamespace(lookup=lambda key: cfg))
    if cleaned_override is not None:
        native_parse = server._parse_tool_calls_with_parser

        def with_cleaned_suffix(text, request):
            # Preserve the real parser/validation/tool AST; vary only its
            # cleaned-prose result to exercise the finalizer's suffix contract.
            _, calls = native_parse(text, request)
            return cleaned_override, calls

        monkeypatch.setattr(server, "_parse_tool_calls_with_parser", with_cleaned_suffix)
    request = ResponsesRequest(
        model="tool-finalization-test", input="Read the fixture.",
        enable_thinking=True, max_output_tokens=128, stream=True,
        tools=[{"type": "function", "name": "read_file", "parameters": {
            "type": "object", "properties": {"path": {"type": "string"}},
            "required": ["path"], "additionalProperties": False,
        }}],
    )
    events = []
    async for frame in server.stream_responses_api(
        Engine(), [{"role": "user", "content": request.input}], request,
    ):
        for line in frame.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    terminal, = [e for e in events if e["type"] in {"response.completed", "response.failed", "response.incomplete"}]
    assert terminal["type"] == "response.completed"
    output = terminal["response"]["output"]
    call, = [i for i in output if i["type"] == "function_call"]
    assert call["name"] == "read_file"
    assert json.loads(call["arguments"]) == {"path": "/fixture.json"}
    assert call["arguments"] == "".join(e["delta"] for e in events if e["type"] == "response.function_call_arguments.delta")
    reason, = [i for i in output if i["type"] == "reasoning"]
    assert reason["summary"][0]["text"] == reasoning
    assert "".join(e["delta"] for e in events if e["type"] == "response.reasoning_summary_text.delta") == reasoning
    deltas = [e for e in events if e["type"] == "response.output_text.delta"]
    assert "".join(e["delta"] for e in deltas) == expected
    assert terminal["response"]["output_text"] == expected
    messages = [i for i in output if i["type"] == "message"]
    if not expected:
        assert not messages and not deltas
        return
    message, = messages
    assert message["content"][0]["text"] == expected
    assert all(e["item_id"] == message["id"] for e in deltas)
    assert [e["text"] for e in events if e["type"] == "response.output_text.done"] == [expected]
    assert [e["part"]["text"] for e in events if e["type"] == "response.content_part.done"] == [expected]
    assert [e["item"] for e in events if e["type"] == "response.output_item.done" and e["item"]["type"] == "message"] == [message]
    text_done = next(i for i, e in enumerate(events) if e["type"] == "response.output_text.done")
    tool_start = next(i for i, e in enumerate(events) if e["type"] == "response.function_call_arguments.delta")
    assert all(events.index(e) < text_done < tool_start for e in deltas)
