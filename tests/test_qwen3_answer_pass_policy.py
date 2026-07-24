"""Shared Chat/Responses contract for bounded plain-Qwen3 reasoning."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import vmlx_engine.model_config_registry as registry
import vmlx_engine.server as server
from vmlx_engine.api.models import (
    ChatCompletionRequest,
    Message,
    ResponsesRequest,
    StreamOptions,
)
from vmlx_engine.engine.base import GenerationOutput
from vmlx_engine.reasoning.qwen3_parser import Qwen3ReasoningParser


class _Qwen3BudgetEngine:
    tokenizer = SimpleNamespace(has_thinking=False)

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def stream_chat(self, *, messages, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("enable_thinking") is False:
            text = ""
            for index, (delta, finished) in enumerate(
                (("Q3-", False), ("STREAM-DONE", True)), start=1
            ):
                text += delta
                yield GenerationOutput(
                    text=text,
                    new_text=delta,
                    tokens=[],
                    prompt_tokens=9,
                    completion_tokens=index,
                    finished=finished,
                    finish_reason="stop" if finished else None,
                )
            return

        reasoning = "<think>plain qwen3 reasoning pass hit the explicit budget"
        yield GenerationOutput(
            text=reasoning,
            new_text=reasoning,
            tokens=[],
            prompt_tokens=7,
            completion_tokens=int(kwargs["max_tokens"]),
            finished=True,
            finish_reason="length",
        )


class _Qwen35AutoBudgetOverrunEngine:
    tokenizer = SimpleNamespace(has_thinking=False)
    is_mllm = False
    preserve_native_tool_format = False

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def stream_chat(self, *, messages, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("enable_thinking") is False:
            text = ""
            for index, delta in enumerate(("Q35-", "VISIBLE-DONE"), start=1):
                text += delta
                yield GenerationOutput(
                    text=text,
                    new_text=delta,
                    tokens=[],
                    prompt_tokens=11,
                    completion_tokens=index,
                    finished=index == 2,
                    finish_reason="stop" if index == 2 else None,
                )
            return

        reasoning = "<think>qwen3.5 auto reasoning overran the implicit UI cap"
        yield GenerationOutput(
            text=reasoning,
            new_text=reasoning,
            tokens=[],
            prompt_tokens=17,
            completion_tokens=int(kwargs["max_tokens"]),
            finished=True,
            finish_reason="length",
        )

    async def chat(self, *, messages, **kwargs):
        self.calls.append(kwargs)
        reasoning = "<think>qwen3.5 auto reasoning overran the implicit UI cap"
        return GenerationOutput(
            text=reasoning,
            raw_text=reasoning,
            tokens=[],
            prompt_tokens=17,
            completion_tokens=int(kwargs["max_tokens"]),
            finished=True,
            finish_reason="length",
        )


class _Qwen35NativeTransitionEngine:
    """One native decode crosses from reasoning into progressive visible text."""

    tokenizer = SimpleNamespace(has_thinking=False)
    is_mllm = False
    preserve_native_tool_format = False

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def stream_chat(self, *, messages, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("enable_thinking") is False:
            text = ""
            for index, delta in enumerate(
                ("BANANA8426\n", "Q35-PARTIAL-", "DONE"), start=1
            ):
                text += delta
                yield GenerationOutput(
                    text=text,
                    new_text=delta,
                    tokens=[],
                    prompt_tokens=11,
                    completion_tokens=index,
                    finished=index == 3,
                    finish_reason="stop" if index == 3 else None,
                )
            return

        reasoning = "<think>private planning</think>"
        visible_prefix = "BANANA8426\nQ35-PARTIAL-"
        visible_suffix = "DONE"
        yield GenerationOutput(
            text=reasoning,
            new_text=reasoning,
            tokens=[],
            prompt_tokens=17,
            completion_tokens=1,
            finished=False,
            finish_reason=None,
        )
        yield GenerationOutput(
            text=reasoning + visible_prefix,
            new_text=visible_prefix,
            tokens=[],
            prompt_tokens=17,
            completion_tokens=2,
            finished=False,
            finish_reason=None,
        )
        yield GenerationOutput(
            text=reasoning + visible_prefix + visible_suffix,
            new_text=visible_suffix,
            tokens=[],
            prompt_tokens=17,
            completion_tokens=3,
            finished=True,
            finish_reason="stop",
        )

    async def chat(self, *, messages, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("enable_thinking") is False:
            visible = "BANANA8426\nQ35-PARTIAL-DONE"
            return GenerationOutput(
                text=visible,
                raw_text=visible,
                tokens=[],
                prompt_tokens=11,
                completion_tokens=3,
                finished=True,
                finish_reason="stop",
            )

        reasoning = "<think>private planning</think>"
        visible = "BANANA8426\nQ35-PARTIAL-DONE"
        raw = reasoning + visible
        return GenerationOutput(
            text=raw,
            raw_text=raw,
            tokens=[],
            prompt_tokens=17,
            completion_tokens=3,
            finished=True,
            finish_reason="stop",
        )


class _Qwen35TruncatedAnswerPassEngine:
    """Reasoning pass is rescued by a visible-answer pass that also truncates."""

    tokenizer = SimpleNamespace(has_thinking=False)
    is_mllm = False
    preserve_native_tool_format = False

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def stream_chat(self, *, messages, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("enable_thinking") is False:
            text = ""
            for index, delta in enumerate(("Q35-CUT-", "OFF"), start=1):
                text += delta
                yield GenerationOutput(
                    text=text,
                    new_text=delta,
                    tokens=[],
                    prompt_tokens=11,
                    completion_tokens=(
                        int(kwargs.get("max_tokens") or index)
                        if index == 2
                        else index
                    ),
                    finished=index == 2,
                    finish_reason="length" if index == 2 else None,
                )
            return

        reasoning = "<think>private reasoning ran out of budget"
        yield GenerationOutput(
            text=reasoning,
            new_text=reasoning,
            tokens=[],
            prompt_tokens=17,
            completion_tokens=int(kwargs["max_tokens"]),
            finished=True,
            finish_reason="length",
        )


class _Qwen35SuppressedRepeatToolEngine:
    """Post-tool reasoning emits a forbidden second native call, then direct answer."""

    tokenizer = SimpleNamespace(has_thinking=False)

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def stream_chat(self, *, messages, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("enable_thinking") is False:
            text = ""
            for index, delta in enumerate(("Q35-", "SUPPRESSED-DONE"), start=1):
                text += delta
                yield GenerationOutput(
                    text=text,
                    new_text=delta,
                    tokens=[],
                    prompt_tokens=11,
                    completion_tokens=index,
                    finished=index == 2,
                    finish_reason="stop" if index == 2 else None,
                )
            return

        raw = (
            "I should call the tool again even though it is disabled."
            "</think>\n\n<tool_call>\n<function=file_info>\n"
            "<parameter=path>\npanel/package.json\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        yield GenerationOutput(
            text=raw,
            raw_text=raw,
            new_text=raw,
            tokens=[],
            prompt_tokens=17,
            completion_tokens=24,
            finished=True,
            finish_reason="stop",
        )


class _Qwen35IncompleteReasoningToolSuffixEngine:
    """Post-tool reasoning ends in a rejected native-call prefix."""

    tokenizer = SimpleNamespace(has_thinking=False)

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def stream_chat(self, *, messages, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("enable_thinking") is False:
            text = ""
            for index, delta in enumerate(("Q35-", "POSTTOOL-DONE"), start=1):
                text += delta
                yield GenerationOutput(
                    text=text,
                    new_text=delta,
                    tokens=[],
                    prompt_tokens=11,
                    completion_tokens=index,
                    finished=index == 2,
                    finish_reason="stop" if index == 2 else None,
                )
            return

        raw = "<think>private post-tool planning must stay private\n<tool_call>"
        yield GenerationOutput(
            text=raw,
            raw_text=raw,
            new_text=raw,
            tokens=[],
            prompt_tokens=17,
            completion_tokens=int(kwargs["max_tokens"]),
            finished=True,
            finish_reason="length",
        )


def _install_qwen_policy(monkeypatch, family_name: str = "qwen3") -> None:
    config = SimpleNamespace(
        family_name=family_name,
        think_in_template=True,
        reasoning_parser="qwen3",
        tool_parser="qwen",
        supports_thinking=True,
    )
    monkeypatch.setattr(server, "_default_timeout", 5.0)
    monkeypatch.setattr(server, "_model_name", "qwen3-policy-test")
    monkeypatch.setattr(server, "_model_path", None)
    monkeypatch.setattr(server, "_reasoning_parser", Qwen3ReasoningParser())
    monkeypatch.setattr(server, "_tool_call_parser", "qwen")
    monkeypatch.setattr(
        registry,
        "get_model_config_registry",
        lambda *args, **kwargs: SimpleNamespace(lookup=lambda *a, **k: config),
    )


def _install_qwen3_policy(monkeypatch) -> None:
    _install_qwen_policy(monkeypatch, "qwen3")


def _install_hy3_policy(monkeypatch) -> None:
    config = SimpleNamespace(
        family_name="hy_v3",
        think_in_template=False,
        reasoning_parser="qwen3",
        tool_parser="hunyuan",
        supports_thinking=True,
    )
    monkeypatch.setattr(server, "_default_timeout", 5.0)
    monkeypatch.setattr(server, "_model_name", "hy3-policy-test")
    monkeypatch.setattr(server, "_model_path", None)
    monkeypatch.setattr(server, "_reasoning_parser", Qwen3ReasoningParser())
    monkeypatch.setattr(server, "_tool_call_parser", "hunyuan")
    monkeypatch.setattr(
        registry,
        "get_model_config_registry",
        lambda *args, **kwargs: SimpleNamespace(lookup=lambda *a, **k: config),
    )


def _data_events(chunks: list[str]) -> list[dict]:
    events: list[dict] = []
    for chunk in chunks:
        for line in chunk.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line.removeprefix("data: ")))
    return events


def _completed_response_message_texts(completed: dict) -> list[str]:
    texts: list[str] = []
    for item in completed.get("output", []):
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if part.get("type") == "output_text":
                texts.append(part.get("text", ""))
    return texts


def _responses_file_info_tool() -> dict:
    return {
        "type": "function",
        "name": "file_info",
        "description": "Inspect one path",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    }


def _chat_file_info_tool() -> dict:
    tool = _responses_file_info_tool()
    return {
        "type": "function",
        "function": {key: value for key, value in tool.items() if key != "type"},
    }


@pytest.mark.asyncio
async def test_qwen3_responses_streams_answer_after_explicit_thinking_budget(
    monkeypatch,
):
    _install_qwen3_policy(monkeypatch)
    engine = _Qwen3BudgetEngine()
    request = ResponsesRequest(
        model="qwen3-policy-test",
        input="say the marker",
        stream=True,
        enable_thinking=True,
        max_thinking_tokens=32,
        max_output_tokens=112,
        stream_options=StreamOptions(include_usage=True),
    )

    chunks = []
    async for chunk in server.stream_responses_api(
        engine,
        [{"role": "user", "content": "say the marker"}],
        request,
        fastapi_request=None,
        max_tokens=112,
    ):
        chunks.append(chunk)
    events = _data_events(chunks)

    assert engine.calls[0]["max_tokens"] == 32
    assert engine.calls[1]["enable_thinking"] is False
    assert engine.calls[1]["max_tokens"] == 80
    assert [
        event["delta"]
        for event in events
        if event.get("type") == "response.output_text.delta"
    ] == ["Q3-", "STREAM-DONE"]
    completed = next(
        event["response"] for event in events if event.get("type") == "response.completed"
    )
    assert _completed_response_message_texts(completed) == ["Q3-STREAM-DONE"]


@pytest.mark.asyncio
async def test_qwen3_chat_streams_answer_after_explicit_thinking_budget(monkeypatch):
    _install_qwen3_policy(monkeypatch)
    engine = _Qwen3BudgetEngine()
    messages = [Message(role="user", content="say the marker")]
    request = ChatCompletionRequest(
        model="qwen3-policy-test",
        messages=messages,
        stream=True,
        enable_thinking=True,
        max_thinking_tokens=32,
        max_tokens=112,
        stream_options=StreamOptions(include_usage=True),
    )

    chunks = []
    async for chunk in server.stream_chat_completion(
        engine,
        messages,
        request,
        fastapi_request=None,
        max_tokens=112,
    ):
        chunks.append(chunk)
    events = _data_events(chunks)

    assert engine.calls[0]["max_tokens"] == 32
    assert engine.calls[1]["enable_thinking"] is False
    assert engine.calls[1]["max_tokens"] == 80
    content_deltas = [
        choice["delta"].get("content", "")
        for event in events
        for choice in event.get("choices", [])
        if choice.get("delta", {}).get("content")
    ]
    assert content_deltas == ["Q3-", "STREAM-DONE"]


@pytest.mark.asyncio
async def test_qwen3_responses_auto_uses_one_native_full_cap_generation(monkeypatch):
    _install_qwen3_policy(monkeypatch)
    engine = _Qwen35NativeTransitionEngine()
    request = ResponsesRequest(
        model="qwen3-policy-test",
        input="say the marker",
        stream=True,
        enable_thinking=True,
        max_output_tokens=112,
    )

    chunks = []
    async for chunk in server.stream_responses_api(
        engine,
        [{"role": "user", "content": "say the marker"}],
        request,
        fastapi_request=None,
        max_tokens=112,
    ):
        chunks.append(chunk)

    assert len(engine.calls) == 1
    assert engine.calls[0]["max_tokens"] == 112
    events = _data_events(chunks)
    assert any(
        event.get("type") == "response.reasoning_summary_text.delta"
        for event in events
    )
    assert [
        event["delta"]
        for event in events
        if event.get("type") == "response.output_text.delta"
    ] == ["BANANA8426\nQ35-PARTIAL-", "DONE"]
    assert any(event.get("type") == "response.completed" for event in events)


@pytest.mark.asyncio
async def test_qwen3_responses_auto_tools_stay_in_one_native_generation(monkeypatch):
    """An Auto tool catalog must not trigger a synthetic tools-free retry."""
    _install_qwen3_policy(monkeypatch)
    engine = _Qwen35NativeTransitionEngine()
    tool = _responses_file_info_tool()
    request = ResponsesRequest(
        model="qwen3-policy-test",
        input="say the marker without using a tool",
        stream=True,
        enable_thinking=True,
        tools=[tool],
        tool_choice="auto",
        max_output_tokens=112,
    )

    chunks = []
    async for chunk in server.stream_responses_api(
        engine,
        [{"role": "user", "content": "say the marker without using a tool"}],
        request,
        fastapi_request=None,
        max_tokens=112,
        tools=[tool],
    ):
        chunks.append(chunk)

    assert len(engine.calls) == 1
    assert engine.calls[0]["max_tokens"] == 112
    events = _data_events(chunks)
    assert [
        event["delta"]
        for event in events
        if event.get("type") == "response.output_text.delta"
    ] == ["BANANA8426\nQ35-PARTIAL-", "DONE"]
    assert any(event.get("type") == "response.completed" for event in events)


@pytest.mark.asyncio
async def test_qwen3_chat_auto_tools_stay_in_one_native_generation(monkeypatch):
    _install_qwen3_policy(monkeypatch)
    engine = _Qwen35NativeTransitionEngine()
    tool = _chat_file_info_tool()
    messages = [Message(role="user", content="say the marker without using a tool")]
    request = ChatCompletionRequest(
        model="qwen3-policy-test",
        messages=messages,
        stream=True,
        enable_thinking=True,
        tools=[tool],
        tool_choice="auto",
        max_tokens=112,
        stream_options=StreamOptions(include_usage=True),
    )

    chunks = []
    async for chunk in server.stream_chat_completion(
        engine,
        messages,
        request,
        fastapi_request=None,
        max_tokens=112,
        tools=[{"name": "file_info", "parameters": tool["function"]["parameters"]}],
    ):
        chunks.append(chunk)

    assert len(engine.calls) == 1
    assert engine.calls[0]["max_tokens"] == 112
    events = _data_events(chunks)
    assert any(
        choice["delta"].get("reasoning_content")
        or choice["delta"].get("reasoning")
        for event in events
        for choice in event.get("choices", [])
    )
    content = "".join(
        choice["delta"].get("content", "")
        for event in events
        for choice in event.get("choices", [])
    )
    assert content == "BANANA8426\nQ35-PARTIAL-DONE"


@pytest.mark.asyncio
async def test_qwen3_responses_explicit_tool_intent_remains_fail_closed(monkeypatch):
    """A failed explicit tool request cannot become a tools-free prose retry."""
    _install_qwen3_policy(monkeypatch)
    engine = _Qwen3BudgetEngine()
    tool = _responses_file_info_tool()
    request = ResponsesRequest(
        model="qwen3-policy-test",
        input="Call the file_info tool for panel/package.json",
        stream=True,
        enable_thinking=True,
        tools=[tool],
        tool_choice="auto",
        max_output_tokens=112,
    )

    chunks = []
    async for chunk in server.stream_responses_api(
        engine,
        [{"role": "user", "content": "Call the file_info tool for panel/package.json"}],
        request,
        fastapi_request=None,
        max_tokens=112,
        tools=[tool],
    ):
        chunks.append(chunk)

    assert len(engine.calls) == 1
    assert engine.calls[0]["max_tokens"] == 112
    events = _data_events(chunks)
    assert not any(
        event.get("type") == "response.output_text.delta" for event in events
    )
    assert any(event.get("type") == "response.incomplete" for event in events)


@pytest.mark.asyncio
async def test_qwen3_chat_explicit_tool_intent_remains_fail_closed(monkeypatch):
    _install_qwen3_policy(monkeypatch)
    engine = _Qwen3BudgetEngine()
    tool = _chat_file_info_tool()
    messages = [
        Message(role="user", content="Call the file_info tool for panel/package.json")
    ]
    request = ChatCompletionRequest(
        model="qwen3-policy-test",
        messages=messages,
        stream=True,
        enable_thinking=True,
        tools=[tool],
        tool_choice="auto",
        max_tokens=112,
        stream_options=StreamOptions(include_usage=True),
    )

    chunks = []
    async for chunk in server.stream_chat_completion(
        engine,
        messages,
        request,
        fastapi_request=None,
        max_tokens=112,
        tools=[{"name": "file_info", "parameters": tool["function"]["parameters"]}],
    ):
        chunks.append(chunk)

    assert len(engine.calls) == 1
    assert engine.calls[0]["max_tokens"] == 112
    events = _data_events(chunks)
    assert not any(
        choice.get("delta", {}).get("content")
        for event in events
        for choice in event.get("choices", [])
    )


@pytest.mark.asyncio
async def test_qwen35_responses_suppressed_repeat_tool_streams_direct_answer(
    monkeypatch,
):
    """tool_choice=none markup is hidden and cannot block the answer pass."""
    _install_qwen_policy(monkeypatch, "qwen3_5")
    engine = _Qwen35SuppressedRepeatToolEngine()
    request = ResponsesRequest(
        model="dealignai/Qwen3.6-35B-A3B-JANGTQ-CRACK",
        input=[
            {
                "type": "function_call_output",
                "call_id": "call_prior",
                "output": "Size: 5.2 KB",
            }
        ],
        stream=True,
        enable_thinking=True,
        max_thinking_tokens=56,
        tool_choice="none",
        tools=[
            {
                "type": "function",
                "name": "file_info",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ],
        max_output_tokens=112,
    )

    chunks = []
    async for chunk in server.stream_responses_api(
        engine,
        [
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "tool", "tool_call_id": "call_prior", "content": "Size: 5.2 KB"},
        ],
        request,
        fastapi_request=None,
        max_tokens=112,
    ):
        chunks.append(chunk)

    events = _data_events(chunks)
    assert len(engine.calls) == 2
    assert engine.calls[1]["enable_thinking"] is False
    assert "tools" not in engine.calls[1]
    assert [
        event["delta"]
        for event in events
        if event.get("type") == "response.output_text.delta"
    ] == ["Q35-", "SUPPRESSED-DONE"]
    completed = next(
        event["response"] for event in events if event.get("type") == "response.completed"
    )
    assert completed["output_text"] == "Q35-SUPPRESSED-DONE"
    assert "<tool_call>" not in json.dumps(events)


@pytest.mark.asyncio
async def test_qwen35_responses_incomplete_reasoning_tool_suffix_stays_private(
    monkeypatch,
):
    """A rejected post-tool suffix cannot promote reasoning into output_text."""
    _install_qwen_policy(monkeypatch, "qwen3_5")
    engine = _Qwen35IncompleteReasoningToolSuffixEngine()
    tool = _responses_file_info_tool()
    request = ResponsesRequest(
        model="jangq-ai/Bonsai-27b-1bit-JANG",
        input=[
            {
                "type": "function_call_output",
                "call_id": "call_prior",
                "output": "Size: 5.2 KB",
            }
        ],
        stream=True,
        enable_thinking=True,
        max_thinking_tokens=56,
        tool_choice="auto",
        tools=[tool],
        max_output_tokens=112,
    )

    chunks = []
    async for chunk in server.stream_responses_api(
        engine,
        [
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "tool", "tool_call_id": "call_prior", "content": "Size: 5.2 KB"},
        ],
        request,
        fastapi_request=None,
        max_tokens=112,
        tools=[tool],
    ):
        chunks.append(chunk)

    events = _data_events(chunks)
    assert len(engine.calls) == 2
    assert engine.calls[1]["enable_thinking"] is False
    assert "tools" not in engine.calls[1]
    assert [
        event["delta"]
        for event in events
        if event.get("type") == "response.output_text.delta"
    ] == ["Q35-", "POSTTOOL-DONE"]
    completed = next(
        event["response"] for event in events if event.get("type") == "response.completed"
    )
    assert completed["output_text"] == "Q35-POSTTOOL-DONE"
    assert "private post-tool planning" not in completed["output_text"]
    assert not any(event.get("type") == "response.incomplete" for event in events)


@pytest.mark.asyncio
async def test_qwen35_chat_auto_blank_budget_preserves_native_length(monkeypatch):
    """Absent max_thinking_tokens means one native full-cap generation."""
    _install_qwen_policy(monkeypatch, "qwen3_5")
    engine = _Qwen35AutoBudgetOverrunEngine()
    messages = [Message(role="user", content="say the marker")]
    request = ChatCompletionRequest(
        model="dealignai/Qwen3.6-27B-MXFP8-CRACK-MTP",
        messages=messages,
        stream=True,
        stream_options=StreamOptions(include_usage=True),
    )

    chunks = []
    async for chunk in server.stream_chat_completion(
        engine,
        messages,
        request,
        fastapi_request=None,
        max_tokens=256,
    ):
        chunks.append(chunk)

    assert len(engine.calls) == 1
    assert engine.calls[0]["max_tokens"] == 256
    events = _data_events(chunks)
    assert any(
        choice["delta"].get("reasoning_content")
        or choice["delta"].get("reasoning")
        for event in events
        for choice in event.get("choices", [])
    )
    content_deltas = [
        choice["delta"].get("content", "")
        for event in events
        for choice in event.get("choices", [])
        if choice.get("delta", {}).get("content")
    ]
    assert content_deltas == []
    assert [
        choice.get("finish_reason")
        for event in events
        for choice in event.get("choices", [])
        if choice.get("finish_reason")
    ] == ["length"]


@pytest.mark.asyncio
async def test_qwen35_responses_auto_blank_budget_preserves_native_incomplete(
    monkeypatch,
):
    """Responses reports an honest incomplete native reasoning-only turn."""
    _install_qwen_policy(monkeypatch, "qwen3_5")
    engine = _Qwen35AutoBudgetOverrunEngine()
    request = ResponsesRequest(
        model="dealignai/Qwen3.6-27B-MXFP8-CRACK-MTP",
        input="say the marker",
        stream=True,
    )

    chunks = []
    async for chunk in server.stream_responses_api(
        engine,
        [{"role": "user", "content": "say the marker"}],
        request,
        fastapi_request=None,
        max_tokens=256,
    ):
        chunks.append(chunk)

    assert len(engine.calls) == 1
    assert engine.calls[0]["max_tokens"] == 256
    events = _data_events(chunks)
    assert any(
        event.get("type") == "response.reasoning_summary_text.delta"
        for event in events
    )
    assert not any(
        event.get("type") == "response.output_text.delta" for event in events
    )
    assert any(event.get("type") == "response.incomplete" for event in events)
    assert not any(event.get("type") == "response.completed" for event in events)


@pytest.mark.asyncio
async def test_qwen35_nonstream_chat_reasoning_only_preserves_native_length(
    monkeypatch,
):
    _install_qwen_policy(monkeypatch, "qwen3_5")
    engine = _Qwen35AutoBudgetOverrunEngine()
    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setattr(
        server, "_served_model_name", "dealignai/Qwen3.6-27B-MXFP8-CRACK-MTP"
    )
    request = ChatCompletionRequest(
        model="dealignai/Qwen3.6-27B-MXFP8-CRACK-MTP",
        messages=[Message(role="user", content="say the marker")],
        stream=False,
        max_tokens=256,
    )

    response = await server.create_chat_completion(request, fastapi_request=None)

    assert len(engine.calls) == 1
    assert engine.calls[0]["max_tokens"] == 256
    assert response.choices[0].message.content in (None, "")
    assert response.choices[0].message.reasoning_content
    assert response.choices[0].finish_reason == "length"


@pytest.mark.asyncio
async def test_qwen35_nonstream_responses_reasoning_only_is_incomplete(monkeypatch):
    _install_qwen_policy(monkeypatch, "qwen3_5")
    engine = _Qwen35AutoBudgetOverrunEngine()
    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setattr(
        server, "_served_model_name", "dealignai/Qwen3.6-27B-MXFP8-CRACK-MTP"
    )
    monkeypatch.setattr(server, "_model_type", "llm")
    monkeypatch.setattr(server, "_mcp_manager", None)
    request = ResponsesRequest(
        model="dealignai/Qwen3.6-27B-MXFP8-CRACK-MTP",
        input="say the marker",
        stream=False,
        max_output_tokens=256,
    )

    response = await server.create_response(request, fastapi_request=None)

    assert len(engine.calls) == 1
    assert engine.calls[0]["max_tokens"] == 256
    assert response.output_text in (None, "")
    assert response.status == "incomplete"


@pytest.mark.asyncio
async def test_qwen35_chat_auto_streams_native_reasoning_to_answer_transition(
    monkeypatch,
):
    _install_qwen_policy(monkeypatch, "qwen3_5")
    engine = _Qwen35NativeTransitionEngine()
    messages = [Message(role="user", content="read the marker")]
    request = ChatCompletionRequest(
        model="dealignai/Qwen3.6-27B-MXFP4-CRACK-MTP",
        messages=messages,
        stream=True,
        enable_thinking=True,
        max_tokens=512,
        stream_options=StreamOptions(include_usage=True),
    )

    chunks = []
    async for chunk in server.stream_chat_completion(
        engine,
        messages,
        request,
        fastapi_request=None,
        max_tokens=512,
    ):
        chunks.append(chunk)

    assert len(engine.calls) == 1
    assert engine.calls[0]["max_tokens"] == 512
    events = _data_events(chunks)
    content_deltas = [
        choice["delta"].get("content", "")
        for event in events
        for choice in event.get("choices", [])
        if choice.get("delta", {}).get("content")
    ]
    assert "".join(content_deltas) == "BANANA8426\nQ35-PARTIAL-DONE"
    assert content_deltas.count("BANANA8426\nQ35-PARTIAL-") == 1
    finish_reasons = [
        choice.get("finish_reason")
        for event in events
        for choice in event.get("choices", [])
        if choice.get("finish_reason")
    ]
    assert finish_reasons == ["stop"]


@pytest.mark.asyncio
async def test_qwen35_chat_truncated_answer_pass_reports_length_terminal(
    monkeypatch,
):
    """A cut-off visible-answer retry must not masquerade as finish_reason=stop."""
    _install_qwen_policy(monkeypatch, "qwen3_5")
    engine = _Qwen35TruncatedAnswerPassEngine()
    messages = [Message(role="user", content="read the marker")]
    request = ChatCompletionRequest(
        model="dealignai/Qwen3.6-27B-MXFP4-CRACK-MTP",
        messages=messages,
        stream=True,
        enable_thinking=True,
        max_thinking_tokens=256,
        max_tokens=512,
        stream_options=StreamOptions(include_usage=True),
    )

    chunks = []
    async for chunk in server.stream_chat_completion(
        engine,
        messages,
        request,
        fastapi_request=None,
        max_tokens=512,
    ):
        chunks.append(chunk)

    assert engine.calls[0]["max_tokens"] == 256
    assert engine.calls[1]["enable_thinking"] is False
    events = _data_events(chunks)
    content_deltas = [
        choice["delta"].get("content", "")
        for event in events
        for choice in event.get("choices", [])
        if choice.get("delta", {}).get("content")
    ]
    assert "".join(content_deltas) == "Q35-CUT-OFF"
    finish_reasons = [
        choice.get("finish_reason")
        for event in events
        for choice in event.get("choices", [])
        if choice.get("finish_reason")
    ]
    assert finish_reasons == ["length"]


@pytest.mark.asyncio
async def test_qwen35_nonstream_chat_preserves_native_completed_terminal(
    monkeypatch,
):
    """Non-stream Chat uses one native generation when no thinking cap exists."""
    _install_qwen_policy(monkeypatch, "qwen3_5")
    engine = _Qwen35NativeTransitionEngine()
    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setattr(
        server, "_served_model_name", "dealignai/Qwen3.6-27B-MXFP4-CRACK-MTP"
    )
    request = ChatCompletionRequest(
        model="dealignai/Qwen3.6-27B-MXFP4-CRACK-MTP",
        messages=[Message(role="user", content="read the marker")],
        stream=False,
        enable_thinking=True,
        max_tokens=512,
    )

    response = await server.create_chat_completion(request, fastapi_request=None)

    assert len(engine.calls) == 1
    assert engine.calls[0]["max_tokens"] == 512
    assert response.choices[0].message.content == "BANANA8426\nQ35-PARTIAL-DONE"
    assert response.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_qwen35_nonstream_responses_preserves_native_completed_terminal(
    monkeypatch,
):
    """Non-stream Responses uses one native generation at the full cap."""
    _install_qwen_policy(monkeypatch, "qwen3_5")
    engine = _Qwen35NativeTransitionEngine()
    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setattr(
        server, "_served_model_name", "dealignai/Qwen3.6-27B-MXFP4-CRACK-MTP"
    )
    monkeypatch.setattr(server, "_model_type", "llm")
    monkeypatch.setattr(server, "_mcp_manager", None)
    request = ResponsesRequest(
        model="dealignai/Qwen3.6-27B-MXFP4-CRACK-MTP",
        input="read the marker",
        stream=False,
        enable_thinking=True,
        max_output_tokens=512,
    )

    response = await server.create_response(request, fastapi_request=None)

    assert len(engine.calls) == 1
    assert engine.calls[0]["max_tokens"] == 512
    assert response.output_text == "BANANA8426\nQ35-PARTIAL-DONE"
    assert response.status == "completed"


@pytest.mark.asyncio
async def test_qwen35_responses_auto_streams_native_reasoning_to_answer_transition(
    monkeypatch,
):
    _install_qwen_policy(monkeypatch, "qwen3_5")
    engine = _Qwen35NativeTransitionEngine()
    request = ResponsesRequest(
        model="dealignai/Qwen3.6-27B-MXFP4-CRACK-MTP",
        input="read the marker",
        stream=True,
        enable_thinking=True,
        max_output_tokens=512,
    )

    chunks = []
    async for chunk in server.stream_responses_api(
        engine,
        [{"role": "user", "content": "read the marker"}],
        request,
        fastapi_request=None,
        max_tokens=512,
    ):
        chunks.append(chunk)

    assert len(engine.calls) == 1
    assert engine.calls[0]["max_tokens"] == 512
    events = _data_events(chunks)
    content_deltas = [
        event["delta"]
        for event in events
        if event.get("type") == "response.output_text.delta"
    ]
    assert "".join(content_deltas) == "BANANA8426\nQ35-PARTIAL-DONE"
    completed = next(
        event["response"] for event in events if event.get("type") == "response.completed"
    )
    assert completed["output_text"] == "BANANA8426\nQ35-PARTIAL-DONE"


@pytest.mark.asyncio
async def test_hy3_responses_auto_uses_one_native_full_cap_generation(
    monkeypatch,
):
    """Hy3 Auto is native one-pass when max_thinking_tokens is absent."""
    _install_hy3_policy(monkeypatch)
    engine = _Qwen35NativeTransitionEngine()
    request = ResponsesRequest(
        model="jangq-ai/Hy3-JANG_2K-MTP",
        input="say the marker",
        stream=True,
        max_output_tokens=512,
        tools=[_responses_file_info_tool()],
        tool_choice="none",
    )

    chunks = []
    async for chunk in server.stream_responses_api(
        engine,
        [{"role": "user", "content": "say the marker"}],
        request,
        fastapi_request=None,
        max_tokens=512,
    ):
        chunks.append(chunk)

    assert len(engine.calls) == 1
    assert engine.calls[0]["max_tokens"] == 512
    assert engine.calls[0].get("reasoning_effort") != "no_think"
    events = _data_events(chunks)
    assert [
        event["delta"]
        for event in events
        if event.get("type") == "response.output_text.delta"
    ] == ["BANANA8426\nQ35-PARTIAL-", "DONE"]
    completed = next(
        event["response"] for event in events if event.get("type") == "response.completed"
    )
    assert _completed_response_message_texts(completed) == [
        "BANANA8426\nQ35-PARTIAL-DONE"
    ]
