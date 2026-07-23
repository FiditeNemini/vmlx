"""Laguna honors an explicit thinking cap without changing Auto policy."""

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
from vmlx_engine.reasoning.deepseek_r1_parser import DeepSeekR1ReasoningParser


class _LagunaBudgetEngine:
    """Run out the explicit reasoning cap, then answer on the direct rail."""

    class _Tokenizer:
        has_thinking = False

        def apply_chat_template(self, messages, **kwargs):
            suffix = "<think>" if kwargs.get("enable_thinking") else "</think>"
            return f"<user>{messages[-1]['content']}</user><assistant>{suffix}"

    tokenizer = _Tokenizer()

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def stream_chat(self, *, messages, **kwargs):
        self.calls.append(dict(kwargs))
        if kwargs.get("enable_thinking") is False:
            text = ""
            for index, delta in enumerate(("LAGUNA-", "VISIBLE-DONE"), start=1):
                text += delta
                yield GenerationOutput(
                    text=text,
                    raw_text=text,
                    new_text=delta,
                    tokens=[],
                    prompt_tokens=9,
                    completion_tokens=index,
                    finished=index == 2,
                    finish_reason="stop" if index == 2 else None,
                )
            return

        reasoning = "private Laguna reasoning filled the explicit budget"
        yield GenerationOutput(
            text=reasoning,
            raw_text=reasoning,
            new_text=reasoning,
            tokens=[],
            prompt_tokens=7,
            completion_tokens=int(kwargs["max_tokens"]),
            finished=True,
            finish_reason="length",
        )


def _install_laguna_policy(monkeypatch) -> None:
    config = SimpleNamespace(
        family_name="laguna",
        model_type="laguna",
        supports_thinking=True,
        supports_instruct_mode=True,
        reasoning_parser="deepseek_r1",
        tool_parser="glm47",
        think_in_template=True,
        architecture_hints={"default_enable_thinking": True},
    )
    monkeypatch.setattr(server, "_default_timeout", 5.0)
    monkeypatch.setattr(server, "_model_name", "Laguna-S-2.1-JANG_4M")
    monkeypatch.setattr(server, "_model_path", None)
    monkeypatch.setattr(server, "_reasoning_parser", DeepSeekR1ReasoningParser())
    monkeypatch.setattr(server, "_tool_call_parser", "glm47")
    monkeypatch.setattr(server, "_default_enable_thinking", None)
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


@pytest.mark.asyncio
async def test_laguna_chat_explicit_thinking_budget_caps_first_pass_and_answers(
    monkeypatch,
):
    _install_laguna_policy(monkeypatch)
    engine = _LagunaBudgetEngine()
    messages = [Message(role="user", content="reason, then answer visibly")]
    request = ChatCompletionRequest(
        model="Laguna-S-2.1-JANG_4M",
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

    assert engine.calls[0]["max_tokens"] == 32
    assert engine.calls[1]["enable_thinking"] is False
    assert engine.calls[1]["chat_template_kwargs"]["enable_thinking"] is False
    assert engine.calls[1]["max_tokens"] == 80
    events = _data_events(chunks)
    reasoning = [
        choice["delta"].get("reasoning_content", "")
        for event in events
        for choice in event.get("choices", [])
        if choice.get("delta", {}).get("reasoning_content")
    ]
    content = [
        choice["delta"].get("content", "")
        for event in events
        for choice in event.get("choices", [])
        if choice.get("delta", {}).get("content")
    ]
    assert reasoning == ["private Laguna reasoning filled the explicit budget"]
    assert content == ["LAGUNA-", "VISIBLE-DONE"]


@pytest.mark.asyncio
async def test_laguna_responses_explicit_thinking_budget_caps_first_pass_and_answers(
    monkeypatch,
):
    _install_laguna_policy(monkeypatch)
    engine = _LagunaBudgetEngine()
    request = ResponsesRequest(
        model="Laguna-S-2.1-JANG_4M",
        input="reason, then answer visibly",
        stream=True,
        enable_thinking=True,
        max_thinking_tokens=32,
        max_output_tokens=112,
        stream_options=StreamOptions(include_usage=True),
    )

    chunks = []
    async for chunk in server.stream_responses_api(
        engine,
        [{"role": "user", "content": "reason, then answer visibly"}],
        request,
        fastapi_request=None,
        max_tokens=112,
    ):
        chunks.append(chunk)

    assert engine.calls[0]["max_tokens"] == 32
    assert engine.calls[1]["enable_thinking"] is False
    assert engine.calls[1]["chat_template_kwargs"]["enable_thinking"] is False
    assert engine.calls[1]["max_tokens"] == 80
    events = _data_events(chunks)
    assert [
        event["delta"]
        for event in events
        if event.get("type") == "response.reasoning_summary_text.delta"
    ] == ["private Laguna reasoning filled the explicit budget"]
    assert [
        event["delta"]
        for event in events
        if event.get("type") == "response.output_text.delta"
    ] == ["LAGUNA-", "VISIBLE-DONE"]
    completed = next(
        event["response"]
        for event in events
        if event.get("type") == "response.completed"
    )
    assert completed["output_text"] == "LAGUNA-VISIBLE-DONE"


def test_laguna_remains_outside_omitted_budget_auto_partition_allowlist():
    assert "laguna" in server._THINKING_BUDGET_CAP_FAMILIES
    assert "laguna" not in server._AUTO_THINKING_PARTITION_FAMILIES

