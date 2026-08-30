# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for terminal MLLM streaming detokenizer flushes."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from vmlx_engine.api.models import ChatCompletionRequest, Message, ResponsesRequest
from vmlx_engine.engine.batched import BatchedEngine, _reconcile_mllm_terminal_delta
from vmlx_engine.engine.base import GenerationOutput
from vmlx_engine.mllm_batch_generator import MLLMBatchResponse
from vmlx_engine.mllm_scheduler import MLLMRequest, MLLMScheduler
from vmlx_engine.reasoning.qwen3_parser import Qwen3ReasoningParser
from vmlx_engine.request import RequestOutput, SamplingParams
from vmlx_engine.server import _terminal_visible_stream_suffix


class _PendingTailDetokenizer:
    """Expose ``A`` immediately and hold ``]`` until ``finalize()``."""

    def __init__(self):
        self._text = ""
        self._pending = ""
        self.offset = 0

    @property
    def text(self):
        return self._text

    @property
    def last_segment(self):
        segment = self._text[self.offset :]
        self.offset = len(self._text)
        return segment

    def add_token(self, token):
        if token == 0:
            self._text += "A"
            self._pending += "]"

    def finalize(self):
        self._text += self._pending
        self._pending = ""


def _scheduler(request_id: str):
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.processor = object()
    scheduler.uid_to_request_id = {0: request_id}
    scheduler.running = {
        request_id: MLLMRequest(
            request_id=request_id,
            prompt="",
            sampling_params=SamplingParams(max_tokens=8),
        )
    }
    detokenizer = _PendingTailDetokenizer()
    scheduler._get_detokenizer = lambda _request_id, _tokenizer: detokenizer
    scheduler.total_prompt_tokens = 0
    scheduler.total_completion_tokens = 0
    scheduler.num_requests_processed = 0
    scheduler._record_cache_hit = lambda *_args, **_kwargs: None
    scheduler.batch_generator = None
    return scheduler


def _response(request_id: str, token: int, finish_reason=None):
    return MLLMBatchResponse(
        uid=0,
        request_id=request_id,
        token=token,
        logprobs=None,
        finish_reason=finish_reason,
        prompt_token_ids=[7, 8],
    )


def test_single_response_terminal_flush_reaches_new_text():
    scheduler = _scheduler("single-final-tail")

    first, finished = scheduler._process_batch_responses(
        [_response("single-final-tail", 0)]
    )
    assert finished == set()
    assert first[0].new_text == "A"

    terminal, finished = scheduler._process_batch_responses(
        [_response("single-final-tail", 99, "stop")]
    )
    assert finished == {"single-final-tail"}
    assert terminal[0].new_text == "]"
    assert terminal[0].output_text == "A]"


def test_coalesced_terminal_flush_is_appended_to_burst_delta():
    scheduler = _scheduler("coalesced-final-tail")

    outputs, finished = scheduler._process_batch_responses(
        [
            _response("coalesced-final-tail", 0),
            _response("coalesced-final-tail", 99, "stop"),
        ]
    )

    assert finished == {"coalesced-final-tail"}
    assert len(outputs) == 1
    assert outputs[0].new_text == "A]"
    assert outputs[0].output_text == "A]"


def test_terminal_reconciliation_appends_only_an_authoritative_suffix():
    assert _reconcile_mllm_terminal_delta(
        "assert value == [",
        "",
        "assert value == []",
        finished=True,
    ) == "]"
    assert _reconcile_mllm_terminal_delta(
        "assert value == [",
        "]",
        "assert value == []",
        finished=True,
    ) == "]"


def test_terminal_reconciliation_never_rewrites_a_nonmonotonic_stream():
    assert _reconcile_mllm_terminal_delta(
        "already streamed",
        " bytes",
        "different final text",
        finished=True,
    ) == " bytes"


def test_terminal_reconciliation_uses_the_existing_display_normalization():
    assert _reconcile_mllm_terminal_delta(
        "<|im_start|>assert value == [",
        "",
        "assert value == []",
        finished=True,
    ) == "]"


class _TerminalSuffixScheduler:
    async def add_request_async(self, **_kwargs):
        return "terminal-suffix"

    async def stream_outputs(self, _request_id):
        yield RequestOutput(
            request_id="terminal-suffix",
            new_text="assert value == [",
            output_text="",
            finished=False,
            finish_reason=None,
        )
        yield RequestOutput(
            request_id="terminal-suffix",
            new_text="",
            output_text="assert value == []",
            finished=True,
            finish_reason="stop",
        )


def test_batched_mllm_stream_generate_reconciles_terminal_suffix():
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._loaded = True
    engine._is_mllm = True
    engine._mllm_scheduler = _TerminalSuffixScheduler()

    async def _collect():
        return [
            output
            async for output in engine.stream_generate(
                prompt="ignored",
                max_tokens=8,
                temperature=0,
            )
        ]

    outputs = asyncio.run(_collect())
    assert "".join(output.new_text for output in outputs) == "assert value == []"
    assert outputs[-1].text == "assert value == []"
    assert outputs[-1].finished is True


def test_terminal_visible_stream_suffix_reconciles_qwen_content():
    parser = Qwen3ReasoningParser()
    parser.reset_state(think_in_prompt=False)

    assert _terminal_visible_stream_suffix(
        "assert value == []",
        "assert value == [",
        parser=parser,
    ) == "]"


def test_terminal_visible_stream_suffix_keeps_qwen_reasoning_private():
    parser = Qwen3ReasoningParser()
    parser.reset_state(think_in_prompt=False)

    assert _terminal_visible_stream_suffix(
        "<think>private</think>answer []",
        "answer [",
        parser=parser,
    ) == "]"


def test_terminal_visible_stream_suffix_never_rewrites_nonmonotonic_text():
    assert _terminal_visible_stream_suffix(
        "different final text",
        "already streamed",
    ) == ""


class _TerminalVisibleSuffixEngine:
    tokenizer = SimpleNamespace(has_thinking=False)

    async def stream_chat(self, *, messages, **kwargs):
        del messages, kwargs
        yield GenerationOutput(
            text="assert value == [",
            new_text="assert value == [",
            prompt_tokens=3,
            completion_tokens=4,
            finished=False,
            finish_reason=None,
        )
        yield GenerationOutput(
            text="assert value == []",
            new_text="",
            prompt_tokens=3,
            completion_tokens=5,
            finished=True,
            finish_reason="stop",
        )


def _configure_terminal_suffix_server(monkeypatch):
    import vmlx_engine.server as server

    monkeypatch.setattr(server, "_default_timeout", 5.0)
    monkeypatch.setattr(server, "_model_name", "qwen4-terminal-suffix")
    monkeypatch.setattr(server, "_model_path", None)
    monkeypatch.setattr(server, "_reasoning_parser", Qwen3ReasoningParser())
    monkeypatch.setattr(server, "_tool_call_parser", None)
    return server


@pytest.mark.asyncio
async def test_chat_stream_emits_terminal_visible_suffix(monkeypatch):
    server = _configure_terminal_suffix_server(monkeypatch)
    request = ChatCompletionRequest(
        model="qwen4-terminal-suffix",
        messages=[Message(role="user", content="return code")],
        stream=True,
        enable_thinking=False,
    )

    chunks = []
    async for line in server.stream_chat_completion(
        _TerminalVisibleSuffixEngine(),
        [message.model_dump(exclude_none=True) for message in request.messages],
        request,
    ):
        if line.startswith("data: ") and line.strip() != "data: [DONE]":
            chunks.append(json.loads(line.removeprefix("data: ")))

    assert "".join(
        chunk["choices"][0]["delta"].get("content") or ""
        for chunk in chunks
        if chunk.get("choices")
    ) == "assert value == []"
    assert any(
        chunk["choices"][0].get("finish_reason") == "stop"
        for chunk in chunks
        if chunk.get("choices")
    )


@pytest.mark.asyncio
async def test_responses_stream_emits_terminal_visible_suffix(monkeypatch):
    server = _configure_terminal_suffix_server(monkeypatch)
    request = ResponsesRequest(
        model="qwen4-terminal-suffix",
        input="return code",
        stream=True,
        enable_thinking=False,
    )

    events = []
    async for chunk in server.stream_responses_api(
        _TerminalVisibleSuffixEngine(),
        [{"role": "user", "content": "return code"}],
        request,
    ):
        for line in chunk.splitlines():
            if line.startswith("data: "):
                data = line.removeprefix("data: ")
                if data != "[DONE]":
                    events.append(json.loads(data))

    deltas = "".join(
        event.get("delta", "")
        for event in events
        if event.get("type") == "response.output_text.delta"
    )
    done = [
        event.get("text", "")
        for event in events
        if event.get("type") == "response.output_text.done"
    ]
    assert deltas == "assert value == []"
    assert done == ["assert value == []"]
