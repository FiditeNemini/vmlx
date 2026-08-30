# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for terminal MLLM streaming detokenizer flushes."""

import asyncio

from vmlx_engine.engine.batched import BatchedEngine, _reconcile_mllm_terminal_delta
from vmlx_engine.mllm_batch_generator import MLLMBatchResponse
from vmlx_engine.mllm_scheduler import MLLMRequest, MLLMScheduler
from vmlx_engine.request import RequestOutput, SamplingParams


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
