# SPDX-License-Identifier: Apache-2.0
"""Explicit MLLM abort must preserve FIFO text and carry a truthful terminal.

No model/Metal work: real scheduler queue/abort/stream methods, following the
__new__ fixtures in test_mllm_streaming_finalization.py. These are not GPU,
durability-publication, or live B2 numerical tests.
"""

import asyncio
from collections import deque
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vmlx_engine.api.models import ResponsesRequest
from vmlx_engine.engine.batched import BatchedEngine
from vmlx_engine.mllm_scheduler import MLLMRequest, MLLMScheduler, MLLMSchedulerOutput
from vmlx_engine.reasoning.qwen3_parser import Qwen3ReasoningParser
from vmlx_engine.request import RequestOutput, RequestStatus


def _scheduler(monkeypatch, request_id="cancel-target", *, capacity=8, waiting=False):
    import vmlx_engine.mllm_scheduler as module

    # Cancel of an unscheduled request must not touch Metal through cache GC.
    monkeypatch.setattr(module, "clear_mlx_memory_cache", lambda **_kwargs: None)
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    request = MLLMRequest(request_id=request_id, prompt="fixture")
    request.status = RequestStatus.WAITING if waiting else RequestStatus.RUNNING
    request.num_prompt_tokens = 64
    request._cached_tokens = 32
    request._cache_detail = "block-disk+ssm"
    request._extracted_cache = None
    scheduler._queue_lock = threading.RLock()
    scheduler._batch_lock = threading.RLock()
    scheduler.requests = {request_id: request}
    scheduler.running = {} if waiting else {request_id: request}
    scheduler.waiting = deque([request] if waiting else [])
    scheduler.request_id_to_uid = {} if waiting else {request_id: 7}
    scheduler.uid_to_request_id = {} if waiting else {7: request_id}
    scheduler._pending_aborts = set()
    scheduler.finished_req_ids = set()
    scheduler.output_queues = {request_id: asyncio.Queue(maxsize=capacity)}
    scheduler._terminal_cleanup_complete = asyncio.Event()
    scheduler._terminal_cleanup_complete.set()
    scheduler._cleanup_aborted_paged_request = Mock()
    scheduler._cleanup_detokenizer = Mock()
    scheduler.batch_generator = SimpleNamespace(
        remove=Mock(), _stats={},
        _clean_boundary_snapshots={request_id: object()},
        _mixed_swa_boundary_snapshots={request_id: object()},
    )
    return scheduler, request


def _queue_partial(scheduler, request, parts):
    for index, part in enumerate(parts, 1):
        scheduler.output_queues[request.request_id].put_nowait(RequestOutput(
            request_id=request.request_id,
            new_text=part,
            new_token_ids=[100 + index],
            output_token_ids=list(range(101, 101 + index)),
            prompt_tokens=64,
            completion_tokens=index,
            cached_tokens=32,
            cache_detail="block-disk+ssm",
        ))
    request.output_tokens = list(range(101, 101 + len(parts)))
    request.num_output_tokens = len(parts)
    request.total_output_tokens = len(parts)
    # Production sets this field only on natural finalization. A cancellation
    # must not mistake this empty stale value for the already-queued text.
    assert request.output_text == ""


async def _collect(scheduler, request_id):
    async def drain():
        return [output async for output in scheduler.stream_outputs(request_id)]
    return await asyncio.wait_for(drain(), timeout=1)


def _assert_aborted(outputs, request_id, text, tokens):
    assert "".join(output.new_text for output in outputs) == text
    terminals = [output for output in outputs if output.finished]
    assert len(terminals) == 1
    terminal = terminals[0]
    assert outputs[-1] is terminal
    assert terminal.request_id == request_id
    assert terminal.finish_reason == "aborted"
    assert terminal.prompt_tokens == 64
    assert terminal.completion_tokens == tokens
    assert terminal.cached_tokens == 32
    assert terminal.cache_detail == "block-disk+ssm"
    assert not terminal.error


@pytest.mark.asyncio
@pytest.mark.parametrize("capacity", [8, 2])
async def test_explicit_abort_preserves_partial_fifo_even_when_queue_full(monkeypatch, capacity):
    scheduler, request = _scheduler(monkeypatch, capacity=capacity)
    _queue_partial(scheduler, request, ["1\n", "2\n"])
    assert scheduler.abort_request(request.request_id) is True
    assert request.cancel_event.is_set()
    assert request.status == RequestStatus.FINISHED_ABORTED
    outputs = await _collect(scheduler, request.request_id)
    _assert_aborted(outputs, request.request_id, "1\n2\n", 2)
    assert [output.new_text for output in outputs if not output.finished] == ["1\n", "2\n"]
    assert request.request_id not in scheduler.output_queues


@pytest.mark.asyncio
@pytest.mark.parametrize("waiting", [False, True])
async def test_explicit_abort_before_first_output_has_empty_aborted_terminal(monkeypatch, waiting):
    scheduler, request = _scheduler(monkeypatch, waiting=waiting)
    assert scheduler.abort_request(request.request_id) is True
    outputs = await _collect(scheduler, request.request_id)
    _assert_aborted(outputs, request.request_id, "", 0)
    assert not scheduler.waiting


@pytest.mark.asyncio
async def test_explicit_abort_unblocks_an_already_waiting_consumer(monkeypatch):
    scheduler, request = _scheduler(monkeypatch)
    pending = asyncio.create_task(_collect(scheduler, request.request_id))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not pending.done()
    assert scheduler.abort_request(request.request_id) is True
    _assert_aborted(await pending, request.request_id, "", 0)


@pytest.mark.asyncio
async def test_abort_ignores_late_worker_output_and_does_not_claim_publication(monkeypatch):
    import vmlx_engine.mllm_scheduler as module

    scheduler, request = _scheduler(monkeypatch, capacity=2)
    _queue_partial(scheduler, request, ["kept", " text"])
    assert scheduler.abort_request(request.request_id) is True
    take = Mock(side_effect=AssertionError("Abort must not consume a normal publication outcome"))
    monkeypatch.setattr(module, "_PERSIST", SimpleNamespace(take=take))
    scheduler._dispatch_outputs(MLLMSchedulerOutput(outputs=[RequestOutput(
        request_id=request.request_id, new_text=" LATE", output_text="kept text LATE",
        finished=True, finish_reason="stop", prompt_tokens=64, completion_tokens=3,
    )]))
    outputs = await _collect(scheduler, request.request_id)
    _assert_aborted(outputs, request.request_id, "kept text", 2)
    assert outputs[-1].output_text == "kept text"
    take.assert_not_called()
    scheduler.batch_generator.remove.assert_not_called()


@pytest.mark.asyncio
async def test_repeated_abort_is_idempotent_and_keeps_worker_cleanup_deferred(monkeypatch):
    scheduler, request = _scheduler(monkeypatch)
    _queue_partial(scheduler, request, ["partial"])
    short = scheduler.batch_generator._clean_boundary_snapshots[request.request_id]
    mixed = scheduler.batch_generator._mixed_swa_boundary_snapshots[request.request_id]
    assert scheduler.abort_request(request.request_id) is True
    assert scheduler.abort_request(request.request_id) is False
    assert scheduler.abort_request("not-this-request") is False
    assert scheduler._pending_aborts == {request.request_id}
    assert scheduler.request_id_to_uid == {request.request_id: 7}
    scheduler.batch_generator.remove.assert_not_called()
    scheduler._cleanup_aborted_paged_request.assert_not_called()
    scheduler._cleanup_detokenizer.assert_not_called()
    assert scheduler.batch_generator._clean_boundary_snapshots[request.request_id] is short
    assert scheduler.batch_generator._mixed_swa_boundary_snapshots[request.request_id] is mixed
    _assert_aborted(await _collect(scheduler, request.request_id), request.request_id, "partial", 1)
    # Explicitly exercise the existing CPU-only worker cleanup boundary. No
    # model cache is constructed, published, or mutated by the abort consumer.
    scheduler._process_pending_aborts()
    scheduler.batch_generator.remove.assert_called_once_with([7])
    scheduler._cleanup_aborted_paged_request.assert_called_once_with(request.request_id)
    scheduler._cleanup_detokenizer.assert_called_once_with(request.request_id)
    assert not scheduler._pending_aborts
    assert request.request_id not in scheduler.requests
    assert request.request_id not in scheduler.batch_generator._clean_boundary_snapshots
    assert request.request_id not in scheduler.batch_generator._mixed_swa_boundary_snapshots
    assert scheduler.abort_request(request.request_id) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
async def test_cancelled_row_does_not_change_sibling_natural_terminal(monkeypatch, finish_reason):
    scheduler, request = _scheduler(monkeypatch)
    sibling = MLLMRequest(request_id="sibling", prompt="other history")
    sibling.status = (RequestStatus.FINISHED_STOPPED if finish_reason == "stop"
                      else RequestStatus.FINISHED_LENGTH_CAPPED)
    scheduler.requests[sibling.request_id] = sibling
    scheduler.running[sibling.request_id] = sibling
    scheduler.output_queues[sibling.request_id] = asyncio.Queue(maxsize=3)
    first = RequestOutput(request_id="sibling", new_text="SURVIVOR", completion_tokens=1)
    terminal = RequestOutput(request_id="sibling", output_text="SURVIVOR", finished=True,
                             finish_reason=finish_reason, prompt_tokens=91, completion_tokens=1)
    for output in (first, terminal, None):
        scheduler.output_queues[sibling.request_id].put_nowait(output)
    assert scheduler.abort_request(request.request_id) is True
    cancelled, survived = await asyncio.gather(
        _collect(scheduler, request.request_id), _collect(scheduler, sibling.request_id))
    _assert_aborted(cancelled, request.request_id, "", 0)
    assert survived == [first, terminal]
    assert not sibling.cancel_event.is_set()
    assert terminal.finish_reason == finish_reason and terminal.output_text == "SURVIVOR"
    assert scheduler._pending_aborts == {request.request_id}


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason,status", [
    ("stop", RequestStatus.FINISHED_STOPPED),
    ("length", RequestStatus.FINISHED_LENGTH_CAPPED),
    ("error", RequestStatus.FINISHED_ERROR),
])
async def test_abort_after_natural_terminal_queued_preserves_original_terminal(
    monkeypatch, finish_reason, status,
):
    scheduler, request = _scheduler(monkeypatch, capacity=3)
    request.status = status
    request.finish_reason = finish_reason
    first = RequestOutput(request_id=request.request_id, new_text="retained", completion_tokens=1)
    terminal = RequestOutput(
        request_id=request.request_id, output_text="retained", finished=True,
        finish_reason=finish_reason, prompt_tokens=64, completion_tokens=1,
        error="original error" if finish_reason == "error" else None,
    )
    for output in (first, terminal, None):
        scheduler.output_queues[request.request_id].put_nowait(output)
    assert scheduler.abort_request(request.request_id) is False
    assert not request.cancel_event.is_set()
    assert request.status == status and request.finish_reason == finish_reason
    assert not scheduler._pending_aborts and not scheduler.finished_req_ids
    assert scheduler.request_id_to_uid == {request.request_id: 7}
    assert await _collect(scheduler, request.request_id) == [first, terminal]
    assert terminal.finish_reason == finish_reason
    scheduler.batch_generator.remove.assert_not_called()
    scheduler._cleanup_aborted_paged_request.assert_not_called()
    scheduler._cleanup_detokenizer.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", [False, True])
async def test_responses_explicit_mllm_abort_is_incomplete_without_partial_history(monkeypatch, partial):
    import vmlx_engine.server as server

    request_id = "resp_b01_cancel_fixture"
    scheduler, owned = _scheduler(monkeypatch, request_id=request_id)
    if partial:
        _queue_partial(scheduler, owned, ["1\n", "2\n"])

    async def existing_request(**kwargs):
        assert kwargs["request_id"] == request_id
        return request_id

    # Only request setup is stubbed. Stream/abort propagate through the actual
    # scheduler and BatchedEngine bridge, not a fabricated aborted engine chunk.
    scheduler.add_request_async = existing_request
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._loaded = True
    engine._is_mllm = True
    engine._mllm_scheduler = scheduler

    class Bridge:
        tokenizer = SimpleNamespace(has_thinking=False)

        async def abort_request(self, rid):
            return await engine.abort_request(rid)

        async def stream_chat(self, **kwargs):
            if not partial:
                assert (await server.cancel_response(request_id))["success"] is True
            cancelled = not partial
            async for output in engine.stream_generate(
                prompt="fixture", request_id=kwargs["request_id"], max_tokens=16, temperature=0,
            ):
                yield output
                if not cancelled:
                    assert (await server.cancel_response(request_id))["success"] is True
                    cancelled = True

    bridge = Bridge()
    monkeypatch.setattr(server, "_engine", bridge)
    monkeypatch.setattr(server, "_default_timeout", 2.0)
    monkeypatch.setattr(server, "_model_name", "qwen4-terminal-cancel")
    monkeypatch.setattr(server, "_model_path", None)
    monkeypatch.setattr(server, "_reasoning_parser", Qwen3ReasoningParser())
    monkeypatch.setattr(server, "_tool_call_parser", None)
    store = Mock()
    monkeypatch.setattr(server, "_responses_store_history", store)
    request = ResponsesRequest(model="qwen4-terminal-cancel", input="count", stream=True,
                               enable_thinking=False, max_output_tokens=16)
    events = []
    async for chunk in server.stream_responses_api(
        bridge, [{"role": "user", "content": "count"}], request, response_id=request_id,
    ):
        for line in chunk.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line.removeprefix("data: ")))
    expected_text = "1\n2\n" if partial else ""
    assert "".join(event.get("delta", "") for event in events
                   if event.get("type") == "response.output_text.delta") == expected_text
    terminals = [event for event in events if event.get("type") in
                 {"response.completed", "response.incomplete", "response.failed"}]
    assert len(terminals) == 1 and terminals[0]["type"] == "response.incomplete"
    response = terminals[0]["response"]
    assert response["id"] == request_id and response["status"] == "incomplete"
    assert response["incomplete_details"] == {"reason": "cancelled"}
    assert response["output_text"] == expected_text
    assert response["usage"]["input_tokens"] == 64
    assert response["usage"]["output_tokens"] == (2 if partial else 0)
    assert all(item.get("status") == "incomplete" for item in response["output"] if "status" in item)
    assert not any(event.get("type") == "error" for event in events)
    store.assert_not_called()
