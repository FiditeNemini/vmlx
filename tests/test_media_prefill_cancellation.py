# SPDX-License-Identifier: Apache-2.0
"""Stop is acknowledged before, not instead of, worker-safe cancellation."""

import asyncio
from collections import deque
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vmlx_engine.mllm_batch_generator import MLLMBatchRequest
from vmlx_engine.mllm_scheduler import MLLMRequest, MLLMScheduler
from vmlx_engine.request import RequestStatus


def _scheduler(monkeypatch, *, waiting=False):
    import vmlx_engine.mllm_scheduler as module

    request = MLLMRequest(request_id="cancelled", prompt="private prompt")
    request.cancel_event = threading.Event()
    request.status = RequestStatus.WAITING if waiting else RequestStatus.RUNNING
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler._queue_lock = threading.RLock()
    scheduler._batch_lock = threading.RLock()
    scheduler.requests = {request.request_id: request}
    scheduler.waiting = deque([request] if waiting else [])
    scheduler.running = {} if waiting else {request.request_id: request}
    scheduler.request_id_to_uid = {} if waiting else {request.request_id: 7}
    scheduler.uid_to_request_id = {} if waiting else {7: request.request_id}
    scheduler._pending_aborts = set()
    scheduler.output_queues = {request.request_id: asyncio.Queue()}
    scheduler.finished_req_ids = set()
    scheduler._terminal_cleanup_complete = asyncio.Event()
    scheduler._terminal_cleanup_complete.set()
    scheduler.batch_generator = SimpleNamespace(remove=Mock())
    scheduler._cleanup_aborted_paged_request = Mock()
    scheduler._cleanup_detokenizer = Mock()
    monkeypatch.setattr(module, "clear_mlx_memory_cache", Mock())
    return scheduler, request, module.clear_mlx_memory_cache


@pytest.mark.parametrize("kind", [MLLMRequest, MLLMBatchRequest])
def test_request_cancel_signal_is_not_shared_by_default(kind):
    args = {"uid": 1} if kind is MLLMBatchRequest else {}
    first = kind(request_id="first", prompt="one", **args)
    second = kind(request_id="second", prompt="two", **args)
    first.cancel_event.set()
    assert not second.cancel_event.is_set()


def test_admission_shares_exact_request_signal(monkeypatch):
    scheduler, request, _ = _scheduler(monkeypatch, waiting=True)
    scheduler.config = SimpleNamespace(max_num_seqs=1)
    scheduler._ensure_batch_generator = lambda params: None
    captured = []
    scheduler.batch_generator.insert = lambda rows: captured.extend(rows) or [7]
    scheduler._schedule_waiting()
    assert captured[0].cancel_event is request.cancel_event


def test_running_abort_signals_but_does_not_clear_worker_cache(monkeypatch):
    scheduler, request, clear = _scheduler(monkeypatch)
    assert scheduler.abort_request(request.request_id)
    assert request.cancel_event.is_set()
    assert scheduler._pending_aborts == {request.request_id}
    assert scheduler.has_requests(), "cancelled worker is not idle until removed"
    scheduler.batch_generator.remove.assert_not_called()
    scheduler._cleanup_aborted_paged_request.assert_not_called()
    clear.assert_not_called()
    assert scheduler.output_queues[request.request_id].get_nowait() is None
    assert not scheduler.abort_request(request.request_id)


def test_deferred_abort_lifecycle_ends_only_after_worker_removal(monkeypatch):
    from vmlx_engine.server import _live_mllm_request_lifecycle_snapshot

    scheduler, request, clear = _scheduler(monkeypatch)
    scheduler.abort_request(request.request_id)
    scheduler.output_queues.clear()  # HTTP collector has already exited.

    def remove(uids):
        assert uids == [7]
        lifecycle = _live_mllm_request_lifecycle_snapshot(scheduler)
        assert lifecycle["active_request_count"] == 1
        assert lifecycle["scheduler_running_requests"][0]["status"] == "CANCELLING"
        assert lifecycle["terminal_cleanup_pending"] is True
        assert scheduler.has_requests()
        scheduler._cleanup_aborted_paged_request.assert_not_called()

    scheduler.batch_generator.remove.side_effect = remove
    scheduler._process_pending_aborts()
    scheduler._cleanup_aborted_paged_request.assert_called_once_with(request.request_id)
    assert not scheduler.has_requests()
    assert not scheduler._pending_aborts
    assert not scheduler.requests
    assert not scheduler.request_id_to_uid
    assert not scheduler.uid_to_request_id
    lifecycle = _live_mllm_request_lifecycle_snapshot(scheduler)
    assert lifecycle["active_request_count"] == 0
    assert lifecycle["terminal_cleanup_pending"] is False
    clear.assert_not_called()


def test_failed_worker_removal_cannot_report_idle(monkeypatch):
    scheduler, request, _ = _scheduler(monkeypatch)
    scheduler.abort_request(request.request_id)
    scheduler.batch_generator.remove.side_effect = RuntimeError("remove failed")
    scheduler._process_pending_aborts()
    assert scheduler.has_requests()
    assert request.request_id in scheduler._pending_aborts
    assert scheduler.request_id_to_uid[request.request_id] == 7
    scheduler._cleanup_aborted_paged_request.assert_not_called()
    scheduler._schedule_waiting = Mock(side_effect=AssertionError("admission before abort removal"))
    scheduler.step()
    scheduler._schedule_waiting.assert_not_called()


@pytest.mark.parametrize("fail_first", [False, True])
def test_deferred_abort_retires_native_handoffs_only_after_removal(monkeypatch, fail_first):
    scheduler, request, _ = _scheduler(monkeypatch)
    snapshots = [object()]
    names = ("_clean_boundary_snapshots", "_mixed_swa_boundary_snapshots")
    for name in names:
        setattr(scheduler.batch_generator, name, {request.request_id: snapshots, "sibling": snapshots})

    def remove(uids):
        assert uids == [7]
        for name in names:
            assert getattr(scheduler.batch_generator, name)[request.request_id] is snapshots
        if fail_first:
            raise RuntimeError("worker still owns the request")

    scheduler.batch_generator.remove.side_effect = remove
    scheduler.abort_request(request.request_id)
    scheduler._process_pending_aborts()
    if fail_first:
        assert request.request_id in scheduler._pending_aborts
        for name in names:
            assert request.request_id in getattr(scheduler.batch_generator, name)
        fail_first = False
        scheduler._process_pending_aborts()
    for name in names:
        assert getattr(scheduler.batch_generator, name) == {"sibling": snapshots}
    assert not scheduler._pending_aborts


def test_waiting_abort_needs_no_deferred_worker_removal(monkeypatch):
    scheduler, request, _ = _scheduler(monkeypatch, waiting=True)
    assert scheduler.abort_request(request.request_id)
    assert request.cancel_event.is_set()
    assert not scheduler._pending_aborts
    assert not scheduler.has_requests()
    scheduler._cleanup_aborted_paged_request.assert_called_once_with(request.request_id)
    assert request.request_id not in scheduler.requests


@pytest.mark.parametrize("cancel_phase", ["queued", "preprocess", "forward"])
def test_cancelled_prefill_is_not_an_error_or_a_sibling_retry(monkeypatch, cancel_phase):
    import vmlx_engine.mllm_batch_generator as module
    monkeypatch.setattr(module.MLLMBatchGenerator, "_stream", module.mx.default_stream(module.mx.gpu))

    class LM:
        layers = []

        def __call__(self, ids, cache=None):
            return module.mx.zeros((1, 8, 4))

        def make_cache(self):
            return []

    generator = module.MLLMBatchGenerator.__new__(module.MLLMBatchGenerator)
    generator.language_model = generator._cache_model = LM()
    generator._stats = SimpleNamespace(prompt_tokens=0, prompt_time=0)
    generator._is_hybrid = generator._ssm_companion_enabled = False
    generator._prefix_cache_enabled = generator._decode_trace = False
    generator.block_aware_cache = generator.memory_aware_cache = None
    generator.prefix_cache = generator.disk_cache = None
    generator._hybrid_kv_positions = []
    generator._prefill_errors = []
    generator._drain_tight_memory_allocator = lambda *args: None
    generator._media_scoped_cache_extra_keys = lambda *args: {}
    generator._request_has_media_cache_context = lambda *args: False
    generator._media_prefix_cache_allowed = lambda *args: False
    generator._prepare_native_mtp_prompt_priming = lambda *args: None
    generator._seed_native_mtp_from_prefill = lambda *args: None
    generator._make_request_sampler = lambda req: lambda logits: module.mx.array([2])
    cancelled = MLLMBatchRequest(uid=1, request_id="cancelled", prompt="one")
    sibling = MLLMBatchRequest(uid=2, request_id="sibling", prompt="two")
    forwards = []
    preprocessed = []

    def preprocess(req):
        preprocessed.append(req.request_id)
        req.input_ids = module.mx.arange(8)[None, :]
        if req is cancelled and cancel_phase == "preprocess":
            req.cancel_event.set()

    def forward(req, cache):
        forwards.append(req.request_id)
        if req is cancelled:
            req.cancel_event.set()
            module._raise_if_prefill_cancelled(req)
        return generator.language_model(req.input_ids, cache=cache)

    generator._preprocess_request = preprocess
    generator._run_vision_encoding = forward
    if cancel_phase == "queued":
        cancelled.cancel_event.set()
    batch = generator._process_prompts([cancelled, sibling])
    assert generator._prefill_errors == []
    assert batch.request_ids == ["sibling"]
    assert batch.uids == [2]
    assert batch.y.tolist() == [2]
    assert batch.cache == []
    assert forwards == (["cancelled", "sibling"] if cancel_phase == "forward" else ["sibling"])
    assert preprocessed == (["sibling"] if cancel_phase == "queued" else ["cancelled", "sibling"])
    assert cancelled.prompt_cache is None


def test_busy_health_keeps_cancelled_worker_visible_without_heavy_stats(monkeypatch):
    from vmlx_engine import server

    scheduler, request, _ = _scheduler(monkeypatch)
    scheduler.abort_request(request.request_id)
    scheduler.output_queues.clear()
    engine = SimpleNamespace(get_stats=Mock(side_effect=AssertionError("heavy stats during cancellation")))
    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setattr(server, "_get_scheduler", lambda: scheduler)
    monkeypatch.setattr(server, "_standby_state", None)
    monkeypatch.setattr(server, "_health_snapshot_cache", {"result": {"scheduler": {}}})
    result = asyncio.run(server.health())
    assert result["health_gauges_cached"] is True
    assert result["scheduler"]["num_running"] == 1
    assert result["request_lifecycle"]["active_request_count"] == 1
    assert result["request_lifecycle"]["terminal_cleanup_pending"] is True
    engine.get_stats.assert_not_called()
