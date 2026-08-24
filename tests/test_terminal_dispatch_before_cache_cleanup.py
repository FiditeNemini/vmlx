"""Terminal streaming must not wait behind synchronous cache persistence."""

from __future__ import annotations

import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from types import MethodType, SimpleNamespace

import pytest

from vmlx_engine.engine_core import EngineCore
from vmlx_engine.mllm_scheduler import MLLMScheduler, MLLMSchedulerOutput
from vmlx_engine.request import RequestOutput
from vmlx_engine.scheduler import Scheduler, SchedulerOutput


def test_engine_loop_failure_uses_structured_error_not_assistant_text() -> None:
    engine = EngineCore.__new__(EngineCore)
    seen: list[RequestOutput] = []

    class _Collector:
        def put(self, output: RequestOutput) -> None:
            seen.append(output)

    class _Scheduler:
        running = {"req": object()}
        waiting = []

        def abort_request(self, request_id: str) -> None:
            assert request_id == "req"

    done = asyncio.Event()
    engine.scheduler = _Scheduler()
    engine._output_collectors = {"req": _Collector()}
    engine._finished_events = {"req": done}

    engine._fail_active_requests("[full] Negative dimensions not allowed.")

    assert done.is_set()
    assert len(seen) == 1
    output = seen[0]
    assert output.finished is True
    assert output.finish_reason == "error"
    assert output.error == "Engine loop error: [full] Negative dimensions not allowed."
    assert output.error_code == "engine_loop_error"
    assert output.error_source == "engine_loop"
    assert output.new_text == ""
    assert "[Engine error:" not in output.output_text


@pytest.mark.asyncio
async def test_llm_engine_dispatches_terminal_before_deferred_cleanup() -> None:
    order: list[str] = []
    engine = EngineCore.__new__(EngineCore)

    class _Collector:
        def put(self, output: RequestOutput) -> None:
            assert output.finished
            assert not engine._terminal_cleanup_complete.is_set()
            order.append("dispatch")

    class _Scheduler:
        _step_executor = None
        running = {}

        def has_requests(self) -> bool:
            return engine._running

        def step(self, *, defer_finished_cleanup: bool = False) -> SchedulerOutput:
            assert defer_finished_cleanup is True
            return SchedulerOutput(
                outputs=[RequestOutput(request_id="req", finished=True)],
                finished_request_ids={"req"},
            )

        def _cleanup_finished(self, finished_ids: set[str]) -> None:
            assert finished_ids == {"req"}
            assert not engine._terminal_cleanup_complete.is_set()
            order.append("cleanup")
            engine._running = False

    engine.scheduler = _Scheduler()
    engine.config = SimpleNamespace(step_interval=0.0, stream_interval=1)
    engine._running = True
    engine._steps_executed = 0
    engine._output_collectors = {"req": _Collector()}
    engine._stream_states = {}
    engine._finished_events = {}
    engine._terminal_cleanup_complete = asyncio.Event()
    engine._terminal_cleanup_complete.set()

    await engine._engine_loop()

    assert order == ["dispatch", "cleanup"]
    assert engine._terminal_cleanup_complete.is_set()


@pytest.mark.asyncio
async def test_mllm_loop_dispatches_terminal_before_worker_cleanup() -> None:
    order: list[str] = []
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler._running = True
    scheduler._step_executor = ThreadPoolExecutor(max_workers=1)
    scheduler._terminal_cleanup_complete = asyncio.Event()
    scheduler._terminal_cleanup_complete.set()

    def _has_requests(self) -> bool:
        return self._running

    def _step(self, *, defer_finished_cleanup: bool = False) -> MLLMSchedulerOutput:
        assert defer_finished_cleanup is True
        return MLLMSchedulerOutput(
            outputs=[RequestOutput(request_id="req", finished=True)],
            finished_request_ids={"req"},
        )

    def _dispatch(self, output: MLLMSchedulerOutput) -> None:
        assert output.outputs[0].finished
        assert not self._terminal_cleanup_complete.is_set()
        order.append("dispatch")

    def _cleanup(self, finished_ids: set[str]) -> None:
        assert finished_ids == {"req"}
        assert not self._terminal_cleanup_complete.is_set()
        order.append("cleanup")
        self._running = False

    scheduler.has_requests = MethodType(_has_requests, scheduler)
    scheduler.step = MethodType(_step, scheduler)
    scheduler._dispatch_outputs = MethodType(_dispatch, scheduler)
    scheduler._cleanup_finished_after_terminal_dispatch = MethodType(
        _cleanup, scheduler
    )

    try:
        await scheduler._process_loop()
    finally:
        scheduler._step_executor.shutdown(wait=True)

    assert order == ["dispatch", "cleanup"]
    assert scheduler._terminal_cleanup_complete.is_set()


def test_mllm_deferred_cleanup_clears_allocator_after_cleanup_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final clear must run after cache-owning cleanup locals are gone."""
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.running = {}
    scheduler._queue_lock = Lock()
    order: list[str] = []

    def _cleanup(self, finished_ids: set[str]) -> None:
        assert finished_ids == {"req"}
        order.append("cleanup-returned")

    scheduler._cleanup_finished = MethodType(_cleanup, scheduler)
    monkeypatch.setattr(
        "vmlx_engine.mllm_scheduler.clear_mlx_memory_cache",
        lambda **_kwargs: order.append("allocator-cleared"),
    )

    scheduler._cleanup_finished_after_terminal_dispatch({"req"})

    assert order == ["cleanup-returned", "allocator-cleared"]


@pytest.mark.asyncio
async def test_llm_request_admission_waits_for_terminal_cache_cleanup() -> None:
    added: list[str] = []
    engine = EngineCore.__new__(EngineCore)

    class _Scheduler:
        def add_request(self, request) -> None:
            added.append(request.request_id)

    engine.scheduler = _Scheduler()
    engine.config = SimpleNamespace(stream_interval=1)
    engine._output_collectors = {}
    engine._stream_states = {}
    engine._finished_events = {}
    engine._terminal_cleanup_complete = asyncio.Event()

    pending = asyncio.create_task(
        engine.add_request(prompt="next turn", request_id="next")
    )
    await asyncio.sleep(0)
    assert not pending.done()
    assert added == []

    engine._terminal_cleanup_complete.set()
    assert await pending == "next"
    assert added == ["next"]


@pytest.mark.asyncio
async def test_mllm_async_admission_waits_for_terminal_cache_cleanup() -> None:
    added: list[str] = []
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler._terminal_cleanup_complete = asyncio.Event()
    scheduler.output_queues = {}

    def _add_request(self, **kwargs) -> str:
        added.append(kwargs["prompt"])
        return "next-vl"

    scheduler.add_request = MethodType(_add_request, scheduler)
    pending = asyncio.create_task(scheduler.add_request_async(prompt="next media turn"))
    await asyncio.sleep(0)
    assert not pending.done()
    assert added == []

    scheduler._terminal_cleanup_complete.set()
    assert await pending == "next-vl"
    assert added == ["next media turn"]


@pytest.mark.asyncio
async def test_llm_stop_waits_for_terminal_cache_cleanup_before_cancelling() -> None:
    order: list[str] = []
    engine = EngineCore.__new__(EngineCore)

    class _Scheduler:
        def shutdown(self) -> None:
            order.append("shutdown")

    async def _loop() -> None:
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            order.append("cancel")
            raise

    engine.scheduler = _Scheduler()
    engine._running = True
    engine._terminal_cleanup_complete = asyncio.Event()
    engine._task = asyncio.create_task(_loop())
    await asyncio.sleep(0)

    stopping = asyncio.create_task(engine.stop())
    await asyncio.sleep(0)
    assert not stopping.done()
    assert order == []

    engine._terminal_cleanup_complete.set()
    await stopping

    assert order == ["cancel", "shutdown"]
    assert engine._task is None


@pytest.mark.asyncio
async def test_mllm_stop_waits_for_terminal_cache_cleanup_before_cancelling() -> None:
    order: list[str] = []
    scheduler = MLLMScheduler.__new__(MLLMScheduler)

    async def _loop() -> None:
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            order.append("cancel")
            raise

    scheduler._running = True
    scheduler._terminal_cleanup_complete = asyncio.Event()
    scheduler._processing_task = asyncio.create_task(_loop())
    scheduler.batch_generator = None
    await asyncio.sleep(0)

    stopping = asyncio.create_task(scheduler.stop())
    await asyncio.sleep(0)
    assert not stopping.done()
    assert order == []

    scheduler._terminal_cleanup_complete.set()
    await stopping

    assert order == ["cancel"]
    assert scheduler._running is False


def test_direct_scheduler_steps_retain_synchronous_cleanup_default() -> None:
    llm_source = inspect.getsource(Scheduler.step)
    mllm_source = inspect.getsource(MLLMScheduler.step)

    assert "defer_finished_cleanup: bool = False" in llm_source
    assert "if not defer_finished_cleanup" in llm_source
    assert "defer_finished_cleanup: bool = False" in mllm_source
    assert "if not defer_finished_cleanup" in mllm_source


def test_finished_stream_consumers_do_not_abort_deferred_cleanup() -> None:
    llm_source = inspect.getsource(EngineCore._cleanup_request)
    mllm_source = inspect.getsource(MLLMScheduler.stream_outputs)

    assert "RequestStatus.is_finished(request.status)" in llm_source
    assert "RequestStatus.is_finished(request.status)" in mllm_source


def test_m3_hit_rederive_materializes_in_post_dispatch_cleanup_shape() -> None:
    scheduler = Scheduler.__new__(Scheduler)
    order: list[str] = []

    def _prefill(tokens):
        order.append("rederive")
        assert tokens == [11, 12, 13]
        return ["native-m3-cache"]

    def _extract(cache):
        order.append("extract")
        assert cache == ["native-m3-cache"]
        return [{"class_name": "MiniMaxM3SparseCache", "state": "typed"}]

    scheduler._prefill_for_prompt_only_cache = _prefill
    scheduler._extract_cache_states = _extract
    scheduler._kv_cache_bits = 0
    scheduler.disk_cache = None
    scheduler._is_hybrid = False
    request = SimpleNamespace(
        prompt_token_ids=[11, 12, 13, 14],
        _deferred_prompt_cache={
            "family": "MiniMax-M3",
            "mode": "paged",
            "key_tokens": [11, 12, 13],
        },
    )

    # Terminal delivery happens in EngineCore before this cleanup helper is
    # called; the descriptor itself must not do any eager work.
    order.append("terminal-dispatch")
    scheduler._materialize_deferred_prompt_cache("m3", request)

    assert order == ["terminal-dispatch", "rederive", "extract"]
    assert request._deferred_prompt_cache is None
    assert request._extracted_cache_key_tokens == [11, 12, 13]
    assert request._extracted_cache_from_prompt_snapshot is True
    assert request._extracted_cache == [
        {"class_name": "MiniMaxM3SparseCache", "state": "typed"}
    ]


def test_m3_response_finalization_only_schedules_clean_rederive() -> None:
    source = inspect.getsource(Scheduler._process_batch_responses)
    paged_marker = '"mode": "paged"'
    object_marker = '"mode": "object"'

    assert source.count("request._deferred_prompt_cache = {") == 6
    assert paged_marker in source
    assert object_marker in source

    # Both MiniMax-M3 hit branches now leave the expensive prefill for
    # _cleanup_finished. Other architecture-specific rederive branches are
    # audited separately and may still call this helper in finalization.
    for marker in (paged_marker, object_marker):
        branch = source[source.index(marker) - 900 : source.index(marker) + 250]
        assert "_prefill_for_prompt_only_cache" not in branch


def test_all_llm_terminal_response_paths_defer_clean_prompt_reprefill() -> None:
    source = inspect.getsource(Scheduler._process_batch_responses)

    assert "_prefill_for_prompt_only_cache" not in source
    for family in ("DSV4", "ZAYA", "Mixed-SWA", "MiniMax-M3"):
        assert f'"family": "{family}"' in source
    assert source.count('"mode": "paged"') == 4
    assert source.count('"mode": "object"') == 2
