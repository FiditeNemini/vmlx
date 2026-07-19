"""Terminal streaming must not wait behind synchronous cache persistence."""

from __future__ import annotations

import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from types import MethodType, SimpleNamespace

import pytest

from vmlx_engine.engine_core import EngineCore
from vmlx_engine.mllm_scheduler import MLLMScheduler, MLLMSchedulerOutput
from vmlx_engine.request import RequestOutput
from vmlx_engine.scheduler import Scheduler, SchedulerOutput


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
