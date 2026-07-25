"""Regression tests for BatchedEngine loader/step executor ownership."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from vmlx_engine.engine.batched import BatchedEngine


class _FakeAsyncCore:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.engine = SimpleNamespace(close=self._close)

    async def stop(self) -> None:
        self._events.append("engine.stop")

    def _close(self) -> None:
        self._events.append("engine.close")


class _FailingAsyncCore(_FakeAsyncCore):
    async def stop(self) -> None:
        self._events.append("engine.stop")
        raise RuntimeError("stop failed")


class _RecordingExecutor:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.calls: list[tuple[bool, bool]] = []

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        self.calls.append((wait, cancel_futures))
        self._events.append("executor.shutdown")


def _bare_engine(*, executor, scheduler_config, async_core=None) -> BatchedEngine:
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._step_executor = executor
    engine._scheduler_config = scheduler_config
    engine._mllm_scheduler = None
    engine._engine = async_core
    engine._model = object()
    engine._tokenizer = object()
    engine._processor = object()
    engine._mllm_instance = object()
    engine._loaded = True
    return engine


def test_stop_shuts_executor_after_engine_and_clears_reusable_config() -> None:
    events: list[str] = []
    executor = _RecordingExecutor(events)
    scheduler_config = SimpleNamespace(step_executor=executor)
    engine = _bare_engine(
        executor=executor,
        scheduler_config=scheduler_config,
        async_core=_FakeAsyncCore(events),
    )

    asyncio.run(engine.stop())

    assert events == ["engine.stop", "engine.close", "executor.shutdown"]
    assert executor.calls == [(True, True)]
    assert scheduler_config.step_executor is None
    assert engine._step_executor is None
    assert engine._engine is None
    assert engine._model is None
    assert engine._loaded is False

    # Teardown is idempotent; a second stop must not re-shutdown the pool.
    asyncio.run(engine.stop())
    assert executor.calls == [(True, True)]


def test_stop_joins_real_loader_worker() -> None:
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-worker")
    assert executor.submit(lambda: "ready").result(timeout=2) == "ready"
    scheduler_config = SimpleNamespace(step_executor=executor)
    engine = _bare_engine(executor=executor, scheduler_config=scheduler_config)

    asyncio.run(engine.stop())

    assert executor._shutdown is True
    assert all(not thread.is_alive() for thread in executor._threads)
    assert scheduler_config.step_executor is None


def test_stop_failure_still_releases_loader_worker() -> None:
    events: list[str] = []
    executor = _RecordingExecutor(events)
    scheduler_config = SimpleNamespace(step_executor=executor)
    engine = _bare_engine(
        executor=executor,
        scheduler_config=scheduler_config,
        async_core=_FailingAsyncCore(events),
    )

    try:
        asyncio.run(engine.stop())
    except RuntimeError as exc:
        assert str(exc) == "stop failed"
    else:
        raise AssertionError("expected stop failure")

    assert events == ["engine.stop", "executor.shutdown"]
    assert scheduler_config.step_executor is None
    assert engine._engine is None
    assert engine._model is None
