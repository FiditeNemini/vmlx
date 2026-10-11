"""Native SSD lifecycle checks without loading a model or creating MLX arrays."""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("mlx.core")
from vmlx_engine import glm_dflash2 as runtime
from vmlx_engine.engine.simple import SimpleEngine


def test_close_drains_queued_writes_and_terminates_worker(tmp_path, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    written = []

    def write(self, name, key, snapshot):
        entered.set()
        assert release.wait(3)
        written.append(name)

    monkeypatch.setattr(runtime.PrefixDiskTier, "_write", write)
    disk = runtime.PrefixDiskTier(tmp_path, 1024)
    disk._queue.put(("first", (1,), None))
    disk._queue.put(("second", (2,), None))
    assert entered.wait(1)
    closing = threading.Thread(target=disk.close)
    closing.start()
    try:
        closing.join(0.05)
        assert closing.is_alive(), "close returned while a snapshot was unwritten"
    finally:
        release.set()
        closing.join(3)
    assert not closing.is_alive()
    assert written == ["first", "second"]
    assert not disk._thread.is_alive()
    assert disk._queue.unfinished_tasks == 0
    disk.close()  # idempotent teardown


def test_simple_stop_drains_native_store_before_releasing_model(monkeypatch):
    events = []
    store = SimpleNamespace(disk=SimpleNamespace(close=lambda: events.append("disk")))
    target = SimpleNamespace(close=lambda: events.append("adapter"))
    monkeypatch.setattr(runtime, "_PREFIX_STORES", {1: store})
    monkeypatch.setattr(runtime, "_TARGETS", {2: target})

    async def run():
        with patch("vmlx_engine.engine.simple.is_mllm_model", return_value=True):
            engine = SimpleEngine("fake")
        engine._loaded = True
        engine._model = object()

        async def on_model_worker(fn):
            assert engine._model is not None
            fn()
        engine._run_model_call = on_model_worker
        await engine.stop()
        assert engine._model is None
        assert not engine._loaded

    asyncio.run(run())
    assert events == ["disk", "adapter"]
    assert runtime._PREFIX_STORES == {}
    assert runtime._TARGETS == {}
