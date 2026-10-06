"""CPU-only ownership checks for the DFlash2 write-behind worker."""
import gc
import queue
import sys
import threading
import types
import weakref
from concurrent.futures import Future

import pytest

from vmlx_engine.dflash2_session_disk import DFlash2SessionSSD


@pytest.mark.parametrize("settled", [False, True])
def test_clear_does_not_delete_while_writer_is_unsettled(settled):
    ssd = DFlash2SessionSSD.__new__(DFlash2SessionSSD)
    deleted = []
    ssd.store = types.SimpleNamespace(clear=lambda: deleted.append(True) or 3)
    ssd.flush = lambda timeout: settled
    if settled:
        assert ssd.clear() == 3
        assert deleted == [True]
    else:
        with pytest.raises(TimeoutError, match="pending SSD writes"):
            ssd.clear()
        assert not deleted


@pytest.mark.parametrize("save_fails", [False, True])
def test_idle_writer_releases_finished_snapshot(monkeypatch, tmp_path, save_fails):
    class Payload:
        pass

    idle = threading.Event()
    stop = threading.Event()

    class ControlledQueue(queue.Queue):
        def get(self, *args, **kwargs):
            if self.empty():
                idle.set()
                stop.wait(5)
                raise SystemExit
            return super().get(*args, **kwargs)

    core = types.ModuleType("mlx.core")
    core.save_safetensors = lambda path, arrays, metadata: None
    mlx = types.ModuleType("mlx")
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    path = tmp_path / "snapshot"
    path.write_bytes(b"snapshot")

    class Store:
        def save(self, signature, writer):
            if save_fails:
                raise OSError("write failed")
            writer(path)
            return path

    ssd = DFlash2SessionSSD.__new__(DFlash2SessionSSD)
    ssd.store = Store()
    ssd._q = ControlledQueue()
    ssd.stats = {"writes": 0, "write_bytes": 0, "write_s": 0.0, "write_errors": 0}
    payload = Payload()
    reference = weakref.ref(payload)
    receipt = Future()
    ssd._q.put(("3:digest", {"t0.0": payload}, {"kind": "turn", "cache_len": 2}, receipt))
    del payload
    worker = threading.Thread(target=ssd._run, daemon=True)
    worker.start()
    try:
        assert idle.wait(5), "writer did not settle"
        assert ssd._q.unfinished_tasks == 0
        gc.collect()
        assert reference() is None, "idle writer retained a completed snapshot"
        assert ssd.stats["write_errors"] == int(save_fails)
        assert ssd.stats["writes"] == int(not save_fails)
        assert receipt.result()["durable"] is (not save_fails)
        assert receipt.result()["outcome"] == ("failed" if save_fails else "stored")
    finally:
        stop.set()
        worker.join(5)
        assert not worker.is_alive()


def test_turn_has_reserved_queue_capacity_and_refusals_are_explicit(monkeypatch):
    core = types.ModuleType("mlx.core")
    core.array = lambda tokens, dtype: tokens
    core.uint32 = object()
    core.eval = lambda arrays: None
    mlx = types.ModuleType("mlx")
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    ssd = DFlash2SessionSSD.__new__(DFlash2SessionSSD)
    ssd._q = queue.Queue(maxsize=ssd.queue_depth + 1)
    ssd.stats = {"dropped": 0}
    entry = {"tokens": [1, 2], "cache_len": 2, "target_cache": []}
    for kind in ("system", "boundary"):
        assert not ssd.put(dict(entry, kind=kind)).done()
    refused = ssd.put(dict(entry, kind="boundary"))
    assert refused.result()["outcome"] == "refused"
    turn = ssd.put(dict(entry, kind="turn"))
    assert not turn.done() and not turn.cancel()
    assert ssd._q.qsize() == 3
    assert ssd.stats["dropped"] == 1
