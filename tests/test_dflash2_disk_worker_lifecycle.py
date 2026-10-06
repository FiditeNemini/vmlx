"""CPU-only ownership checks for the DFlash2 write-behind worker."""
import gc
import queue
import sys
import threading
import types
import weakref

import pytest

from vmlx_engine.dflash2_session_disk import DFlash2SessionSSD


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
    ssd._q.put(("3:digest", {"t0.0": payload}, {"kind": "turn"}))
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
    finally:
        stop.set()
        worker.join(5)
        assert not worker.is_alive()
