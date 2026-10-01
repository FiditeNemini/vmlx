"""Exercise the production writer loop's ownership without MLX or model loading."""
import ast
import gc
import logging
from pathlib import Path
import queue
import threading
from types import SimpleNamespace
import weakref

import pytest


def writer_loop():
    source = Path(__file__).parents[1] / 'vmlx_engine/utils/ssm_companion_disk_store.py'
    tree = ast.parse(source.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SSMCompanionDiskStore')
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == '_background_writer')
    namespace = {'queue': queue, 'logger': logging.getLogger(__name__)}
    # Execute the unchanged production loop, isolating only serializer/IO dependencies.
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['_background_writer']


@pytest.mark.parametrize('fail_publish', [False, True])
def test_settled_writer_releases_payload_while_thread_stays_alive(fail_publish):
    class Payload(bytearray):
        pass

    condition = threading.Condition()
    disk = SimpleNamespace(
        _stop_event=threading.Event(), _write_queue=queue.Queue(),
        _idle_maintenance=None, _stats_lock=threading.Lock(),
        _write_condition=condition, _write_inflight=0, _pending_write_jobs=1,
        _latest_write_by_key={}, _last_completed_write=0, _stores=0,
        _write_failures=0, _global_budget=object(), _pending_write_bytes=2048,
    )
    observed = []

    def publish(job, key, data, sidecar):
        assert len(data) == len(sidecar) == 1024
        assert disk._pending_write_bytes == 2048
        if fail_publish:
            raise OSError('injected publication failure')
        return True

    def release(amount):
        assert all(ref() is None for ref in refs), "reservation released before payload"
        disk._pending_write_bytes -= amount

    def completed(job, ok):
        observed.append((job, ok))
        disk._last_completed_write = job

    disk._publish_entry = publish
    disk._release_pending_bytes = release
    disk._record_write_result_locked = completed
    data, sidecar = Payload(1024), Payload(1024)
    refs = weakref.ref(data), weakref.ref(sidecar)
    disk._write_queue.put((1, 'key', data, sidecar, 8, 2048))
    del data, sidecar
    worker = threading.Thread(target=writer_loop(), args=(disk,))
    worker.start()
    try:
        with condition:
            assert condition.wait_for(lambda: disk._pending_write_jobs == 0, timeout=2)
        gc.collect()
        assert worker.is_alive()
        assert disk._pending_write_bytes == disk._write_inflight == 0
        assert observed == [(1, not fail_publish)]
        assert all(ref() is None for ref in refs), 'idle writer retains settled payload'
    finally:
        disk._stop_event.set()
        worker.join(timeout=2)
        assert not worker.is_alive()
