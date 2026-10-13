"""Qwen DFlash2 exposes the same guarded SSD-pool action as other engines."""
import asyncio
import os
from types import SimpleNamespace

import pytest


@pytest.fixture
def context(tmp_path, monkeypatch):
    from vmlx_engine import server, speculative, dflash2_runtime
    from vmlx_engine.utils.omni_session_disk_store import OmniSessionDiskStore

    store = OmniSessionDiskStore(root=tmp_path, model_key="qwen-clear-test",
                                max_size_bytes=65536, schema="test")
    lock = asyncio.Lock()
    drained = []

    def flush(timeout):
        assert lock.locked()
        drained.append(timeout)
        return True

    async def worker(fn):
        assert lock.locked()
        return fn()

    disk = SimpleNamespace(store=store, flush=flush)
    monkeypatch.setattr(server, "_engine", SimpleNamespace(_generation_lock=lock, _run_model_call=worker))
    monkeypatch.setattr(server, "_loaded_dspark_health_status", lambda _: None)
    monkeypatch.setattr(server, "_current_model_config", lambda: SimpleNamespace(family_name="qwen3_5"))
    monkeypatch.setattr(speculative, "is_dflash2_enabled", lambda: True)
    monkeypatch.setattr(dflash2_runtime, "_SESSION_SSD", disk)
    store.save("a" * 40, lambda path: path.write_bytes(b"owned-state"))
    (tmp_path / "NOTES.md").write_text("preserve")
    try:
        yield server, lock, disk, drained
    finally:
        store.close()


@pytest.mark.asyncio
async def test_qwen_pool_clear_routes_without_scheduler_and_preserves_unmanaged(context, tmp_path):
    server, lock, disk, drained = context
    assert server._get_scheduler() is None
    result = await server.clear_cache("ssd_pool", str(tmp_path), os.getpid())
    assert result["status"] == "eligible_cleared"
    assert result["freed_bytes"] > 0 and result["remaining_bytes"] == 0
    assert result["resident_cache_preserved"] and drained == [10.0]
    assert not disk.store.paths("a" * 40)[0].exists()
    assert (tmp_path / "NOTES.md").read_text() == "preserve"
    assert not lock.locked()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["busy", "pid", "root", "writer", "missing_disk"])
async def test_qwen_pool_clear_refuses_before_deletion(context, tmp_path, monkeypatch, failure):
    from vmlx_engine import dflash2_runtime
    server, lock, disk, drained = context
    pid, root = os.getpid(), str(tmp_path)
    if failure == "busy":
        await lock.acquire()
    elif failure == "pid":
        pid += 1
    elif failure == "root":
        root += "-wrong"
    elif failure == "writer":
        disk.flush = lambda timeout: False
    else:
        monkeypatch.setattr(dflash2_runtime, "_SESSION_SSD", None)
    try:
        with pytest.raises(server.HTTPException) as exc:
            await server.clear_cache("ssd_pool", root, pid)
        assert exc.value.status_code == 409
        assert disk.store.paths("a" * 40)[0].exists()
        assert not drained
    finally:
        if failure == "busy":
            lock.release()


@pytest.mark.asyncio
async def test_qwen_pool_clear_keeps_admission_until_cancelled_worker_finishes(context, tmp_path):
    server, lock, _, _ = context
    entered, release = asyncio.Event(), asyncio.Event()

    async def worker(fn):
        entered.set()
        await release.wait()
        return fn()

    server._engine._run_model_call = worker
    task = asyncio.create_task(server.clear_cache("ssd_pool", str(tmp_path), os.getpid()))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    await asyncio.sleep(0)
    assert lock.locked() and not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not lock.locked()
