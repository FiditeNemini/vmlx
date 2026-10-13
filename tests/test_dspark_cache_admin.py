"""DSpark administration addresses its SSD tier without changing generation."""
import asyncio
import os
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_dspark_pool_clear_checks_identity_and_drains_writer(tmp_path, monkeypatch):
    from vmlx_engine import server
    from vmlx_engine.dspark import session
    from vmlx_engine.utils.omni_session_disk_store import OmniSessionDiskStore

    store = OmniSessionDiskStore(root=tmp_path, model_key="dspark-test", max_size_bytes=65536, schema="test")
    lock = asyncio.Lock()
    drained = []
    def flush(timeout):
        assert lock.locked()
        drained.append(True)
        return True
    async def worker(fn):
        assert lock.locked()
        return fn()
    monkeypatch.setattr(server, "_engine", SimpleNamespace(_generation_lock=lock, _run_model_call=worker))
    monkeypatch.setattr(session, "SSD", SimpleNamespace(store=store, flush=flush))
    try:
        store.save("a" * 40, lambda path: path.write_bytes(b"owned-state"))
        (tmp_path / "NOTES.md").write_text("preserve")
        async with lock:
            with pytest.raises(server.HTTPException) as e:
                await server._clear_dspark_cache(pool=True, expected_root=str(tmp_path), expected_pid=os.getpid())
            assert e.value.status_code == 409
        with pytest.raises(server.HTTPException) as e:
            await server._clear_dspark_cache(pool=True, expected_root=str(tmp_path), expected_pid=os.getpid()+1)
        assert e.value.status_code == 409 and not drained
        assert store.paths("a" * 40)[0].exists()
        result = await server._clear_dspark_cache(pool=True, expected_root=str(tmp_path), expected_pid=os.getpid())
        assert result["freed_bytes"] > 0 and result["remaining_bytes"] == 0
        assert result["resident_cache_preserved"] and drained == [True]
        assert not store.paths("a" * 40)[0].exists()
        assert (tmp_path / "NOTES.md").read_text() == "preserve"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_dspark_clear_holds_admission_after_client_disconnect(monkeypatch):
    from vmlx_engine import server
    lock = asyncio.Lock()
    entered, release = asyncio.Event(), asyncio.Event()
    async def worker(fn):
        entered.set()
        await release.wait()
        return 0
    monkeypatch.setattr(server, "_engine", SimpleNamespace(_generation_lock=lock, _run_model_call=worker))
    task = asyncio.create_task(server._clear_dspark_cache(True))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert lock.locked() and not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not lock.locked()


@pytest.mark.asyncio
async def test_dspark_ram_clear_does_not_clear_ssd(monkeypatch):
    from vmlx_engine import server
    from vmlx_engine.dspark import session
    calls = []
    async def worker(fn):
        return fn()
    monkeypatch.setattr(server, "_engine", SimpleNamespace(_generation_lock=asyncio.Lock(), _run_model_call=worker))
    monkeypatch.setattr(session, "clear", lambda include_ssd: calls.append(include_ssd))
    await server._clear_dspark_cache(False)
    assert calls == [False]


@pytest.mark.asyncio
async def test_dspark_pool_clear_preserves_entries_when_writer_is_busy(tmp_path, monkeypatch):
    from vmlx_engine import server
    from vmlx_engine.dspark import session
    from vmlx_engine.utils.omni_session_disk_store import OmniSessionDiskStore

    store = OmniSessionDiskStore(root=tmp_path, model_key="busy", max_size_bytes=65536)
    async def worker(fn):
        return fn()
    monkeypatch.setattr(server, "_engine", SimpleNamespace(_generation_lock=asyncio.Lock(), _run_model_call=worker))
    monkeypatch.setattr(session, "SSD", SimpleNamespace(store=store, flush=lambda timeout: False))
    try:
        store.save("busy", lambda path: path.write_bytes(b"preserve pending publication"))
        with pytest.raises(server.HTTPException) as exc:
            await server._clear_dspark_cache(pool=True, expected_root=str(tmp_path), expected_pid=os.getpid())
        assert exc.value.status_code == 409
        assert store.load("busy", lambda path: path.read_bytes()) == b"preserve pending publication"
    finally:
        store.close()


def test_dspark_stats_report_existing_pool_and_pending_writes(tmp_path, monkeypatch):
    from vmlx_engine.dspark import session
    from vmlx_engine.utils.omni_session_disk_store import OmniSessionDiskStore

    store = OmniSessionDiskStore(root=tmp_path, model_key="stats", max_size_bytes=65536)
    monkeypatch.setattr(session, "STORE", SimpleNamespace(stats=lambda: {"entries": 0, "max_entries": 4}))
    monkeypatch.setattr(session, "SSD", SimpleNamespace(store=store, stats={"writes": 1}, _q=SimpleNamespace(unfinished_tasks=2)))
    try:
        store.save("stats", lambda path: path.write_bytes(b"state"))
        stats = session.cache_stats()
        assert stats["ram"]["entries"] == 0
        assert stats["ssd"]["pending_writes"] == 2
        assert stats["ssd"]["global_budget"]["root"] == str(tmp_path)
        assert stats["ssd"]["global_budget"]["bytes_after"] > 0
        monkeypatch.setattr(session, "SSD", None)
        assert session.cache_stats()["ssd"] is None
    finally:
        store.close()
