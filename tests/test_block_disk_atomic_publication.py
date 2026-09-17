"""New block index+fence pins must commit together and fail closed."""
import hashlib
import sqlite3
import threading
import time

import mlx.core as mx
import pytest

from vmlx_engine.block_disk_store import BlockDiskStore


def _data(value=2):
    result = [("kv", mx.full((1, 1, 8, 16), value, mx.float16),
               mx.full((1, 1, 8, 16), value + 1, mx.float16))]
    mx.eval(result)
    return result


def _hash(value):
    return hashlib.sha256(value.encode()).digest()


def _fence(store, fence_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        pipeline = store.get_stats()["write_pipeline"]
        row = next(x for x in pipeline["recent_fences"] if x["fence_id"] == fence_id)
        if row["post_eviction_complete"] and not pipeline["inflight"] and not pipeline["queue_depth"]:
            return row
        time.sleep(0.005)
    raise AssertionError("fence did not settle")


def _assert_restores(store, block_hash, expected):
    got = store.read_block(block_hash)
    assert got and got[0][0] == "kv"
    for a, b in zip(got[0][1:], expected[0][1:]):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert mx.array_equal(a, b).item()


def test_new_row_and_pin_are_visible_in_one_commit(tmp_path, monkeypatch):
    target = _hash("atomic")
    real_connect = sqlite3.connect
    observations = []
    published = threading.Event()

    class Connection(sqlite3.Connection):
        pending = False

        def execute(self, sql, parameters=(), /):
            result = super().execute(sql, parameters)
            if sql.lstrip().startswith("INSERT OR IGNORE INTO blocks") and parameters[0] == target.hex():
                self.pending = True
            return result

        def commit(self):
            if self.pending:
                path = self.execute("PRAGMA database_list").fetchone()[2]
                with real_connect(path) as peer:
                    before = tuple(peer.execute(sql, (target.hex(),)).fetchone()[0] for sql in (
                        "SELECT count(*) FROM blocks WHERE block_hash = ?",
                        "SELECT count(*) FROM block_write_pins WHERE block_hash = ?"))
                super().commit()
                with real_connect(path) as peer:
                    after = tuple(peer.execute(sql, (target.hex(),)).fetchone()[0] for sql in (
                        "SELECT count(*) FROM blocks WHERE block_hash = ?",
                        "SELECT count(*) FROM block_write_pins WHERE block_hash = ?"))
                observations.append((before, after))
                self.pending = False
                published.set()
            else:
                super().commit()

    def connect(*args, **kwargs):
        return real_connect(*args, factory=Connection, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    try:
        fence = store.begin_write_fence("atomic")
        data = _data()
        assert store.write_block_async(target, data, 8, request_id="atomic", fence_id=fence)
        assert published.wait(5)
        assert observations == [((0, 0), (1, 1))]
        assert store.seal_write_fence(fence)
        final = _fence(store, fence)
        assert final["completed"] == final["retained"] == 1
        assert not final["failed"]
        _assert_restores(store, target, data)
    finally:
        store.shutdown()


@pytest.mark.parametrize("failure", [
    "pin_insert", "before_commit", "after_commit", "rollback",
    "after_commit_rollback", "reopen", "recovery_exhausted",
])
def test_new_publication_failure_cannot_leak_into_later_commit(tmp_path, monkeypatch, failure):
    root, child, good = map(_hash, ("failed", "child", "good"))
    real_connect = sqlite3.connect
    injected = {"commit": False, "rollback": False, "reopen": False}
    permit_recovery = threading.Event()
    start = threading.Event()
    original_writer = BlockDiskStore._background_writer

    def delayed_writer(self):
        assert start.wait(5)
        original_writer(self)

    class Connection(sqlite3.Connection):
        pending = False

        def execute(self, sql, parameters=(), /):
            if (failure == "pin_insert" and not injected["commit"]
                    and sql.lstrip().startswith("INSERT OR IGNORE INTO block_write_pins")
                    and parameters[0] == root.hex()):
                injected["commit"] = True
                raise sqlite3.OperationalError("injected pin INSERT failure")
            result = super().execute(sql, parameters)
            if sql.lstrip().startswith("INSERT OR IGNORE INTO blocks") and parameters[0] == root.hex():
                self.pending = True
            return result

        def commit(self):
            if self.pending and failure != "pin_insert" and not injected["commit"]:
                injected["commit"] = True
                if failure in {"after_commit", "after_commit_rollback", "recovery_exhausted"}:
                    super().commit()
                    self.pending = False
                raise sqlite3.OperationalError("injected COMMIT failure")
            result = super().commit()
            self.pending = False
            return result

        def rollback(self):
            if (failure in {"rollback", "after_commit_rollback", "reopen", "recovery_exhausted"}
                    and injected["commit"] and not injected["rollback"]):
                injected["rollback"] = True
                raise sqlite3.OperationalError("injected rollback failure")
            result = super().rollback()
            self.pending = False
            return result

    def connect(*args, **kwargs):
        if (failure == "recovery_exhausted" and injected["rollback"]
                and not permit_recovery.is_set()
                and threading.current_thread().name == "block-disk-writer"):
            raise sqlite3.OperationalError("injected recovery connections unavailable")
        if (failure == "reopen" and injected["rollback"] and not injected["reopen"]
                and threading.current_thread().name == "block-disk-writer"):
            injected["reopen"] = True
            raise sqlite3.OperationalError("injected one-shot reconnect failure")
        return real_connect(*args, factory=Connection, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)
    monkeypatch.setattr(BlockDiskStore, "_background_writer", delayed_writer)
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    data = _data()
    try:
        fence = store.begin_write_fence("failed-chain")
        for block_hash, parent in ((root, None), (child, root)):
            assert store.write_block_async(block_hash, data, 8, parent_hash=parent,
                                           request_id="failed-chain", fence_id=fence)
        assert store.seal_write_fence(fence)
        start.set()
        final = _fence(store, fence)
        assert injected["commit"]
        assert final["failed"] == 2 and final["completed"] == final["retained"] == 0
        assert store.read_block(root) is None and store.read_block(child) is None
        if failure == "recovery_exhausted":
            assert store._pending_publication_recovery is not None
            assert not store._maybe_recover_global_budget_writes()
            assert not store._hash_to_path(root.hex()).exists()
            permit_recovery.set()
            deadline = time.monotonic() + 5
            while store._pending_publication_recovery is not None and time.monotonic() < deadline:
                time.sleep(0.005)
            assert store._pending_publication_recovery is None
        with real_connect(str(store._db_path)) as conn:
            assert conn.execute("SELECT count(*) FROM blocks").fetchone()[0] == 0
            assert conn.execute("SELECT count(*) FROM block_write_pins").fetchone()[0] == 0
        if failure in {"rollback", "after_commit_rollback", "reopen", "recovery_exhausted"}:
            assert injected["rollback"] and not store._global_budget_write_enabled
            if failure == "reopen":
                assert injected["reopen"]
            store._budget_recovery_interval_ns = 0
            assert store._maybe_recover_global_budget_writes()
        healthy = store.begin_write_fence("healthy")
        assert store.write_block_async(good, data, 8, request_id="healthy", fence_id=healthy)
        assert store.seal_write_fence(healthy)
        finished = _fence(store, healthy)
        assert finished["completed"] == finished["retained"] == 1 and not finished["failed"]
        _assert_restores(store, good, data)
        assert store.read_block(root) is None and store.read_block(child) is None
        assert store.get_stats()["disk_writes"] == 1
        assert store._writer_thread.is_alive()
    finally:
        permit_recovery.set()
        start.set()
        store.shutdown()


def test_poisoned_later_batch_releases_already_committed_fence_pins(tmp_path, monkeypatch):
    root, child = map(_hash, ("pinned-root", "poisoned-child"))
    real_connect = sqlite3.connect
    injected = {"commit": False, "rollback": False}

    class Connection(sqlite3.Connection):
        pending = False

        def execute(self, sql, parameters=(), /):
            result = super().execute(sql, parameters)
            if sql.lstrip().startswith("INSERT OR IGNORE INTO blocks") and parameters[0] == child.hex():
                self.pending = True
            return result

        def commit(self):
            if self.pending and not injected["commit"]:
                injected["commit"] = True
                super().commit()
                raise sqlite3.OperationalError("injected ambiguous COMMIT")
            result = super().commit()
            self.pending = False
            return result

        def rollback(self):
            if self.pending and not injected["rollback"]:
                injected["rollback"] = True
                raise sqlite3.OperationalError("injected rollback failure")
            return super().rollback()

    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: real_connect(*a, factory=Connection, **k))
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    try:
        fence = store.begin_write_fence("split-failure")
        assert store.write_block_async(root, _data(), 8, request_id="split-failure", fence_id=fence)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with real_connect(str(store._db_path)) as conn:
                pins = conn.execute("SELECT count(*) FROM block_write_pins").fetchone()[0]
            if pins == 1:
                break
            time.sleep(0.005)
        assert pins == 1
        assert store.write_block_async(child, _data(), 8, parent_hash=root,
                                      request_id="split-failure", fence_id=fence)
        assert store.seal_write_fence(fence)
        final = _fence(store, fence)
        assert final["failed"] == 1 and final["completed"] == 1
        assert final.get("post_eviction_error")
        with real_connect(str(store._db_path)) as conn:
            assert conn.execute("SELECT count(*) FROM block_write_pins").fetchone()[0] == 0
            assert conn.execute("SELECT count(*) FROM blocks WHERE block_hash=?", (child.hex(),)).fetchone()[0] == 0
        assert not store._hash_to_path(child.hex()).exists()
        assert store._writer_thread.is_alive()
    finally:
        store.shutdown()
