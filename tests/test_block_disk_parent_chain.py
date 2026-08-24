"""Persistent block ancestry and chain-safe L2 eviction regressions."""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import mlx.core as mx

from vmlx_engine.block_disk_store import BlockDiskStore
from vmlx_engine.global_disk_cache_budget import (
    GlobalDiskCacheBudget,
    ensure_managed_block_cache_namespace,
)


def _cache_data(value: float) -> list[tuple]:
    keys = mx.full((1, 1, 8, 4096), value, dtype=mx.float16)
    values = mx.full((1, 1, 8, 4096), value + 1, dtype=mx.float16)
    mx.eval(keys, values)
    return [("kv", keys, values)]


def _wait_for_fence(
    store: BlockDiskStore,
    fence_id: str,
    *,
    timeout: float = 10.0,
) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pipeline = store.get_stats()["write_pipeline"]
        fence = next(
            (
                item
                for item in pipeline["recent_fences"]
                if item["fence_id"] == fence_id
            ),
            None,
        )
        if (
            fence is not None
            and fence["post_eviction_complete"]
            and pipeline["queue_depth"] == 0
            and pipeline["inflight"] == 0
        ):
            return fence
        time.sleep(0.01)
    raise AssertionError(f"write fence {fence_id} did not settle")


def _write_chain(
    store: BlockDiskStore,
    edges: list[tuple[bytes, bytes | None]],
    *,
    request_id: str,
) -> dict:
    fence_id = store.begin_write_fence(request_id)
    for index, (block_hash, parent_hash) in enumerate(edges):
        assert store.write_block_async(
            block_hash,
            _cache_data(float(index + 1)),
            8,
            parent_hash=parent_hash,
            request_id=request_id,
            fence_id=fence_id,
        )
    assert store.seal_write_fence(fence_id)
    return _wait_for_fence(store, fence_id)


def _rows(store: BlockDiskStore) -> list[tuple[str, str | None, int]]:
    with sqlite3.connect(str(store._db_path)) as connection:
        return [
            (str(block_hash), parent_hash, int(ancestry_known))
            for block_hash, parent_hash, ancestry_known in connection.execute(
                "SELECT block_hash, parent_hash, ancestry_known "
                "FROM blocks ORDER BY created_at"
            ).fetchall()
        ]


def _set_access_order(store: BlockDiskStore, hashes: list[bytes]) -> None:
    base = time.time() - 1000
    with sqlite3.connect(str(store._db_path)) as connection:
        for index, block_hash in enumerate(hashes):
            accessed = base + index
            row = connection.execute(
                "SELECT file_name FROM blocks WHERE block_hash = ?",
                (block_hash.hex(),),
            ).fetchone()
            assert row is not None
            path = store.cache_dir / str(row[0])
            os.utime(path, (accessed, accessed))
            connection.execute(
                "UPDATE blocks SET last_accessed = ? WHERE block_hash = ?",
                (accessed, block_hash.hex()),
            )
        connection.commit()


def _trim_below_current_size(store: BlockDiskStore) -> None:
    before = store.global_budget.enforce(force=True)
    assert before.bytes_after > 1
    cap = before.bytes_after - 1
    store.max_size_bytes = cap
    store.global_budget._requested_max_size_bytes = cap
    store.global_budget._publish_budget(cap)
    result = store.global_budget.enforce(force=True)
    assert result.compliant is True
    assert result.evicted_entries >= 1


def test_linear_chain_cap_evicts_suffix_and_preserves_causal_head(
    tmp_path: Path,
) -> None:
    hashes = [bytes([value]) * 32 for value in range(1, 5)]
    edges = [
        (hashes[0], None),
        (hashes[1], hashes[0]),
        (hashes[2], hashes[1]),
        (hashes[3], hashes[2]),
    ]
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    try:
        fence = _write_chain(store, edges, request_id="linear")
        assert fence["retained"] == len(edges)
        _set_access_order(store, hashes)
        _trim_below_current_size(store)

        remaining = {block_hash for block_hash, _, _ in _rows(store)}
        remaining_ordinals = [
            index for index, block_hash in enumerate(hashes) if block_hash.hex() in remaining
        ]
        assert 0 < len(remaining_ordinals) < len(hashes)
        assert remaining_ordinals == list(range(len(remaining_ordinals)))
    finally:
        store.shutdown()


def test_branching_chain_cap_never_orphans_shared_prefix(
    tmp_path: Path,
) -> None:
    root, left, left_tail, right, right_tail = [
        bytes([value]) * 32 for value in range(10, 15)
    ]
    edges = [
        (root, None),
        (left, root),
        (left_tail, left),
        (right, root),
        (right_tail, right),
    ]
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    try:
        _write_chain(store, edges, request_id="branch")
        _set_access_order(
            store,
            [root, left, right, left_tail, right_tail],
        )
        _trim_below_current_size(store)

        rows = _rows(store)
        remaining = {block_hash for block_hash, _, _ in rows}
        assert remaining
        for block_hash, parent_hash, ancestry_known in rows:
            assert ancestry_known == 1
            if parent_hash is not None:
                assert parent_hash in remaining, block_hash
        if len(remaining) > 1:
            assert root.hex() in remaining
    finally:
        store.shutdown()


def test_schema_migration_keeps_legacy_null_explicitly_unknown(
    tmp_path: Path,
) -> None:
    cache_dir = ensure_managed_block_cache_namespace(tmp_path)
    blocks = cache_dir / "blocks" / "aa"
    blocks.mkdir(parents=True)
    block_hash = "aa" * 32
    payload = blocks / f"{block_hash}.safetensors"
    payload.write_bytes(b"legacy")
    database = cache_dir / "block_index.db"
    with sqlite3.connect(str(database)) as connection:
        connection.execute(
            "CREATE TABLE blocks ("
            "block_hash TEXT PRIMARY KEY, file_name TEXT NOT NULL, "
            "num_tokens INTEGER NOT NULL, num_layers INTEGER NOT NULL, "
            "dtype TEXT NOT NULL, file_size INTEGER NOT NULL, "
            "created_at REAL NOT NULL, last_accessed REAL NOT NULL, "
            "access_count INTEGER DEFAULT 0)"
        )
        now = time.time()
        connection.execute(
            "INSERT INTO blocks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                block_hash,
                str(payload.relative_to(cache_dir)),
                8,
                1,
                "float16",
                payload.stat().st_size,
                now,
                now,
                0,
            ),
        )
        connection.commit()

    store = BlockDiskStore(str(cache_dir), max_size_gb=0)
    try:
        with sqlite3.connect(str(database)) as connection:
            columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(blocks)"
                ).fetchall()
            }
            row = connection.execute(
                "SELECT parent_hash, ancestry_known FROM blocks "
                "WHERE block_hash = ?",
                (block_hash,),
            ).fetchone()
        assert {"parent_hash", "ancestry_known"}.issubset(columns)
        assert row == (None, 0)
    finally:
        store.shutdown()


def test_cap_pressure_invalidates_complete_legacy_unknown_set(
    tmp_path: Path,
) -> None:
    root, child = [bytes([value]) * 32 for value in range(40, 42)]
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    try:
        _write_chain(
            store,
            [(root, None), (child, root)],
            request_id="legacy-pressure",
        )
        with sqlite3.connect(str(store._db_path)) as connection:
            connection.execute(
                "UPDATE blocks SET parent_hash = NULL, ancestry_known = 0"
            )
            connection.commit()

        _trim_below_current_size(store)

        assert _rows(store) == []
        assert list(store.blocks_dir.rglob("*.safetensors")) == []
    finally:
        store.shutdown()


def test_cap_pressure_invalidates_complete_broken_known_cycle(
    tmp_path: Path,
) -> None:
    root, child = [bytes([value]) * 32 for value in range(50, 52)]
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    try:
        _write_chain(
            store,
            [(root, None), (child, root)],
            request_id="cycle-pressure",
        )
        with sqlite3.connect(str(store._db_path)) as connection:
            connection.execute(
                "UPDATE blocks SET parent_hash = ? WHERE block_hash = ?",
                (child.hex(), root.hex()),
            )
            connection.commit()

        _trim_below_current_size(store)

        assert _rows(store) == []
        assert list(store.blocks_dir.rglob("*.safetensors")) == []
    finally:
        store.shutdown()


def test_ordered_write_upgrades_legacy_rows_root_to_tail(
    tmp_path: Path,
) -> None:
    root, child = [bytes([value]) * 32 for value in range(60, 62)]
    edges = [(root, None), (child, root)]
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    try:
        _write_chain(store, edges, request_id="legacy-seed")
        with sqlite3.connect(str(store._db_path)) as connection:
            connection.execute(
                "UPDATE blocks SET parent_hash = NULL, ancestry_known = 0"
            )
            connection.commit()

        fence = _write_chain(store, edges, request_id="legacy-upgrade")

        assert fence["failed"] == 0
        assert fence["retained"] == len(edges)
        assert _rows(store) == [
            (root.hex(), None, 1),
            (child.hex(), root.hex(), 1),
        ]
    finally:
        store.shutdown()


def test_failed_parent_publication_rejects_child_and_rolls_back_batch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = b"r" * 32
    child = b"c" * 32
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    original_write = store._write_payload_file
    attempts = 0

    def fail_first(path: Path, payload: bytes | Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("forced root publication failure")
        original_write(path, payload)

    monkeypatch.setattr(store, "_write_payload_file", fail_first)
    try:
        fence = _write_chain(
            store,
            [(root, None), (child, root)],
            request_id="failed-parent",
        )
        assert fence["failed"] == 2
        assert fence["retained"] == 0
        assert _rows(store) == []
        assert list(store.blocks_dir.rglob("*.safetensors")) == []
    finally:
        store.shutdown()


def test_stale_parent_row_cannot_authorize_new_child(
    tmp_path: Path,
) -> None:
    root, existing_child, new_child = [
        bytes([value]) * 32 for value in range(20, 23)
    ]
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    try:
        _write_chain(
            store,
            [(root, None), (existing_child, root)],
            request_id="stale-seed",
        )
        with sqlite3.connect(str(store._db_path)) as connection:
            row = connection.execute(
                "SELECT file_name FROM blocks WHERE block_hash = ?",
                (root.hex(),),
            ).fetchone()
        assert row is not None
        (store.cache_dir / str(row[0])).unlink()

        fence = _write_chain(
            store,
            [(new_child, root)],
            request_id="stale-child",
        )
        assert fence["failed"] == 1
        assert fence["retained"] == 0
        # Cleaning the unreadable parent also invalidates every old branch
        # below it; no indexed suffix survives.
        assert _rows(store) == []
    finally:
        store.shutdown()


def test_cleanup_of_shared_root_cascades_all_descendants(
    tmp_path: Path,
) -> None:
    root, left, right = [bytes([value]) * 32 for value in range(30, 33)]
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    try:
        _write_chain(
            store,
            [(root, None), (left, root), (right, root)],
            request_id="cleanup",
        )
        store._queue_index_cleanup(root.hex())
        deadline = time.monotonic() + 5.0
        with store._write_lifecycle:
            while store._pending_write_items > 0:
                remaining = deadline - time.monotonic()
                assert remaining > 0
                store._write_lifecycle.wait(timeout=remaining)
        assert _rows(store) == []
        assert list(store.blocks_dir.rglob("*.safetensors")) == []
    finally:
        store.shutdown()


def test_global_scan_handles_more_than_python_recursion_limit(
    tmp_path: Path,
) -> None:
    namespace = ensure_managed_block_cache_namespace(tmp_path / "aaaaaaaaaaaa")
    blocks = namespace / "blocks"
    database = namespace / "block_index.db"
    with sqlite3.connect(str(database)) as connection:
        connection.execute(
            "CREATE TABLE blocks ("
            "block_hash TEXT PRIMARY KEY, parent_hash TEXT, "
            "ancestry_known INTEGER NOT NULL, file_name TEXT NOT NULL, "
            "last_accessed REAL NOT NULL)"
        )
        parent_hash = None
        now = time.time()
        for index in range(1600):
            block_hash = f"{index + 1:064x}"
            path = blocks / block_hash[:2] / f"{block_hash}.safetensors"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
            connection.execute(
                "INSERT INTO blocks VALUES (?, ?, 1, ?, ?)",
                (
                    block_hash,
                    parent_hash,
                    str(path.relative_to(namespace)),
                    now + index,
                ),
            )
            parent_hash = block_hash
        connection.commit()

    budget = GlobalDiskCacheBudget(tmp_path, 0, orphan_grace_seconds=0)
    try:
        result = budget.enforce(force=True)
        assert result.compliant is True
        assert result.evicted_entries == 0
    finally:
        budget.close()
