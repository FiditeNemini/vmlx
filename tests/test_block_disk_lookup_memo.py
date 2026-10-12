"""SSD lookup fast paths: candidate metadata is checked once per payload file; fresh writes skip the TQ probe.

Every SSD-only lookup of a mixed-SWA or nested chain validated each block's safetensors header and re-read its
packed metadata before reconstruction (~0.24 s of a 37k Naive-layout lookup, on every tool continuation). The
result depends only on the immutable payload file, so it is memoized under the file identity; a rewrite (rename)
changes the identity and is checked again. Cold writes in explicit non-TQ mode probed has_block() under the
aggregate mutation guard for every new hash, waiting on the busy writer; a hash with no index row is now skipped.
"""

import os
import time

import pytest

mx = pytest.importorskip("mlx.core")

import vmlx_engine.block_disk_store as bds
from vmlx_engine.block_disk_store import BlockDiskStore


def _entry(value: float):
    """A Naive-shaped block: nested CacheList attention/indexer KV plus a pending rotating layer."""
    keys = mx.full((1, 2, 8, 16), value, dtype=mx.bfloat16)
    values = mx.full((1, 2, 8, 16), value + 1, dtype=mx.bfloat16)
    ik = mx.full((1, 1, 8, 8), value, dtype=mx.float32)
    iv = mx.zeros((1, 1, 8, 0), dtype=mx.float32)
    mx.eval(keys, values, ik, iv)
    return [("cache_list", [("kv", keys, values), ("kv", ik, iv)]), ("rotating_kv_pending", "RotatingKVCache")]


def _write(store, block_hash, value):
    assert store.write_block_async(block_hash, _entry(value), 8)
    deadline = time.time() + 10
    while time.time() < deadline and store.get_stats()["write_pipeline"]["pending_items"]:
        time.sleep(0.01)
    assert store.has_block_record(block_hash)


def test_candidate_metadata_is_checked_once_per_payload_file(tmp_path, monkeypatch):
    store = BlockDiskStore(str(tmp_path), max_size_gb=1, allow_tq_native=False)
    try:
        block_hash = bytes(range(32))
        _write(store, block_hash, 1.0)
        loads = []
        original = bds._load_block_validation_entries

        def counting(path, member=None):
            loads.append(path)
            return original(path, member)

        monkeypatch.setattr(bds, "_load_block_validation_entries", counting)
        first = store.read_block_validation_entries(block_hash)
        second = store.read_block_validation_entries(block_hash)
        assert first is not None and second is not None
        assert len(loads) == 1, "the second lookup of an unchanged file must reuse its checked metadata"
        assert [e[0] for e in first] == [e[0] for e in second] == ["cache_list", "rotating_kv_pending"]
        assert [sub[1].shape for sub in first[0][1]] == [sub[1].shape for sub in second[0][1]]
        assert second is not first, "callers get their own list"

        # A rewrite publishes a new file (new identity): it is checked again.
        row = store._read_conn.execute(
            "SELECT file_name FROM blocks WHERE block_hash = ?", (block_hash.hex(),)
        ).fetchone()
        path = store.cache_dir / row[0]
        clone = path.with_suffix(".clone")
        clone.write_bytes(path.read_bytes())
        os.replace(clone, path)
        third = store.read_block_validation_entries(block_hash)
        assert third is not None and len(loads) == 2
    finally:
        store.shutdown()


def test_forgotten_payload_checks_drop_the_metadata_memo(tmp_path, monkeypatch):
    store = BlockDiskStore(str(tmp_path), max_size_gb=1, allow_tq_native=False)
    try:
        block_hash = bytes(range(1, 33))
        _write(store, block_hash, 2.0)
        assert store.read_block_validation_entries(block_hash) is not None
        assert block_hash.hex() in store._validation_entries_memo
        store._forget_payload_checks(block_hash.hex())
        assert block_hash.hex() not in store._validation_entries_memo
    finally:
        store.shutdown()


def test_fresh_non_tq_writes_skip_the_guarded_tq_probe(tmp_path, monkeypatch):
    store = BlockDiskStore(str(tmp_path), max_size_gb=1, allow_tq_native=False)
    try:
        probes = []
        original = BlockDiskStore.has_block

        def counting(self, block_hash):
            probes.append(block_hash)
            return original(self, block_hash)

        monkeypatch.setattr(BlockDiskStore, "has_block", counting)
        fresh = bytes(range(2, 34))
        _write(store, fresh, 3.0)
        assert fresh not in probes, "a hash with no index row has no TQ record to inspect"
        # An existing row is still inspected before a replacement is queued.
        store.write_block_async(fresh, _entry(3.0), 8)
        assert fresh in probes
    finally:
        store.shutdown()
