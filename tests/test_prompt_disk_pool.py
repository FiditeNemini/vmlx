"""Whole-prompt SSD records in the managed aggregate pool (--disk-cache-pool).

openPangu-2.0 typed snapshots (~3 GB per 4.5k-token prompt) had their own 10 GB budget (~3 records): it evicted on
almost every turn, a restarted engine missed (box tw row 2026-10-11, r1/r2), and the app's SSD pool notice never saw
it. In pool mode the records are published, accounted and evicted oldest-first by the same pool as block files and
GLM native snapshots.
"""
import threading
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")


def _cache(value, n=32):
    from mlx_lm.models.cache import KVCache

    state = KVCache()
    state.update_and_fetch(mx.full((1, 1, n, 8), value, dtype=mx.bfloat16),
                           mx.full((1, 1, n, 8), value, dtype=mx.bfloat16))
    return [state]


def _manager(tmp_path, pool_gb=1.0, name="model_abc"):
    from vmlx_engine.disk_cache import DiskCacheManager

    return DiskCacheManager(cache_dir=str(tmp_path / "index" / name), max_size_gb=10,
                            pool_root=str(tmp_path / "pool"), pool_max_size_gb=pool_gb, pool_model_key=name)


def test_records_live_in_the_pool_and_survive_restart(tmp_path):
    from vmlx_engine.disk_cache import PROMPT_POOL_SCHEMA

    tokens = list(range(40))
    m = _manager(tmp_path)
    try:
        assert m.store(tokens, _cache(4))
        assert m.flush_pending_writes(tokens)
        files = list((tmp_path / "pool").rglob("*.safetensors"))
        assert len(files) == 1 and "native_sessions" in files[0].parts
        side = files[0].with_suffix(".json").read_text()
        assert PROMPT_POOL_SCHEMA in side
        assert not list((tmp_path / "index").rglob("*.safetensors"))   # nothing outside the pool
        assert m.stats()["ssd_pool"] is True
    finally:
        m.shutdown()
    reader = _manager(tmp_path)                                         # engine restart
    try:
        restored = reader.fetch(tokens)
        assert restored is not None and mx.all(restored[0].keys == 4).item()
        cache, matched = reader.fetch_longest_prefix(tokens + [7, 8])
        assert cache is not None and matched == tokens
    finally:
        reader.shutdown()


def test_pool_evicts_the_oldest_record_and_the_index_follows(tmp_path):
    old, new = list(range(40)), list(range(100, 140))
    m = _manager(tmp_path, pool_gb=1.0)
    try:
        assert m.store(old, _cache(1, n=64))
        assert m.flush_pending_writes(old)
        one = sum(p.stat().st_size for p in (tmp_path / "pool").rglob("*") if p.is_file()
                  and p.suffix in (".safetensors", ".json") and "native_sessions" in p.parts)
        # Shrink the pool so it holds one record but not two.
        m._pool_store.budget._publish_budget(int(one * 1.5))
        m._pool_store.max_size_bytes = int(one * 1.5)
        assert m.store(new, _cache(2, n=64))
        m._write_queue.join()
        m._pool_store.budget.enforce(force=True)
        assert m.fetch(old) is None                                     # oldest evicted by the pool
        assert m.fetch(new) is not None
    finally:
        m.shutdown()


def test_rejected_record_is_never_deleted_behind_the_pool(tmp_path):
    tokens = list(range(40))
    m = _manager(tmp_path)
    try:
        assert m.store(tokens, _cache(3))
        assert m.flush_pending_writes(tokens)
        data = next((tmp_path / "pool").rglob("*.safetensors"))
        m._required_cache_classes = frozenset({"SomeOtherCache"})       # force a class-mismatch rejection
        assert m.fetch(tokens) is None
        assert data.is_file()                                           # left for pool eviction/accounting
    finally:
        m.shutdown()


def test_pool_mode_never_writes_tq_native_files(tmp_path):
    from vmlx_engine.disk_cache import DiskCacheManager

    m = DiskCacheManager(cache_dir=str(tmp_path / "index"), allow_tq_native=True,
                         pool_root=str(tmp_path / "pool"), pool_max_size_gb=1.0, pool_model_key="k")
    try:
        assert m._allow_tq_native is False
    finally:
        m.shutdown()


def test_shutdown_drains_queued_writes_of_every_shape(tmp_path, monkeypatch):
    """Standard queue items are 6-tuples; the old drain unpacked 5 and dropped them ("too many values to unpack")."""
    from vmlx_engine.disk_cache import DiskCacheManager

    m = DiskCacheManager(cache_dir=str(tmp_path / "plain"), max_size_gb=1)
    gate = threading.Event()
    real = DiskCacheManager._write_cache
    first = {"seen": False}

    def slow(self, *a, **kw):
        if not first["seen"]:
            first["seen"] = True
            gate.wait(5)
        return real(self, *a, **kw)

    monkeypatch.setattr(DiskCacheManager, "_write_cache", slow)
    keys = [list(range(s, s + 40)) for s in (0, 100, 200)]
    for i, k in enumerate(keys):
        assert m.store(k, _cache(i + 1))
    threading.Timer(0.3, gate.set).start()
    m.shutdown()                                                         # writer stops after item 1; drain does 2-3
    reader = DiskCacheManager(cache_dir=str(tmp_path / "plain"), max_size_gb=1)
    try:
        assert all(reader.fetch(k) is not None for k in keys)
    finally:
        reader.shutdown()


def test_cli_resolves_the_pool_like_the_block_store(tmp_path, monkeypatch):
    from vmlx_engine import cli

    off = SimpleNamespace(disk_cache_pool=False)
    assert cli.resolve_prompt_disk_pool(off) == (None, 0.0)
    on = SimpleNamespace(disk_cache_pool=True, block_disk_cache_dir=str(tmp_path / "b"),
                         block_disk_cache_max_gb=None, block_disk_cache_max_percent=10)
    root, gb = cli.resolve_prompt_disk_pool(on)
    assert root == str((tmp_path / "b").resolve()) and gb > 0          # percent of the volume, not a flat 10 GB
    explicit = SimpleNamespace(disk_cache_pool=True, block_disk_cache_dir=None, block_disk_cache_max_gb=0.0,
                               block_disk_cache_max_percent=10)
    assert cli.resolve_prompt_disk_pool(explicit)[1] == 0.0             # explicit 0 = unlimited, as for blocks
