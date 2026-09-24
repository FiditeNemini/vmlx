"""Whole-prompt SSD snapshots must not starve new tool continuations."""
import pytest

mx = pytest.importorskip("mlx.core")


@pytest.mark.parametrize("old_role", ["system", "user", "assistant"])
def test_recent_tool_snapshot_survives_pressure_and_restart(tmp_path, old_role):
    from mlx_lm.models.cache import KVCache
    from vmlx_engine.disk_cache import DiskCacheManager

    def cache(value):
        state = KVCache()
        state.update_and_fetch(mx.full((1, 1, 32, 8), value, dtype=mx.bfloat16),
                               mx.full((1, 1, 32, 8), value, dtype=mx.bfloat16))
        return [state]

    old, recent = list(range(32)), list(range(100, 132))
    manager = DiskCacheManager(cache_dir=str(tmp_path), max_size_gb=1)
    try:
        assert manager.store(old, cache(1), cache_type=old_role)
        assert manager.flush_pending_writes(old)
        # Budget holds either snapshot, but not both. A whole conversation
        # containing a system message must not permanently pin the old file.
        manager.max_size_bytes = manager._total_size() + 256
        assert manager.store(recent, cache(2), cache_type="assistant")
        assert manager.flush_pending_writes(recent)
        assert manager._total_size() <= manager.max_size_bytes
        assert manager.fetch(old) is None
        assert len(list(tmp_path.glob("*.safetensors"))) == 1
    finally:
        manager.shutdown()
    reader = DiskCacheManager(cache_dir=str(tmp_path), max_size_gb=1)
    try:
        restored = reader.fetch(recent)
        assert restored is not None
        assert restored[0].offset == 32
        assert restored[0].keys.dtype == mx.bfloat16
        assert mx.all(restored[0].keys == 2).item()
    finally:
        reader.shutdown()


def test_fetch_refreshes_eviction_recency(tmp_path):
    from mlx_lm.models.cache import KVCache
    from vmlx_engine.disk_cache import DiskCacheManager

    state = KVCache()
    state.update_and_fetch(mx.ones((1, 1, 32, 8)), mx.ones((1, 1, 32, 8)))
    keys = [list(range(start, start + 32)) for start in (0, 100, 200)]
    manager = DiskCacheManager(cache_dir=str(tmp_path), max_size_gb=1)
    try:
        for tokens in keys[:2]:
            assert manager.store(tokens, [state])
            assert manager.flush_pending_writes(tokens)
        manager.max_size_bytes = manager._total_size() + 256
        assert manager.fetch(keys[0]) is not None
        assert manager.store(keys[2], [state])
        assert manager.flush_pending_writes(keys[2])
        assert manager.fetch(keys[1]) is None
        assert manager.fetch(keys[0]) is not None
        assert manager._total_size() <= manager.max_size_bytes
    finally:
        manager.shutdown()
