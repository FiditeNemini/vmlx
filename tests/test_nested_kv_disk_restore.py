"""Nested attention/indexer state must survive the production SSD write path."""
import pytest
from types import SimpleNamespace

mx = pytest.importorskip('mlx.core')
pytest.importorskip('mlx_lm')


@pytest.mark.parametrize('disk_only', [False, True])
@pytest.mark.parametrize('declared_layout', [None, ((2, 1),), ((1, 1),), ((2, 2),)],
                         ids=['legacy', 'declared', 'bad-attention', 'bad-indexer'])
def test_nested_kv_survives_disk_restart(tmp_path, disk_only, declared_layout):
    from mlx_lm.models.cache import CacheList, KVCache
    from vmlx_engine.block_disk_store import BlockDiskStore
    from vmlx_engine.paged_cache import PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    # Attention uses bf16 K/V; the DSA indexer uses fp32 K and zero-width V.
    k = mx.arange(128).reshape(1, 2, 16, 4).astype(mx.bfloat16)
    v = (k + 1).astype(mx.bfloat16)
    ik = mx.arange(64).reshape(1, 1, 16, 4).astype(mx.float32) / 7
    iv = mx.zeros((1, 1, 16, 0), dtype=mx.float32)
    mx.eval(k, v, ik, iv)
    state = [{'class_name': 'CacheList', 'state': None, 'meta_state': None,
              'sub_caches': [
                  {'class_name': 'KVCache', 'state': (k, v), 'meta_state': ()},
                  {'class_name': 'KVCache', 'state': (ik, iv), 'meta_state': ()}]}]
    tokens = list(range(16))
    model = None if declared_layout is None else SimpleNamespace(
        args=SimpleNamespace(num_key_value_heads=2),
        cache_list_head_counts=declared_layout,
    )
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    try:
        manager = PagedCacheManager(block_size=8, max_blocks=32,
                                    disk_store=store, disk_only=disk_only)
        cache = BlockAwarePrefixCache(model=model, paged_cache_manager=manager)
        cache.store_cache('store', tokens, state)
    finally:
        store.shutdown()

    # New managers and store, with no possible L1 payload fallback.
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    try:
        manager = PagedCacheManager(block_size=8, max_blocks=32, disk_store=store)
        cache = BlockAwarePrefixCache(model=model, paged_cache_manager=manager)
        table, remaining = cache.fetch_cache('restore', tokens + [99])
        assert table is not None and table.num_tokens == 16 and remaining == [99]
        restored = cache.reconstruct_cache(table)
        if declared_layout not in (None, ((2, 1),)):
            assert restored is None
            return
        assert restored is not None and len(restored) == 1
        assert isinstance(restored[0], CacheList)
        for sub, expected in zip(restored[0].caches, [(k, v), (ik, iv)]):
            assert isinstance(sub, KVCache) and sub.offset == 16
            for actual, wanted in zip(sub.state, expected):
                assert actual.dtype == wanted.dtype
                assert actual.shape == wanted.shape
                assert bool(mx.array_equal(actual, wanted))
    finally:
        store.shutdown()


def test_mixed_indexer_rotating_append_after_disk_restart(tmp_path):
    from mlx_lm.models.cache import CacheList, KVCache, RotatingKVCache
    from vmlx_engine.block_disk_store import BlockDiskStore
    from vmlx_engine.paged_cache import PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache
    from vmlx_engine.scheduler import Scheduler

    model = SimpleNamespace(
        args=SimpleNamespace(num_key_value_heads=2, swa_num_key_value_heads=4),
        cache_list_head_counts=((2, 1), None),
    )
    native = [CacheList(KVCache(), KVCache()), RotatingKVCache(max_size=8)]

    def append(caches, positions):
        base = mx.array(positions, dtype=mx.float32).reshape(1, 1, -1, 1)
        outputs = []
        for cache, heads, width, dtype in [
            (caches[0][0], 2, 4, mx.bfloat16),
            (caches[0][1], 1, 0, mx.float32),
            (caches[1], 4, 4, mx.bfloat16),
        ]:
            keys = mx.broadcast_to(base, (1, heads, len(positions), 4)).astype(dtype)
            values = mx.broadcast_to(base + 100, (1, heads, len(positions), width)).astype(dtype)
            outputs.extend(cache.update_and_fetch(keys, values))
        mx.eval(*outputs)
        return outputs

    append(native, list(range(12)))
    append(native, list(range(12, 16)))
    state = Scheduler.__new__(Scheduler)._extract_cache_states(native)
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    try:
        manager = PagedCacheManager(block_size=4, max_blocks=32, disk_store=store, disk_only=True)
        cache = BlockAwarePrefixCache(model=model, paged_cache_manager=manager)
        assert cache.store_cache('mixed-store', list(range(16)), state) is not None
    finally:
        store.shutdown()
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    try:
        manager = PagedCacheManager(block_size=4, max_blocks=32, disk_store=store, disk_only=True)
        cache = BlockAwarePrefixCache(model=model, paged_cache_manager=manager)
        table, rest = cache.fetch_cache('mixed-restore', list(range(20)))
        assert table is not None and table.num_tokens == 16 and rest == list(range(16, 20))
        restored = cache.reconstruct_cache(table)
        assert restored is not None
        assert isinstance(restored[0], CacheList)
        assert isinstance(restored[1], RotatingKVCache)
        expected, actual = append(native, rest), append(restored, rest)
        for a, b in zip(actual, expected):
            assert a.dtype == b.dtype and a.shape == b.shape
            assert bool(mx.array_equal(a, b))
        assert restored[1].offset == native[1].offset == 20
        assert restored[1]._idx == native[1]._idx
        before = [mx.array(x) for x in native[0][0].state]
        append(restored, [20])
        assert native[0][0].offset == 20 and restored[0][0].offset == 21
        for a, b in zip(native[0][0].state, before):
            assert bool(mx.array_equal(a, b))
    finally:
        store.shutdown()
