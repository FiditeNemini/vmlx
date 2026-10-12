"""SSD-only write-behind: a disk-only store returns before its SSD write is durable.

The next request after a cold long prompt (a tool continuation) used to wait at admission for the whole SSD write
(Naive 42k: 1.2 GB, 1.40 s). With write-behind each admitted block keeps the payload it handed to the writer until
its fence settles; a lookup inside that window restores from RAM, and the model thread drops the copies once the
blocks are durable. A block the writer did not retain is retired exactly like the synchronous fence retires it.
"""

import threading
import time

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.cache import KVCache, RotatingKVCache

from vmlx_engine.block_disk_store import BlockDiskStore
from vmlx_engine.paged_cache import PagedCacheManager
import vmlx_engine.prefix_cache as prefix_cache

TOKENS = 200  # 64-token blocks: 64, 64, 64, 8 -> the last block is partial on purpose


def _caches(tokens=TOKENS):
    mx.random.seed(11)
    caches = [KVCache(), RotatingKVCache(max_size=64, keep=0), KVCache()]
    dtypes = [mx.bfloat16, mx.bfloat16, mx.float16]
    fed = 0
    while fed < tokens:
        n = min(48, tokens - fed)
        for c, dt in zip(caches, dtypes):
            c.update_and_fetch(
                (mx.random.normal((1, 2, n, 32)) * 3).astype(dt),
                (mx.random.normal((1, 2, n, 32)) * 3).astype(dt),
            )
        fed += n
    mx.eval([c.state for c in caches])
    return caches


def _extracted(caches):
    return [{"state": c.state, "meta_state": c.meta_state, "class_name": type(c).__name__} for c in caches]


@pytest.fixture
def slow_writer(monkeypatch):
    """Make every staged SSD block take 0.15 s on the writer thread."""
    original = BlockDiskStore._stage_numpy_safetensors_file
    gate = {"delay": 0.15, "fail_from": None, "calls": 0}

    def staged(self, block_hash, tensors):
        gate["calls"] += 1
        if gate["fail_from"] is not None and gate["calls"] >= gate["fail_from"]:
            raise OSError("simulated SSD write failure")
        time.sleep(gate["delay"])
        return original(self, block_hash, tensors)

    monkeypatch.setattr(BlockDiskStore, "_stage_numpy_safetensors_file", staged)
    return gate


def _manager(tmp_path):
    store = BlockDiskStore(str(tmp_path), max_size_gb=1, allow_tq_native=False)
    manager = PagedCacheManager(block_size=64, max_blocks=64, disk_store=store, disk_only=True)
    return store, manager, prefix_cache.BlockAwarePrefixCache(model=None, paged_cache_manager=manager)


def _blocks(manager):
    return [manager.cached_block_hash_to_block.get_block(h) for h in manager.cached_block_hash_to_block._cache]


def _assert_restores(cache, caches, request_id):
    table, remaining = cache.fetch_cache(request_id, list(range(TOKENS)) + [999])
    assert table is not None and table.num_tokens == TOKENS and remaining == [999]
    restored = cache.reconstruct_cache(table)
    assert restored is not None and len(restored) == 3
    for idx in (0, 2):  # positional layers restore bit-exactly
        for got, want in zip(restored[idx].state, caches[idx].state):
            assert got.dtype == want.dtype and got.shape == want.shape
            assert mx.array_equal(got.view(mx.uint16), want.view(mx.uint16)).item()
    # The rotating layer restores its exact terminal window at the stored offset.
    assert restored[1].offset == TOKENS
    cache.release_cache(request_id)
    return [tuple(mx.array(a) for a in c.state) for c in restored]


def test_store_returns_before_the_write_and_restores_from_ram(tmp_path, monkeypatch, slow_writer):
    monkeypatch.setenv("VMLX_SSD_WRITE_BEHIND", "1")
    slow_writer["delay"] = 0.3  # 4 blocks: the synchronous fence would take >= 1.2 s
    store, manager, cache = _manager(tmp_path)
    try:
        caches = _caches()
        t0 = time.perf_counter()
        assert cache.store_cache("cold", list(range(TOKENS)), _extracted(caches)) is not None
        store_s = time.perf_counter() - t0
        assert store_s < 0.9, f"write-behind store waited for the writer ({store_s:.2f} s for 4 x 0.3 s writes)"
        assert cache.write_behind_pending() == 1
        held = [b for b in _blocks(manager) if b.cache_data is not None]
        assert len(held) == 4, "every admitted block keeps its writer payload until durable"
        # A lookup inside the window (the tool continuation) restores from the RAM copies.
        from_ram = _assert_restores(cache, caches, "continuation-in-window")
        assert store.get_stats()["write_pipeline"]["pending_items"] > 0, "lookup must have run inside the window"
        cache.settle_write_behind(wait=True)
        assert cache.write_behind_pending() == 0
        assert all(b.cache_data is None for b in _blocks(manager)), "RAM copies must be gone once durable"
        # After release the same lookup reads the durable SSD records: every layer, the rotating window
        # included, is bit-identical to the in-window RAM restore.
        from_ssd = _assert_restores(cache, caches, "continuation-after-durable")
        for ram_layer, ssd_layer in zip(from_ram, from_ssd):
            for a, b in zip(ram_layer, ssd_layer):
                assert a.dtype == b.dtype and a.shape == b.shape
                assert mx.array_equal(a.view(mx.uint8), b.view(mx.uint8)).item()
    finally:
        store.shutdown()


def test_switch_off_keeps_the_synchronous_fence(tmp_path, monkeypatch, slow_writer):
    monkeypatch.setenv("VMLX_SSD_WRITE_BEHIND", "0")
    store, manager, cache = _manager(tmp_path)
    try:
        t0 = time.perf_counter()
        assert cache.store_cache("cold", list(range(TOKENS)), _extracted(_caches())) is not None
        assert time.perf_counter() - t0 >= 0.55, "VMLX_SSD_WRITE_BEHIND=0 must wait for every block"
        assert cache.write_behind_pending() == 0
        assert all(b.cache_data is None for b in _blocks(manager))
    finally:
        store.shutdown()


@pytest.mark.parametrize("write_behind", ["1", "0"])
def test_blocks_the_writer_did_not_retain_are_retired(tmp_path, monkeypatch, slow_writer, write_behind):
    """A failed write ends in the same state with and without write-behind."""
    monkeypatch.setenv("VMLX_SSD_WRITE_BEHIND", write_behind)
    slow_writer["fail_from"] = 3  # the third block onward fails to stage
    store, manager, cache = _manager(tmp_path)
    try:
        cache.store_cache("cold", list(range(TOKENS)), _extracted(_caches()))
        cache.settle_write_behind(wait=True)
        survivors = _blocks(manager)
        assert len(survivors) == 2, "failed blocks must be retired from lookup"
        assert all(b.cache_data is None for b in survivors), "no RAM copy may outlive a settled fence"
        table, _remaining = cache.fetch_cache("after-failure", list(range(TOKENS)) + [999])
        # No exact rotating anchor survives below the failure, so the mixed-SWA prompt misses.
        assert table is None
    finally:
        store.shutdown()


def test_settle_runs_on_the_scheduler_idle_hook(tmp_path, monkeypatch, slow_writer):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    monkeypatch.setenv("VMLX_SSD_WRITE_BEHIND", "1")
    store, manager, cache = _manager(tmp_path)
    try:
        assert cache.store_cache("cold", list(range(TOKENS)), _extracted(_caches())) is not None
        sched = SimpleNamespace(block_aware_cache=cache)
        deadline = time.time() + 10
        while not prefix_cache._write_behind_ready(sched) and time.time() < deadline:
            time.sleep(0.02)
        assert prefix_cache._write_behind_ready(sched)
        prefix_cache._settle_write_behind_on_model_thread(sched)
        assert cache.write_behind_pending() == 0
        assert not prefix_cache._write_behind_ready(sched)
        # Test doubles never look ready (no idle-loop spin on a MagicMock cache).
        assert not prefix_cache._write_behind_ready(SimpleNamespace(block_aware_cache=MagicMock()))
    finally:
        store.shutdown()


def test_hybrid_store_also_writes_behind(tmp_path, monkeypatch, slow_writer):
    monkeypatch.setenv("VMLX_SSD_WRITE_BEHIND", "1")
    store, manager, cache = _manager(tmp_path)
    try:
        data = _extracted(_caches(128))
        conv = mx.ones((1, 3, 16), dtype=mx.bfloat16)
        ssm = mx.ones((1, 2, 8, 8), dtype=mx.float32)
        mx.eval(conv, ssm)
        data.append({"state": [conv, ssm], "meta_state": "", "class_name": "ArraysCache"})
        assert cache.store_cache("hybrid", list(range(128)), data) is not None
        # The non-waiting branch ran: one tracked fence and every block still holds its RAM copy.
        assert cache.write_behind_pending() == 1
        assert all(b.cache_data is not None for b in _blocks(manager))
        cache.settle_write_behind(wait=True)
        assert all(b.cache_data is None for b in _blocks(manager))
    finally:
        store.shutdown()


def test_nested_cache_list_layers_write_behind(tmp_path, monkeypatch, slow_writer):
    """Naive N0.5 full-attention layers are CacheList(KVCache, KVCache): they must write behind too."""
    monkeypatch.setenv("VMLX_SSD_WRITE_BEHIND", "1")
    from mlx_lm.models.cache import CacheList

    mx.random.seed(21)
    k = mx.random.normal((1, 2, 128, 16)).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, 128, 16)).astype(mx.bfloat16)
    ik = (mx.random.normal((1, 1, 128, 8)) / 7).astype(mx.float32)
    iv = mx.zeros((1, 1, 128, 0), dtype=mx.float32)
    mx.eval(k, v, ik, iv)
    state = [{"class_name": "CacheList", "state": None, "meta_state": None,
              "sub_caches": [{"class_name": "KVCache", "state": (k, v), "meta_state": ()},
                             {"class_name": "KVCache", "state": (ik, iv), "meta_state": ()}]}]
    store, manager, cache = _manager(tmp_path)
    try:
        assert cache.store_cache("nested", list(range(128)), state) is not None
        assert cache.write_behind_pending() == 1, "CacheList stores must take the write-behind branch"
        assert all(b.cache_data is not None for b in _blocks(manager))

        def restore(rid):
            table, rest = cache.fetch_cache(rid, list(range(128)) + [7])
            assert table is not None and table.num_tokens == 128 and rest == [7]
            restored = cache.reconstruct_cache(table)
            assert restored is not None and isinstance(restored[0], CacheList)
            got = [tuple(mx.array(a) for a in sub.state) for sub in restored[0].caches]
            cache.release_cache(rid)
            return got

        from_ram = restore("in-window")
        cache.settle_write_behind(wait=True)
        assert all(b.cache_data is None for b in _blocks(manager))
        from_ssd = restore("after-durable")
        for (rk, rv), (sk, sv), (wk, wv) in zip(from_ram, from_ssd, [(k, v), (ik, iv)]):
            for a, b, w in ((rk, sk, wk), (rv, sv, wv)):
                assert a.dtype == w.dtype and a.shape == w.shape
                assert mx.array_equal(a.view(mx.uint8), w.view(mx.uint8)).item()
                assert mx.array_equal(b.view(mx.uint8), w.view(mx.uint8)).item()
    finally:
        store.shutdown()
