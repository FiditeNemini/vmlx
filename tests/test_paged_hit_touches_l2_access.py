"""A paged/L1 hit must refresh the reused chain's L2 LRU access.

Campaign #181 (r5_evict_B, measured live): in the paged lane nothing touched
an L2 entry after store time — the chain's reuse lived entirely in L1 — so
under a bounded L2 the global LRU ranked the HOT recent conversation's disk
backing as cold as the stale old chain and two filler bursts swept BOTH
(recent chain 382/382 readable -> 0/0). The disk-only lane never sees this
because its reads physically touch L2.

fetch_cache now enqueues best-effort access touches for every full reused
block, using the same chain-hash scheme the store path writes with, so the
touched hashes match the on-disk keys exactly.
"""

from __future__ import annotations

from types import SimpleNamespace

from vmlx_engine.block_disk_store import BlockDiskStore
from vmlx_engine.paged_cache import compute_block_hash
from vmlx_engine.prefix_cache import BlockAwarePrefixCache


def _make_cache(disk_store, block_size=4):
    cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
    cache.paged_cache = SimpleNamespace(_disk_store=disk_store)
    cache.block_size = block_size
    return cache


def test_touch_enqueues_store_matching_chain_hashes():
    touched = []
    disk_store = SimpleNamespace(
        _queue_access_update=lambda hash_hex: touched.append(hash_hex)
    )
    cache = _make_cache(disk_store)
    tokens = list(range(11))  # 2 full blocks of 4, partial tail ignored
    cache._touch_disk_chain_access(tokens, 8, cache_extra_keys=None)

    parent = None
    expected = []
    for idx in range(2):
        parent = compute_block_hash(
            parent, tokens[idx * 4 : idx * 4 + 4], extra_keys=None
        )
        expected.append(parent.hex())
    assert touched == expected, (
        "touched hashes must equal the store path's chain hashes byte for "
        "byte — anything else touches nonexistent rows and the hot chain "
        "still gets evicted"
    )


def test_touch_is_a_noop_without_disk_store_or_full_blocks():
    cache = _make_cache(None)
    cache._touch_disk_chain_access(list(range(8)), 8)  # no store: no raise

    touched = []
    disk_store = SimpleNamespace(
        _queue_access_update=lambda hash_hex: touched.append(hash_hex)
    )
    cache = _make_cache(disk_store)
    cache._touch_disk_chain_access(list(range(3)), 3)  # sub-block hit
    assert touched == []


def test_fetch_cache_hit_paths_call_the_touch():
    import inspect

    src = inspect.getsource(BlockAwarePrefixCache.fetch_cache)
    assert src.count("_touch_disk_chain_access") == 2, (
        "both hit-return paths (chain-hash and prefix-index) must refresh "
        "L2 access — fixing one of two is the documented default failure"
    )


def test_real_store_access_queue_accepts_touches(tmp_path):
    """The touch target is the real store's non-blocking access queue."""
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    try:
        cache = _make_cache(store)
        cache._touch_disk_chain_access(list(range(8)), 8)
    finally:
        store.shutdown()
