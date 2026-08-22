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


def test_touch_is_a_noop_without_a_disk_store_or_without_tokens():
    cache = _make_cache(None)
    cache._touch_disk_chain_access(list(range(8)), 8)  # no store: no raise

    touched = []
    disk_store = SimpleNamespace(
        _queue_access_update=lambda hash_hex: touched.append(hash_hex)
    )
    cache = _make_cache(disk_store)
    cache._touch_disk_chain_access([], 0)
    assert touched == []
    # NOTE: a sub-block hit is NOT a no-op any more. Partial blocks are
    # durable rows (paged_cache._partial_block_sizes probes them on restart),
    # so leaving them untouched is exactly the eviction hazard this file is
    # about. See test_touch_still_covers_a_sub_block_only_chain.


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


def test_touch_includes_the_terminal_partial_block():
    """The partial tail is a LEAF, and the L2 trim evicts leaves oldest-first.

    Skipping it meant that on an L1-hot chain every full block had its
    timestamp refreshed each warm turn while the partial kept its store-time
    stamp — so the block that COMPLETES the chain was the first casualty of any
    budget pressure. The two pre-existing tests here used 8 (exact multiple of
    the block size) and 3 (sub-block), the two shapes where this is invisible
    by construction. This one uses 11 with block_size 4: full blocks present
    AND a tail.
    """
    touched = []
    disk_store = SimpleNamespace(
        _queue_access_update=lambda hash_hex: touched.append(hash_hex)
    )
    cache = _make_cache(disk_store)
    tokens = list(range(11))
    cache._touch_disk_chain_access(tokens, 11, cache_extra_keys=None)

    parent = None
    expected = []
    for idx in range(2):
        parent = compute_block_hash(
            parent, tokens[idx * 4 : idx * 4 + 4], extra_keys=None
        )
        expected.append(parent.hex())
    # The partial hashes against the running parent exactly as paged_cache's
    # own partial-hit probe does (paged_cache.py compute_block_hash on
    # token_ids[start : start + partial_size]).
    expected.append(
        compute_block_hash(parent, tokens[8:11], extra_keys=None).hex()
    )

    assert touched == expected, (
        "the terminal partial was not touched — it is the first row the LRU "
        "trim will take, and losing it dead-ends the whole chain"
    )


def test_touch_still_covers_a_sub_block_only_chain():
    """A chain shorter than one block is ALL partial — it must still touch."""
    touched = []
    disk_store = SimpleNamespace(
        _queue_access_update=lambda hash_hex: touched.append(hash_hex)
    )
    cache = _make_cache(disk_store)
    tokens = [7, 8, 9]
    cache._touch_disk_chain_access(tokens, 3, cache_extra_keys=None)
    assert touched == [
        compute_block_hash(None, tokens[:3], extra_keys=None).hex()
    ]
