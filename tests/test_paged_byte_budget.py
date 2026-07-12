# SPDX-License-Identifier: Apache-2.0
"""Byte-budget RAM ceiling for the paged KV cache (Wave-18).

The paged cache previously had no RAM-byte ceiling — the block pool grew to a
fixed ``max_blocks`` regardless of per-model KV size, and freed blocks never
released their KV mirror, so resident GPU memory ratcheted upward with distinct
prefixes (measured +3.7 GB vs +98 MB for the memory-aware path on the same
workload). ``max_resident_bytes`` + ``enforce_byte_budget()`` give paged the
same RAM discipline the memory-aware path already has: evict FREE (ref_count==0)
cached blocks — LRU first, disk-L2 write-through first — until under budget.

These tests drive the accounting/eviction directly (no model needed).
"""

from vmlx_engine.paged_cache import PagedCacheManager


def _cache_a_block(mgr, block, block_hash, nbytes, ref_count=0, last_access=0.0):
    """Register a block as a cached, materialized RAM mirror."""
    block.block_hash = block_hash
    block.cache_data = [("k", "v")]  # sentinel non-None payload
    block.token_count = mgr.block_size
    block.ref_count = ref_count
    block.last_access = last_access
    mgr.cached_block_hash_to_block.insert(block_hash, block)
    mgr._note_resident(block, nbytes)


def test_disabled_budget_is_noop():
    mgr = PagedCacheManager(block_size=4, max_blocks=10, max_resident_bytes=0)
    blk = mgr.blocks[1]
    _cache_a_block(mgr, blk, block_hash=111, nbytes=10_000)
    # Ceiling disabled → no accounting, no eviction.
    assert mgr.resident_bytes == 0
    assert mgr.enforce_byte_budget() == 0
    assert blk.cache_data is not None


def test_note_and_release_accounting():
    mgr = PagedCacheManager(block_size=4, max_blocks=10, max_resident_bytes=100_000)
    b1, b2 = mgr.blocks[1], mgr.blocks[2]
    _cache_a_block(mgr, b1, 111, 400)
    _cache_a_block(mgr, b2, 222, 600)
    assert mgr.resident_bytes == 1000
    # Re-noting the same block replaces (no double count).
    mgr._note_resident(b1, 500)
    assert mgr.resident_bytes == 1100
    mgr._release_resident(b1)
    assert mgr.resident_bytes == 600
    assert b1.resident_bytes == 0


def test_enforce_evicts_lru_until_under_budget():
    mgr = PagedCacheManager(block_size=4, max_blocks=10, max_resident_bytes=1000)
    # 3 free cached blocks, 400 each = 1200 > 1000 budget.
    _cache_a_block(mgr, mgr.blocks[1], 111, 400, last_access=30.0)  # newest
    _cache_a_block(mgr, mgr.blocks[2], 222, 400, last_access=10.0)  # oldest (LRU)
    _cache_a_block(mgr, mgr.blocks[3], 333, 400, last_access=20.0)
    assert mgr.resident_bytes == 1200
    evicted = mgr.enforce_byte_budget()
    # Must drop exactly one (1200-400=800 <= 1000) and it must be the LRU one.
    assert evicted == 1
    assert mgr.resident_bytes == 800
    assert mgr.blocks[2].cache_data is None  # LRU (last_access=10) evicted
    assert mgr.blocks[1].cache_data is not None
    assert mgr.blocks[3].cache_data is not None


def test_enforce_never_evicts_referenced_blocks():
    mgr = PagedCacheManager(block_size=4, max_blocks=10, max_resident_bytes=500)
    # One in-flight (ref_count=1) big block + one free small block; over budget.
    _cache_a_block(mgr, mgr.blocks[1], 111, 900, ref_count=1, last_access=1.0)
    _cache_a_block(mgr, mgr.blocks[2], 222, 100, ref_count=0, last_access=2.0)
    assert mgr.resident_bytes == 1000
    evicted = mgr.enforce_byte_budget()
    # The referenced block cannot be evicted even though it is the LRU + biggest;
    # only the free one is eligible. Budget may remain exceeded — that is correct:
    # never corrupt an in-flight sequence to hit a RAM target.
    assert mgr.blocks[1].cache_data is not None  # in-flight preserved
    assert evicted == 1
    assert mgr.blocks[2].cache_data is None
    assert mgr.resident_bytes == 900  # only the free 100 reclaimed


def test_enforce_noop_when_within_budget():
    mgr = PagedCacheManager(block_size=4, max_blocks=10, max_resident_bytes=10_000)
    _cache_a_block(mgr, mgr.blocks[1], 111, 400)
    assert mgr.enforce_byte_budget() == 0
    assert mgr.blocks[1].cache_data is not None
