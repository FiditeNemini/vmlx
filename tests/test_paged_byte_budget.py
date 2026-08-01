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

from vmlx_engine.paged_cache import BlockTable, PagedCacheManager


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


def test_release_resident_payload_clears_bytes_flags_and_data():
    """Disk reconstruction cleanup must not leave phantom RAM attribution."""
    mgr = PagedCacheManager(block_size=4, max_blocks=10, max_resident_bytes=100_000)
    block = mgr.blocks[1]
    _cache_a_block(mgr, block, 111, 4000)
    block.cache_data_from_disk = True
    block.cache_data_transient = True
    block.keep_resident = True

    mgr.release_resident_payload(block)

    assert block.cache_data is None
    assert block.cache_data_from_disk is False
    assert block.cache_data_transient is False
    assert block.keep_resident is False
    assert block.resident_bytes == 0
    assert mgr.resident_bytes == 0


def test_durable_fallback_release_waits_for_active_request_ref():
    """Late L2 completion must not clear an in-flight native fallback."""

    mgr = PagedCacheManager(block_size=4, max_blocks=10, max_resident_bytes=100_000)
    block = mgr.allocate_block()
    assert block is not None
    _cache_a_block(mgr, block, b"d" * 32, 4000, ref_count=1)
    payload = block.cache_data
    table = BlockTable(
        request_id="active-native-fallback",
        block_ids=[block.block_id],
        num_tokens=4,
    )

    assert mgr.release_resident_payload_when_unreferenced(block) is False
    assert block.cache_data is payload
    assert block.release_resident_when_unreferenced is True

    assert mgr.release_request_refs(table) == 1
    assert block.cache_data is None
    assert block.release_resident_when_unreferenced is False
    assert block.resident_bytes == 0
    assert mgr.resident_bytes == 0


def test_make_resident_payload_evictable_keeps_data_and_accounting():
    """A restored native payload becomes a normal RAM-tier LRU entry."""
    mgr = PagedCacheManager(block_size=4, max_blocks=10, max_resident_bytes=100_000)
    block = mgr.blocks[1]
    _cache_a_block(mgr, block, 111, 4000)
    block.cache_data_from_disk = True
    block.cache_data_transient = False
    block.keep_resident = True

    mgr.make_resident_payload_evictable(block)

    assert block.cache_data is not None
    assert block.cache_data_from_disk is False
    assert block.cache_data_transient is False
    assert block.keep_resident is False
    assert block.resident_bytes == 4000
    assert mgr.resident_bytes == 4000


def test_hash_reset_does_not_leak_keep_resident_to_reused_block():
    block = PagedCacheManager(block_size=4, max_blocks=4).blocks[1]
    block.block_hash = 111
    block.keep_resident = True

    block.reset_hash()

    assert block.block_hash is None
    assert block.keep_resident is False


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


def test_request_ref_release_immediately_enforces_lru_byte_ceiling():
    """Completed native blocks must rotate out of RAM without a later request."""
    mgr = PagedCacheManager(block_size=4, max_blocks=10, max_resident_bytes=500)
    first = mgr.allocate_block()
    second = mgr.allocate_block()
    assert first is not None and second is not None
    _cache_a_block(mgr, first, b"a" * 32, 400, ref_count=1, last_access=1.0)
    _cache_a_block(mgr, second, b"b" * 32, 400, ref_count=1, last_access=2.0)
    table = BlockTable(
        request_id="native-complete",
        block_ids=[first.block_id, second.block_id],
        num_tokens=8,
    )

    assert mgr.release_request_refs(table) == 2

    assert mgr.resident_bytes == 400
    assert first.cache_data is None
    assert second.cache_data is not None
    assert mgr.stats.evictions == 1


def test_enforce_noop_when_within_budget():
    mgr = PagedCacheManager(block_size=4, max_blocks=10, max_resident_bytes=10_000)
    _cache_a_block(mgr, mgr.blocks[1], 111, 400)
    assert mgr.enforce_byte_budget() == 0
    assert mgr.blocks[1].cache_data is not None


def test_disk_promotion_uses_transient_payload_beyond_persistent_l1_cap():
    """A long SSD prefix must not be truncated by the persistent RAM cap."""

    class _Arr:
        def __init__(self, nbytes):
            self.nbytes = nbytes

    mgr = PagedCacheManager(
        block_size=4,
        max_blocks=10,
        max_resident_bytes=1000,
    )
    first = mgr._promote_from_disk(b"a" * 32, [_Arr(600)], 4)
    second = mgr._promote_from_disk(b"b" * 32, [_Arr(600)], 4)

    assert first is not None
    assert first.ref_count == 1
    assert first.cache_data_transient is False
    assert second is not None
    assert second.ref_count == 1
    assert second.cache_data_transient is True
    assert mgr.resident_bytes == 1200
    assert sum(b.cache_data is not None for b in mgr.blocks) == 2
    assert mgr.transient_disk_promotions == 1
    assert mgr.transient_disk_peak_bytes == 1200

    mgr.release_resident_payload(second)

    assert mgr.resident_bytes == 600
    assert second.cache_data is None


def test_disk_promotion_evicts_free_lru_mirror_to_make_room():
    """Admission may evict an unreferenced L1 mirror before promoting L2."""

    class _Arr:
        def __init__(self, nbytes):
            self.nbytes = nbytes

    mgr = PagedCacheManager(
        block_size=4,
        max_blocks=10,
        max_resident_bytes=1000,
    )
    old = mgr._promote_from_disk(b"a" * 32, [_Arr(700)], 4)
    assert old is not None
    old.ref_count = 0

    new = mgr._promote_from_disk(b"b" * 32, [_Arr(600)], 4)

    assert new is not None
    assert old.cache_data is None
    assert new.cache_data is not None
    assert mgr.resident_bytes == 600
    assert mgr.stats.evictions == 1


def test_enforce_skips_keep_resident_native_state():
    """DSV4/ZAYA/rotating-SWA composite blocks are flagged keep_resident and must
    survive the byte ceiling — their RAM mirror has to outlive the async L2 write
    so an immediate same-process repeat can reconstruct without corruption."""
    mgr = PagedCacheManager(block_size=4, max_blocks=10, max_resident_bytes=500)
    # Oldest is the protected composite block; the plain block is newer.
    _cache_a_block(mgr, mgr.blocks[1], 111, 600, last_access=1.0)  # LRU, protected
    mgr.blocks[1].keep_resident = True
    _cache_a_block(mgr, mgr.blocks[2], 222, 400, last_access=2.0)  # plain
    assert mgr.resident_bytes == 1000
    evicted = mgr.enforce_byte_budget()
    # Only the plain block is eligible even though the protected one is LRU+biggest.
    assert mgr.blocks[1].cache_data is not None  # keep_resident preserved
    assert mgr.blocks[2].cache_data is None
    assert evicted == 1
    assert mgr.resident_bytes == 600


def test_clear_resets_resident_accounting():
    """clear() recreates the block pool (fresh resident_bytes=0 per block); the
    global counter must follow or it stays a phantom positive that makes every
    future store over-evict."""
    mgr = PagedCacheManager(block_size=4, max_blocks=10, max_resident_bytes=10_000)
    _cache_a_block(mgr, mgr.blocks[1], 111, 4000)
    _cache_a_block(mgr, mgr.blocks[2], 222, 4000)
    assert mgr.resident_bytes == 8000
    mgr.clear()
    assert mgr.resident_bytes == 0
    # A fresh store after clear accounts only its own bytes (no phantom carry).
    _cache_a_block(mgr, mgr.blocks[1], 333, 1000)
    assert mgr.resident_bytes == 1000


def test_reset_prefix_cache_resets_resident_accounting():
    mgr = PagedCacheManager(block_size=4, max_blocks=10, max_resident_bytes=10_000)
    _cache_a_block(mgr, mgr.blocks[1], 111, 4000)
    assert mgr.resident_bytes == 4000
    assert mgr.reset_prefix_cache() is True
    assert mgr.resident_bytes == 0
    assert all(b.resident_bytes == 0 for b in mgr.blocks)


def test_estimate_block_nbytes_recurses_dicts():
    """DSV4 composite state nests its largest arrays under mapping leaves; without
    dict recursion the estimate undercounts to zero and the ceiling never fires."""

    class _Arr:
        def __init__(self, nbytes):
            self.nbytes = nbytes

    # tuple → dict → list → array leaves (DSV4-style pytree state)
    cache_data = (
        "deepseek_v4",
        {"layer0": [_Arr(1000), _Arr(2000)], "layer1": {"k": _Arr(500)}},
        "meta",
    )
    assert PagedCacheManager.estimate_block_nbytes(cache_data) == 3500
    # Plain (keys, values) list path still works.
    assert PagedCacheManager.estimate_block_nbytes([(_Arr(10), _Arr(20))]) == 30
    # Self-referential dict must not infinite-loop.
    d = {}
    d["self"] = d
    d["arr"] = _Arr(42)
    assert PagedCacheManager.estimate_block_nbytes(d) == 42


def test_both_schedulers_pass_max_resident_bytes_to_paged_manager():
    """#98 parity: the MLLM/VL scheduler must give its PagedCacheManager the
    same RAM-byte ceiling the text scheduler does. Before the fix the MLLM path
    instantiated PagedCacheManager without max_resident_bytes, so the VL/MLLM
    paged pool (e.g. Step-3.7 video, forced-paged under paged-default-ON) was
    bounded only by max_cache_blocks — the exact gap the text path closed in
    Wave-18. This source-parity guard prevents a silent regression."""
    import re

    text_src = open("vmlx_engine/scheduler.py").read()
    mllm_src = open("vmlx_engine/mllm_scheduler.py").read()

    # Both call sites must exist and both must thread max_resident_bytes.
    for name, src in (("scheduler.py", text_src), ("mllm_scheduler.py", mllm_src)):
        calls = re.findall(r"PagedCacheManager\((.*?)\)", src, re.DOTALL)
        # The production instantiation is the one that also passes max_blocks.
        prod = [c for c in calls if "max_blocks" in c and "block_size" in c]
        assert prod, f"{name}: no production PagedCacheManager(...) call found"
        assert any("max_resident_bytes=" in c for c in prod), (
            f"{name}: production PagedCacheManager must pass max_resident_bytes "
            f"(RAM byte ceiling parity, #98)"
        )
        assert any("compute_memory_limit()" in src for _ in prod), (
            f"{name}: must derive the ceiling from MemoryCacheConfig.compute_memory_limit()"
        )


def test_finished_paged_store_reenforces_budget_after_ref_release():
    """The L1 ceiling must run after completed-request blocks become evictable."""
    source = open("vmlx_engine/scheduler.py").read()
    release = source.index(
        "self.block_aware_cache.paged_cache.release_request_refs("
    )
    post_store_release = source.index(
        "self.block_aware_cache.paged_cache.release_request_refs(",
        release + 1,
    )
    next_handler = source.index("except Exception as _rel_e:", post_store_release)
    branch = source[post_store_release:next_handler]

    assert "self.block_aware_cache.paged_cache.enforce_byte_budget()" in branch
