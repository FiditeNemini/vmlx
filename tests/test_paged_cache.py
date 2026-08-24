# SPDX-License-Identifier: Apache-2.0
"""Tests for Paged KV Cache Manager."""

import platform
import sys
import threading
import time

import pytest

# Skip all tests if not on Apple Silicon
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


class TestCacheBlock:
    """Test CacheBlock dataclass."""

    def test_cache_block_creation(self):
        """Test creating a CacheBlock."""
        from vmlx_engine.paged_cache import CacheBlock

        block = CacheBlock(block_id=0)
        assert block.block_id == 0
        assert block.token_count == 0
        assert block.ref_count == 0  # vLLM style: starts at 0, set to 1 when allocated
        assert block.hash_value is None
        assert block.cache_data is None

    def test_cache_block_is_full(self):
        """Test is_full method."""
        from vmlx_engine.paged_cache import CacheBlock

        block = CacheBlock(block_id=0, token_count=64)
        assert block.is_full(64) is True
        assert block.is_full(128) is False

        block.token_count = 32
        assert block.is_full(64) is False

    def test_cache_block_is_shared(self):
        """Test is_shared method."""
        from vmlx_engine.paged_cache import CacheBlock

        block = CacheBlock(block_id=0, ref_count=1)
        assert block.is_shared() is False

        block.ref_count = 2
        assert block.is_shared() is True

    def test_cache_block_touch(self):
        """Test touch updates last_access."""
        from vmlx_engine.paged_cache import CacheBlock

        block = CacheBlock(block_id=0)
        old_time = block.last_access
        time.sleep(0.01)
        block.touch()
        assert block.last_access > old_time


class TestBlockTable:
    """Test BlockTable dataclass."""

    def test_block_table_creation(self):
        """Test creating a BlockTable."""
        from vmlx_engine.paged_cache import BlockTable

        table = BlockTable(request_id="req-1")
        assert table.request_id == "req-1"
        assert table.block_ids == []
        assert table.num_tokens == 0
        assert len(table) == 0

    def test_block_table_copy(self):
        """Test copying a BlockTable."""
        from vmlx_engine.paged_cache import BlockTable

        table = BlockTable(
            request_id="req-1",
            block_ids=[0, 1, 2],
            num_tokens=192,
        )

        copied = table.copy("req-2")
        assert copied.request_id == "req-2"
        assert copied.block_ids == [0, 1, 2]
        assert copied.num_tokens == 192

        # Verify independence
        copied.block_ids.append(3)
        assert table.block_ids == [0, 1, 2]


class TestPagedCacheManager:
    """Test PagedCacheManager class."""

    def test_initialization(self):
        """Test manager initialization."""
        from vmlx_engine.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=100)

        assert manager.block_size == 64
        assert manager.max_blocks == 100
        # vLLM style: free_blocks is an int property, and null block takes 1 slot
        assert manager.free_blocks == 99  # 100 - 1 (null block)
        assert len(manager.allocated_blocks) == 1  # null block is allocated

        stats = manager.get_stats()
        assert stats.total_blocks == 100
        assert stats.free_blocks == 99
        assert stats.allocated_blocks == 1  # null block

    def test_allocate_block(self):
        """Test block allocation."""
        from vmlx_engine.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=10)
        # Initial: 10 blocks, 1 null block, so 9 free

        block = manager.allocate_block()
        assert block is not None
        assert block.block_id in manager.allocated_blocks
        assert manager.free_blocks == 8  # 9 - 1

        stats = manager.get_stats()
        assert stats.allocated_blocks == 2  # null block + 1 allocated
        assert stats.free_blocks == 8

    def test_allocate_all_blocks(self):
        """Test allocating all available blocks."""
        from vmlx_engine.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=5)
        # With null block taking 1 slot, we have 4 allocatable blocks

        blocks = []
        for _ in range(4):  # Can only allocate 4 (5 - 1 null block)
            block = manager.allocate_block()
            assert block is not None
            blocks.append(block)

        # Should return None when out of blocks
        assert manager.allocate_block() is None
        assert manager.free_blocks == 0

    def test_free_block(self):
        """Test block deallocation."""
        from vmlx_engine.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=10)
        initial_free = manager.free_blocks  # 9 (10 - 1 null block)

        block = manager.allocate_block()
        block_id = block.block_id
        assert manager.free_blocks == initial_free - 1

        result = manager.free_block(block_id)
        assert result is True
        assert block_id not in manager.allocated_blocks
        # Block should be back in free queue
        assert manager.free_blocks == initial_free

    def test_reference_counting(self):
        """Test reference counting."""
        from vmlx_engine.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=10)

        block = manager.allocate_block()
        block_id = block.block_id
        assert block.ref_count == 1

        # Increment ref
        manager.increment_ref(block_id)
        assert block.ref_count == 2

        # Free should decrement, not remove
        result = manager.free_block(block_id)
        assert result is False  # Still referenced
        assert block.ref_count == 1
        assert block_id in manager.allocated_blocks

        # Free again should remove
        result = manager.free_block(block_id)
        assert result is True
        assert block_id not in manager.allocated_blocks

    def test_allocate_blocks_for_tokens(self):
        """Test allocating blocks for a token count."""
        from vmlx_engine.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=100)

        # 100 tokens should need 2 blocks (ceil(100/64) = 2)
        blocks = manager.allocate_blocks_for_tokens(100)
        assert len(blocks) == 2

        # 64 tokens should need 1 block
        blocks = manager.allocate_blocks_for_tokens(64)
        assert len(blocks) == 1

        # 65 tokens should need 2 blocks
        blocks = manager.allocate_blocks_for_tokens(65)
        assert len(blocks) == 2

    def test_allocate_blocks_for_tokens_rollback(self):
        """Test rollback when allocation fails."""
        from vmlx_engine.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=3)
        # With null block, we have 2 allocatable blocks
        initial_free = manager.free_blocks  # 2

        # Try to allocate more than available (300 tokens needs 5 blocks)
        # vLLM style: raises ValueError instead of returning empty list
        try:
            blocks = manager.allocate_blocks_for_tokens(300)
            assert False, "Expected ValueError"
        except ValueError:
            pass

        # All blocks should be unchanged (no rollback needed since allocation failed)
        assert manager.free_blocks == initial_free


class TestHashBasedDeduplication:
    """Test hash-based deduplication."""

    def test_compute_block_hash(self):
        """Test hash computation."""
        from vmlx_engine.paged_cache import PagedCacheManager

        tokens1 = [1, 2, 3, 4, 5]
        tokens2 = [1, 2, 3, 4, 5]
        tokens3 = [1, 2, 3, 4, 6]

        hash1 = PagedCacheManager.compute_block_hash(tokens1)
        hash2 = PagedCacheManager.compute_block_hash(tokens2)
        hash3 = PagedCacheManager.compute_block_hash(tokens3)

        assert hash1 == hash2  # Same tokens = same hash
        assert hash1 != hash3  # Different tokens = different hash
        assert len(hash1) == 16  # 16 char hex string

    def test_find_cached_block(self):
        """Test finding cached block by tokens."""
        from vmlx_engine.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=10)

        tokens = list(range(64))

        # Initially not found
        result = manager.find_cached_block(tokens)
        assert result is None

        # Register a block
        block = manager.allocate_block()
        manager.register_block_hash(block, tokens)

        # Now should find it
        result = manager.find_cached_block(tokens)
        assert result is not None
        assert result.block_id == block.block_id


class TestBlockTableManagement:
    """Test block table management."""

    def test_create_block_table(self):
        """Test creating a block table."""
        from vmlx_engine.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=10)

        table = manager.create_block_table("req-1")
        assert table.request_id == "req-1"
        assert "req-1" in manager.request_tables

    def test_get_block_table(self):
        """Test getting a block table."""
        from vmlx_engine.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=10)

        manager.create_block_table("req-1")

        table = manager.get_block_table("req-1")
        assert table is not None
        assert table.request_id == "req-1"

        # Non-existent table
        assert manager.get_block_table("req-999") is None

    def test_delete_block_table(self):
        """Test deleting a block table frees blocks."""
        from vmlx_engine.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=10)
        # Initial: 9 free (10 - 1 null block), 1 allocated (null block)

        table = manager.create_block_table("req-1")
        block1 = manager.allocate_block()
        block2 = manager.allocate_block()
        manager.add_block_to_table(table, block1, 64)
        manager.add_block_to_table(table, block2, 64)

        assert len(manager.allocated_blocks) == 3  # null block + 2

        manager.delete_block_table("req-1")

        assert "req-1" not in manager.request_tables
        assert len(manager.allocated_blocks) == 1  # only null block remains
        assert manager.free_blocks == 9  # all non-null blocks free


class TestPrefixSharing:
    """Test prefix sharing functionality."""

    def test_find_shared_prefix_no_cache(self):
        """Test finding shared prefix with empty cache."""
        from vmlx_engine.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=10)

        tokens = list(range(200))
        shared_blocks, remaining = manager.find_shared_prefix(tokens)

        assert len(shared_blocks) == 0
        assert remaining == tokens

    def test_find_shared_prefix_with_cache(self):
        """Test finding shared prefix with cached blocks."""
        from vmlx_engine.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=10)

        # Cache the first block
        first_block_tokens = list(range(64))
        block = manager.allocate_block()
        block.token_count = 64
        manager.register_block_hash(block, first_block_tokens)

        # Search with tokens that start with cached prefix
        tokens = list(range(128))  # 64 cached + 64 new
        shared_blocks, remaining = manager.find_shared_prefix(tokens)

        assert len(shared_blocks) == 1
        assert shared_blocks[0] == block.block_id
        assert remaining == list(range(64, 128))

    def test_fork_block_table(self):
        """Test forking a block table (COW)."""
        from vmlx_engine.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=10)

        # Create source table with blocks
        source_table = manager.create_block_table("req-1")
        block1 = manager.allocate_block()
        block2 = manager.allocate_block()
        manager.add_block_to_table(source_table, block1, 64)
        manager.add_block_to_table(source_table, block2, 64)

        # Fork to new request
        forked_table = manager.fork_block_table(source_table, "req-2")

        assert forked_table.request_id == "req-2"
        assert forked_table.block_ids == source_table.block_ids
        assert forked_table.num_tokens == source_table.num_tokens

        # Blocks should now have ref_count = 2
        assert block1.ref_count == 2
        assert block2.ref_count == 2


class TestCopyOnWrite:
    """Test Copy-on-Write functionality."""

    def test_get_blocks_no_cow_needed(self):
        """Test getting blocks when no COW is needed."""
        from vmlx_engine.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=10)

        table = manager.create_block_table("req-1")
        block = manager.allocate_block()
        manager.add_block_to_table(table, block, 64)

        blocks, was_copied = manager.get_blocks_for_generation(table)

        assert len(blocks) == 1
        assert was_copied is False
        assert blocks[0].block_id == block.block_id

    def test_get_blocks_with_cow(self):
        """Test getting blocks triggers COW for shared blocks."""
        from vmlx_engine.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=10)

        # Create and fork table
        source_table = manager.create_block_table("req-1")
        block = manager.allocate_block()
        manager.add_block_to_table(source_table, block, 64)

        forked_table = manager.fork_block_table(source_table, "req-2")
        assert block.ref_count == 2

        # Get blocks for forked table - should trigger COW
        blocks, was_copied = manager.get_blocks_for_generation(forked_table)

        assert len(blocks) == 1
        assert was_copied is True
        assert blocks[0].block_id != block.block_id  # New block created
        assert block.ref_count == 1  # Original block ref decreased

        stats = manager.get_stats()
        assert stats.cow_copies == 1


class TestEviction:
    """Test LRU eviction."""

    def test_evict_lru_blocks(self):
        """Test LRU eviction."""
        from vmlx_engine.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=5)
        # With null block, we have 4 allocatable blocks

        # Allocate all blocks
        blocks = []
        for _ in range(4):  # 4 allocatable (5 - 1 null block)
            block = manager.allocate_block()
            block.token_count = 64
            blocks.append(block)
            time.sleep(0.01)  # Ensure different timestamps

        assert manager.free_blocks == 0

        # Free 2 blocks first (they go to free queue)
        manager.free_block(blocks[0].block_id)
        manager.free_block(blocks[1].block_id)
        assert manager.free_blocks == 2

        # Now evict_lru_blocks rotates them to clear cache data
        evicted = manager.evict_lru_blocks(2)

        assert evicted == 2
        assert manager.free_blocks == 2
        assert len(manager.allocated_blocks) == 3  # null block + 2 remaining

    def test_handle_memory_pressure(self):
        """Test handling memory pressure."""
        from vmlx_engine.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=5)
        # With null block, we have 4 allocatable blocks

        # Allocate 3 blocks
        allocated = []
        for _ in range(3):
            block = manager.allocate_block()
            block.token_count = 64
            allocated.append(block)

        assert manager.free_blocks == 1  # 4 - 3 = 1

        # Free 2 blocks to put them in free queue (they can be evicted from cache)
        manager.free_block(allocated[0].block_id)
        manager.free_block(allocated[1].block_id)
        assert manager.free_blocks == 3

        # Request 3 blocks - should already have enough
        result = manager.handle_memory_pressure(3)

        assert result is True
        assert manager.free_blocks >= 3


class TestStatistics:
    """Test statistics and monitoring."""

    def test_get_memory_usage(self):
        """Test memory usage reporting."""
        from vmlx_engine.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=100)
        # Initial: 99 free (100 - 1 null block), 1 allocated (null block)

        # Allocate 25 blocks
        for _ in range(25):
            block = manager.allocate_block()
            block.token_count = 64

        usage = manager.get_memory_usage()

        assert usage["block_size"] == 64
        assert usage["max_blocks"] == 100
        assert usage["usable_blocks"] == 99
        assert usage["capacity_tokens"] == 99 * 64
        assert usage["allocated_blocks"] == 26  # null block + 25
        assert usage["free_blocks"] == 74  # 99 - 25
        assert usage["utilization"] == pytest.approx(25 / 99)
        assert usage["total_tokens_cached"] == 25 * 64

    def test_reserved_null_block_is_zero_utilization(self):
        """An idle pool must not report the reserved null block as user cache."""
        from vmlx_engine.paged_cache import PagedCacheManager

        usage = PagedCacheManager(block_size=64, max_blocks=4).get_memory_usage()

        assert usage["usable_blocks"] == 3
        assert usage["capacity_tokens"] == 192
        assert usage["allocated_blocks"] == 1
        assert usage["utilization"] == 0.0

    def test_reset_stats(self):
        """Test resetting statistics."""
        from vmlx_engine.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=10)

        # Generate some stats
        manager.find_cached_block([1, 2, 3])  # Cache miss
        manager.stats.cow_copies = 5

        manager.reset_stats()

        assert manager.stats.cache_hits == 0
        assert manager.stats.cache_misses == 0
        assert manager.stats.cow_copies == 0

    def test_clear(self):
        """Test clearing all cache."""
        from vmlx_engine.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=10)

        # Allocate and populate
        table = manager.create_block_table("req-1")
        block = manager.allocate_block()
        manager.add_block_to_table(table, block, 64)

        # A finished request drops its refs; blocks stay cached at ref 0. clear()
        # refuses while a request is still LIVE, because rebuilding the pool
        # recycles its block ids into another request's table.
        manager.release_request_refs(table)
        assert manager.clear() is True

        # After clear, null block is re-reserved
        assert manager.free_blocks == 9  # 10 - 1 null block
        assert len(manager.allocated_blocks) == 1  # only null block
        assert len(manager.request_tables) == 0
        assert len(manager.hash_to_block) == 0


class TestThreadSafety:
    """Test thread safety."""

    def test_concurrent_allocation(self):
        """Test concurrent block allocation."""
        import threading
        from vmlx_engine.paged_cache import PagedCacheManager

        # Use 101 blocks so we have 100 allocatable (after null block)
        manager = PagedCacheManager(block_size=64, max_blocks=101)
        results = []
        errors = []

        def allocate_blocks():
            try:
                for _ in range(10):
                    block = manager.allocate_block()
                    if block:
                        results.append(block.block_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=allocate_blocks) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 50
        assert len(set(results)) == 50  # All unique block IDs


# =============================================================================
# BlockAwarePrefixCache Tests
# =============================================================================


class TestBlockAwarePrefixCache:
    """Test BlockAwarePrefixCache class."""

    def test_initialization(self):
        """Test cache initialization."""
        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        paged_manager = PagedCacheManager(block_size=64, max_blocks=100)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

        assert cache.block_size == 64
        assert len(cache) == 0

    def test_store_and_fetch_cache(self):
        """Test storing and fetching cache."""
        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        paged_manager = PagedCacheManager(block_size=64, max_blocks=100)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

        # Store cache for first request
        tokens1 = list(range(128))  # 2 blocks worth
        cache_data1 = ["cache_data_1"]
        block_table = cache.store_cache("req-1", tokens1, cache_data1)

        assert block_table is not None
        assert block_table.num_tokens == 128
        assert len(block_table.block_ids) == 2

        # Fetch cache for second request with same prefix
        block_table2, remaining = cache.fetch_cache("req-2", tokens1 + [999, 1000])

        # Should hit the prefix
        assert remaining == [999, 1000]

    def test_fetched_block_table_is_registered_for_completion_ref_release(self):
        """A paged hit must not leave one permanent request ref behind.

        Live agent loops fetch a prefix, generate, and then store the refreshed
        prompt snapshot under the same request ID.  Scheduler completion cleanup
        finds the fetched ownership through ``_request_tables``.  If fetch does
        not register its table, every hit pins its blocks and a small pool can no
        longer evict or allocate after the first tool iteration.
        """
        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        paged = PagedCacheManager(block_size=4, max_blocks=4)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged)
        tokens = list(range(12))

        stored = cache.store_cache("write", tokens, ["cache-data"])
        assert stored is not None

        # Mirror scheduler post-store cleanup: keep the hash entries, release
        # the request ownership, and make all three usable blocks reclaimable.
        write_entry = cache._request_tables.pop("write")
        assert paged.release_request_refs(write_entry.block_table) == 3
        paged.detach_request("write")
        assert [paged.blocks[i].ref_count for i in stored.block_ids] == [0, 0, 0]

        hit, remaining = cache.fetch_cache("read", tokens + [99])

        assert hit is not None
        assert remaining == [99]
        assert cache._request_tables["read"].block_table is hit
        assert [paged.blocks[i].ref_count for i in hit.block_ids] == [1, 1, 1]

        read_entry = cache._request_tables.pop("read")
        assert paged.release_request_refs(read_entry.block_table) == 3
        paged.detach_request("read")
        assert [paged.blocks[i].ref_count for i in hit.block_ids] == [0, 0, 0]
        assert paged.free_block_queue.num_free_blocks == 3

    def test_fetch_prefers_exact_partial_prefix_over_shorter_block_hit(self):
        """A cached terminal partial block must win over a shorter full-block hit.

        Hybrid SSM models restore companion state by absolute prompt boundary.
        If a 70-token prompt was cached as one full block plus a 6-token
        terminal partial, then a later request with that prompt plus a long
        tail must resume at 70, not at 64. Resuming at the shorter boundary
        can force the model through a different warm pass and break recall
        even though cached_tokens is non-zero.
        """
        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        paged_manager = PagedCacheManager(block_size=64, max_blocks=100)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

        prefix_tokens = list(range(70))
        long_tail = list(range(1000, 1070))
        cache.store_cache("req-1", prefix_tokens, ["cache_data_1"])

        block_table, remaining = cache.fetch_cache("req-2", prefix_tokens + long_tail)

        assert block_table is not None
        assert block_table.num_tokens == len(prefix_tokens)
        assert remaining == long_tail

    def test_prefix_index_rejects_block_ids_reused_for_another_chain(self):
        """A fallback prefix entry must not follow recycled paged block IDs."""
        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        paged = PagedCacheManager(block_size=4, max_blocks=3)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged)
        original_tokens = [1, 2, 3, 4, 5, 6]

        original = cache.store_cache(
            "original",
            original_tokens,
            ["original-state"],
        )
        assert original is not None
        original_block_ids = list(original.block_ids)
        assert cache._prefix_index_blocks_are_current(
            original_tokens,
            original_block_ids,
        )

        terminal = paged.allocated_blocks[original_block_ids[-1]]
        original_terminal_count = terminal.token_count
        terminal.token_count = original_terminal_count - 1
        assert not cache._prefix_index_blocks_are_current(
            original_tokens,
            original_block_ids,
        )
        terminal.token_count = original_terminal_count

        original_entry = cache._request_tables.pop("original")
        assert paged.release_request_refs(original_entry.block_table) == 2
        paged.detach_request("original")

        replacement = cache.store_cache(
            "replacement",
            [20, 21, 22, 23, 24, 25],
            ["replacement-state"],
        )
        assert replacement is not None
        assert set(replacement.block_ids) == set(original_block_ids)
        assert not cache._prefix_index_blocks_are_current(
            original_tokens,
            original_block_ids,
        )

        requested = original_tokens + [99]
        hit, remaining = cache.fetch_cache("reader", requested)

        assert hit is None
        assert remaining == requested
        assert (
            cache._prefix_index_hash(original_tokens)
            not in cache._prefix_index
        )

    def test_prefix_index_validation_pins_before_concurrent_reallocation(
        self,
        monkeypatch,
    ):
        """Validation and ref pinning must share one paged-cache lock hold."""
        from vmlx_engine.paged_cache import BlockTable, PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        paged = PagedCacheManager(block_size=4, max_blocks=3)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged)
        tokens = [1, 2, 3, 4, 5, 6]
        stored = cache.store_cache("writer", tokens, ["state"])
        assert stored is not None
        block_ids = list(stored.block_ids)

        writer_entry = cache._request_tables.pop("writer")
        assert paged.release_request_refs(writer_entry.block_table) == 2
        paged.detach_request("writer")
        assert [paged.blocks[block_id].ref_count for block_id in block_ids] == [0, 0]

        validation_entered = threading.Event()
        allow_validation_to_return = threading.Event()
        contender_started = threading.Event()
        contender_acquired_lock = threading.Event()
        results = {}
        original_validator = cache._prefix_index_blocks_are_current

        def blocking_validator(
            cached_tokens,
            candidate_block_ids,
            *,
            cache_extra_keys=None,
        ):
            valid = original_validator(
                cached_tokens,
                candidate_block_ids,
                cache_extra_keys=cache_extra_keys,
            )
            validation_entered.set()
            if not allow_validation_to_return.wait(timeout=2):
                raise AssertionError("timed out waiting to release validation")
            return valid

        monkeypatch.setattr(
            cache,
            "_prefix_index_blocks_are_current",
            blocking_validator,
        )

        def lookup():
            try:
                results["match"] = cache._find_best_prefix_match(tokens + [99])
            except BaseException as exc:  # pragma: no cover - surfaced below
                results["lookup_error"] = exc

        def contend_for_reallocation():
            contender_started.set()
            with paged._lock:
                contender_acquired_lock.set()
                results["refs_seen_by_contender"] = [
                    paged.blocks[block_id].ref_count for block_id in block_ids
                ]

        lookup_thread = threading.Thread(target=lookup)
        lookup_thread.start()
        assert validation_entered.wait(timeout=2)

        contender_thread = threading.Thread(target=contend_for_reallocation)
        contender_thread.start()
        assert contender_started.wait(timeout=2)
        assert not contender_acquired_lock.wait(timeout=0.05)

        allow_validation_to_return.set()
        lookup_thread.join(timeout=2)
        contender_thread.join(timeout=2)

        assert not lookup_thread.is_alive()
        assert not contender_thread.is_alive()
        assert "lookup_error" not in results
        assert results["match"] == (tokens, block_ids)
        assert results["refs_seen_by_contender"] == [1, 1]

        assert paged.release_request_refs(
            BlockTable(
                request_id="reader",
                block_ids=block_ids,
                num_tokens=len(tokens),
            )
        ) == 2

    def test_stale_prefix_cleanup_does_not_delete_concurrent_replacement(
        self,
        monkeypatch,
    ):
        """A serialized fresh index write must survive stale-entry cleanup."""
        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        paged = PagedCacheManager(block_size=4, max_blocks=3)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged)
        stale_tokens = [1, 2, 3, 4, 5, 6]
        stored = cache.store_cache("stale", stale_tokens, ["stale-state"])
        assert stored is not None
        # Ask for the key through the accessor that honours the active
        # derivation scheme; calling _prefix_index_hash directly computes the
        # from-scratch key, which is not what the writer stored when
        # VMLX_CHAINED_PREFIX_INDEX_HASH is on.
        stale_key = cache._prefix_index_key(stale_tokens)
        stale_entry = cache._prefix_index[stale_key]

        stale_request = cache._request_tables.pop("stale")
        assert paged.release_request_refs(stale_request.block_table) == 2
        paged.detach_request("stale")
        replacement = cache.store_cache(
            "replacement-blocks",
            [20, 21, 22, 23, 24, 25],
            ["replacement-state"],
        )
        assert replacement is not None
        assert not cache._prefix_index_blocks_are_current(
            stale_tokens,
            list(stale_entry[1]),
        )

        validation_entered = threading.Event()
        allow_cleanup = threading.Event()
        writer_started = threading.Event()
        writer_finished = threading.Event()
        results = {}
        original_validator = cache._prefix_index_blocks_are_current
        fresh_entry = (
            list(stale_tokens),
            list(replacement.block_ids),
            "fresh-replacement-sentinel",
        )

        def blocking_validator(
            cached_tokens,
            candidate_block_ids,
            *,
            cache_extra_keys=None,
        ):
            valid = original_validator(
                cached_tokens,
                candidate_block_ids,
                cache_extra_keys=cache_extra_keys,
            )
            if list(cached_tokens) == stale_tokens:
                validation_entered.set()
                if not allow_cleanup.wait(timeout=2):
                    raise AssertionError("timed out waiting to release cleanup")
            return valid

        monkeypatch.setattr(
            cache,
            "_prefix_index_blocks_are_current",
            blocking_validator,
        )

        def lookup():
            try:
                results["match"] = cache._find_best_prefix_match(
                    stale_tokens + [99]
                )
            except BaseException as exc:  # pragma: no cover - surfaced below
                results["lookup_error"] = exc

        def replace_index_entry():
            writer_started.set()
            with paged._lock:
                cache._prefix_index[stale_key] = fresh_entry
            writer_finished.set()

        lookup_thread = threading.Thread(target=lookup)
        lookup_thread.start()
        assert validation_entered.wait(timeout=2)

        writer_thread = threading.Thread(target=replace_index_entry)
        writer_thread.start()
        assert writer_started.wait(timeout=2)
        assert not writer_finished.wait(timeout=0.05)

        allow_cleanup.set()
        lookup_thread.join(timeout=2)
        writer_thread.join(timeout=2)

        assert not lookup_thread.is_alive()
        assert not writer_thread.is_alive()
        assert "lookup_error" not in results
        assert results["match"] is None
        assert cache._prefix_index[stale_key] is fresh_entry

    def test_restart_restores_short_partial_prefix_for_longer_prompt(self):
        """L2 must discover a short root partial after process restart."""
        from vmlx_engine.paged_cache import PagedCacheManager, compute_block_hash
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        partial_tokens = [10, 11, 12]
        partial_hash = compute_block_hash(None, partial_tokens)

        class _Disk:
            def partial_token_counts(self, block_size):
                assert block_size == 4
                return [3]

            def read_block(self, block_hash):
                return ["disk-state"] if block_hash == partial_hash else None

            def write_block_async(self, *_args, **_kwargs):
                return None

        paged = PagedCacheManager(
            block_size=4,
            max_blocks=8,
            disk_store=_Disk(),
        )
        cache = BlockAwarePrefixCache(
            model=None,
            paged_cache_manager=paged,
        )

        table, remaining = cache.fetch_cache(
            "after-restart",
            partial_tokens + [13, 14, 15],
        )

        assert table is not None
        assert table.num_tokens == 3
        assert remaining == [13, 14, 15]
        assert paged.stats.disk_hits >= 1

    def test_disk_only_manager_requires_durable_store(self):
        from vmlx_engine.paged_cache import PagedCacheManager

        with pytest.raises(ValueError, match="disk_only requires a disk_store"):
            PagedCacheManager(block_size=4, max_blocks=8, disk_only=True)

    def test_paged_with_l2_defaults_to_resident_ram_tier(self, monkeypatch):
        """Block L2 must not silently turn Paged On into SSD-only caching."""
        from vmlx_engine.paged_cache import PagedCacheManager

        class _Disk:
            def partial_token_counts(self, _block_size):
                return []

        monkeypatch.delenv("VMLX_PAGED_FRUGAL", raising=False)
        manager = PagedCacheManager(
            block_size=4,
            max_blocks=8,
            disk_store=_Disk(),
        )

        assert manager.disk_only is False
        assert manager.paged_frugal is False
        assert manager.ram_mirror_policy == "resident"
        stats = manager.get_memory_usage()
        assert stats["paged_ram_enabled"] is True
        assert stats["paged_frugal"] is False
        assert stats["ram_mirror_policy"] == "resident"

    def test_explicit_frugal_override_and_disk_only_are_truthful(self, monkeypatch):
        from vmlx_engine.paged_cache import PagedCacheManager

        class _Disk:
            def partial_token_counts(self, _block_size):
                return []

        monkeypatch.setenv("VMLX_PAGED_FRUGAL", "1")
        frugal = PagedCacheManager(
            block_size=4,
            max_blocks=8,
            disk_store=_Disk(),
        )
        assert frugal.disk_only is False
        assert frugal.paged_frugal is True
        assert frugal.ram_mirror_policy == "frugal_env"

        monkeypatch.setenv("VMLX_PAGED_FRUGAL", "0")
        disk_only = PagedCacheManager(
            block_size=4,
            max_blocks=8,
            disk_store=_Disk(),
            disk_only=True,
        )
        assert disk_only.disk_only is True
        assert disk_only.paged_frugal is True
        assert disk_only.ram_mirror_policy == "disk_only"

    def test_disk_only_store_and_restart_restore_exact_partial_prefix(self, tmp_path):
        """SSD-only mode writes through, drops RAM payloads, and restores 7/8 tokens."""
        mx = pytest.importorskip("mlx.core")

        from vmlx_engine.block_disk_store import BlockDiskStore
        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        cache_dir = tmp_path / "disk-only-blocks"
        tokens = list(range(7))
        keys = mx.arange(14, dtype=mx.float32).reshape(1, 1, 7, 2)
        values = (keys + 100).astype(mx.float32)
        state = [{
            "class_name": "KVCache",
            "state": (keys, values),
            "meta_state": ("7",),
        }]

        store = BlockDiskStore(
            cache_dir=str(cache_dir),
            max_size_gb=0.01,
            expected_num_layers=1,
        )
        manager = PagedCacheManager(
            block_size=4,
            max_blocks=8,
            disk_store=store,
            max_resident_bytes=0,
            disk_only=True,
        )
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=manager)

        table = cache.store_cache("writer", tokens, state)
        assert table is not None
        assert table.num_tokens == 7
        assert all(
            manager.allocated_blocks[block_id].cache_data is None
            for block_id in table.block_ids
        )
        assert manager.resident_bytes == 0
        store.shutdown()

        restarted_store = BlockDiskStore(
            cache_dir=str(cache_dir),
            max_size_gb=0.01,
            expected_num_layers=1,
        )
        restarted_manager = PagedCacheManager(
            block_size=4,
            max_blocks=8,
            disk_store=restarted_store,
            max_resident_bytes=0,
            disk_only=True,
        )
        restarted = BlockAwarePrefixCache(
            model=None,
            paged_cache_manager=restarted_manager,
        )
        try:
            hit, remaining = restarted.fetch_cache("reader", tokens + [99])
            assert hit is not None
            assert hit.num_tokens == 7
            assert remaining == [99]
            rebuilt = restarted.reconstruct_cache(hit)
            assert rebuilt is not None
            assert len(rebuilt) == 1
            assert int(rebuilt[0].offset) == 7
            assert restarted_manager.stats.disk_hits >= 2
            public_stats = restarted.get_stats()
            store_stats = restarted_store.get_stats()
            assert public_stats["disk_counter_source"] == "block_disk_store"
            assert public_stats["disk_hits"] == store_stats["disk_hits"]
            assert public_stats["disk_misses"] == store_stats["disk_misses"]
            assert public_stats["disk_promotion_hits"] >= 2
            assert restarted_manager.resident_bytes == 0
            assert all(
                restarted_manager.allocated_blocks[block_id].cache_data is None
                for block_id in hit.block_ids
            )
        finally:
            restarted_store.shutdown()

    def test_known_plain_kv_disk_only_hit_reads_each_block_once(self, tmp_path):
        """Generic KV/hybrid-KV hits must not preload native-only validators.

        In SSD-only mode an in-process block remains indexed with
        ``cache_data=None`` after publication.  Running the DSV4, ZAYA and
        rotating-KV terminal validators unconditionally therefore loads the
        whole chain once, discards those validation payloads, and makes worker
        reconstruction load the same chain again.  Production passes the
        instantiated runtime cache kinds explicitly; a known ordinary KV lane
        must perform exactly one physical L2 read per reconstructed block.
        """
        mx = pytest.importorskip("mlx.core")

        from vmlx_engine.block_disk_store import BlockDiskStore
        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        tokens = list(range(8))
        keys = mx.arange(16, dtype=mx.float32).reshape(1, 1, 8, 2)
        values = (keys + 100).astype(mx.float32)
        state = [{
            "class_name": "KVCache",
            "state": (keys, values),
            "meta_state": ("8",),
        }]
        store = BlockDiskStore(
            cache_dir=str(tmp_path / "single-read-disk-only"),
            max_size_gb=0.01,
            expected_num_layers=1,
        )
        manager = PagedCacheManager(
            block_size=4,
            max_blocks=8,
            disk_store=store,
            max_resident_bytes=0,
            disk_only=True,
        )
        cache = BlockAwarePrefixCache(
            model=None,
            paged_cache_manager=manager,
            uses_dsv4_cache=False,
            uses_zaya_cache=False,
            mixed_attention_cache_model=False,
        )
        try:
            stored = cache.store_cache("writer", tokens, state)
            assert stored is not None
            entry = cache._request_tables.pop("writer")
            manager.release_request_refs(entry.block_table)
            manager.detach_request("writer")
            assert all(
                manager.allocated_blocks[block_id].cache_data is None
                for block_id in stored.block_ids
            )

            reads_before = store.get_stats()["disk_hits"]
            hit, remaining = cache.fetch_cache("reader", tokens + [99])
            assert hit is not None
            assert hit.num_tokens == len(tokens)
            assert remaining == [99]
            reads_after_fetch = store.get_stats()["disk_hits"]
            rebuilt = cache.reconstruct_cache(hit)
            assert rebuilt is not None
            reads_after = store.get_stats()["disk_hits"]

            assert reads_after_fetch - reads_before == 0
            assert reads_after - reads_after_fetch == len(hit.block_ids)
            assert cache._last_reconstruct_disk_blocks == len(hit.block_ids)
            assert manager.resident_bytes == 0
        finally:
            store.shutdown()

    def test_block_l2_reads_run_on_model_owner_executor(self, tmp_path):
        """API-thread L2 lookup must create MLX arrays on the model worker."""
        from concurrent.futures import ThreadPoolExecutor
        import threading

        from vmlx_engine.block_disk_store import BlockDiskStore

        store = BlockDiskStore(
            cache_dir=str(tmp_path / "owner-thread-blocks"),
            max_size_gb=0.01,
            expected_num_layers=1,
        )
        observed_threads = []

        def fake_read(_block_hash):
            observed_threads.append(threading.current_thread().name)
            return [("skip",)]

        store._read_block_impl = fake_read
        try:
            with ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="model-owner",
            ) as executor:
                store.set_load_executor(executor)

                assert store.read_block(b"a" * 32) == [("skip",)]
                assert executor.submit(
                    store.read_block,
                    b"b" * 32,
                ).result() == [("skip",)]

            assert observed_threads == [
                "model-owner_0",
                "model-owner_0",
            ]
        finally:
            store.shutdown()

    def test_disk_only_rotating_terminal_restores_across_restart(self, tmp_path):
        """SSD-only L2 preserves pending chain nodes plus the exact SWA window."""
        mx = pytest.importorskip("mlx.core")

        from vmlx_engine.block_disk_store import BlockDiskStore
        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        cache_dir = tmp_path / "disk-only-rotating-blocks"
        tokens = list(range(12))
        positions = mx.arange(4, 12, dtype=mx.float32).reshape(1, 1, 8, 1)
        state = [{
            "class_name": "RotatingKVCache",
            "state": (positions, positions + 100),
            "meta_state": (0, 8, 12, 8),
        }]

        store = BlockDiskStore(
            cache_dir=str(cache_dir),
            max_size_gb=0.01,
            expected_num_layers=1,
        )
        manager = PagedCacheManager(
            block_size=4,
            max_blocks=8,
            disk_store=store,
            max_resident_bytes=0,
            disk_only=True,
        )
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=manager)
        table = cache.store_cache("writer", tokens, state)
        assert table is not None
        assert table.num_tokens == 12
        store.shutdown()

        restarted_store = BlockDiskStore(
            cache_dir=str(cache_dir),
            max_size_gb=0.01,
            expected_num_layers=1,
        )
        restarted_manager = PagedCacheManager(
            block_size=4,
            max_blocks=8,
            disk_store=restarted_store,
            max_resident_bytes=0,
            disk_only=True,
        )
        restarted = BlockAwarePrefixCache(
            model=None,
            paged_cache_manager=restarted_manager,
        )
        try:
            hit, remaining = restarted.fetch_cache("reader", tokens + [99])
            assert hit is not None
            assert hit.num_tokens == 12
            assert remaining == [99]
            rebuilt = restarted.reconstruct_cache(hit)
            assert rebuilt is not None
            assert rebuilt[0].offset == 12
            assert rebuilt[0]._idx == 8
            assert mx.array_equal(
                rebuilt[0].keys.reshape(-1),
                mx.arange(4, 12, dtype=mx.float32),
            ).item()
            next_key = mx.array([[[[12.0]]]], dtype=mx.float32)
            next_value = mx.array([[[[112.0]]]], dtype=mx.float32)
            fetched_keys, fetched_values = rebuilt[0].update_and_fetch(
                next_key, next_value
            )
            mx.eval(fetched_keys, fetched_values)
            assert rebuilt[0].offset == 13
            assert restarted_manager.stats.disk_hits >= 3
        finally:
            restarted_store.shutdown()

    def test_rotating_cache_partial_window_hit_downgrades_to_miss(self):
        """Unsafe mixed-SWA partial hits must not restore impossible ring state.

        A long prompt can have a RotatingKVCache logical offset far beyond the
        sliding-window max_size while only the most recent physical window is
        materialized. If a later request matches only part of that tail, the
        block chain may contain fewer physical tokens than max_size but still
        carry offset=target_tokens. Restoring that state makes mlx-lm allocate
        ``max_size - offset`` elements on the next decode token, which is the
        live Electron Laguna failure:
        ``[full] Negative dimensions not allowed``.
        """
        mx = pytest.importorskip("mlx.core")

        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        tokens = list(range(256))
        keys = mx.arange(128, dtype=mx.float32).reshape(1, 1, 64, 2)
        values = (keys + 1000).astype(mx.float32)
        state = [{
            "class_name": "RotatingKVCache",
            "state": (keys, values),
            "meta_state": (0, 128, 256, 64),
        }]

        manager = PagedCacheManager(block_size=64, max_blocks=16)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=manager)
        table = cache.store_cache("writer", tokens, state)

        assert table is not None
        assert table.num_tokens == 256
        assert cache.reconstruct_cache(table) is None

    def test_rotating_cache_stores_exact_terminal_window(self):
        """Long mixed-SWA stores one complete terminal ring checkpoint."""
        mx = pytest.importorskip("mlx.core")

        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        tokens = list(range(256))
        # A clean chunked prefill may retain more than max_size physical tokens.
        # The terminal record must normalize it to the exact last 128 positions.
        positions = mx.arange(64, 256, dtype=mx.float32).reshape(1, 1, 192, 1)
        state = [{
            "class_name": "RotatingKVCache",
            "state": (positions, positions + 1000),
            "meta_state": (0, 128, 256, 192),
        }]

        manager = PagedCacheManager(block_size=64, max_blocks=16)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=manager)
        table = cache.store_cache("writer", tokens, state)

        assert table is not None
        assert table.num_tokens == 256
        entries = [
            manager.allocated_blocks[block_id].cache_data[0]
            for block_id in table.block_ids
        ]
        assert [entry[0] for entry in entries[:-2]] == [
            "rotating_kv_pending",
            "rotating_kv_pending",
        ]
        assert entries[-2][0] == "rotating_kv"
        assert entries[-2][3:] == (128, 0, 192, 128)
        assert entries[-1][0] == "rotating_kv"
        assert entries[-1][3:] == (128, 0, 256, 128)
        assert entries[-1][1].shape[2] == 128

        rebuilt = cache.reconstruct_cache(table)
        assert rebuilt is not None
        assert len(rebuilt) == 1
        assert rebuilt[0].offset == 256
        assert rebuilt[0]._idx == 128
        assert rebuilt[0].keys.shape[2] == 128
        assert mx.array_equal(
            rebuilt[0].keys.reshape(-1),
            mx.arange(128, 256, dtype=mx.float32),
        ).item()
        next_key = mx.array([[[[256.0]]]], dtype=mx.float32)
        next_value = mx.array([[[[1256.0]]]], dtype=mx.float32)
        fetched_keys, fetched_values = rebuilt[0].update_and_fetch(
            next_key, next_value
        )
        mx.eval(fetched_keys, fetched_values)
        assert rebuilt[0].offset == 257
        assert fetched_keys.shape[2] == 128
        assert float(fetched_keys[0, 0, rebuilt[0]._idx - 1, 0].item()) == 256.0

    @pytest.mark.parametrize("keep", [0, 2])
    def test_rotating_cache_changed_tail_reuses_previous_block_checkpoint(
        self,
        keep,
    ):
        """A changed final block resumes mixed-SWA at the shared boundary."""
        mx = pytest.importorskip("mlx.core")

        from mlx_lm.models.cache import RotatingKVCache
        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        def append(cache, positions):
            keys = mx.array(positions, dtype=mx.float32).reshape(1, 1, -1, 1)
            values = keys + 1000
            fetched = cache.update_and_fetch(keys, values)
            mx.eval(*fetched)
            return fetched

        # Match mlx-lm's real concat shape: after an 8-token saturated window,
        # appending four tokens retains positions 1..11 (max_size + S - 1).
        live = RotatingKVCache(max_size=8, keep=keep)
        append(live, list(range(8)))
        append(live, list(range(8, 12)))
        assert live.keys.shape[2] == 11

        tokens = list(range(12))
        state = [{
            "class_name": "RotatingKVCache",
            "state": live.state,
            "meta_state": live.meta_state,
        }]
        manager = PagedCacheManager(block_size=4, max_blocks=8)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=manager)
        table = cache.store_cache("writer", tokens, state)
        assert table is not None

        changed_tokens = list(range(8)) + [80, 81, 82, 83]
        hit, remaining = cache.fetch_cache("changed", changed_tokens)
        assert hit is not None
        assert hit.num_tokens == 8
        assert remaining == [80, 81, 82, 83]

        restored = cache.reconstruct_cache(hit)
        assert restored is not None
        assert restored[0].offset == 8
        assert restored[0]._idx == 8
        restored_fetch = append(restored[0], remaining)

        cold = RotatingKVCache(max_size=8, keep=keep)
        append(cold, list(range(8)))
        cold_fetch = append(cold, remaining)
        assert mx.array_equal(restored_fetch[0], cold_fetch[0]).item()
        assert mx.array_equal(restored_fetch[1], cold_fetch[1]).item()
        assert restored[0].offset == cold.offset == 12
        assert restored[0]._idx == cold._idx == 11

    def test_disk_only_rotating_changed_tail_restores_previous_block(self, tmp_path):
        """The preceding mixed-SWA checkpoint survives SSD-only restart."""
        mx = pytest.importorskip("mlx.core")

        from mlx_lm.models.cache import RotatingKVCache
        from vmlx_engine.block_disk_store import BlockDiskStore
        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        def append(cache, positions):
            keys = mx.array(positions, dtype=mx.float32).reshape(1, 1, -1, 1)
            values = keys + 1000
            fetched = cache.update_and_fetch(keys, values)
            mx.eval(*fetched)
            return fetched

        live = RotatingKVCache(max_size=8)
        append(live, list(range(8)))
        append(live, list(range(8, 12)))
        tokens = list(range(12))
        state = [{
            "class_name": "RotatingKVCache",
            "state": live.state,
            "meta_state": live.meta_state,
        }]
        cache_dir = tmp_path / "disk-only-rotating-partial"
        store = BlockDiskStore(
            cache_dir=str(cache_dir),
            max_size_gb=0.01,
            expected_num_layers=1,
        )
        manager = PagedCacheManager(
            block_size=4,
            max_blocks=8,
            disk_store=store,
            max_resident_bytes=0,
            disk_only=True,
        )
        writer = BlockAwarePrefixCache(model=None, paged_cache_manager=manager)
        assert writer.store_cache("writer", tokens, state) is not None
        store.shutdown()

        restarted_store = BlockDiskStore(
            cache_dir=str(cache_dir),
            max_size_gb=0.01,
            expected_num_layers=1,
        )
        restarted_manager = PagedCacheManager(
            block_size=4,
            max_blocks=8,
            disk_store=restarted_store,
            max_resident_bytes=0,
            disk_only=True,
        )
        restarted = BlockAwarePrefixCache(
            model=None,
            paged_cache_manager=restarted_manager,
        )
        try:
            changed_tokens = list(range(8)) + [80, 81, 82, 83]
            hit, remaining = restarted.fetch_cache("reader", changed_tokens)
            assert hit is not None
            assert hit.num_tokens == 8
            assert remaining == [80, 81, 82, 83]
            restored = restarted.reconstruct_cache(hit)
            assert restored is not None
            restored_fetch = append(restored[0], remaining)

            cold = RotatingKVCache(max_size=8)
            append(cold, list(range(8)))
            cold_fetch = append(cold, remaining)
            assert mx.array_equal(restored_fetch[0], cold_fetch[0]).item()
            assert mx.array_equal(restored_fetch[1], cold_fetch[1]).item()
            assert restarted_manager.stats.disk_hits >= 2
            assert restarted_manager.resident_bytes == 0
        finally:
            restarted_store.shutdown()

    @pytest.mark.parametrize("keep", [0, 4])
    def test_rotating_cache_changed_full_tail_reuses_second_checkpoint(
        self,
        keep,
    ):
        """A terminal partial must not hide the preceding shared full block.

        This reproduces the live Laguna boundary exactly: a 1,026-token stored
        prompt ended in a two-token partial block, while a changed full block
        matched only through token 960.  The final offset was therefore 66
        tokens beyond the 960 boundary -- just outside the former one-block
        checkpoint limit even though the exact rotating window was retained.
        """
        mx = pytest.importorskip("mlx.core")

        from mlx_lm.models.cache import RotatingKVCache
        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        def append(cache, positions):
            keys = mx.array(positions, dtype=mx.float32).reshape(1, 1, -1, 1)
            values = keys + 1000
            fetched = cache.update_and_fetch(keys, values)
            mx.eval(*fetched)
            return fetched

        live = RotatingKVCache(max_size=512, keep=keep)
        append(live, list(range(1026)))
        tokens = list(range(1026))
        state = [{
            "class_name": "RotatingKVCache",
            "state": live.state,
            "meta_state": live.meta_state,
        }]
        manager = PagedCacheManager(block_size=64, max_blocks=32)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=manager)
        table = cache.store_cache("writer", tokens, state)
        assert table is not None

        entries = [
            manager.allocated_blocks[block_id].cache_data[0]
            for block_id in table.block_ids
        ]
        assert all(entry[0] == "rotating_kv_pending" for entry in entries[:-3])
        assert [entry[0] for entry in entries[-3:]] == [
            "rotating_kv",
            "rotating_kv",
            "rotating_kv",
        ]
        assert entries[-3][3:] == (512, keep, 960, 512)

        changed_tail = list(range(10_000, 10_066))
        changed_tokens = list(range(960)) + changed_tail
        hit, remaining = cache.fetch_cache("changed", changed_tokens)
        assert hit is not None
        assert hit.num_tokens == 960
        assert remaining == changed_tail

        restored = cache.reconstruct_cache(hit)
        assert restored is not None
        assert type(restored[0]).__name__ == "RotatingKVCache"
        assert restored[0].offset == 960
        assert restored[0]._idx == 512
        restored_fetch = append(restored[0], remaining)

        cold = RotatingKVCache(max_size=512, keep=keep)
        append(cold, list(range(960)))
        cold_fetch = append(cold, remaining)
        assert mx.array_equal(restored_fetch[0], cold_fetch[0]).item()
        assert mx.array_equal(restored_fetch[1], cold_fetch[1]).item()

    def test_rotating_checkpoint_fanout_stays_bounded_to_two_blocks(self):
        """Do not duplicate a full rotating window into arbitrary old pages."""
        mx = pytest.importorskip("mlx.core")

        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        tokens = list(range(18))
        positions = mx.arange(18, dtype=mx.float32).reshape(1, 1, 18, 1)
        state = [{
            "class_name": "RotatingKVCache",
            "state": (positions, positions + 1000),
            "meta_state": (0, 16, 18, 18),
        }]
        manager = PagedCacheManager(block_size=4, max_blocks=8)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=manager)
        table = cache.store_cache("writer", tokens, state)
        assert table is not None

        tags = [
            manager.allocated_blocks[block_id].cache_data[0][0]
            for block_id in table.block_ids
        ]
        assert tags == [
            "rotating_kv_pending",
            "rotating_kv_pending",
            "rotating_kv",
            "rotating_kv",
            "rotating_kv",
        ]

    def test_rotating_cache_extension_uses_new_terminal_not_old_snapshot(self):
        """A longer chain must not combine rotating state from two snapshots."""
        mx = pytest.importorskip("mlx.core")

        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        def snapshot(offset):
            start = max(0, offset - 128)
            positions = mx.arange(start, offset, dtype=mx.float32).reshape(
                1, 1, offset - start, 1
            )
            return [{
                "class_name": "RotatingKVCache",
                "state": (positions, positions + 1000),
                "meta_state": (0, 128, offset, offset - start),
            }]

        manager = PagedCacheManager(block_size=64, max_blocks=24)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=manager)
        short_tokens = list(range(192))
        long_tokens = list(range(384))
        cache.store_cache("short", short_tokens, snapshot(192))

        hit, remaining = cache.fetch_cache("long", long_tokens)
        assert hit is not None
        assert hit.num_tokens == 192
        assert remaining == long_tokens[192:]

        table = cache.store_cache("long", long_tokens, snapshot(384))
        assert table is not None
        rebuilt = cache.reconstruct_cache(table)
        assert rebuilt is not None
        assert rebuilt[0].offset == 384
        assert rebuilt[0]._idx == 128
        assert mx.array_equal(
            rebuilt[0].keys.reshape(-1),
            mx.arange(256, 384, dtype=mx.float32),
        ).item()

    def test_rotating_cache_promotes_interior_block_to_terminal_checkpoint(self):
        """An interior match misses before a clean prefill promotes the boundary."""
        mx = pytest.importorskip("mlx.core")

        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        def snapshot(offset):
            start = max(0, offset - 128)
            positions = mx.arange(start, offset, dtype=mx.float32).reshape(
                1, 1, offset - start, 1
            )
            return [{
                "class_name": "RotatingKVCache",
                "state": (positions, positions + 1000),
                "meta_state": (0, 128, offset, offset - start),
            }]

        manager = PagedCacheManager(block_size=64, max_blocks=24)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=manager)
        long_tokens = list(range(384))
        short_tokens = long_tokens[:192]
        cache.store_cache("long", long_tokens, snapshot(384))

        hit, remaining = cache.fetch_cache("short", short_tokens)
        assert hit is None
        assert remaining == short_tokens
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 1
        assert stats["tokens_saved"] == 0
        assert "short" not in cache._request_tables

        # The clean miss computes a real 192-token state. Storing that state
        # promotes the formerly interior chain boundary to an exact terminal
        # RotatingKV checkpoint, after which a changed-tail request may reuse it.
        table = cache.store_cache("short", short_tokens, snapshot(192))
        assert table is not None
        rebuilt = cache.reconstruct_cache(table)
        assert rebuilt is not None
        assert rebuilt[0].offset == 192
        assert rebuilt[0]._idx == 128
        assert mx.array_equal(
            rebuilt[0].keys.reshape(-1),
            mx.arange(64, 192, dtype=mx.float32),
        ).item()

        promoted_hit, promoted_remaining = cache.fetch_cache(
            "short-changed-tail",
            short_tokens + [999],
        )
        assert promoted_hit is not None
        assert promoted_hit.num_tokens == 192
        assert promoted_remaining == [999]

    def test_disk_only_mixed_swa_pending_boundary_is_fetch_time_miss(
        self,
        tmp_path,
    ):
        """A 12-full/36-SWA L2 chain must not report an unusable partial hit."""
        mx = pytest.importorskip("mlx.core")

        from vmlx_engine.block_disk_store import BlockDiskStore
        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        def mixed_snapshot(offset):
            full = mx.arange(offset, dtype=mx.float32).reshape(1, 1, offset, 1)
            rotating = mx.arange(
                max(0, offset - 128),
                offset,
                dtype=mx.float32,
            ).reshape(1, 1, min(offset, 128), 1)
            return [
                {
                    "class_name": "KVCache",
                    "state": (full, full + 1000),
                    "meta_state": (),
                }
                for _ in range(12)
            ] + [
                {
                    "class_name": "RotatingKVCache",
                    "state": (rotating, rotating + 1000),
                    "meta_state": (0, 128, offset, min(offset, 128)),
                }
                for _ in range(36)
            ]

        cache_dir = tmp_path / "disk-only-mixed-swa-pending-boundary"
        long_tokens = list(range(384))
        short_tokens = long_tokens[:192]
        store = BlockDiskStore(
            cache_dir=str(cache_dir),
            max_size_gb=0.05,
            expected_num_layers=48,
        )
        manager = PagedCacheManager(
            block_size=64,
            max_blocks=16,
            disk_store=store,
            max_resident_bytes=0,
            disk_only=True,
        )
        writer = BlockAwarePrefixCache(model=None, paged_cache_manager=manager)
        assert writer.store_cache(
            "writer",
            long_tokens,
            mixed_snapshot(len(long_tokens)),
        ) is not None
        store.shutdown()

        restarted_store = BlockDiskStore(
            cache_dir=str(cache_dir),
            max_size_gb=0.05,
            expected_num_layers=48,
        )
        restarted_manager = PagedCacheManager(
            block_size=64,
            max_blocks=16,
            disk_store=restarted_store,
            max_resident_bytes=0,
            disk_only=True,
        )
        restarted = BlockAwarePrefixCache(
            model=None,
            paged_cache_manager=restarted_manager,
        )
        try:
            hit, remaining = restarted.fetch_cache("reader", short_tokens)
            assert hit is None
            assert remaining == short_tokens
            stats = restarted.get_stats()
            assert stats["hits"] == 0
            assert stats["misses"] == 1
            assert stats["tokens_saved"] == 0
            assert stats["active_requests"] == 0
        finally:
            restarted_store.shutdown()

    @pytest.mark.parametrize("read_len", [320, 300, 301, 255, 193])
    def test_rotating_pending_boundary_walks_back_to_an_earlier_anchor(
        self,
        tmp_path,
        read_len,
    ):
        """A pending boundary must trim to the newest exact anchor, not go cold.

        The store keeps exact ``rotating_kv`` records only for a stored
        prompt's terminal cluster; older boundaries are deliberately
        ``rotating_kv_pending``.  A multi-turn follow-up therefore matches FAR
        from any terminal, lands on a pending marker, and used to discard the
        WHOLE candidate -- going cold even though an earlier turn left an exact
        anchor in the very same chain.

        Real shape reproduced here: turn 1 stores 192 tokens (exact anchor at
        192), turn 2 stores 384 (192 is now interior), then a follow-up matches
        past 192.  Read lengths deliberately include values that are NOT
        multiples of the 64-token block size -- an aligned-only matrix hides
        this whole class of defect by construction.
        """
        mx = pytest.importorskip("mlx.core")

        from vmlx_engine.block_disk_store import BlockDiskStore
        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        def mixed_snapshot(offset):
            full = mx.arange(offset, dtype=mx.float32).reshape(1, 1, offset, 1)
            rotating = mx.arange(
                max(0, offset - 128),
                offset,
                dtype=mx.float32,
            ).reshape(1, 1, min(offset, 128), 1)
            return [
                {
                    "class_name": "KVCache",
                    "state": (full, full + 1000),
                    "meta_state": (),
                }
                for _ in range(12)
            ] + [
                {
                    "class_name": "RotatingKVCache",
                    "state": (rotating, rotating + 1000),
                    "meta_state": (0, 128, offset, min(offset, 128)),
                }
                for _ in range(36)
            ]

        cache_dir = tmp_path / "rotating-anchor-walk"
        tokens = list(range(384))
        store = BlockDiskStore(
            cache_dir=str(cache_dir),
            max_size_gb=0.05,
            expected_num_layers=48,
        )
        manager = PagedCacheManager(
            block_size=64,
            max_blocks=32,
            disk_store=store,
            max_resident_bytes=0,
            disk_only=True,
        )
        writer = BlockAwarePrefixCache(model=None, paged_cache_manager=manager)
        assert writer.store_cache("turn1", tokens[:192], mixed_snapshot(192))
        assert writer.store_cache("turn2", tokens, mixed_snapshot(384))
        store.shutdown()

        restarted_store = BlockDiskStore(
            cache_dir=str(cache_dir),
            max_size_gb=0.05,
            expected_num_layers=48,
        )
        restarted_manager = PagedCacheManager(
            block_size=64,
            max_blocks=32,
            disk_store=restarted_store,
            max_resident_bytes=0,
            disk_only=True,
        )
        restarted = BlockAwarePrefixCache(
            model=None,
            paged_cache_manager=restarted_manager,
        )
        try:
            hit, remaining = restarted.fetch_cache("reader", tokens[:read_len])
            assert hit is not None, (
                "pending boundary went cold instead of trimming back to the "
                "exact 192-token anchor left by turn 1"
            )
            assert hit.num_tokens == 192
            assert len(remaining) == read_len - 192
            assert remaining == tokens[192:read_len]
            assert restarted.get_stats()["tokens_saved"] == 192
        finally:
            restarted_store.shutdown()

    def test_rotating_pending_boundary_with_no_anchor_still_goes_cold(
        self,
        tmp_path,
    ):
        """The walk must not invent an anchor where the chain has none.

        Same construction WITHOUT the earlier short store, so every boundary
        below the terminal is pending.  Trimming to a boundary that cannot be
        reconstructed would restore a wrong SWA window and yield
        plausible-but-wrong output, which is far worse than a clean re-prefill.
        """
        mx = pytest.importorskip("mlx.core")

        from vmlx_engine.block_disk_store import BlockDiskStore
        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        def mixed_snapshot(offset):
            full = mx.arange(offset, dtype=mx.float32).reshape(1, 1, offset, 1)
            rotating = mx.arange(
                max(0, offset - 128),
                offset,
                dtype=mx.float32,
            ).reshape(1, 1, min(offset, 128), 1)
            return [
                {
                    "class_name": "KVCache",
                    "state": (full, full + 1000),
                    "meta_state": (),
                }
                for _ in range(12)
            ] + [
                {
                    "class_name": "RotatingKVCache",
                    "state": (rotating, rotating + 1000),
                    "meta_state": (0, 128, offset, min(offset, 128)),
                }
                for _ in range(36)
            ]

        cache_dir = tmp_path / "rotating-anchor-walk-none"
        tokens = list(range(384))
        store = BlockDiskStore(
            cache_dir=str(cache_dir),
            max_size_gb=0.05,
            expected_num_layers=48,
        )
        manager = PagedCacheManager(
            block_size=64,
            max_blocks=32,
            disk_store=store,
            max_resident_bytes=0,
            disk_only=True,
        )
        writer = BlockAwarePrefixCache(model=None, paged_cache_manager=manager)
        assert writer.store_cache("only", tokens, mixed_snapshot(384))
        store.shutdown()

        restarted_store = BlockDiskStore(
            cache_dir=str(cache_dir),
            max_size_gb=0.05,
            expected_num_layers=48,
        )
        restarted_manager = PagedCacheManager(
            block_size=64,
            max_blocks=32,
            disk_store=restarted_store,
            max_resident_bytes=0,
            disk_only=True,
        )
        restarted = BlockAwarePrefixCache(
            model=None,
            paged_cache_manager=restarted_manager,
        )
        try:
            hit, remaining = restarted.fetch_cache("reader", tokens[:192])
            assert hit is None
            assert remaining == tokens[:192]
            assert restarted.get_stats()["misses"] == 1
            assert restarted.get_stats()["tokens_saved"] == 0
        finally:
            restarted_store.shutdown()

    @staticmethod
    def _mixed_swa_snapshot(mx, offset, *, seq_len=None):
        """One KVCache + one RotatingKVCache layer (max_size=128, keep=0).

        ``offset`` is the ring's logical offset; ``seq_len`` (default
        ``offset``) is the positional mirror length for the full-attention
        layer, so a post-generation snapshot can be modeled by passing a ring
        offset larger than the stored token count.
        """
        if seq_len is None:
            seq_len = offset
        full = mx.arange(seq_len, dtype=mx.float32).reshape(1, 1, seq_len, 1)
        rotating = mx.arange(
            max(0, offset - 128),
            offset,
            dtype=mx.float32,
        ).reshape(1, 1, min(offset, 128), 1)
        return [
            {
                "class_name": "KVCache",
                "state": (full, full + 1000),
                "meta_state": (),
            },
            {
                "class_name": "RotatingKVCache",
                "state": (rotating, rotating + 1000),
                "meta_state": (0, 128, offset, min(offset, 128)),
            },
        ]

    def test_rotating_partial_terminal_block_stores_and_restores_exact_anchor(
        self,
    ):
        """A terminal PARTIAL block is a valid mixed-SWA store boundary.

        206 = 3*64 + 14 — the live defect shape (2638 = 41*64 + 14) scaled
        down.  Every earlier rotating test used lengths divisible by the
        block size, where a terminal partial cannot occur BY CONSTRUCTION.
        A clean snapshot whose ring offset equals the store key must produce
        an exact terminal anchor inside the partial block, and the follow-up
        turn must restore all 206 tokens from it.
        """
        mx = pytest.importorskip("mlx.core")

        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        tokens = list(range(206))
        manager = PagedCacheManager(block_size=64, max_blocks=16)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=manager)
        table = cache.store_cache(
            "writer", tokens, self._mixed_swa_snapshot(mx, 206)
        )
        assert table is not None
        assert table.num_tokens == 206
        terminal = manager.allocated_blocks[table.block_ids[-1]]
        assert terminal.token_count == 14
        assert terminal.cache_data[1][0] == "rotating_kv"
        assert int(terminal.cache_data[1][5]) == 206

        hit, remaining = cache.fetch_cache("reader", tokens + [901, 902, 903])
        assert hit is not None
        assert hit.num_tokens == 206
        assert remaining == [901, 902, 903]
        rebuilt = cache.reconstruct_cache(hit)
        assert rebuilt is not None
        assert rebuilt[0].offset == 206
        assert rebuilt[1].offset == 206
        assert rebuilt[1]._idx == 128
        assert mx.array_equal(
            rebuilt[1].keys.reshape(-1),
            mx.arange(78, 206, dtype=mx.float32),
        ).item()

    def test_warm_extension_with_rolled_ring_walks_back_to_exact_anchor(self):
        """A warm-store extension past the ring must trim, not restore nothing.

        Live gemma4 turn 3: a WARM media turn has no clean boundary capture,
        so the scheduler stores the POST-GENERATION extracted cache under the
        pre-generation N-1 key (2638 = 41*64 + 14).  The ring has rolled past
        every appended boundary, so each new interior block goes pending and
        the terminal cut fails outright.  The old cutters wrote a bare
        ("skip",) terminal — invisible to the fetch-side guard — so the next
        turn credited the full 2638-token hit and reconstructed an EMPTY
        cache.  The terminal must be stored as rotating_kv_pending and the
        fetch must walk back to the last exact anchor left by the cold turn.
        """
        mx = pytest.importorskip("mlx.core")

        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        tokens = list(range(398))  # 6*64 + 14 — terminal partial of 14
        manager = PagedCacheManager(block_size=64, max_blocks=32)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=manager)
        # Cold turn: clean snapshot at the store boundary — exact anchor @192.
        assert cache.store_cache(
            "turn1", tokens[:192], self._mixed_swa_snapshot(mx, 192)
        ) is not None
        # Warm turn: ring already rolled to 430 (as if 32 tokens were
        # generated past the 398-token store key).
        table2 = cache.store_cache(
            "turn2",
            tokens,
            self._mixed_swa_snapshot(mx, 430, seq_len=430),
        )
        assert table2 is not None
        assert table2.num_tokens == 398
        terminal = manager.allocated_blocks[table2.block_ids[-1]]
        assert terminal.token_count == 14
        # Pins the cutter contract: a terminal the snapshot cannot cut is a
        # VISIBLE pending marker, never a silent ("skip",).
        assert terminal.cache_data[1][0] == "rotating_kv_pending"

        query = tokens + [901, 902, 903]
        hit, remaining = cache.fetch_cache("turn3", query)
        assert hit is not None, (
            "the warm extension's unrestorable boundary went cold instead of "
            "walking back to the exact 192-token anchor"
        )
        assert hit.num_tokens == 192
        assert remaining == query[192:]
        assert cache.get_stats()["tokens_saved"] == 192

        rebuilt = cache.reconstruct_cache(hit)
        assert rebuilt is not None
        assert rebuilt[0].offset == 192
        assert rebuilt[1].offset == 192
        assert rebuilt[1]._idx == 128
        assert mx.array_equal(
            rebuilt[1].keys.reshape(-1),
            mx.arange(64, 192, dtype=mx.float32),
        ).item()

    def test_warm_extension_with_no_anchor_declines_the_hit(self):
        """No anchor anywhere: decline explicitly, never credit the hit.

        The same rolled-ring warm store WITHOUT a preceding cold turn leaves
        every rotating boundary pending.  The fetch must release the blocks
        and report a miss — a credited hit that reconstructs to nothing is
        the silent-empty-cache failure this campaign closes.
        """
        mx = pytest.importorskip("mlx.core")

        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        tokens = list(range(398))
        manager = PagedCacheManager(block_size=64, max_blocks=32)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=manager)
        assert cache.store_cache(
            "only",
            tokens,
            self._mixed_swa_snapshot(mx, 430, seq_len=430),
        ) is not None

        hit, remaining = cache.fetch_cache("reader", tokens)
        assert hit is None
        assert remaining == tokens
        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["tokens_saved"] == 0
        assert "reader" not in cache._request_tables

    def test_legacy_skip_terminal_chain_walks_back_via_previous_block_probe(
        self,
    ):
        """Chains persisted by older builds carry ("skip",) terminals.

        Shipped cutters wrote a bare skip when the terminal rotating cut
        failed, so durable chains with that shape exist on user disks.  A
        skip entry names no family, which is exactly how the terminal guard
        was blinded; the repair probes the PREVIOUS block (rotating families
        stamp every non-terminal block with rotating_kv or
        rotating_kv_pending) and then walks back like any pending boundary.
        """
        mx = pytest.importorskip("mlx.core")

        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        tokens = list(range(398))
        manager = PagedCacheManager(block_size=64, max_blocks=32)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=manager)
        assert cache.store_cache(
            "turn1", tokens[:192], self._mixed_swa_snapshot(mx, 192)
        ) is not None
        table2 = cache.store_cache(
            "turn2",
            tokens,
            self._mixed_swa_snapshot(mx, 430, seq_len=430),
        )
        assert table2 is not None
        # Rewrite the terminal block payload into the legacy shape.
        terminal = manager.allocated_blocks[table2.block_ids[-1]]
        terminal.cache_data = [
            ("skip",)
            if (
                isinstance(entry, (tuple, list))
                and entry
                and entry[0] == "rotating_kv_pending"
            )
            else entry
            for entry in terminal.cache_data
        ]
        assert terminal.cache_data[1] == ("skip",)

        query = tokens + [901]
        hit, remaining = cache.fetch_cache("turn3", query)
        assert hit is not None, (
            "legacy skip-terminal chain went cold (or was fully credited) "
            "instead of walking back to the 192-token anchor"
        )
        assert hit.num_tokens == 192
        assert remaining == query[192:]
        rebuilt = cache.reconstruct_cache(hit)
        assert rebuilt is not None
        assert rebuilt[1].offset == 192
        assert rebuilt[1]._idx == 128

    def test_both_fetch_lanes_normalize_rotating_candidates(self):
        """Both fetch lanes must call the anchor walk.

        A fix applied to one of the two lanes and not the other is the single
        most repeated failure in this cache: the paged chain-hash lane and the
        prefix-index lane are separate code paths that both reach the mixed-SWA
        terminal guard.  Pin structurally, because the prefix-index lane needs
        a terminal-PARTIAL match to be selected and is not reachable from the
        same fixture as the paged lane.
        """
        import inspect

        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        source = inspect.getsource(BlockAwarePrefixCache.fetch_cache)
        guard_sites = source.count("_rotating_l2_chain_missing_terminal_state(")
        walk_sites = source.count("_normalize_rotating_candidate(")
        assert guard_sites == 2, (
            "expected exactly two mixed-SWA terminal guards (paged lane + "
            f"prefix-index lane), found {guard_sites}"
        )
        assert walk_sites == guard_sites, (
            f"{guard_sites} terminal guards but only {walk_sites} anchor "
            "walks -- one lane still goes cold instead of trimming"
        )

    def test_extending_partial_prefix_realigns_durable_block_chain(self):
        """An extended partial tail must be replaced at block boundaries."""
        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        paged = PagedCacheManager(block_size=4, max_blocks=16)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged)
        cache.store_cache("same-request", [1, 2, 3], ["short-state"])

        extended = cache.store_cache(
            "same-request",
            [1, 2, 3, 4, 5, 6, 7],
            ["extended-state"],
        )

        assert extended is not None
        assert extended.num_tokens == 7
        assert [
            paged.allocated_blocks[block_id].token_count
            for block_id in extended.block_ids
        ] == [4, 3]

        hit, remaining = cache.fetch_cache(
            "follow-up",
            [1, 2, 3, 4, 5, 6, 7, 8],
        )
        assert hit is not None
        assert hit.num_tokens == 7
        assert remaining == [8]

    def test_oversized_prompt_stores_capacity_limited_partial_prefix(self, caplog):
        """A prompt larger than Max Cache Blocks should degrade to partial reuse.

        Long agent/system prompts must not turn cache pressure into an opaque
        failure. The cache stores the largest block-aligned prefix that fits,
        indexes only that prefix, and reports the partial capacity in logs.
        """
        import logging

        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        caplog.set_level(logging.WARNING, logger="vmlx_engine.prefix_cache")

        paged_manager = PagedCacheManager(block_size=4, max_blocks=3)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

        tokens = list(range(12))
        stored = cache.store_cache("req-too-large", tokens, ["cache_data"])

        assert stored is not None
        assert stored.num_tokens == 8
        assert len(stored.block_ids) == 2

        hit_table, remaining = cache.fetch_cache("req-repeat", tokens + [99])
        assert hit_table is not None
        assert hit_table.num_tokens == 8
        assert remaining == [8, 9, 10, 11, 99]
        assert "stored partial prefix" in caplog.text
        assert "8/12 tokens" in caplog.text

    def test_block_aware_fetch_ignores_context_free_legacy_block_hash(self):
        """Repeated block bytes under a different parent are not a prefix hit.

        The legacy PagedCacheManager.find_shared_prefix path keys a block only
        by the current chunk's token content. Real KV state depends on the
        previous context too, so BlockAwarePrefixCache must use the chain hash
        path instead.
        """
        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        paged_manager = PagedCacheManager(block_size=4, max_blocks=100)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

        # Second block is [9, 9, 9, 9], but it was computed after [1,2,3,4].
        cache.store_cache("req-1", [1, 2, 3, 4, 9, 9, 9, 9], ["data"])

        # A new prompt starting with [9,9,9,9] must miss; otherwise it would
        # restore hidden/KV state produced under the wrong parent context.
        block_table, remaining = cache.fetch_cache("req-2", [9, 9, 9, 9, 7])

        assert block_table is None
        assert remaining == [9, 9, 9, 9, 7]

    def test_tensor_store_does_not_reuse_legacy_content_hash_for_repeated_blocks(self):
        """Tensor KV blocks must be keyed by full prefix history, not bytes alone."""
        import mlx.core as mx

        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        paged_manager = PagedCacheManager(block_size=4, max_blocks=100)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

        keys = mx.zeros((1, 1, 8, 1))
        values = mx.ones((1, 1, 8, 1))
        cache_data = [
            {
                "class_name": "KVCache",
                "state": (keys, values),
            }
        ]
        tokens = [1, 2, 3, 4, 1, 2, 3, 4]

        block_table = cache.store_cache("req-repeated-tensor", tokens, cache_data)

        assert block_table is not None
        assert block_table.num_tokens == len(tokens)
        assert len(block_table.block_ids) == 2
        assert len(set(block_table.block_ids)) == 2

    def test_media_extra_key_isolates_identical_placeholder_tokens(self):
        """Media prompts need a side key; token ids alone collide by image size.

        VLM templates emit the same placeholder token sequence for two images
        with the same grid shape. Reusing a paged cache entry across different
        image bytes would replay the first image's vision-conditioned KV state.
        The cache key must therefore include a media fingerprint while keeping
        the cached token count equal to the real model-token count.
        """
        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        cache = BlockAwarePrefixCache(
            model=object(),
            paged_cache_manager=PagedCacheManager(block_size=4, max_blocks=16),
        )
        tokens = [101, 262147, 262147, 102, 103, 104]

        stored = cache.store_cache(
            "media-a",
            tokens,
            ["cache-a"],
            cache_extra_keys={"media": "image-a"},
        )
        assert stored is not None

        miss_table, miss_remaining = cache.fetch_cache(
            "media-b",
            tokens,
            cache_extra_keys={"media": "image-b"},
        )
        assert miss_table is None
        assert miss_remaining == tokens

        hit_table, hit_remaining = cache.fetch_cache(
            "media-a-repeat",
            tokens + [105],
            cache_extra_keys={"media": "image-a"},
        )
        assert hit_table is not None
        assert hit_table.num_tokens == len(tokens)
        assert hit_remaining == [105]

    def test_release_cache(self):
        """Test releasing cache."""
        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        paged_manager = PagedCacheManager(block_size=64, max_blocks=100)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

        tokens = list(range(64))
        cache.store_cache("req-1", tokens, ["data"])

        assert len(cache) == 1

        cache.release_cache("req-1")

        assert len(cache) == 0

    def test_fork_cache(self):
        """Test forking cache (COW)."""
        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        paged_manager = PagedCacheManager(block_size=64, max_blocks=100)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

        tokens = list(range(128))
        cache.store_cache("req-1", tokens, ["shared_data"])

        # Fork to new request
        forked_table = cache.fork_cache("req-1", "req-2")

        assert forked_table is not None
        assert len(cache) == 2

        # Both should share the same blocks
        stats = cache.get_stats()
        assert stats["shared_blocks"] > 0

    def test_get_cache_for_generation(self):
        """Test getting cache for generation with COW."""
        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        paged_manager = PagedCacheManager(block_size=64, max_blocks=100)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

        tokens = list(range(64))
        cache.store_cache("req-1", tokens, ["data"])

        # Get cache for generation (no COW needed)
        cache_data, was_copied = cache.get_cache_for_generation("req-1")

        assert cache_data == ["data"]
        assert was_copied is False

    def test_get_cache_for_generation_with_cow(self):
        """Test COW is triggered for shared blocks."""
        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        paged_manager = PagedCacheManager(block_size=64, max_blocks=100)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

        tokens = list(range(64))
        cache.store_cache("req-1", tokens, ["shared_data"])
        cache.fork_cache("req-1", "req-2")

        # Get cache for forked request - should trigger COW
        cache_data, was_copied = cache.get_cache_for_generation("req-2")

        assert cache_data is not None
        assert was_copied is True

    def test_stats(self):
        """Test statistics."""
        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        paged_manager = PagedCacheManager(block_size=64, max_blocks=100)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

        # Miss
        cache.fetch_cache("req-1", [1, 2, 3])

        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

    def test_clear(self):
        """Test clearing cache."""
        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        paged_manager = PagedCacheManager(block_size=64, max_blocks=100)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

        tokens = list(range(128))
        cache.store_cache("req-1", tokens, ["data"])
        cache.store_cache("req-2", tokens, ["data2"])

        assert len(cache) == 2

        # Model the real lifecycle: both requests finish and drop their refs.
        # Their blocks stay cached at ref 0, which is what makes them clearable.
        for request_id in ("req-1", "req-2"):
            table = paged_manager.request_tables.get(request_id)
            if table is not None:
                paged_manager.release_request_refs(table)
        assert cache.clear() is True

        assert len(cache) == 0
        stats = cache.get_stats()
        # After clear, null block is still allocated (vLLM style)
        assert stats["allocated_blocks"] == 1  # only null block
