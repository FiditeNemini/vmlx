# SPDX-License-Identifier: Apache-2.0
# Base paged cache from waybarrios/vllm-mlx. Block disk store L2 tier,
# partial block matching, COW fork, and MLA-aware block extraction added
# by Jinho Jang (eric@jangq.ai) for vMLX (github.com/jjang-ai/vmlx).
"""
Paged KV Cache Manager for vmlx-engine.

This module implements block-based paged KV cache management following vLLM's
architecture (vllm/v1/core/block_pool.py), adapted for MLX on Apple Silicon.

Key components:
- KVCacheBlock: Metadata for each cache block with doubly linked list pointers
- FreeKVCacheBlockQueue: O(1) doubly linked list for LRU block allocation
- BlockHashToBlockMap: Hash-to-block cache for prefix caching
- PagedCacheManager: Main manager with block allocation, prefix caching, and COW

Features:
- Block-based allocation (configurable tokens per block)
- Reference counting for shared blocks
- Copy-on-Write (COW) for efficient prefix sharing
- LRU eviction using doubly linked list (O(1) operations)
- Chain hashing for prefix caching (hash depends on parent block)

Reference: vLLM v1 - vllm/v1/core/block_pool.py, vllm/v1/core/kv_cache_utils.py
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Dict, List, NewType, Optional, Tuple

logger = logging.getLogger(__name__)

_CACHE_HASH_DEBUG = os.environ.get("VMLX_CACHE_HASH_DEBUG", "") == "1"

# Type alias for block hash (content-based hash for prefix caching)
BlockHash = NewType("BlockHash", bytes)

# Metal-pressure-aware L1 eviction. The static resident ceiling
# (max_resident_bytes) derives from a system-RAM percent and never sees the
# Metal working set, so on a machine whose weights nearly fill unified memory
# the L1 pool can push Metal active memory into the working-set limit long
# before the RAM ceiling is reached (measured on the 128GB box: 96.5GB weights
# + 4.9GB retained L1 + ~3.4GB prefill transient crossed the 107.5GB limit at
# ~430k tokens with zero evictions against a ~23GB static ceiling). When active
# Metal memory plus a transient margin exceeds the guard threshold, the LRU
# write-through eviction below sheds the overage regardless of the static
# ceiling. Blocks are disk-mirrored before dropping RAM, so this trades
# re-promotion latency for process survival.
_PRESSURE_EVICT_ENV = "VMLX_PAGED_METAL_PRESSURE_EVICT"
_PRESSURE_MARGIN_ENV = "VMLX_PAGED_METAL_PRESSURE_MARGIN_GB"
# Default margin must cover the largest observed prefill transient (~3.4GB on
# the box 1M gate) with slack.
_DEFAULT_PRESSURE_MARGIN_BYTES = 4 * 1024**3


def paged_metal_pressure_evict_enabled() -> bool:
    raw = os.environ.get(_PRESSURE_EVICT_ENV, "").strip().lower()
    if not raw:
        return True
    return raw not in ("0", "false", "no", "off")


def paged_metal_pressure_margin_bytes() -> int:
    raw = os.environ.get(_PRESSURE_MARGIN_ENV, "").strip()
    if raw:
        try:
            parsed = float(raw)
        except ValueError:
            parsed = -1.0
        if parsed >= 0:
            return int(parsed * 1024**3)
    return _DEFAULT_PRESSURE_MARGIN_BYTES


def compute_metal_pressure_overage_bytes(
    active_bytes: int,
    max_ws_bytes: int,
    threshold_pct: float,
    margin_bytes: int,
) -> int:
    """Bytes the resident pool must shed so ``active + margin`` fits back
    under ``threshold_pct%`` of the Metal working-set limit. Pure so tests can
    pin the arithmetic; 0 when unpressured or when either byte figure is
    unavailable (<= 0)."""
    if active_bytes <= 0 or max_ws_bytes <= 0:
        return 0
    limit = int(max_ws_bytes * (threshold_pct / 100.0))
    return max(0, int(active_bytes) + max(0, int(margin_bytes)) - limit)


def compute_block_hash(
    parent_hash: Optional[BlockHash],
    token_ids: List[int],
    extra_keys: Optional[Any] = None,
) -> BlockHash:
    """
    Compute hash for a block based on its content and parent block.

    This enables prefix caching by creating a chain of hashes where
    each block's hash depends on all previous blocks (similar to vLLM).

    Args:
        parent_hash: Hash of the previous block, or None for first block
        token_ids: Token IDs in this block
        extra_keys: Additional keys (e.g., LoRA, multimodal arrays)

    Returns:
        Content-based hash for this block
    """
    hasher = hashlib.sha256()

    # Include parent hash for chain
    if parent_hash:
        hasher.update(parent_hash)
    else:
        # Use fixed seed for reproducibility
        hasher.update(b"vmlx-engine-root")

    # Include token content
    hasher.update(bytes(str(tuple(token_ids)), "utf-8"))

    # Include extra keys if present.
    #
    # The encoding is canonical: every node contributes a type tag and a
    # length-prefixed payload, and containers contribute their element count
    # before their elements. Without this, adjacent fields concatenate
    # ambiguously and distinct conditions collide — {'a':'bc'} vs {'ab':'c'},
    # [1,23] vs [12,3] — which would let one request condition receive another
    # condition's cached KV blocks.
    if extra_keys is not None:
        import struct

        import mlx.core as mx

        def _update_sized(tag: bytes, payload: bytes) -> None:
            hasher.update(tag)
            hasher.update(struct.pack("<Q", len(payload)))
            hasher.update(payload)

        def _hash_extra(obj):
            if isinstance(obj, mx.array):
                # dtype is part of identity: the same bytes under a different
                # dtype are a different condition. np.array(obj) copies to
                # CPU; for small vision embeddings this is acceptable.
                import numpy as np

                _update_sized(
                    b"A", bytes(f"{obj.dtype}|{tuple(obj.shape)}", "utf-8")
                )
                _update_sized(b"a", np.array(obj).tobytes())
            elif isinstance(obj, dict):
                hasher.update(b"D")
                hasher.update(struct.pack("<Q", len(obj)))
                for k in sorted(
                    obj.keys(),
                    key=lambda value: (
                        type(value).__module__,
                        type(value).__qualname__,
                        repr(value),
                    ),
                ):
                    hasher.update(b"K")
                    _hash_extra(k)
                    hasher.update(b"V")
                    _hash_extra(obj[k])
            elif isinstance(obj, (list, tuple)):
                hasher.update(b"L")
                hasher.update(struct.pack("<Q", len(obj)))
                for item in obj:
                    _hash_extra(item)
            else:
                _update_sized(
                    b"S", bytes(f"{type(obj).__name__}|{obj}", "utf-8")
                )

        _hash_extra(extra_keys)

    return BlockHash(hasher.digest())


def _native_dsv4_interval_from_payload(
    cache_data: Optional[List[Tuple[Any, ...]]],
) -> Optional[Tuple[int, int]]:
    """Read a validated native DSV4 interval without retaining tensor state."""
    interval: Optional[Tuple[int, int]] = None
    saw_delta = False
    rotating_ends: List[int] = []
    for entry in cache_data or ():
        if not isinstance(entry, (tuple, list)) or not entry:
            return None
        tag = entry[0]
        if tag == "deepseek_v4_delta_v1":
            if (
                len(entry) < 4
                or not isinstance(entry[1], dict)
                or entry[1].get("schema") != "deepseek_v4_block_delta_v1"
                or not isinstance(entry[2], str)
                or not isinstance(entry[3], dict)
            ):
                return None
            try:
                current = (
                    int(entry[1]["start_token"]),
                    int(entry[1]["end_token"]),
                )
            except (KeyError, TypeError, ValueError):
                return None
            if current[1] <= current[0]:
                return None
            if interval is None:
                interval = current
            elif current != interval:
                return None
            saw_delta = True
        elif tag == "rotating_kv":
            if len(entry) < 7:
                return None
            try:
                rotating_ends.append(int(entry[5]))
            except (TypeError, ValueError):
                return None
        elif tag == "rotating_kv_pending":
            if len(entry) < 2:
                return None
        else:
            return None
    if (
        not saw_delta
        or interval is None
        or any(end != interval[1] for end in rotating_ends)
    ):
        return None
    return interval


# =============================================================================
# KVCacheBlock - Following vLLM's design
# =============================================================================


@dataclass
class CacheBlock:
    """
    KV cache block metadata following vLLM's design.

    Each block represents a fixed number of tokens (block_size) worth
    of KV cache data. Blocks can be shared across requests via
    reference counting for prefix caching.

    Attributes:
        block_id: Physical block index (0 to num_blocks - 1)
        ref_count: Reference count for sharing (0 = can be evicted)
        block_hash: Content hash for prefix caching (None if not cached)
        prev_free_block: Previous block in free list (doubly linked)
        next_free_block: Next block in free list (doubly linked)
        is_null: True if this is the null/placeholder block
        cache_data: Actual KV tensor data stored in this block
        token_count: Number of tokens stored in this block
    """

    block_id: int
    ref_count: int = 0
    block_hash: Optional[BlockHash] = None
    # Immediate predecessor in the content-addressed chain.  Persisted L2
    # eviction needs this independently of any request-scoped block table so it
    # can retain a restorable root-to-tail prefix across RAM eviction/restart.
    parent_hash: Optional[BlockHash] = None

    # Doubly linked list pointers for FreeKVCacheBlockQueue
    prev_free_block: Optional["CacheBlock"] = None
    next_free_block: Optional["CacheBlock"] = None

    # Special flags
    is_null: bool = False

    # Actual tensor data for this block
    # List of (keys, values) per layer, shape: (1, n_kv_heads, block_tokens, head_dim)
    cache_data: Optional[List[Tuple[Any, Any]]] = None
    cache_data_from_disk: bool = False
    # True when an L2 payload is request-owned reconstruction workspace rather
    # than an admitted persistent L1 mirror. This lets a prefix larger than
    # the RAM slider refault completely from SSD without lying that those bytes
    # are resident cache; reconstruction drops the payload before request refs
    # are released.
    cache_data_transient: bool = False
    # Native DSV4 payload contract retained independently of the resident
    # tensors. Disk-only extension stores can validate a long parent chain in
    # O(blocks) metadata checks without reloading every page from SSD.
    dsv4_native_interval: Optional[Tuple[int, int]] = None
    # A richer native representation superseded this block's payload under
    # the same token hash. Active request refs may continue reading it, but a
    # later eviction must never republish the stale representation to L2.
    suppress_l2_republish: bool = False
    # Resident RAM bytes currently attributed to this block's cache_data mirror
    # (0 when cache_data is None). Tracked so the manager can hold a byte ceiling
    # like the memory-aware path instead of an unbounded block count.
    resident_bytes: int = 0
    # Native path-dependent composite state (DSV4 SWA+CSA/HCA, ZAYA CCA
    # conv_state+prev_hs, mixed-SWA rotating-window) is deliberately kept
    # resident by the store path: an immediate same-process repeat can hit the
    # block table before the async L2 write is index-readable and would
    # otherwise reconstruct as None. Blocks flagged here are excluded from the
    # byte-ceiling LRU so enforce_byte_budget can't drop that guarantee.
    keep_resident: bool = False
    # Native composite records cannot be recycled until their asynchronous L2
    # copy is index-readable. A successful queue admission sets this marker;
    # allocation then applies bounded backpressure instead of repeatedly
    # serializing the same payload or discarding its only restorable copy.
    durability_write_pending: bool = False
    durability_retry_after: float = 0.0
    # A native disk-only fallback became durable while this block still had
    # request references. The final ref release performs the deferred RAM drop
    # under the manager lock instead of racing an active reconstruction.
    release_resident_when_unreferenced: bool = False

    # Metadata
    token_count: int = 0
    hash_value: Optional[str] = None  # Legacy string hash for compatibility
    last_access: float = field(default_factory=time.time)

    def is_full(self, block_size: int) -> bool:
        """Check if block is at capacity."""
        return self.token_count >= block_size

    def is_shared(self) -> bool:
        """Check if block is shared (ref_count > 1)."""
        return self.ref_count > 1

    def reset_hash(self) -> None:
        """Reset block hash when evicted from cache."""
        self.block_hash = None
        self.parent_hash = None
        self.hash_value = None
        self.cache_data_from_disk = False
        self.cache_data_transient = False
        self.dsv4_native_interval = None
        self.suppress_l2_republish = False
        # Protection is scoped to the native composite payload currently held
        # by this block. Once the hash/payload is evicted, carrying the flag
        # into a later ordinary KV/TQ allocation makes that unrelated block
        # permanently ineligible for byte-budget eviction.
        self.keep_resident = False
        self.durability_write_pending = False
        self.durability_retry_after = 0.0
        self.release_resident_when_unreferenced = False

    def touch(self) -> None:
        """Update last access time."""
        self.last_access = time.time()

    def __repr__(self) -> str:
        prev_id = self.prev_free_block.block_id if self.prev_free_block else None
        next_id = self.next_free_block.block_id if self.next_free_block else None
        return (
            f"CacheBlock(id={self.block_id}, ref={self.ref_count}, "
            f"tokens={self.token_count}, prev={prev_id}, next={next_id})"
        )


# Alias for backwards compatibility
KVCacheBlock = CacheBlock


# =============================================================================
# FreeKVCacheBlockQueue - O(1) Doubly Linked List (vLLM style)
# =============================================================================


class FreeKVCacheBlockQueue:
    """
    Doubly linked list of free blocks following vLLM's design.

    Provides O(1) operations for:
    - popleft(): Allocate block from front (LRU order)
    - remove(): Remove block from middle (when touched by cache hit)
    - append(): Return block to end (when freed)

    The queue maintains LRU eviction order:
    - Front = least recently used (evict first)
    - Back = most recently used (evict last)

    Uses fake head/tail sentinels to simplify edge cases.
    """

    def __init__(self, blocks: List[CacheBlock]) -> None:
        """
        Initialize queue with all blocks as free.

        Args:
            blocks: List of all CacheBlock objects
        """
        self.num_free_blocks = len(blocks)

        # Initialize doubly linked list
        for i in range(len(blocks)):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < len(blocks) - 1:
                blocks[i].next_free_block = blocks[i + 1]

        # Create sentinel nodes (never popped)
        self.fake_head = CacheBlock(block_id=-1)
        self.fake_tail = CacheBlock(block_id=-2)

        if blocks:
            self.fake_head.next_free_block = blocks[0]
            blocks[0].prev_free_block = self.fake_head
            self.fake_tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self.fake_tail
        else:
            self.fake_head.next_free_block = self.fake_tail
            self.fake_tail.prev_free_block = self.fake_head

    def popleft(self) -> CacheBlock:
        """
        Pop and return the first (LRU) free block.

        Raises:
            ValueError: If no free blocks available
        """
        if self.fake_head.next_free_block is self.fake_tail:
            raise ValueError("No free blocks available")

        block = self.fake_head.next_free_block
        assert block is not None

        # Remove from list
        self.fake_head.next_free_block = block.next_free_block
        if block.next_free_block:
            block.next_free_block.prev_free_block = self.fake_head

        block.prev_free_block = None
        block.next_free_block = None
        self.num_free_blocks -= 1

        return block

    def popleft_n(self, n: int) -> List[CacheBlock]:
        """
        Pop n blocks from the front.

        Args:
            n: Number of blocks to allocate

        Returns:
            List of n free blocks

        Raises:
            AssertionError: If not enough free blocks
        """
        if n == 0:
            return []

        assert (
            self.num_free_blocks >= n
        ), f"Need {n} blocks, have {self.num_free_blocks}"

        result = []
        curr = self.fake_head.next_free_block

        for _ in range(n):
            assert curr is not None and curr is not self.fake_tail
            result.append(curr)
            last = curr
            curr = curr.next_free_block
            # Clear pointers
            last.prev_free_block = None
            last.next_free_block = None

        # Reconnect list
        self.fake_head.next_free_block = curr
        if curr:
            curr.prev_free_block = self.fake_head

        self.num_free_blocks -= n
        return result

    def remove(self, block: CacheBlock) -> None:
        """
        Remove a block from the middle of the queue.

        Used when a free block is "touched" (reused by prefix cache hit).

        Args:
            block: Block to remove

        Raises:
            RuntimeError: If block not in queue
        """
        if block.prev_free_block is None or block.next_free_block is None:
            raise RuntimeError(f"Block {block.block_id} not in free queue")

        # Unlink
        block.prev_free_block.next_free_block = block.next_free_block
        block.next_free_block.prev_free_block = block.prev_free_block
        block.prev_free_block = None
        block.next_free_block = None

        self.num_free_blocks -= 1

    def append(self, block: CacheBlock) -> None:
        """
        Append a block to the end (MRU position).

        Args:
            block: Block to append
        """
        last = self.fake_tail.prev_free_block
        assert last is not None

        last.next_free_block = block
        block.prev_free_block = last
        block.next_free_block = self.fake_tail
        self.fake_tail.prev_free_block = block

        self.num_free_blocks += 1

    def append_n(self, blocks: List[CacheBlock]) -> None:
        """
        Append multiple blocks to the end.

        Args:
            blocks: Blocks to append (in order)
        """
        if not blocks:
            return

        last = self.fake_tail.prev_free_block
        assert last is not None

        for block in blocks:
            block.prev_free_block = last
            last.next_free_block = block
            last = block

        last.next_free_block = self.fake_tail
        self.fake_tail.prev_free_block = last

        self.num_free_blocks += len(blocks)

    def get_all_free_blocks(self) -> List[CacheBlock]:
        """Get all free blocks (for testing)."""
        result = []
        curr = self.fake_head.next_free_block
        while curr and curr is not self.fake_tail:
            result.append(curr)
            curr = curr.next_free_block
        return result


# =============================================================================
# BlockHashToBlockMap - Hash-based prefix cache (vLLM style)
# =============================================================================


class BlockHashToBlockMap:
    """
    Cache mapping block hashes to blocks for prefix caching.

    Follows vLLM's design where the same hash can map to multiple
    blocks (for different KV cache groups in hybrid models).
    """

    def __init__(self) -> None:
        self._cache: Dict[BlockHash, CacheBlock | Dict[int, CacheBlock]] = {}

    def get_block(self, block_hash: BlockHash) -> Optional[CacheBlock]:
        """Get any block with the given hash."""
        blocks = self._cache.get(block_hash)
        if blocks is None:
            return None
        if isinstance(blocks, CacheBlock):
            return blocks
        if isinstance(blocks, dict):
            # A later store may deliberately promote the same token-chain
            # boundary with richer native state (for example replacing a
            # mixed-SWA ``rotating_kv_pending`` interior record with an exact
            # terminal RotatingKV checkpoint). Prefer the most recently
            # inserted live block so the promoted boundary becomes
            # discoverable; returning the oldest duplicate made promotion
            # ineffective and forced every later request to miss again.
            return next(reversed(blocks.values()))
        return None

    def insert(self, block_hash: BlockHash, block: CacheBlock) -> None:
        """Insert a block into the cache."""
        existing = self._cache.get(block_hash)
        if existing is None:
            self._cache[block_hash] = block
        elif isinstance(existing, CacheBlock):
            self._cache[block_hash] = {
                existing.block_id: existing,
                block.block_id: block,
            }
        elif isinstance(existing, dict):
            existing[block.block_id] = block

    def pop(self, block_hash: BlockHash, block_id: int) -> Optional[CacheBlock]:
        """Remove and return a specific block from the cache."""
        blocks = self._cache.pop(block_hash, None)
        if blocks is None:
            return None

        if isinstance(blocks, CacheBlock):
            if blocks.block_id == block_id:
                return blocks
            # Wrong block ID, put it back
            self._cache[block_hash] = blocks
            return None

        if isinstance(blocks, dict):
            block = blocks.pop(block_id, None)
            if blocks:  # Still has other blocks
                self._cache[block_hash] = blocks
            return block

        return None

    def pop_all(self, block_hash: BlockHash) -> List[CacheBlock]:
        """Retire every discoverable representation for one content hash.

        Blocks remain allocated and active request tables retain their refs;
        only future content-addressed lookup is removed. This is required when
        a richer native payload supersedes any number of generic siblings.
        """
        blocks = self._cache.pop(block_hash, None)
        if blocks is None:
            return []
        if isinstance(blocks, CacheBlock):
            return [blocks]
        if isinstance(blocks, dict):
            return list(blocks.values())
        return []

    def __len__(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()


# =============================================================================
# BlockTable - Per-request block mapping
# =============================================================================


@dataclass
class BlockTable:
    """
    Per-request block table mapping logical to physical blocks.

    Similar to vLLM's block table, this maps a request's token positions
    to physical cache blocks.

    Attributes:
        request_id: Unique request identifier
        block_ids: List of physical block IDs
        num_tokens: Total number of cached tokens
    """

    request_id: str
    block_ids: List[int] = field(default_factory=list)
    num_tokens: int = 0
    # Native path-dependent caches can discover a longer token prefix than
    # they can safely restore without replay. ``num_tokens`` remains the
    # actual checkpoint consumed by generation; these fields preserve the
    # independently attested longest match and replay cost for telemetry.
    matched_tokens: Optional[int] = None
    checkpoint_tokens: Optional[int] = None
    replayed_tokens: int = 0

    def add_block(self, block_id: int, num_tokens: int) -> None:
        """Add a block to the table."""
        self.block_ids.append(block_id)
        self.num_tokens += num_tokens

    def __len__(self) -> int:
        return len(self.block_ids)

    def copy(self, new_request_id: str) -> "BlockTable":
        """Create a copy with new request ID."""
        return BlockTable(
            request_id=new_request_id,
            block_ids=self.block_ids.copy(),
            num_tokens=self.num_tokens,
            matched_tokens=self.matched_tokens,
            checkpoint_tokens=self.checkpoint_tokens,
            replayed_tokens=self.replayed_tokens,
        )


# =============================================================================
# CacheStats - Statistics for monitoring
# =============================================================================


@dataclass
class CacheStats:
    """Statistics for cache monitoring."""

    total_blocks: int = 0
    allocated_blocks: int = 0
    free_blocks: int = 0
    shared_blocks: int = 0  # Blocks with ref_count > 1
    total_tokens_cached: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cow_copies: int = 0
    evictions: int = 0
    # L2 disk cache stats
    disk_hits: int = 0
    disk_misses: int = 0


# =============================================================================
# PagedCacheManager - Main manager (vLLM BlockPool style)
# =============================================================================


class PagedCacheManager:
    """
    Paged KV cache manager following vLLM's BlockPool architecture.

    Features:
    - Block allocation/deallocation with reference counting
    - Prefix sharing via chain-based hash deduplication
    - Copy-on-Write for efficient forking
    - O(1) LRU eviction using doubly linked list

    Args:
        block_size: Number of tokens per block (default: 64)
        max_blocks: Maximum number of blocks to allocate (default: 1000)
        enable_caching: Whether to enable prefix caching (default: True)
    """

    def __init__(
        self,
        block_size: int = 64,
        max_blocks: int = 1000,
        enable_caching: bool = True,
        disk_store: "Optional[Any]" = None,
        max_resident_bytes: int = 0,
        disk_only: bool = False,
    ):
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        if max_blocks < 2:
            raise ValueError(f"max_blocks must be >= 2 (1 reserved for null block), got {max_blocks}")
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.enable_caching = enable_caching
        # Optional L2 disk store for block persistence (BlockDiskStore)
        self._disk_store = disk_store
        # A block-aware hash/index is still required to discover exact and
        # partial durable prefixes when the user explicitly disables paged RAM
        # caching. In disk-only mode the manager owns only block metadata and
        # transient reconstruction payloads; successful stores and restores do
        # not leave KV tensors resident in the block pool.
        self.disk_only = bool(disk_only)
        if self.disk_only and self._disk_store is None:
            raise ValueError("disk_only requires a disk_store")
        # Paged RAM and block-disk L2 are independent tiers.  Enabling L2 must
        # not silently turn a user-selected Paged On session into SSD-only
        # reconstruction.  The legacy frugal behavior remains available as an
        # explicit diagnostic/low-RAM override, while true disk-only launches
        # always suppress persistent RAM payloads.
        frugal_env = os.environ.get("VMLX_PAGED_FRUGAL", "").strip().lower()
        frugal_requested = bool(frugal_env) and frugal_env not in (
            "0",
            "false",
            "no",
            "off",
        )
        self.paged_frugal = self.disk_only or frugal_requested
        self.ram_mirror_policy = (
            "disk_only"
            if self.disk_only
            else "frugal_env"
            if frugal_requested
            else "resident"
        )
        # RAM byte ceiling for the in-RAM block KV mirror. 0 = unbounded (legacy
        # behavior: the pool grows to max_blocks with no byte ceiling). When > 0,
        # enforce_byte_budget() evicts free (ref_count==0) cached blocks — writing
        # them to L2 disk first if a store is present — to hold this ceiling,
        # mirroring MemoryAwarePrefixCache. Prevents the paged pool from
        # ratcheting resident GPU memory upward with distinct prefixes.
        self.max_resident_bytes = max(0, int(max_resident_bytes or 0))
        self.resident_bytes = 0
        self.transient_disk_promotions = 0
        self.transient_disk_peak_bytes = 0

        # Create all blocks
        self.blocks: List[CacheBlock] = [
            CacheBlock(block_id=i) for i in range(max_blocks)
        ]

        # Free block queue (doubly linked list for O(1) LRU)
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # Hash-to-block cache for prefix caching
        self.cached_block_hash_to_block = BlockHashToBlockMap()

        # Legacy hash index for compatibility
        self.hash_to_block: Dict[str, int] = {}

        # Request to block table mapping
        self.request_tables: Dict[str, BlockTable] = {}

        # Allocated blocks (for fast lookup)
        self.allocated_blocks: Dict[int, CacheBlock] = {}

        # Reserve null block (block 0) - never freed
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True
        self.null_block.ref_count = 1
        self.allocated_blocks[self.null_block.block_id] = self.null_block

        # Track known partial block sizes for sub-prefix matching
        self._partial_block_sizes: set = set()
        if self._disk_store is not None:
            persisted_partial_sizes = getattr(
                self._disk_store, "partial_token_counts", None
            )
            if callable(persisted_partial_sizes):
                try:
                    self._partial_block_sizes.update(
                        int(size)
                        for size in persisted_partial_sizes(self.block_size)
                        if 0 < int(size) < self.block_size
                    )
                except Exception as exc:
                    logger.debug(
                        "Unable to restore persisted partial block sizes: %s",
                        exc,
                    )

        # Statistics
        self.stats = CacheStats(
            total_blocks=max_blocks,
            allocated_blocks=1,  # null block
            free_blocks=max_blocks - 1,
        )

        # Thread safety
        self._lock = threading.RLock()

        logger.info(
            f"PagedCacheManager initialized: block_size={block_size}, "
            f"max_blocks={max_blocks}, usable_blocks={max_blocks - 1}, "
            f"max_tokens={block_size * (max_blocks - 1)}, "
            f"backend={'block-disk-only' if self.disk_only else 'paged'}, "
            f"ram_mirror_policy={self.ram_mirror_policy}"
        )

    # =========================================================================
    # Block Allocation (vLLM style)
    # =========================================================================

    def _pop_recyclable_free_block_locked(self) -> Optional[CacheBlock]:
        """Return one free block whose prior payload can be discarded safely.

        Free native blocks may still own the only restorable SWA/CSA/HCA state
        while their asynchronous L2 write is pending or rejected. Scan the
        current free queue at most once, rotating protected entries to MRU, and
        return ``None`` when every candidate is durability-pending. This is an
        explicit bounded backpressure path: callers can prefill uncached work,
        but cannot recycle path-dependent state behind a still-advertised hash.
        The caller must hold ``self._lock``.
        """
        attempts = self.free_block_queue.num_free_blocks
        for _ in range(attempts):
            block = self.free_block_queue.popleft()
            reusable = block.block_hash is None
            if not reusable:
                reusable = self._maybe_evict_cached_block(block)
            if reusable:
                # Hand back a CLEAN block.
                #
                # A block with block_hash None was returned with its previous
                # tenant's cache_data still attached. The eviction branch above
                # clears the payload, but the hash-None branch did not — and the
                # frugal store path can set a NEW block_hash without setting
                # cache_data, while reconstruct trusts any non-None cache_data.
                # That is a route to serving one request's KV under another
                # request's hash: a correctness bug, not just wasted RAM.
                #
                # Release under the lock we already hold; this also corrects the
                # resident-byte accounting the stale payload was still counted
                # against.
                if block.cache_data is not None:
                    self._release_resident_payload_locked(block)
                return block
            self.free_block_queue.append(block)
        return None

    def allocate_block(self) -> Optional[CacheBlock]:
        """
        Allocate a new cache block.

        Returns:
            CacheBlock if available, None if out of memory.
        """
        with self._lock:
            if self.free_block_queue.num_free_blocks == 0:
                logger.warning("Out of cache blocks")
                return None

            block = self._pop_recyclable_free_block_locked()
            if block is None:
                logger.warning(
                    "Out of recyclable cache blocks: all free candidates are "
                    "waiting for durable L2 admission"
                )
                return None

            block.ref_count = 1
            block.touch()
            self.allocated_blocks[block.block_id] = block

            self.stats.allocated_blocks += 1
            self.stats.free_blocks -= 1

            return block

    def get_new_blocks(self, num_blocks: int) -> List[CacheBlock]:
        """
        Allocate multiple blocks at once (vLLM style).

        Args:
            num_blocks: Number of blocks to allocate

        Returns:
            List of allocated blocks

        Raises:
            ValueError: If not enough free blocks
        """
        with self._lock:
            if num_blocks > self.free_block_queue.num_free_blocks:
                raise ValueError(
                    f"Cannot allocate {num_blocks} blocks, "
                    f"only {self.free_block_queue.num_free_blocks} available"
                )

            blocks = []
            for _ in range(num_blocks):
                block = self._pop_recyclable_free_block_locked()
                if block is None:
                    self.free_block_queue.append_n(blocks)
                    raise ValueError(
                        f"Cannot allocate {num_blocks} recyclable blocks; "
                        "remaining free blocks are waiting for durable L2 admission"
                    )
                blocks.append(block)

            for block in blocks:
                block.ref_count = 1
                block.touch()
                self.allocated_blocks[block.block_id] = block

            self.stats.allocated_blocks += num_blocks
            self.stats.free_blocks -= num_blocks

            return blocks

    @staticmethod
    def _payload_requires_durable_l2(cache_data: Any) -> bool:
        """Identify path-dependent native records that cannot be dropped early."""
        native_tags = {
            "deepseek_v4",
            "deepseek_v4_pending",
            "deepseek_v4_delta_v1",
            "zaya_cca",
            "rotating_kv",
            "rotating_kv_pending",
        }

        def _contains(value: Any) -> bool:
            if not isinstance(value, (tuple, list)):
                return False
            if value and isinstance(value[0], str) and value[0] in native_tags:
                return True
            return any(_contains(item) for item in value)

        try:
            return _contains(cache_data)
        except Exception:
            return False

    def _persist_before_cached_block_eviction(self, block: CacheBlock) -> bool:
        """Fence eviction against L2 admission for path-dependent state.

        Ordinary payloads can be released once the disk writer accepts an
        immutable detached copy. Native composite payloads remain resident
        until ``has_block`` confirms the copy is index-readable. Synchronous
        admission failure retains every payload and its hash mapping, allowing
        a later bounded allocator scan to retry without corrupting ancestry.
        """
        if block.cache_data is None:
            return True
        if block.suppress_l2_republish:
            block.keep_resident = False
            block.durability_write_pending = False
            block.durability_retry_after = 0.0
            return True
        disk_store = self._disk_store
        native = bool(block.keep_resident) or self._payload_requires_durable_l2(
            block.cache_data
        )
        if disk_store is None:
            return not block.keep_resident

        has_block = getattr(disk_store, "has_block", None)
        if callable(has_block):
            try:
                if bool(has_block(block.block_hash)):
                    block.keep_resident = False
                    block.durability_write_pending = False
                    block.durability_retry_after = 0.0
                    return True
            except Exception as exc:
                logger.warning(
                    "Unable to confirm L2 durability for cache block %s: %s",
                    block.block_id,
                    exc,
                )

        now = time.monotonic()
        if (
            native
            and block.durability_write_pending
            and now < block.durability_retry_after
        ):
            block.keep_resident = True
            return False

        try:
            admitted = bool(
                disk_store.write_block_async(
                    block.block_hash,
                    block.cache_data,
                    block.token_count,
                    parent_hash=block.parent_hash,
                )
            )
        except Exception as exc:
            admitted = False
            logger.warning(
                "L2 write admission raised while retaining cache block %s: %s",
                block.block_id,
                exc,
            )
        if not admitted:
            if native:
                block.keep_resident = True
            block.durability_write_pending = False
            block.durability_retry_after = now
            logger.warning(
                "L2 write admission rejected cache block %s; retaining its "
                "RAM payload and hash ancestry",
                block.block_id,
            )
            return False

        if native:
            # Admission detached an immutable writer-owned copy, but the native
            # source remains the only readable checkpoint until publication.
            # Retry at most once per five seconds if publication never arrives.
            block.keep_resident = True
            block.durability_write_pending = True
            block.durability_retry_after = now + 5.0
            return False
        return True

    def _maybe_evict_cached_block(
        self, block: CacheBlock, *, force: bool = False
    ) -> bool:
        """
        Evict a block from the hash cache if present.
        If a disk store is configured, persist the block before freeing RAM.

        Args:
            block: Block to evict
            force: Drop the RAM payload even when the durability fence refuses.
                Persistence is still attempted first, so an L2 copy is written
                whenever the store accepts one. Reserved for the byte budget's
                last-resort pass: losing a cached block only costs a re-prefill,
                while retaining it without a drain path grows RAM without bound.

        Returns:
            True if block was evicted from cache
        """
        if block.block_hash is None:
            return False
        block_hash = block.block_hash
        if not self._persist_before_cached_block_eviction(block) and not force:
            return False

        # Remove the L1 mapping only after persistence is either already
        # readable or safely queued. If this particular duplicate is no longer
        # mapped, its local payload is stale and can still be reclaimed without
        # disturbing another block registered under the same content hash.
        self.cached_block_hash_to_block.pop(block_hash, block.block_id)

        # Also remove from legacy hash index
        if block.hash_value and block.hash_value in self.hash_to_block:
            if self.hash_to_block[block.hash_value] == block.block_id:
                del self.hash_to_block[block.hash_value]

        block.reset_hash()
        block.cache_data = None  # Free tensor memory
        self._release_resident(block)
        block.cache_data_from_disk = False
        self.stats.evictions += 1
        return True

    def _note_resident(self, block: CacheBlock, nbytes: int) -> None:
        """Attribute ``nbytes`` of resident RAM to ``block``'s cache_data mirror.

        Idempotent per block: replaces any prior attribution so re-promotion or
        re-store does not double-count. Normally a no-op when the byte ceiling is
        disabled. Disk-only mode is the exception: its zero byte ceiling means
        "no persistent payloads", so a failed SSD write fallback must still be
        counted and exposed rather than disappearing from health telemetry.
        """
        if self.max_resident_bytes <= 0 and not self.disk_only:
            return
        nbytes = max(0, int(nbytes or 0))
        with self._lock:
            self.resident_bytes += nbytes - block.resident_bytes
            block.resident_bytes = nbytes

    def _release_resident(self, block: CacheBlock) -> None:
        """Drop ``block``'s resident-byte attribution (called when cache_data is freed)."""
        if block.resident_bytes:
            self.resident_bytes = max(0, self.resident_bytes - block.resident_bytes)
            block.resident_bytes = 0

    def release_resident_payload(self, block: CacheBlock) -> None:
        """Atomically drop a block's RAM mirror and its byte attribution.

        Disk reconstruction uses a temporary L1 promotion. Clearing only
        ``cache_data`` releases the arrays but leaves ``resident_bytes`` as a
        phantom positive, causing health to over-report RAM and the byte
        budget to evict unrelated blocks. The native keep-resident guard also
        expires once the corresponding disk payload is readable and removed.
        """
        with self._lock:
            self._release_resident_payload_locked(block)

    def _release_resident_payload_locked(self, block: CacheBlock) -> None:
        """Drop one RAM mirror while ``self._lock`` is already held."""

        block.cache_data = None
        block.cache_data_from_disk = False
        block.cache_data_transient = False
        block.keep_resident = False
        block.durability_write_pending = False
        block.durability_retry_after = 0.0
        block.release_resident_when_unreferenced = False
        self._release_resident(block)

    def release_resident_payload_when_unreferenced(
        self,
        block: CacheBlock,
    ) -> bool:
        """Drop a durable disk-only fallback without racing active readers.

        Returns ``True`` when the payload was released immediately. An active
        block is merely marked; whichever request releases the final reference
        performs the drop under the same paged-cache lock.
        """

        with self._lock:
            block.cache_data_from_disk = False
            block.cache_data_transient = False
            block.keep_resident = False
            block.durability_write_pending = False
            block.durability_retry_after = 0.0
            if block.ref_count > 0:
                block.release_resident_when_unreferenced = True
                return False
            self._release_resident_payload_locked(block)
            return True

    def make_resident_payload_evictable(self, block: CacheBlock) -> None:
        """Retain a restored L1 payload but return it to normal LRU policy.

        Path-dependent native cache records are protected while their async L2
        write becomes readable. After a successful reconstruction, the payload
        should remain a genuine RAM-tier hit, while the byte ceiling must be able
        to evict it later. Clear only the provenance/protection flags; preserve
        ``cache_data`` and its resident-byte attribution.
        """
        with self._lock:
            block.cache_data_from_disk = False
            block.cache_data_transient = False
            block.keep_resident = False
            block.durability_write_pending = False
            block.durability_retry_after = 0.0

    @staticmethod
    def estimate_block_nbytes(cache_data: Any) -> int:
        """Best-effort resident RAM (bytes) of a block's KV mirror.

        Recursively sums ``.nbytes`` of any array-like leaf (mlx arrays, incl.
        quantized) in the nested (keys, values) / composite structure. Non-array
        leaves (marker strings like "skip"/"kv", None, ints) contribute nothing.
        Recurses into tuples/lists AND dict values, because DSV4's native
        composite state (``("deepseek_v4", state_tree, meta, ...)``) and other
        pytree-structured cache payloads nest their largest arrays under mapping
        leaves — without dict recursion those undercount to zero and the byte
        ceiling never triggers for DSV4. Returns 0 on any error — accounting is
        advisory, never fatal.
        """
        total = 0
        _seen: set = set()

        def _add(x: Any) -> None:
            nonlocal total
            nb = getattr(x, "nbytes", None)
            if isinstance(nb, int):
                total += nb
            elif isinstance(x, dict):
                # Guard against pathological self-referential state trees.
                if id(x) in _seen:
                    return
                _seen.add(id(x))
                for e in x.values():
                    _add(e)
            elif isinstance(x, (tuple, list)):
                for e in x:
                    _add(e)

        try:
            _add(cache_data)
        except Exception:
            return 0
        return total

    def enforce_byte_budget(self, required_bytes: int = 0) -> int:
        """Evict free cached blocks until resident RAM is within the byte ceiling.

        Only blocks with ``ref_count == 0`` (not referenced by any in-flight
        request) are eligible, so this can never corrupt an active sequence.
        Eviction goes through ``_maybe_evict_cached_block``, which write-throughs
        to the L2 disk store (when present) before dropping the RAM mirror — so a
        later prefix hit re-promotes from disk instead of full-prefilling.
        ``required_bytes`` reserves room for a payload that is about to be
        admitted (for example an L2 disk promotion). This is important because
        once a promoted block is referenced by the active request it is no
        longer evictable; admission must therefore make room *before* the ref
        becomes active. Evicts least-recently-used first and returns the number
        of blocks evicted. No-op when the ceiling is disabled
        (``max_resident_bytes <= 0`` — resident accounting is off in that
        mode, so neither target can be measured), except in disk-only mode,
        which keeps accounting on (see ``_note_resident``) and is handled
        below.

        Beyond the static RAM ceiling, this also enforces Metal working-set
        pressure: when active Metal memory plus a transient margin exceeds the
        guard threshold (98% of the device working-set limit by default), the
        pool sheds the overage even though the static ceiling isn't hit. The
        static ceiling derives from a system-RAM percent and cannot see how
        much unified memory the lazily-faulted weights occupy, so on
        weight-heavy machines it never fires (measured: 0 evictions while L1
        growth OOM'd the process at ~430k tokens).

        Disk-only mode gets the pressure guard but never the static ceiling.
        Its ceiling is zero by construction ("no persistent payloads"), so
        enforcing it literally would try to evict the transient buffers a
        reconstruction is actively reading. Those buffers are released by the
        prefix cache once reconstruction finishes; only genuine Metal pressure
        justifies shedding them early.
        """
        if self.max_resident_bytes <= 0 and not self.disk_only:
            return 0
        required_bytes = max(0, int(required_bytes or 0))
        # Metal query stays outside the lock: cheap allocator-counter reads,
        # and the device limit is static.
        pressure_overage = self._metal_pressure_overage_bytes()
        evicted = 0
        with self._lock:
            target_resident_bytes = (
                self.resident_bytes
                if self.disk_only
                else max(0, self.max_resident_bytes - required_bytes)
            )
            if pressure_overage > 0:
                pressure_target = max(
                    0,
                    self.resident_bytes - pressure_overage - required_bytes,
                )
                target_resident_bytes = min(
                    target_resident_bytes, pressure_target
                )
            if self.resident_bytes <= target_resident_bytes:
                return 0
            # LRU order: oldest access first. Only free (ref_count==0), still-cached
            # blocks are eligible; null block and in-flight blocks are excluded.
            candidates = [
                b
                for b in self.blocks
                if b.ref_count == 0
                and not b.is_null
                and b.cache_data is not None
                and b.block_hash is not None
                and not b.keep_resident
            ]
            candidates.sort(key=lambda b: b.last_access)
            for block in candidates:
                if self.resident_bytes <= target_resident_bytes:
                    break
                if self._maybe_evict_cached_block(block):
                    evicted += 1

            forced = 0
            if self.resident_bytes > target_resident_bytes:
                forced = self._force_drop_undrainable_locked(
                    target_resident_bytes
                )
                evicted += forced
        if evicted and pressure_overage > 0:
            logger.info(
                "Paged L1 Metal-pressure eviction: evicted %d block(s) "
                "(%.2fGB overage) to keep active Metal memory under the "
                "working-set guard threshold",
                evicted,
                pressure_overage / 1024**3,
            )
        return evicted

    def _force_drop_undrainable_locked(self, target_resident_bytes: int) -> int:
        """Shed ``keep_resident`` payloads that can never reach L2. Lock held.

        A native composite payload is pinned resident so its RAM mirror outlives
        the async L2 write. When that write can never land — no disk store is
        configured, or the store rejected it and the retry deadline has passed —
        the pin is permanent, the block is excluded from every ordinary
        candidate list, and the byte ceiling becomes unenforceable. Production
        reaches this state (``prefix_cache`` pins native state even with no disk
        store, and re-pins it after a write failure), and the pool then grows
        until the process is killed.

        Dropping an unreferenced block costs a re-prefill, never correctness, so
        it is strictly better than that. Two guards keep it safe: blocks whose
        write is still legitimately in flight are left alone, and only leaves are
        dropped — peeling one layer at a time — so a parent is never removed out
        from under a cached delta-chain descendant.
        """
        now = time.monotonic()

        def drainable_soon(block: CacheBlock) -> bool:
            if self._disk_store is None:
                return False
            return bool(block.durability_write_pending) and now < (
                block.durability_retry_after or 0.0
            )

        dropped = 0
        while self.resident_bytes > target_resident_bytes:
            live_parents = {
                b.parent_hash
                for b in self.blocks
                if b.cache_data is not None and b.parent_hash is not None
            }
            leaves = [
                b
                for b in self.blocks
                if b.ref_count == 0
                and not b.is_null
                and b.cache_data is not None
                and b.block_hash is not None
                and b.keep_resident
                and b.block_hash not in live_parents
                and not drainable_soon(b)
            ]
            if not leaves:
                break
            leaves.sort(key=lambda b: b.last_access)
            progressed = False
            for block in leaves:
                if self.resident_bytes <= target_resident_bytes:
                    break
                if self._maybe_evict_cached_block(block, force=True):
                    dropped += 1
                    progressed = True
            if not progressed:
                break

        if dropped:
            logger.warning(
                "Paged L1 byte budget: dropped %d pinned block(s) with no "
                "reachable L2 destination to honor the resident ceiling (%s); "
                "those prefixes re-prefill on next use",
                dropped,
                "no disk cache configured"
                if self._disk_store is None
                else "L2 writes are failing",
            )
        return dropped

    def _metal_pressure_overage_bytes(self) -> int:
        """Query Metal and return the current pressure overage in bytes.

        Returns 0 whenever pressure eviction is disabled, Metal telemetry is
        unavailable, or active memory (plus the transient margin) still fits
        under the guard threshold.
        """
        if not paged_metal_pressure_evict_enabled():
            return 0
        try:
            import mlx.core as mx  # local: keep CPU-only/test paths import-safe
        except Exception:
            return 0
        if not hasattr(mx, "get_active_memory"):
            return 0
        try:
            from .utils.memory_limits import (
                get_effective_metal_working_set_bytes,
                get_metal_ws_guard_threshold,
            )

            active_bytes, max_ws_bytes = get_effective_metal_working_set_bytes(
                mx
            )
            threshold_pct = get_metal_ws_guard_threshold()
        except Exception:
            return 0
        return compute_metal_pressure_overage_bytes(
            active_bytes,
            max_ws_bytes,
            threshold_pct,
            paged_metal_pressure_margin_bytes(),
        )

    def free_block(self, block_id: int) -> bool:
        """
        Free a cache block (decrements ref_count, frees if 0).

        Returns:
            True if block was freed, False if still referenced.
        """
        with self._lock:
            if block_id not in self.allocated_blocks:
                logger.warning(f"Attempted to free unknown block: {block_id}")
                return False

            block = self.allocated_blocks[block_id]
            if block.is_null:
                return False  # Never free null block

            # Same guard as free_blocks() and release_request_refs(); free_block
            # was the only one of the three without it. A released block is left
            # in allocated_blocks at ref_count 0 AND linked into the free queue
            # (release_request_refs, above) — that is a designed steady state, so
            # an unpaired free here drove ref_count to -1 and relinked a block
            # that was already queued. Measured consequences: the counter reports
            # one more free block than the queue can reach, so popleft raises
            # "No free blocks available" while claiming space; mid-queue, stale
            # neighbour pointers orphan the tail; and a negative ref_count defeats
            # the `ref_count == 0` revival guards in touch()/increment_ref(), which
            # hands one physical block to two requests.
            if block.ref_count <= 0:
                logger.warning(
                    f"free_block: block {block_id} already has "
                    f"ref_count={block.ref_count}; refusing to double-free"
                )
                return False

            block.ref_count -= 1

            if block.ref_count <= 0:
                if block.release_resident_when_unreferenced:
                    self._release_resident_payload_locked(block)
                # Remove from allocated
                del self.allocated_blocks[block_id]

                # Only queue a block that is not already linked, mirroring
                # release_request_refs — appending a queued block corrupts the
                # intrusive free list.
                if block.prev_free_block is None and block.next_free_block is None:
                    self.free_block_queue.append(block)
                    self.stats.free_blocks += 1

                self.stats.allocated_blocks -= 1

                return True

            return False

    def free_blocks(self, blocks: Iterable[CacheBlock]) -> None:
        """
        Free multiple blocks (vLLM style).

        Blocks with ref_count=0 are added to the free queue.

        Args:
            blocks: Blocks to free (in eviction order)
        """
        with self._lock:
            blocks_list = list(blocks)
            to_free = []

            for block in blocks_list:
                if block.is_null:
                    continue

                # Guard against double-free (ref_count already 0)
                if block.ref_count <= 0:
                    continue

                block.ref_count -= 1

                if block.ref_count <= 0:
                    if block.release_resident_when_unreferenced:
                        self._release_resident_payload_locked(block)
                    self.allocated_blocks.pop(block.block_id, None)
                    to_free.append(block)
                    self.stats.allocated_blocks -= 1
                    self.stats.free_blocks += 1

            # Add to free queue (back = MRU, evicted last)
            self.free_block_queue.append_n(to_free)

    def touch(self, blocks: Iterable[CacheBlock]) -> None:
        """
        Touch blocks to prevent eviction (cache hit, vLLM style).

        Increments ref_count and removes from free queue if needed.

        Args:
            blocks: Blocks to touch
        """
        with self._lock:
            for block in blocks:
                if block.ref_count == 0 and not block.is_null:
                    # Block is in free queue, remove it
                    try:
                        self.free_block_queue.remove(block)
                        self.stats.free_blocks -= 1
                        self.stats.allocated_blocks += 1
                        self.allocated_blocks[block.block_id] = block
                    except RuntimeError:
                        pass  # Block not in queue

                block.ref_count += 1
                block.touch()

    # =========================================================================
    # Reference Counting
    # =========================================================================

    def increment_ref(self, block_id: int) -> bool:
        """Increment reference count for a block."""
        with self._lock:
            if block_id not in self.allocated_blocks:
                return False

            block = self.allocated_blocks[block_id]
            if block.ref_count == 0 and not block.is_null:
                # Cached-but-free block: revive it for this request and remove
                # it from the free LRU queue so it cannot be reallocated while
                # a block table is actively using it.
                if block.prev_free_block is not None or block.next_free_block is not None:
                    try:
                        self.free_block_queue.remove(block)
                        self.stats.free_blocks = max(0, self.stats.free_blocks - 1)
                        self.stats.allocated_blocks += 1
                    except RuntimeError:
                        pass
            block.ref_count += 1
            block.touch()

            if block.ref_count == 2:
                self.stats.shared_blocks += 1

            return True

    def decrement_ref(self, block_id: int) -> bool:
        """Decrement reference count (alias for free_block)."""
        return self.free_block(block_id)

    # =========================================================================
    # Prefix Caching (vLLM chain-hash style)
    # =========================================================================

    def get_cached_block(self, block_hash: BlockHash) -> Optional[CacheBlock]:
        """
        Get a cached block by its hash (vLLM style).

        Args:
            block_hash: Content hash of the block

        Returns:
            Cached block if found, None otherwise
        """
        if not self.enable_caching:
            return None

        with self._lock:
            block = self.cached_block_hash_to_block.get_block(block_hash)
            if block:
                self.stats.cache_hits += 1
            else:
                self.stats.cache_misses += 1
            return block

    def cache_full_blocks(
        self,
        blocks: List[CacheBlock],
        token_ids: List[int],
        num_cached_blocks: int,
        num_full_blocks: int,
        extra_keys: Optional[Any] = None,
    ) -> None:
        """
        Cache full blocks for prefix caching (vLLM style).

        Computes chain hashes and adds blocks to the cache.

        Args:
            blocks: All blocks for the request
            token_ids: All token IDs for the request
            num_cached_blocks: Number of blocks already cached
            num_full_blocks: Number of full blocks to cache
        """
        if not self.enable_caching:
            return

        if num_cached_blocks >= num_full_blocks:
            return

        with self._lock:
            # Get parent hash from last cached block
            parent_hash = None
            if num_cached_blocks > 0:
                parent_hash = blocks[num_cached_blocks - 1].block_hash

            for i in range(num_cached_blocks, num_full_blocks):
                block = blocks[i]
                if block.block_hash is not None:
                    parent_hash = block.block_hash
                    continue  # Already cached

                # Get tokens for this block
                start = i * self.block_size
                end = start + self.block_size
                block_tokens = token_ids[start:end]

                # Compute chain hash
                block_hash = compute_block_hash(
                    parent_hash,
                    block_tokens,
                    extra_keys=extra_keys,
                )
                block.block_hash = block_hash
                block.parent_hash = parent_hash
                block.token_count = len(block_tokens)

                # Add to cache
                self.cached_block_hash_to_block.insert(block_hash, block)

                # Also maintain legacy hash for compatibility
                legacy_hash = self.compute_block_hash(block_tokens)
                block.hash_value = legacy_hash
                self.hash_to_block[legacy_hash] = block.block_id

                parent_hash = block_hash

    def get_computed_blocks(
        self,
        token_ids: List[int],
        extra_keys: Optional[Any] = None,
    ) -> Tuple[List[CacheBlock], int]:
        """
        Find cached blocks for a token prefix (vLLM style).

        Args:
            token_ids: Token IDs to look up

        Returns:
            Tuple of (cached_blocks, num_cached_tokens)
        """
        if not self.enable_caching:
            return [], 0

        cached_blocks = []
        parent_hash = None
        num_cached_tokens = 0

        num_full_blocks = len(token_ids) // self.block_size

        for i in range(num_full_blocks):
            start = i * self.block_size
            end = start + self.block_size
            block_tokens = token_ids[start:end]

            # Compute expected hash
            block_hash = compute_block_hash(
                parent_hash,
                block_tokens,
                extra_keys=extra_keys,
            )

            if _CACHE_HASH_DEBUG:
                logger.info(
                    "cache-hash-debug FETCH pos=%d hash=%s parent=%s "
                    "tok[:4]=%s tok[-4:]=%s n=%d extra=%r",
                    i,
                    block_hash.hex()[:16],
                    parent_hash.hex()[:16] if parent_hash else None,
                    block_tokens[:4],
                    block_tokens[-4:],
                    len(block_tokens),
                    extra_keys,
                )

            # Look up in L1 cache (under lock)
            with self._lock:
                cached_block = self.cached_block_hash_to_block.get_block(block_hash)
                if cached_block is not None:
                    # Hold a request reference for the block returned to the
                    # caller. This also revives cached-but-free blocks from the
                    # free LRU queue before they can be reallocated.
                    self.touch([cached_block])

            if cached_block is None and self._disk_store is not None:
                # L1 miss — check L2 disk cache (outside lock to avoid blocking)
                disk_data = self._disk_store.read_block(block_hash)
                if disk_data is not None:
                    with self._lock:
                        # Re-check L1 after releasing lock (another thread may have promoted)
                        cached_block = self.cached_block_hash_to_block.get_block(block_hash)
                        if cached_block is None:
                            promoted = self._promote_from_disk(
                                block_hash,
                                disk_data,
                                len(block_tokens),
                                parent_hash=parent_hash,
                            )
                            if promoted is not None:
                                cached_block = promoted
                                self.stats.disk_hits += 1
                        else:
                            self.touch([cached_block])

            if cached_block is None:
                # A previous request may have ended in a partial block. After
                # restart there is no in-memory prefix index, and a longer
                # continuation probes a full block first. Fall back to the
                # durable terminal sizes at this exact chain position before
                # declaring a miss. Return after the partial hit: the caller
                # can safely prefill the tail, while store_cache realigns the
                # extended chain to block boundaries.
                for partial_size in sorted(
                    self._partial_block_sizes, reverse=True
                ):
                    if (
                        partial_size <= 0
                        or partial_size >= self.block_size
                        or start + partial_size > len(token_ids)
                    ):
                        continue
                    partial_tokens = token_ids[start : start + partial_size]
                    partial_hash = compute_block_hash(
                        parent_hash,
                        partial_tokens,
                        extra_keys=extra_keys,
                    )
                    with self._lock:
                        partial_block = (
                            self.cached_block_hash_to_block.get_block(
                                partial_hash
                            )
                        )
                        if partial_block is not None:
                            self.touch([partial_block])
                    if partial_block is None and self._disk_store is not None:
                        disk_data = self._disk_store.read_block(partial_hash)
                        if disk_data is not None:
                            with self._lock:
                                partial_block = (
                                    self.cached_block_hash_to_block.get_block(
                                        partial_hash
                                    )
                                )
                                if partial_block is None:
                                    partial_block = self._promote_from_disk(
                                        partial_hash,
                                        disk_data,
                                        partial_size,
                                        parent_hash=parent_hash,
                                    )
                                    if partial_block is not None:
                                        self.stats.disk_hits += 1
                                else:
                                    self.touch([partial_block])
                    if partial_block is not None:
                        cached_blocks.append(partial_block)
                        num_cached_tokens += partial_size
                        with self._lock:
                            self.stats.cache_hits += 1
                        return cached_blocks, num_cached_tokens
                with self._lock:
                    self.stats.cache_misses += 1
                    if self._disk_store is not None:
                        self.stats.disk_misses += 1
                break  # Cache miss, stop here

            cached_blocks.append(cached_block)
            parent_hash = block_hash
            num_cached_tokens += self.block_size
            with self._lock:
                self.stats.cache_hits += 1

        # Also check partial (trailing) block after all full blocks matched.
        # Without this, short prompts (< block_size) never get disk hits.
        #
        # Important: partial-prefix matching must still use the parent chain
        # hash. The legacy hash_to_block map keys only on the block's token
        # content; that is unsafe for real KV state because a repeated block of
        # text has different hidden/KV values under a different preceding
        # context. Try the exact remaining length first, then known in-memory
        # partial sizes from largest to smallest so a request can reuse a
        # previously stored terminal partial block when extending that prompt.
        remaining_tokens = token_ids[num_cached_tokens:]
        if remaining_tokens and len(remaining_tokens) < self.block_size:
            partial_sizes: List[int] = [len(remaining_tokens)]
            for size in sorted(self._partial_block_sizes, reverse=True):
                if 0 < size <= len(remaining_tokens) and size not in partial_sizes:
                    partial_sizes.append(size)

            for size in partial_sizes:
                block_tokens = remaining_tokens[:size]
                block_hash = compute_block_hash(
                    parent_hash,
                    block_tokens,
                    extra_keys=extra_keys,
                )

                with self._lock:
                    cached_block = self.cached_block_hash_to_block.get_block(block_hash)
                    if cached_block is not None:
                        # Hold a request reference for the block returned to the
                        # caller. This also revives cached-but-free blocks from the
                        # free LRU queue before they can be reallocated.
                        self.touch([cached_block])

                if cached_block is None and self._disk_store is not None:
                    disk_data = self._disk_store.read_block(block_hash)
                    if disk_data is not None:
                        with self._lock:
                            cached_block = self.cached_block_hash_to_block.get_block(block_hash)
                            if cached_block is None:
                                promoted = self._promote_from_disk(
                                    block_hash,
                                    disk_data,
                                    size,
                                    parent_hash=parent_hash,
                                )
                                if promoted is not None:
                                    cached_block = promoted
                                    self.stats.disk_hits += 1
                            else:
                                self.touch([cached_block])

                if cached_block is not None:
                    cached_blocks.append(cached_block)
                    num_cached_tokens += cached_block.token_count
                    with self._lock:
                        self.stats.cache_hits += 1
                    break

        return cached_blocks, num_cached_tokens

    # =========================================================================
    # Disk L2 promotion
    # =========================================================================

    def _promote_from_disk(
        self,
        block_hash: BlockHash,
        cache_data: List[Tuple[Any, ...]],
        token_count: int,
        *,
        parent_hash: Optional[BlockHash] = None,
    ) -> Optional[CacheBlock]:
        """
        Promote a block from disk L2 into L1 RAM.

        Allocates a new CacheBlock, populates it with the loaded tensor data,
        and registers it in the hash cache so future lookups are L1 hits.

        Args:
            block_hash: Chain hash for this block
            cache_data: Deserialized cache data from disk
            token_count: Number of tokens in the block

        Returns:
            The promoted CacheBlock, or None if allocation failed.
        """
        resident_nbytes = self.estimate_block_nbytes(cache_data)
        persistent_l1_admitted = not self.disk_only and not self.paged_frugal
        if self.max_resident_bytes > 0 and resident_nbytes > 0:
            # L2 prefix lookup promotes blocks one at a time. Reserve room
            # before assigning the request ref: after promotion ref_count=1,
            # so normal byte-budget eviction must (correctly) leave the block
            # alone. Without this gate a long disk prefix can promote far past
            # the configured L1 ceiling and thrash during worker reconstruction.
            # If the persistent tier cannot admit another request-owned block,
            # keep the L2 payload transient instead of truncating the longest
            # match. Reconstruction releases transient bytes immediately.
            self.enforce_byte_budget(required_bytes=resident_nbytes)
            with self._lock:
                if (
                    resident_nbytes > self.max_resident_bytes
                    or self.resident_bytes + resident_nbytes
                    > self.max_resident_bytes
                ):
                    persistent_l1_admitted = False
                    logger.info(
                        "Using transient L2 reconstruction payload: persistent "
                        "RAM cache would exceed byte ceiling (%d + %d > %d bytes)",
                        self.resident_bytes,
                        resident_nbytes,
                        self.max_resident_bytes,
                    )

        with self._lock:
            if self.free_block_queue.num_free_blocks == 0:
                return None
            block = self._pop_recyclable_free_block_locked()
            if block is None:
                logger.warning(
                    "Cannot promote L2 block: every free L1 slot is waiting "
                    "for durable write admission"
                )
                return None

        # Populate
        block.ref_count = 1
        block.block_hash = block_hash
        block.parent_hash = parent_hash
        block.cache_data = cache_data
        block.cache_data_from_disk = True
        block.cache_data_transient = not persistent_l1_admitted
        block.dsv4_native_interval = _native_dsv4_interval_from_payload(cache_data)
        block.token_count = token_count
        block.touch()
        if self.max_resident_bytes > 0 or self.disk_only:
            self._note_resident(block, resident_nbytes)
        if block.cache_data_transient:
            self.transient_disk_promotions += 1
            self.transient_disk_peak_bytes = max(
                self.transient_disk_peak_bytes,
                self.resident_bytes,
            )
        self.allocated_blocks[block.block_id] = block

        # Register in hash cache
        self.cached_block_hash_to_block.insert(block_hash, block)

        self.stats.allocated_blocks += 1
        self.stats.free_blocks -= 1

        logger.debug(
            f"Promoted block from disk: id={block.block_id}, "
            f"hash={block_hash.hex()[:12]}, tokens={token_count}"
        )
        return block

    # =========================================================================
    # Legacy hash methods (for backwards compatibility)
    # =========================================================================

    @staticmethod
    def compute_block_hash(tokens: List[int]) -> str:
        """Compute legacy string hash for a sequence of tokens."""
        # Use full token values (not truncated to byte) for accurate hashing
        token_str = ",".join(str(t) for t in tokens)
        return hashlib.sha256(token_str.encode()).hexdigest()[:16]

    def find_cached_block(self, tokens: List[int]) -> Optional[CacheBlock]:
        """
        Find a cached block matching the given tokens (legacy method).
        """
        with self._lock:
            hash_value = self.compute_block_hash(tokens)

            if hash_value in self.hash_to_block:
                block_id = self.hash_to_block[hash_value]
                if block_id in self.allocated_blocks:
                    block = self.allocated_blocks[block_id]
                    block.touch()
                    self.stats.cache_hits += 1
                    return block

            self.stats.cache_misses += 1
            return None

    def register_block_hash(self, block: CacheBlock, tokens: List[int]) -> None:
        """Register a block's hash for deduplication (legacy method)."""
        with self._lock:
            hash_value = self.compute_block_hash(tokens)
            block.hash_value = hash_value
            self.hash_to_block[hash_value] = block.block_id
            # Track partial block sizes for sub-prefix matching
            if len(tokens) < self.block_size:
                self._partial_block_sizes.add(len(tokens))

    # =========================================================================
    # Block Table Management
    # =========================================================================

    def create_block_table(self, request_id: str) -> BlockTable:
        """Create a new block table for a request."""
        with self._lock:
            table = BlockTable(request_id=request_id)
            self.request_tables[request_id] = table
            return table

    def get_block_table(self, request_id: str) -> Optional[BlockTable]:
        """Get block table for a request."""
        with self._lock:
            return self.request_tables.get(request_id)

    def get_or_create_block_table(self, request_id: str) -> BlockTable:
        """Get or create block table for a request."""
        with self._lock:
            if request_id not in self.request_tables:
                self.request_tables[request_id] = BlockTable(request_id=request_id)
            return self.request_tables[request_id]

    def delete_block_table(self, request_id: str) -> None:
        """Delete block table and free associated blocks."""
        with self._lock:
            table = self.request_tables.pop(request_id, None)
            if table:
                for block_id in table.block_ids:
                    self.free_block(block_id)

    def detach_request(self, request_id: str) -> None:
        """Remove request_tables entry without freeing blocks.

        Use after a request completes and its blocks have been stored
        in the cache for future reuse. Unlike delete_block_table(),
        this preserves the blocks for LRU-based sharing/eviction.
        """
        with self._lock:
            self.request_tables.pop(request_id, None)

    def release_request_refs(self, block_table: Optional[BlockTable]) -> int:
        """Release request-held refs while keeping zero-ref blocks cache-resident.

        This implements "cached but free" semantics:
        - decrement one ref per block-table entry,
        - when a block reaches ref_count==0, keep its hash entries intact,
          but move it to the free LRU queue so memory pressure can reclaim it.

        Returns:
            Number of blocks that transitioned to ref_count==0.
        """
        if block_table is None:
            return 0

        with self._lock:
            released = 0

            for block_id in block_table.block_ids:
                block = self.allocated_blocks.get(block_id)
                if block is None or block.is_null:
                    continue

                if block.ref_count <= 0:
                    logger.warning(
                        f"release_request_refs: block {block_id} already has "
                        f"ref_count={block.ref_count}; skipping"
                    )
                    continue

                block.ref_count -= 1

                # Keep cache mapping (block_hash/hash_value) so prefix hits can
                # revive this block, but make it reclaimable via free LRU queue.
                if block.ref_count == 0:
                    if block.release_resident_when_unreferenced:
                        self._release_resident_payload_locked(block)
                    if block.prev_free_block is None and block.next_free_block is None:
                        self.free_block_queue.append(block)
                        self.stats.allocated_blocks = max(0, self.stats.allocated_blocks - 1)
                        self.stats.free_blocks += 1
                    released += 1

        # A just-completed request kept every one of its blocks ineligible for
        # eviction while ref_count was positive. Enforce the configured RAM
        # byte ceiling immediately after those refs become free; otherwise a
        # large native DSV4 prompt can remain above the user's limit until an
        # unrelated later request happens to store or promote another block.
        if released and self.max_resident_bytes > 0:
            self.enforce_byte_budget()
        return released

    def add_block_to_table(
        self,
        table: BlockTable,
        block: CacheBlock,
        tokens_in_block: int,
    ) -> None:
        """Add a block to a block table."""
        with self._lock:
            table.block_ids.append(block.block_id)
            block.token_count = tokens_in_block
            table.num_tokens += tokens_in_block

    # =========================================================================
    # Prefix Sharing & COW
    # =========================================================================

    def find_shared_prefix(
        self,
        tokens: List[int],
    ) -> Tuple[List[int], List[int]]:
        """
        Find shared prefix blocks for a token sequence.

        Greedily matches cached blocks (full and partial) from the start
        of the token sequence. Keeps matching until no more cached blocks
        are found, allowing multi-turn conversations to reuse caches from
        all previous turns.
        """
        with self._lock:
            shared_blocks = []
            remaining_tokens = tokens.copy()

            while remaining_tokens:
                matched = False

                # Try full block match first
                if len(remaining_tokens) >= self.block_size:
                    chunk = remaining_tokens[: self.block_size]
                    cached_block = self.find_cached_block(chunk)
                    if cached_block:
                        shared_blocks.append(cached_block.block_id)
                        remaining_tokens = remaining_tokens[self.block_size :]
                        matched = True

                # Try partial block match (known stored sizes, largest first)
                if not matched and self._partial_block_sizes:
                    for size in sorted(self._partial_block_sizes, reverse=True):
                        if size <= len(remaining_tokens):
                            cached_block = self.find_cached_block(remaining_tokens[:size])
                            if cached_block:
                                shared_blocks.append(cached_block.block_id)
                                remaining_tokens = remaining_tokens[size:]
                                matched = True
                                break

                if not matched:
                    break

            return shared_blocks, remaining_tokens

    def fork_block_table(
        self,
        source_table: BlockTable,
        new_request_id: str,
    ) -> BlockTable:
        """
        Fork a block table for a new request (COW).
        """
        with self._lock:
            new_table = source_table.copy(new_request_id)

            for block_id in new_table.block_ids:
                self.increment_ref(block_id)

            self.request_tables[new_request_id] = new_table

            logger.debug(
                f"Forked block table: {source_table.request_id} -> {new_request_id}, "
                f"blocks={len(new_table.block_ids)}"
            )

            return new_table

    def get_blocks_for_generation(
        self,
        table: BlockTable,
    ) -> Tuple[List[CacheBlock], bool]:
        """
        Get blocks for generation, applying COW if needed.
        """
        with self._lock:
            blocks = []
            was_copied = False

            for i, block_id in enumerate(table.block_ids):
                block = self.allocated_blocks.get(block_id)
                if not block:
                    continue

                if block.is_shared():
                    new_block = self._cow_copy_block(block)
                    if new_block:
                        table.block_ids[i] = new_block.block_id
                        blocks.append(new_block)
                        was_copied = True
                        self.stats.cow_copies += 1
                    else:
                        blocks.append(block)
                else:
                    blocks.append(block)

                block.touch()

            return blocks, was_copied

    def _cow_copy_block(self, source_block: CacheBlock) -> Optional[CacheBlock]:
        """Create a copy of a block for COW."""
        new_block = self.allocate_block()
        if not new_block:
            return None

        new_block.token_count = source_block.token_count
        new_block.cache_data = source_block.cache_data
        new_block.cache_data_from_disk = source_block.cache_data_from_disk

        source_block.ref_count -= 1
        if source_block.ref_count == 1:
            self.stats.shared_blocks -= 1

        logger.debug(f"COW copy: block {source_block.block_id} -> {new_block.block_id}")

        return new_block

    # =========================================================================
    # Legacy allocation methods (for backwards compatibility)
    # =========================================================================

    def allocate_blocks_for_tokens(self, num_tokens: int) -> List[CacheBlock]:
        """Allocate enough blocks to hold num_tokens."""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        return self.get_new_blocks(num_blocks_needed)

    # =========================================================================
    # Eviction
    # =========================================================================

    def evict_lru_blocks(self, num_blocks: int) -> int:
        """
        Evict least recently used blocks.

        With the doubly linked list, LRU blocks are already at the front
        of the free queue. We just need to pop from front.
        """
        with self._lock:
            evicted = 0

            # Get evictable blocks from free queue (they're already LRU ordered)
            for _ in range(min(num_blocks, self.free_block_queue.num_free_blocks)):
                try:
                    block = self.free_block_queue.popleft()
                    was_clean = block.block_hash is None
                    did_evict = self._maybe_evict_cached_block(block)
                    # Put back at MRU. A durability-pending block remains in
                    # the queue but allocator scans will skip it safely.
                    self.free_block_queue.append(block)
                    if was_clean or did_evict:
                        evicted += 1
                except ValueError:
                    break

            if evicted > 0:
                logger.info(f"Evicted {evicted} LRU blocks from cache")

            return evicted

    def handle_memory_pressure(self, requested_blocks: int) -> bool:
        """Handle memory pressure by evicting blocks."""
        with self._lock:
            if self.free_block_queue.num_free_blocks >= requested_blocks:
                return True

            needed = requested_blocks - self.free_block_queue.num_free_blocks
            self.evict_lru_blocks(needed)
            self.stats.free_blocks = self.free_block_queue.num_free_blocks

            return self.free_block_queue.num_free_blocks >= requested_blocks

    # =========================================================================
    # Statistics and Properties
    # =========================================================================

    @property
    def free_blocks(self) -> int:
        """Number of free blocks available."""
        return self.free_block_queue.num_free_blocks

    @property
    def usage(self) -> float:
        """Cache usage ratio (0.0 to 1.0)."""
        total = self.max_blocks - 1  # Exclude null block
        if total == 0:
            return 0.0
        return 1.0 - (self.free_blocks / total)

    def get_stats(self) -> CacheStats:
        """Get current cache statistics."""
        with self._lock:
            self.stats.shared_blocks = sum(
                1 for b in self.allocated_blocks.values() if b.ref_count > 1
            )
            self.stats.free_blocks = self.free_block_queue.num_free_blocks
            # Token totals are derived from the live block table, not maintained
            # incrementally, so reused/free/evicted block paths cannot drift.
            self.stats.total_tokens_cached = sum(
                max(0, int(getattr(b, "token_count", 0) or 0))
                for b in self.allocated_blocks.values()
                if not getattr(b, "is_null", False)
                and (
                    getattr(b, "ref_count", 0) > 0
                    or getattr(b, "block_hash", None) is not None
                    or getattr(b, "cache_data", None) is not None
                )
            )
            return self.stats

    def _live_cached_blocks(self) -> int:
        """Blocks currently holding reusable cached content.

        Derived from the live block table for the same reason
        ``total_tokens_cached`` is: incrementally maintained counters drift
        across the reuse/free/evict paths.

        This is deliberately *not* ``allocated_blocks``. That counter reflects
        the paged allocator's own pin state, which architecture-native caches
        (DSV4's SWA+CSA/HCA composite) never drive — it was observed pinned at
        1 with ``free_blocks`` at capacity across 48 samples taken during a live
        4509-token prefill, while ``total_tokens_cached`` rose from 1751 to
        6257. Occupancy reported from the allocator counters therefore reads
        0% on a cache holding tens of thousands of reusable tokens.
        """
        return sum(
            1
            for b in self.allocated_blocks.values()
            if not getattr(b, "is_null", False)
            and (
                getattr(b, "ref_count", 0) > 0
                or getattr(b, "block_hash", None) is not None
                or getattr(b, "cache_data", None) is not None
            )
        )

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information."""
        with self._lock:
            stats = self.get_stats()
            total_hits = stats.cache_hits + stats.cache_misses
            usable_blocks = max(0, self.max_blocks - 1)
            allocated_usable_blocks = max(0, stats.allocated_blocks - 1)
            cached_blocks = self._live_cached_blocks()
            return {
                "cached_blocks": cached_blocks,
                "cache_occupancy": (
                    cached_blocks / usable_blocks if usable_blocks > 0 else 0.0
                ),
                "backend_mode": "block_disk_only" if self.disk_only else "paged",
                "paged_ram_enabled": not self.disk_only,
                "disk_only": self.disk_only,
                "paged_frugal": self.paged_frugal,
                "ram_mirror_policy": self.ram_mirror_policy,
                "block_size": self.block_size,
                "max_blocks": self.max_blocks,
                "usable_blocks": usable_blocks,
                "capacity_tokens": self.block_size * usable_blocks,
                "allocated_blocks": stats.allocated_blocks,
                "free_blocks": stats.free_blocks,
                "shared_blocks": stats.shared_blocks,
                "total_tokens_cached": stats.total_tokens_cached,
                "utilization": (
                    allocated_usable_blocks / usable_blocks
                    if usable_blocks > 0
                    else 0.0
                ),
                "cache_hit_rate": stats.cache_hits / total_hits if total_hits > 0 else 0,
                "cache_hits": stats.cache_hits,
                "cache_misses": stats.cache_misses,
                "disk_hits": stats.disk_hits,
                "disk_misses": stats.disk_misses,
                "cow_copies": stats.cow_copies,
                "evictions": stats.evictions,
                "transient_disk_promotions": self.transient_disk_promotions,
                "transient_disk_peak_bytes": self.transient_disk_peak_bytes,
            }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        with self._lock:
            self.stats.cache_hits = 0
            self.stats.cache_misses = 0
            self.stats.cow_copies = 0
            self.stats.evictions = 0

    def reset_prefix_cache(self) -> bool:
        """Reset the prefix cache."""
        with self._lock:
            num_used = self.max_blocks - self.free_block_queue.num_free_blocks
            if num_used > 1:  # null_block is always "used"
                logger.warning(f"Cannot reset cache: {num_used - 1} blocks in use")
                return False

            self.cached_block_hash_to_block.clear()
            self.hash_to_block.clear()

            for block in self.blocks:
                block.reset_hash()
                block.cache_data = None
                block.cache_data_from_disk = False
                block.resident_bytes = 0

            # Every cache_data mirror was just dropped; the resident-byte
            # accounting must return to zero or it stays a phantom positive that
            # makes enforce_byte_budget over-evict every future store forever.
            self.resident_bytes = 0

            self.stats.evictions = 0
            self.stats.cache_hits = 0
            self.stats.cache_misses = 0

            logger.info("Prefix cache reset successfully")
            return True

    def blocks_in_use(self) -> int:
        """How many non-null blocks a live request still holds a reference to."""
        with self._lock:
            return sum(
                1
                for block in self.allocated_blocks.values()
                if block is not None and not block.is_null and block.ref_count > 0
            )

    def clear(self, force: bool = False) -> bool:
        """Clear all cached data.

        Refuses while a live request still holds blocks, the same guard
        ``reset_prefix_cache`` has always had. Without it the pool is rebuilt
        with block ids 0..N reused immediately, while a surviving request still
        holds its old BlockTable: its completion path resolves those ids through
        ``allocated_blocks`` and so decrements, frees and re-hands-out blocks that
        now belong to a DIFFERENT request. That is the cross-request wrong-answer
        class, reachable straight from ``DELETE /v1/cache`` mid-generation.

        Teardown callers (batch generator already closed, or a nuclear retry that
        is rebuilding everything anyway) pass ``force=True``.

        Returns True when the pool was cleared, False when it refused.
        """
        if not force:
            in_use = self.blocks_in_use()
            if in_use:
                logger.warning(
                    f"Cannot clear paged cache: {in_use} block(s) still "
                    f"referenced by live requests"
                )
                return False
        with self._lock:
            # Recreate blocks and queue
            self.blocks = [CacheBlock(block_id=i) for i in range(self.max_blocks)]
            self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

            self.cached_block_hash_to_block.clear()
            self.hash_to_block.clear()
            self.request_tables.clear()
            self.allocated_blocks.clear()

            # Fresh block pool holds no cache_data mirror; drop the resident-byte
            # accounting to zero so the byte ceiling doesn't over-evict on the
            # next store (the recreated blocks default resident_bytes=0).
            self.resident_bytes = 0

            # Reserve null block
            self.null_block = self.free_block_queue.popleft()
            self.null_block.is_null = True
            self.null_block.ref_count = 1
            self.allocated_blocks[self.null_block.block_id] = self.null_block

            self.stats = CacheStats(
                total_blocks=self.max_blocks,
                allocated_blocks=1,
                free_blocks=self.max_blocks - 1,
            )

            logger.info("PagedCacheManager cleared")
            return True
