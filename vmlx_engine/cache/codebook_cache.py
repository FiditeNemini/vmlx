"""
Codebook Weight Cache - RAM LRU for experimental codebook-VQ expert weights.

Expert weights are stored as codebook + indices (compressed).
On cache miss, we:
    1. Load codebook and indices from safetensors
    2. Retain the compressed tuple in memory (LRU)
    3. Reconstruct or fuse the requested matmul at use time
    4. On memory pressure, evict LRU entries and reload the original compressed
       model shard on the next miss

Memory budget is unlimited by default.  This cache deliberately has no SSD
spill: the compressed ``(codebook, indices)`` tuple already lives in the model
artifact, so duplicating it under a global unscoped cache root adds correctness,
capacity, and concurrency hazards without preserving unique data.

Usage:
    cache = CodebookWeightCache(config)

    # Get compressed codebook data (lazy load)
    weights = cache.get((layer_idx, "gate_proj"))

    # Cache miss triggers:
    # 1. Load codebook-layer-{layer_idx}-gate_proj.safetensors
    # 2. Cache the compressed tuple in memory
"""

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry for reconstructed expert weights."""

    weights: Any  # mx.array or a nested tuple such as (codebook, indices)
    access_time: float = field(default_factory=time.time)
    size_bytes: int = 0


class CodebookWeightCache:
    """
    RAM-only LRU cache for codebook VQ expert weights.

    Features:
    - Unlimited RAM by default, or bounded RAM LRU when configured
    - Thread-safe operations
    - Evicted entries reload from their original model safetensors shard
    - Memory pressure monitoring
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        disk_cache_dir: str | None = None,
        disk_max_gb: int = 1000,
        eviction_batch_size: int = 4,
    ):
        """
        Initialize codebook weight cache.

        Args:
            config: Configuration dict with memory/disk settings.
            disk_cache_dir: Deprecated compatibility argument; ignored.
            disk_max_gb: Deprecated compatibility argument; ignored.
            eviction_batch_size: Number of entries to evict at once.
        """
        self._config = config or {}

        # Memory cache (LRU OrderedDict)
        self._memory_cache: OrderedDict[tuple[int, str], CacheEntry] = OrderedDict()
        memory_limit_mb = self._config.get("memory_limit_mb")
        normalized_memory_limit_mb = (
            None if memory_limit_mb is None else float(memory_limit_mb)
        )
        self._memory_budget_bytes: int | None = (
            None
            if normalized_memory_limit_mb is None or normalized_memory_limit_mb <= 0
            else int(normalized_memory_limit_mb * 1024 * 1024)
        )
        self._current_memory_bytes: int = 0
        self._eviction_batch_size = eviction_batch_size or 4

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        logger.info(
            "CodebookWeightCache initialized: RAM limit=%s; redundant SSD "
            "spill is disabled",
            "unlimited"
            if self._memory_budget_bytes is None
            else f"{self._memory_budget_bytes / 1024**2:.1f} MB",
        )

    def get(self, key: tuple[int, str]) -> Any | None:
        """
        Get reconstructed expert weights for a layer/tensor_type.

        Args:
            key: Tuple of (layer_idx, tensor_type), e.g., (0, "gate_proj")

        Returns:
            Reconstructed weights as mx.array, or None if not found.
        """
        with self._lock:
            # Check memory cache
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                entry.access_time = time.time()
                # Move to end (most recently used)
                self._memory_cache.move_to_end(key)
                self._hits += 1
                return entry.weights

            # Cache miss
            self._misses += 1
            return None

    def put(self, key: tuple[int, str], weights: Any):
        """
        Store reconstructed weights in cache.

        Args:
            key: Tuple of (layer_idx, tensor_type)
            weights: Reconstructed weight mx.array
        """
        with self._lock:
            self._put_memory(key, weights)

    @classmethod
    def _entry_nbytes(cls, value: Any) -> int:
        """Return physical tensor bytes for nested codebook cache values."""
        if isinstance(value, dict):
            return sum(cls._entry_nbytes(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return sum(cls._entry_nbytes(item) for item in value)
        try:
            return max(0, int(value.nbytes))
        except (AttributeError, TypeError, ValueError):
            return 0

    def _put_memory(self, key: tuple[int, str], weights: Any):
        """Put weights into memory cache, evicting LRU if needed."""
        size_bytes = self._entry_nbytes(weights)

        # Check if already in cache
        if key in self._memory_cache:
            old_entry = self._memory_cache.pop(key)
            self._current_memory_bytes -= old_entry.size_bytes

        # An entry larger than the complete RAM allowance remains usable by the
        # caller but is not retained. The model wrapper will reload its original
        # compressed shard on the next miss.
        if (
            self._memory_budget_bytes is not None
            and size_bytes > self._memory_budget_bytes
        ):
            logger.debug(
                "Skipping oversized codebook cache entry %s (%d > %d bytes)",
                key,
                size_bytes,
                self._memory_budget_bytes,
            )
            return

        # Evict if over budget (unlimited budget if memory_budget_bytes is None)
        while (
            self._memory_budget_bytes is not None
            and self._current_memory_bytes + size_bytes > self._memory_budget_bytes
            and self._memory_cache
        ):
            self._evict_lru()

        # Add to cache
        entry = CacheEntry(weights=weights, size_bytes=size_bytes)
        self._memory_cache[key] = entry
        self._current_memory_bytes += size_bytes

        logger.debug(
            f"Cached weights for {key}: {size_bytes / 1e6:.1f} MB, "
            f"total: {self._current_memory_bytes / 1e9:.2f} GB"
        )

    def _evict_lru(self):
        """Evict least recently used entries from RAM."""
        if not self._memory_cache:
            return

        # Get LRU entries (first N items)
        for _ in range(min(self._eviction_batch_size, len(self._memory_cache))):
            key, entry = self._memory_cache.popitem(last=False)
            self._current_memory_bytes -= entry.size_bytes

            self._evictions += 1

        logger.debug(
            f"Evicted {self._eviction_batch_size} entries, "
            f"freed {entry.size_bytes / 1e6:.1f} MB"
        )

    def prefetch(self, keys: list[tuple[int, str]]):
        """
        Prefetch weights for multiple keys.

        Args:
            keys: List of (layer_idx, tensor_type) tuples
        """
        for key in keys:
            if key not in self._memory_cache:
                # Trigger lazy load
                self.get(key)

    def invalidate(self, key: tuple[int, str]):
        """Invalidate a cache entry."""
        with self._lock:
            if key in self._memory_cache:
                entry = self._memory_cache.pop(key)
                self._current_memory_bytes -= entry.size_bytes

    def clear(self):
        """Clear all retained RAM entries."""
        with self._lock:
            self._memory_cache.clear()
            self._current_memory_bytes = 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "evictions": self._evictions,
            "disk_persistence": False,
            "disk_writes": 0,
            "disk_reads": 0,
            "memory_entries": len(self._memory_cache),
            "memory_bytes": self._current_memory_bytes,
            "disk_entries": 0,
            "disk_bytes": 0,
        }

    @property
    def memory_bytes(self) -> int:
        """Current memory usage in bytes."""
        return self._current_memory_bytes

    @property
    def disk_bytes(self) -> int:
        """Codebook-VQ cache does not duplicate model shards on disk."""
        return 0
