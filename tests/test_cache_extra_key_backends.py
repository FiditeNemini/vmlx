# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for generation-prompt discriminators on cache backends."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from vmlx_engine.cache_key import canonical_cache_extra_marker
from vmlx_engine.disk_cache import DiskCacheManager, _hash_tokens
from vmlx_engine.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig
from vmlx_engine.prefix_cache import PrefixCacheManager


def _kv_cache(tokens: int = 3) -> list[KVCache]:
    layer = KVCache()
    layer.update_and_fetch(
        mx.zeros((1, 1, tokens, 2)),
        mx.zeros((1, 1, tokens, 2)),
    )
    return [layer]


def test_memory_aware_cache_matches_same_extra_key_and_isolates_other_key():
    cache = MemoryAwarePrefixCache(
        MagicMock(),
        MemoryCacheConfig(max_memory_mb=32, max_entries=8),
    )
    on = {"generation_prompt": "v1:1:thinking-on"}
    off = {"generation_prompt": "v1:1:thinking-off"}

    assert cache.store([10, 20, 30], _kv_cache(), cache_extra_keys=on)

    hit, remaining = cache.fetch([10, 20, 30, 40], cache_extra_keys=on)
    assert hit is not None
    assert remaining == [40]

    miss, remaining = cache.fetch([10, 20, 30, 40], cache_extra_keys=off)
    assert miss is None
    assert remaining == [10, 20, 30, 40]


def test_legacy_prefix_cache_scopes_trie_by_extra_key():
    cache = PrefixCacheManager(MagicMock(), max_entries=8)
    on = {"generation_prompt": "v1:1:thinking-on"}
    off = {"generation_prompt": "v1:1:thinking-off"}
    on_payload = [object()]
    off_payload = [object()]

    cache.store_cache([1, 2, 3], on_payload, cache_extra_keys=on)
    cache.store_cache([1, 2, 3], off_payload, cache_extra_keys=off)

    on_hit, on_remaining = cache.fetch_cache(
        [1, 2, 3, 4],
        cache_extra_keys=on,
    )
    off_hit, off_remaining = cache.fetch_cache(
        [1, 2, 3, 5],
        cache_extra_keys=off,
    )
    assert on_hit is on_payload
    assert on_remaining == [4]
    assert off_hit is off_payload
    assert off_remaining == [5]


def test_disk_longest_prefix_filters_and_hashes_with_extra_key(tmp_path, monkeypatch):
    manager = DiskCacheManager(str(tmp_path), max_size_gb=0)
    extra = {"generation_prompt": "v1:1:thinking-on"}
    other = {"generation_prompt": "v1:1:thinking-off"}
    stored_tokens = [7, 8, 9]
    token_hash = _hash_tokens(stored_tokens, extra)
    marker = canonical_cache_extra_marker(extra)
    now = time.time()

    conn = manager._pool.get()
    try:
        conn.execute(
            "INSERT INTO cache_entries "
            "(token_hash, file_name, num_tokens, file_size, created_at, "
            "last_accessed, access_count, payload_prefix_hash, cache_extra_marker) "
            "VALUES (?, ?, ?, 0, ?, ?, 1, ?, ?)",
            (
                token_hash,
                "unused.safetensors",
                len(stored_tokens),
                now,
                now,
                _hash_tokens(stored_tokens[:-1], extra),
                marker,
            ),
        )
        conn.commit()
    finally:
        manager._pool.put(conn)

    sentinel = [object()]
    monkeypatch.setattr(
        manager,
        "_fetch_indexed_hash",
        lambda selected_hash, current_tokens: (
            sentinel if selected_hash == token_hash else None
        ),
    )
    monkeypatch.setattr(
        manager,
        "fetch",
        lambda current_tokens, cache_extra_keys=None: (
            sentinel if cache_extra_keys == extra else None
        ),
    )
    try:
        hit, matched = manager.fetch_longest_prefix(
            stored_tokens + [10, 11],
            cache_extra_keys=extra,
        )
        assert hit is sentinel
        assert matched == stored_tokens

        miss, matched = manager.fetch_longest_prefix(
            stored_tokens + [10, 11],
            cache_extra_keys=other,
        )
        assert miss is None
        assert matched == []
    finally:
        manager.shutdown()
