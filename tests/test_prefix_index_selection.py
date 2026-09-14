# SPDX-License-Identifier: Apache-2.0
"""Longest-prefix selection must not repeatedly revalidate shorter chains."""

import pytest
import mlx.core as mx

from vmlx_engine.paged_cache import BlockTable, PagedCacheManager
from vmlx_engine.prefix_cache import BlockAwarePrefixCache


@pytest.fixture(params=[False, True], ids=["legacy-index", "chained-index"])
def cache(request):
    paged = PagedCacheManager(block_size=4, max_blocks=32)
    result = BlockAwarePrefixCache(model=None, paged_cache_manager=paged)
    result._chained_prefix_index_hash = request.param
    return result


def _store(cache, request_id, tokens, extra=None):
    # Real tensor state uses full chain identity. Plain string placeholders
    # exercise a legacy non-tensor compatibility branch instead.
    keys = mx.zeros((1, 1, len(tokens), 1))
    values = mx.ones((1, 1, len(tokens), 1))
    table = cache.store_cache(
        request_id, tokens, [{"class_name": "KVCache", "state": (keys, values)}],
        cache_extra_keys=extra,
    )
    assert table is not None
    entry = cache._request_tables.pop(request_id)
    cache.paged_cache.release_request_refs(entry.block_table)
    cache.paged_cache.detach_request(request_id)
    return table


def _count_validations(cache, monkeypatch):
    lengths = []
    original = cache._prefix_index_blocks_are_current

    def validate(tokens, blocks, **kwargs):
        lengths.append(len(tokens))
        return original(tokens, blocks, **kwargs)

    monkeypatch.setattr(cache, "_prefix_index_blocks_are_current", validate)
    return lengths


def _release_match(cache, match):
    tokens, block_ids = match
    assert cache.paged_cache.release_request_refs(
        BlockTable(request_id="reader", block_ids=block_ids, num_tokens=len(tokens))
    ) == len(block_ids)
    assert all(cache.paged_cache.blocks[i].ref_count == 0 for i in block_ids)


def test_longest_partial_is_validated_once_and_pinned(cache, monkeypatch):
    tokens = list(range(18))
    table = _store(cache, "writer", tokens)
    keys_before = set(cache._prefix_index)
    validations = _count_validations(cache, monkeypatch)

    match = cache._find_best_prefix_match(tokens + [90, 91, 92, 93, 94])

    assert match == (tokens, table.block_ids)
    assert validations == [18]
    assert set(cache._prefix_index) == keys_before
    assert all(cache.paged_cache.blocks[i].ref_count == 1 for i in table.block_ids)
    _release_match(cache, match)


def test_fetch_does_not_revalidate_an_authoritative_chain_hit(cache, monkeypatch):
    tokens = list(range(18))
    table = _store(cache, "writer", tokens)
    validations = _count_validations(cache, monkeypatch)

    hit, remaining = cache.fetch_cache("reader", tokens + [99])

    assert hit.num_tokens == len(tokens)
    assert hit.block_ids == table.block_ids
    assert remaining == [99]
    assert validations == []
    assert all(cache.paged_cache.blocks[i].ref_count == 1 for i in hit.block_ids)


def test_empty_index_does_not_hash_absent_candidates(cache, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("empty index must not hash a growing prompt")

    monkeypatch.setattr(cache, "_prefix_index_hash", unexpected)
    monkeypatch.setattr(cache, "_prefix_index_hash_sequence", unexpected)
    assert cache._find_best_prefix_match(list(range(1000))) is None


def test_stale_longest_falls_back_without_pinning_recycled_tail(cache, monkeypatch):
    tokens = list(range(18))
    table = _store(cache, "writer", tokens)
    stale_key = cache._prefix_index_key(tokens)
    tail = cache.paged_cache.blocks[table.block_ids[-1]]
    tail.token_count = 1  # No longer the indexed two-token partial block.
    validations = _count_validations(cache, monkeypatch)

    match = cache._find_best_prefix_match(tokens + [99])

    assert match == (tokens[:16], table.block_ids[:4])
    assert validations == [18, 16]
    assert stale_key not in cache._prefix_index
    assert tail.ref_count == 0
    _release_match(cache, match)


def test_equal_length_different_media_never_borrows_state(cache, monkeypatch):
    tokens = list(range(10))
    first = _store(cache, "media-a", tokens, {"media": "a"})
    second = _store(cache, "media-b", tokens, {"media": "b"})
    keys_before = set(cache._prefix_index)
    validations = _count_validations(cache, monkeypatch)

    match = cache._find_best_prefix_match(tokens + [99], cache_extra_keys={"media": "b"})

    assert match == (tokens, second.block_ids)
    assert validations == [10]
    assert all(cache.paged_cache.blocks[i].ref_count == 0 for i in first.block_ids)
    assert set(cache._prefix_index) == keys_before
    _release_match(cache, match)
