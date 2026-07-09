"""Regression: a prefix-cache hit must never hand decode the stored TurboQuantKVCache.

TurboQuantKVCache is not a KVCache subclass, so it used to fall through
_clone_cache_for_fetch's _safe() check to the stored-reference return. Because TQ is
monotonic-growth (update_and_fetch does `self.offset += num_new` and writes into
self.keys), decode then appended straight into the CACHED entry. Live on gemma-4 with
VMLX_SWA_TQ=1 the stored 18-token entry's offset walked 18 -> 34 -> 50 -> 66 across
repeated hits, and a later "capital of France?" hit replayed the polluted prefix and
answered "Berlin."
"""

import pytest

mx = pytest.importorskip("mlx.core")
tq_cache = pytest.importorskip("jang_tools.turboquant.cache")

from vmlx_engine.memory_cache import MemoryAwarePrefixCache

TurboQuantKVCache = tq_cache.TurboQuantKVCache


def _tq_with_tokens(n_tokens: int) -> TurboQuantKVCache:
    tq = TurboQuantKVCache(key_dim=8, value_dim=8, key_bits=3, value_bits=3)
    keys = mx.ones((1, 2, n_tokens, 8), dtype=mx.float16)
    values = mx.ones((1, 2, n_tokens, 8), dtype=mx.float16)
    tq.update_and_fetch(keys, values)
    return tq


def test_truncate_cache_rebuilds_turboquant_as_independent_cache():
    src = _tq_with_tokens(8)
    cloned = MemoryAwarePrefixCache._truncate_cache([src], 8)

    assert cloned is not None
    layer = cloned[0]
    assert type(layer).__name__ == "TurboQuantKVCache"
    assert layer is not src
    assert int(layer.offset) == 8
    # clean single-sequence contract (the live-cache validator whitelists _idx=None)
    assert layer._idx is None


def test_decoding_into_the_clone_does_not_grow_the_stored_entry():
    stored = _tq_with_tokens(8)

    cloned = MemoryAwarePrefixCache._truncate_cache([stored], 8)[0]
    for _ in range(3):
        cloned.update_and_fetch(
            mx.ones((1, 2, 1, 8), dtype=mx.float16),
            mx.ones((1, 2, 1, 8), dtype=mx.float16),
        )

    assert int(cloned.offset) == 11
    assert int(stored.offset) == 8, "decode mutated the cached entry (prefix pollution)"


def test_truncate_cache_can_shorten_a_turboquant_layer():
    src = _tq_with_tokens(8)
    cloned = MemoryAwarePrefixCache._truncate_cache([src], 5)

    assert cloned is not None
    assert int(cloned[0].offset) == 5
    assert int(src.offset) == 8


def test_clone_for_fetch_accepts_a_mixed_swa_layer_list():
    """gemma-4 under VMLX_SWA_TQ: full-attention slots are TQ, sliding stay rotating.

    If _safe() rejects any layer the whole list falls back to the stored reference,
    which is what corrupted the gemma prefix cache.
    """
    rotating = pytest.importorskip("mlx_lm.models.cache").RotatingKVCache

    rot = rotating(max_size=16)
    rot.update_and_fetch(
        mx.ones((1, 2, 8, 8), dtype=mx.float16),
        mx.ones((1, 2, 8, 8), dtype=mx.float16),
    )
    stored = [_tq_with_tokens(8), rot]

    cache = MemoryAwarePrefixCache.__new__(MemoryAwarePrefixCache)
    returned = cache._clone_cache_for_fetch(stored, 8)

    assert returned is not stored
    assert returned[0] is not stored[0], "TQ layer returned by reference"
