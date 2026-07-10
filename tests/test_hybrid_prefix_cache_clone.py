"""F20: a hybrid (KV + cumulative SSM) prefix-cache hit must be isolated.

`TurboQuantKVCache` had the same defect (see test_turboquant_prefix_cache_clone).
Here the rejected layer is cumulative SSM/conv state — MambaCache / ArraysCache —
which `_safe()` refused, so the whole layer list fell through to the stored-reference
return and decode advanced the CACHED entry's offsets in place.

This was latent rather than live: both schedulers auto-switch hybrid models to the
paged cache and disable the memory-aware cache (scheduler.py:719,
mllm_scheduler.py:512), so a hybrid layer list normally never reaches this class.
It gets here only when `_is_hybrid_model()` fails open — it returns False if
`make_cache()` raises. These tests pin the isolation contract at the cache layer so
correctness no longer rests on that one detector.
"""

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache

from vmlx_engine.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig


class _FakeModel:
    def make_cache(self):
        return None


def _kv_layer(n=18, heads=2, dim=4):
    layer = KVCache()
    layer.update_and_fetch(mx.zeros((1, heads, n, dim)), mx.zeros((1, heads, n, dim)))
    return layer


def _ssm_layer(fill=0.0):
    layer = ArraysCache(2)
    layer[0] = mx.full((1, 4, 4), fill)
    layer[1] = mx.full((1, 4, 4), fill)
    return layer


def _hybrid_cache():
    return [_kv_layer(), _ssm_layer(), _kv_layer()]


@pytest.fixture
def prefix_cache():
    return MemoryAwarePrefixCache(
        model=_FakeModel(),
        config=MemoryCacheConfig(max_memory_mb=512),
        model_path="/tmp/hybrid-test",
    )


def _decode_one_token(cache):
    for layer in cache:
        if isinstance(layer, KVCache):
            layer.update_and_fetch(mx.zeros((1, 2, 1, 4)), mx.zeros((1, 2, 1, 4)))


def test_hybrid_hit_returns_independent_objects(prefix_cache):
    tokens = list(range(18))
    stored = _hybrid_cache()
    assert prefix_cache.store(tokens, stored)

    fetched, remaining = prefix_cache.fetch(tokens)

    assert fetched is not None, "hybrid entry must still hit, not degrade to a miss"
    assert remaining == []
    for got, orig in zip(fetched, stored):
        assert got is not orig
        assert type(got) is type(orig)


def test_decoding_into_the_hybrid_clone_does_not_move_the_stored_entry(prefix_cache):
    tokens = list(range(18))
    stored = _hybrid_cache()
    prefix_cache.store(tokens, stored)

    fetched, _ = prefix_cache.fetch(tokens)
    _decode_one_token(fetched)

    # Without the fix the stored KV offsets walk 18 -> 19 here.
    assert [layer.offset for layer in stored if isinstance(layer, KVCache)] == [18, 18]


def test_mutating_the_cloned_ssm_state_does_not_touch_the_stored_state(prefix_cache):
    tokens = list(range(18))
    stored = _hybrid_cache()
    prefix_cache.store(tokens, stored)

    fetched, _ = prefix_cache.fetch(tokens)
    fetched[1][0] = mx.full((1, 4, 4), 9.0)

    assert float(stored[1][0][0, 0, 0]) == 0.0


def test_second_hit_sees_the_original_prefix(prefix_cache):
    """The user-visible symptom: a later hit must replay the clean prefix."""
    tokens = list(range(18))
    stored = _hybrid_cache()
    prefix_cache.store(tokens, stored)

    first, _ = prefix_cache.fetch(tokens)
    _decode_one_token(first)
    second, _ = prefix_cache.fetch(tokens)

    assert second is not None
    assert [layer.offset for layer in second if isinstance(layer, KVCache)] == [18, 18]


def test_reverse_match_on_cumulative_state_still_forces_a_miss(prefix_cache):
    """Cumulative state cannot be REDUCED — a true truncation must not fake it."""
    stored = _hybrid_cache()
    prefix_cache.store(list(range(18)), stored)

    # Ask for a strict prefix of the cached entry -> reverse match -> truncation.
    cache, remaining = prefix_cache.fetch(list(range(9)))

    assert cache is None
    assert remaining == list(range(9))


def test_truncate_cache_refuses_cumulative_reduction_by_default():
    cache = _hybrid_cache()
    assert MemoryAwarePrefixCache._truncate_cache(cache, 9) is None
    # ...and only copies it when the caller asserts full-length (no reduction).
    cloned = MemoryAwarePrefixCache._truncate_cache(cache, 18, True)
    assert cloned is not None
    assert cloned[1] is not cache[1]
