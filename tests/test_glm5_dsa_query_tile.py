"""Focused DSA candidate checks; actual numerical case requires MLX on M5."""
from types import SimpleNamespace
import numpy as np
import pytest
import mlx.core as mx
from vmlx_engine.glm5_dsa_query_tile import try_glm5_dsa_query_tiles


def shaped(shape, dtype=mx.float32):
    return SimpleNamespace(shape=shape, ndim=len(shape), dtype=dtype)


@pytest.mark.parametrize("queries,pools,dtype,enabled", [
    (2048, 2048, mx.float32, False),
    (605, 663, mx.float32, True),
    (1, 8192, mx.float32, True),
    (2048, 2048, mx.bfloat16, True),
])
def test_unsupported_or_small_scores_do_not_dispatch(monkeypatch, queries, pools, dtype, enabled):
    def unexpected(*args, **kwargs):
        pytest.fail("ineligible score path reached GPU contraction")
    monkeypatch.setattr(mx, "einsum", unexpected)
    assert try_glm5_dsa_query_tiles(
        shaped((1, queries, 32, 128), dtype),
        shaped((1, pools, 128), dtype),
        shaped((1, queries, 32), dtype),
        128 ** -0.5, enabled=enabled,
    ) is None


@pytest.mark.parametrize("queries,pools", [(1024, 4096)])
def test_large_native_scores_are_word_exact(queries, pools):
    rng = np.random.default_rng(671)
    q = mx.array(rng.standard_normal((1, queries, 32, 128), dtype=np.float32))
    pool = mx.array(rng.standard_normal((1, pools, 128), dtype=np.float32))
    weights = mx.array(rng.standard_normal((1, queries, 32), dtype=np.float32)) * (32 ** -0.5)
    scale = 128 ** -0.5
    scores = mx.einsum("bshd,bpd->bshp", q, pool)
    reference = mx.einsum("bsh,bshp->bsp", weights, mx.maximum(scores * scale, 0.0))
    mx.eval(reference)
    del scores
    candidate = try_glm5_dsa_query_tiles(q, pool, weights, scale, enabled=True)
    assert candidate is not None
    mx.eval(candidate)
    np.testing.assert_array_equal(np.asarray(reference).view(np.uint32), np.asarray(candidate).view(np.uint32))


def test_opt_in_partitions_model_and_legacy_cache(monkeypatch):
    from vmlx_engine.prefix_cache import compute_model_cache_key, _resolve_runtime_cache_fingerprint
    model = SimpleNamespace(args=SimpleNamespace(model_type="glm5_next"))
    monkeypatch.delenv("VMLX_GLM5_DSA_QUERY_TILE", raising=False)
    off = compute_model_cache_key(model), _resolve_runtime_cache_fingerprint()
    monkeypatch.setenv("VMLX_GLM5_DSA_QUERY_TILE", "1")
    on = compute_model_cache_key(model), _resolve_runtime_cache_fingerprint()
    assert off[0] != on[0]
    assert off[1] != on[1]
