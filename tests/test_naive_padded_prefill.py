"""Native asymmetric attention: independent oracle, policy and cache isolation."""
import importlib

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")


def attention(monkeypatch, enabled):
    from vmlx_engine.models.naive_n05_flash.register import register_naive_n05_flash_runtime

    monkeypatch.setenv("VMLX_NAIVE_PADDED_PREFILL", "1" if enabled else "0")
    register_naive_n05_flash_runtime()
    rt = importlib.import_module("mlx_lm.models.naive_n05_flash")
    return rt.Attention(rt.ModelArgs(hidden_size=32), False)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_ragged_sparse_attention_matches_independent_float32_oracle(monkeypatch, dtype):
    layer = attention(monkeypatch, True)
    rng = np.random.default_rng(928)
    q = mx.array(rng.normal(size=(1, 64, 17, 192)).astype(np.float32)).astype(dtype)
    k = mx.array(rng.normal(size=(1, 4, 257, 192)).astype(np.float32)).astype(dtype)
    v = mx.array(rng.normal(size=(1, 4, 257, 128)).astype(np.float32)).astype(dtype)
    qp = np.arange(240, 257)[:, None]
    kp = np.arange(257)[None, :]
    mask = ((kp <= qp) & ((kp % 3 != 0) | (kp == qp)))[None, None]
    result = layer._full_sdpa(q, k, v, mx.array(mask), None)
    mx.eval(result)
    qf, kf, vf = (np.array(x.astype(mx.float32)) for x in (q, k, v))
    kf, vf = (np.repeat(x, 16, axis=1) for x in (kf, vf))
    scores = (qf @ kf.swapaxes(-1, -2)) * (192 ** -0.5)
    scores = np.where(mask, scores, -np.inf)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    expected = probs @ vf
    actual = np.array(result.astype(mx.float32))
    assert actual.shape == expected.shape == (1, 64, 17, 128)
    assert np.isfinite(actual).all()
    # Bound total relative error by one output unit roundoff. The inputs to
    # the independent FP32 oracle are the actual rounded runtime inputs.
    unit_roundoff = 2 ** (-8 if dtype == mx.bfloat16 else -11)
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < unit_roundoff


@pytest.mark.parametrize("enabled,rows,sinks", [(False, 17, False), (True, 1, False), (True, 17, True)])
def test_unqualified_or_disabled_calls_preserve_native_dispatch(monkeypatch, enabled, rows, sinks):
    layer = attention(monkeypatch, enabled)
    q = mx.zeros((1, 64, rows, 192), mx.bfloat16)
    k = mx.zeros((1, 4, 32, 192), mx.bfloat16)
    v = mx.zeros((1, 4, 32, 128), mx.bfloat16)
    mask = mx.ones((1, 1, rows, 32), mx.bool_)
    sink = mx.zeros((64,), mx.bfloat16) if sinks else None
    calls = []

    def observe(*args, **kwargs):
        calls.append((args, kwargs))
        return mx.zeros((1, 64, rows, 128), mx.bfloat16)

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", observe)
    layer._full_sdpa(q, k, v, mask, sink)
    args, kwargs = calls[0]
    assert args[0] is q and args[1] is k and args[2] is v
    assert kwargs["mask"] is mask and kwargs["sinks"] is sink
    assert "force_fused" not in kwargs


def test_arithmetic_policy_has_distinct_stable_cache_identity(monkeypatch):
    """auto (default), on and off are three distinct, stable cache namespaces."""
    from vmlx_engine import prefix_cache

    monkeypatch.delenv("VMLX_NAIVE_PADDED_PREFILL", raising=False)
    auto = prefix_cache._resolve_runtime_cache_fingerprint()
    assert "naive_padded_prefill_auto_v1:" in auto
    monkeypatch.setenv("VMLX_NAIVE_PADDED_PREFILL", "auto")
    assert prefix_cache._resolve_runtime_cache_fingerprint() == auto
    monkeypatch.setenv("VMLX_NAIVE_PADDED_PREFILL", "true")
    enabled = prefix_cache._resolve_runtime_cache_fingerprint()
    assert "naive_padded_prefill_v1" in enabled and enabled != auto
    monkeypatch.setenv("VMLX_NAIVE_PADDED_PREFILL", "1")
    assert prefix_cache._resolve_runtime_cache_fingerprint() == enabled
    monkeypatch.setenv("VMLX_NAIVE_PADDED_PREFILL", "false")
    stock = prefix_cache._resolve_runtime_cache_fingerprint()
    assert "naive_padded_prefill" not in stock and stock not in (auto, enabled)


def test_auto_keeps_stock_below_the_threshold_and_pads_above(monkeypatch):
    """Default auto: byte-identical stock dispatch while the score tensor fits, fused padded path above."""
    layer = attention(monkeypatch, True)
    monkeypatch.setenv("VMLX_NAIVE_PADDED_PREFILL", "auto")
    q = mx.zeros((1, 64, 17, 192), mx.bfloat16)
    k = mx.zeros((1, 4, 32, 192), mx.bfloat16)
    v = mx.zeros((1, 4, 32, 128), mx.bfloat16)
    mask = mx.ones((1, 1, 17, 32), mx.bool_)
    calls = []

    def observe(*args, **kwargs):
        calls.append(kwargs)
        return mx.zeros((1, 64, 17, args[2].shape[-1]), mx.bfloat16)

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", observe)
    layer._full_sdpa(q, k, v, mask, None)          # 64*17*32*4 bytes, far below the default 4 GiB
    assert "force_fused" not in calls[-1]
    monkeypatch.setenv("VMLX_NAIVE_PADDED_PREFILL_AUTO_GIB", "0.0000001")  # ~107 bytes: this chunk is above it
    layer._full_sdpa(q, k, v, mask, None)
    assert calls[-1].get("force_fused") is True
    monkeypatch.setenv("VMLX_NAIVE_PADDED_PREFILL", "0")  # forced stock wins at any size
    layer._full_sdpa(q, k, v, mask, None)
    assert "force_fused" not in calls[-1]


def test_padding_does_not_change_native_cache_or_next_append(monkeypatch):
    layer = attention(monkeypatch, True)
    layer.set_dtype(mx.bfloat16)
    from mlx_lm.models.cache import CacheList, KVCache

    mx.random.seed(29)
    chunks = [mx.random.normal((1, n, 32)).astype(mx.bfloat16) for n in (17, 1)]
    saved = []
    for enabled in (False, True):
        monkeypatch.setenv("VMLX_NAIVE_PADDED_PREFILL", "1" if enabled else "0")
        cache = CacheList(KVCache(), KVCache())
        for chunk in chunks:
            # This fixture remains below top-k, so provide the same causal
            # visibility a real model supplies to its full-attention layer.
            output = layer(chunk, mask="causal", cache=cache)
            mx.eval(output, cache.state)
        kv, indexer = cache.caches
        assert kv.offset == indexer.offset == 18
        assert kv.state[0].shape[-1] == 192
        assert kv.state[1].shape[-1] == 128
        assert indexer.state[0].dtype == mx.float32
        assert indexer.state[1].shape[-1] == 0
        saved.append([np.array(x.astype(mx.float32)) for child in cache.caches for x in child.state])
    for baseline, candidate in zip(*saved):
        np.testing.assert_array_equal(baseline, candidate)
