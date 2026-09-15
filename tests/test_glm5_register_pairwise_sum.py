# SPDX-License-Identifier: Apache-2.0
"""Exact FP32 reduction/state and fail-closed dispatch for GLM bulk prefill."""
from types import SimpleNamespace

import mlx.core as mx
import pytest

from vmlx_engine.models.glm5_next import kda
from vmlx_engine.metal import glm5_pairwise_sum as fused


def exact(a, b):
    mx.eval(a, b)
    assert a.shape == b.shape and a.dtype == b.dtype
    assert bool(mx.array_equal(a.view(mx.uint32), b.view(mx.uint32)))


@pytest.fixture
def qualified(monkeypatch):
    if not mx.metal.is_available() or not fused._compatible_runtime():
        pytest.skip("register-sum numerical qualification requires MLX0.32.2/M5")
    monkeypatch.setattr(fused, "_FAILED", False)
    monkeypatch.setattr(fused, "_OBSERVED", False)
    monkeypatch.setattr(kda, "_EXACT_PAIRWISE_REQUESTED", True)


@pytest.mark.parametrize("shape", [(1, 1, 64, 128), (1, 64, 64, 128), (2, 2, 64, 128)])
@pytest.mark.parametrize("strided", [False, True])
def test_exact_product_sum_bits(qualified, shape, strided):
    mx.random.seed(819)
    left, right, gates = [mx.random.normal(shape) for _ in range(3)]
    gates = mx.cumsum(-mx.sigmoid(gates), axis=-2)
    if strided:
        left = mx.stack((left, right), axis=2)[:, :, 0]
        right = mx.stack((right, left), axis=2)[:, :, 0]
        gates = mx.stack((gates, -gates), axis=2)[:, :, 0]
    expected = mx.sum(kda._exact_pairwise_product(left, right, gates), axis=-1)
    actual = fused.glm5_pairwise_sum(left, right, gates, enabled=True)
    assert actual is not None
    exact(expected, actual)


@pytest.mark.parametrize("case", ["signed_zero", "cancellation", "underflow", "tiny"])
def test_edge_arithmetic(qualified, case):
    shape = (1, 2, 64, 128)
    left, right, gates = mx.ones(shape), mx.ones(shape), mx.zeros(shape)
    if case == "signed_zero":
        left = mx.broadcast_to(mx.array([-0.0]), shape)
    elif case == "cancellation":
        left = mx.broadcast_to(mx.array([1.0e15, 1.0, -1.0e15, 1.0] * 32), shape)
    elif case == "underflow":
        gates = mx.broadcast_to(mx.linspace(0, -1.0e5, 64)[None, None, :, None], shape)
    else:
        left = mx.full(shape, 1.0e-20)
    expected = mx.sum(kda._exact_pairwise_product(left, right, gates), axis=-1)
    actual = fused.glm5_pairwise_sum(left, right, gates, enabled=True)
    exact(expected, actual)


@pytest.mark.parametrize("tokens", [63, 64, 65, 127, 257, 513])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_product_path_chunked_state_and_decode(qualified, monkeypatch, tokens, dtype):
    mx.random.seed(tokens)
    shape = (1, tokens, 2, 128)
    q = kda.l2norm(mx.random.normal(shape)).astype(dtype)
    k = kda.l2norm(mx.random.normal(shape)).astype(dtype)
    v = mx.random.normal(shape).astype(dtype)
    g = -mx.sigmoid(mx.random.normal(shape))
    beta = mx.sigmoid(mx.random.normal(shape[:-1]))
    initial = mx.random.normal((1, 2, 128, 128)) * .01
    monkeypatch.setattr(kda, "_REGISTER_PAIRWISE_REQUESTED", False)
    expected, se = kda.kda_chunked(q, k, v, g, beta, initial)
    mx.eval(expected, se)
    monkeypatch.setattr(kda, "_REGISTER_PAIRWISE_REQUESTED", True)
    actual, sa = kda.kda_chunked(q, k, v, g, beta, initial)
    exact(expected, actual)
    exact(se, sa)
    assert fused._OBSERVED, "test must execute the real candidate, not its fallback"
    for _ in range(3):
        x = mx.random.normal((1, 2, 128))
        args = (kda.l2norm(x), kda.l2norm(x), x, -mx.ones_like(x), mx.full((1, 2), .5))
        oe, se = kda.kda_step(*args, se)
        oa, sa = kda.kda_step(*args, sa)
        exact(oe, oa)
        exact(se, sa)


def test_unqualified_shapes_dtypes_and_disabled_never_launch(monkeypatch):
    monkeypatch.setattr(fused, "_FAILED", False)
    def forbidden():
        raise AssertionError("ineligible input reached kernel")
    monkeypatch.setattr(fused, "_kernel", forbidden)
    x = mx.zeros((1, 2, 64, 128))
    assert fused.glm5_pairwise_sum(x, x, x, enabled=False) is None
    for y in [x[:, :, :32], x[..., :64], x.astype(mx.bfloat16), x[0], x[:0]]:
        assert fused.glm5_pairwise_sum(y, y, y, enabled=True) is None
    assert fused.glm5_pairwise_sum(x, x[:, :1], x, enabled=True) is None
    monkeypatch.setattr(fused, "_compatible_runtime", lambda: False)
    assert fused.glm5_pairwise_sum(x, x, x, enabled=True) is None


def test_first_launch_failure_disables_candidate_and_preserves_stock(qualified, monkeypatch, caplog):
    def broken():
        raise RuntimeError("controlled shader compilation failure")
    monkeypatch.setattr(fused, "_kernel", broken)
    x = mx.ones((1, 2, 64, 128))
    assert fused.glm5_pairwise_sum(x, x, x, enabled=True) is None
    assert fused._FAILED
    assert "retaining stock reduction" in caplog.text
    monkeypatch.setattr(kda, "_REGISTER_PAIRWISE_REQUESTED", True)
    exact(kda._pairwise_sum(x, x, x), mx.full((1, 2, 64, 64), 128.0))


def test_policy_off_and_namespace_isolation_glm_only(monkeypatch):
    from vmlx_engine.glm5_prefill_policy import glm5_register_pairwise_sum_requested
    from vmlx_engine.prefix_cache import compute_model_cache_key
    monkeypatch.delenv("VMLX_GLM5_REGISTER_PAIRWISE_SUM", raising=False)
    assert glm5_register_pairwise_sum_requested() is False
    for family in ("glm5_next", "glm5_next_text", "qwen4_exp", "qwen3_5"):
        model = SimpleNamespace(args=SimpleNamespace(model_type=family))
        monkeypatch.delenv("VMLX_GLM5_REGISTER_PAIRWISE_SUM", raising=False)
        default = compute_model_cache_key(model)
        monkeypatch.setenv("VMLX_GLM5_REGISTER_PAIRWISE_SUM", "0")
        assert compute_model_cache_key(model) == default
        monkeypatch.setenv("VMLX_GLM5_REGISTER_PAIRWISE_SUM", "1")
        assert glm5_register_pairwise_sum_requested() is True
        assert (compute_model_cache_key(model) != default) == family.startswith("glm5_next")
