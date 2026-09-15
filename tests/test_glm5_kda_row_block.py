"""Guarded AR row blocking: compare the incumbent kernel, not new algebra."""
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from vmlx_engine.metal import glm5_kda_row_block as blocked
from vmlx_engine.metal.kda_step_decode import _kernel, glm5_kda_step_decode

qualified_gpu = pytest.mark.skipif(
    not blocked._hardware_allowed(), reason="row-block arithmetic requires M5 and MLX 0.32.2"
)


def inputs(dtype=mx.float32, gate_dtype=mx.float32, beta_dtype=mx.float32):
    mx.random.seed(7)
    q, k, v = [mx.random.normal((1, 64, 128)) for _ in range(3)]
    q = q * mx.rsqrt(mx.sum(q * q, axis=-1, keepdims=True) + 1e-6)
    k = k * mx.rsqrt(mx.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    g = -5 * mx.sigmoid(mx.random.normal(q.shape))
    beta = mx.sigmoid(mx.random.normal((1, 64)))
    state = mx.random.normal((1, 64, 128, 128)) * .05
    data = (q.astype(dtype), k.astype(dtype), v.astype(dtype),
            g.astype(gate_dtype), beta.astype(beta_dtype), state)
    mx.eval(*data)
    return data


def incumbent(data):
    return _kernel(64, 128, 128)(
        inputs=data, grid=(32 * 64 * 128, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(1, 64, 128), (1, 64, 128, 128)],
        output_dtypes=[mx.float32, mx.float32],
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("gate_dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("beta_dtype", [mx.float16, mx.bfloat16, mx.float32])
@qualified_gpu
def test_exact_connected_output_state_and_dispatch(dtype, gate_dtype, beta_dtype, monkeypatch):
    monkeypatch.setenv("VMLX_GLM5_KDA_ROW_BLOCK", "1")
    data = inputs(dtype, gate_dtype, beta_dtype)
    assert blocked._hardware_allowed(), "this qualification requires the nominated M5/MLX runtime"
    old_state, new_state = data[-1], data[-1]
    before = blocked.observed_calls()
    for _ in range(4):
        expected, old_state = incumbent((*data[:-1], old_state))
        actual, new_state = glm5_kda_step_decode(*data[:-1], new_state, enabled=True)
        mx.eval(expected, old_state, actual, new_state)
        assert mx.array_equal(expected.view(mx.uint32), actual.view(mx.uint32))
        assert mx.array_equal(old_state.view(mx.uint32), new_state.view(mx.uint32))
    assert blocked.observed_calls() - before == 4


@qualified_gpu
def test_offset_strided_views_and_input_ownership():
    data = inputs(mx.bfloat16)
    def bits(a):
        dtype = mx.uint16 if a.dtype in (mx.float16, mx.bfloat16) else mx.uint32
        return np.array(a.view(dtype), copy=True)
    original = [bits(a) for a in data]
    # Same logical arrays through non-contiguous padded storage.
    strided = []
    for a in data:
        storage = mx.stack([a, mx.zeros_like(a)], axis=-1)
        strided.append(storage[..., 0])
    expected = incumbent(data)
    actual = blocked.glm5_kda_row_block(*strided, enabled=True)
    assert actual is not None
    mx.eval(*expected, *actual)
    for a, b in zip(expected, actual):
        assert mx.array_equal(a.view(mx.uint32), b.view(mx.uint32))
    assert all(np.array_equal(bits(a), b) for a, b in zip(data, original))


@pytest.mark.parametrize("flag", [None, "0", "false", "garbage", "2"])
def test_default_and_invalid_flag_keep_incumbent(flag, monkeypatch):
    if flag is None:
        monkeypatch.delenv("VMLX_GLM5_KDA_ROW_BLOCK", raising=False)
    else:
        monkeypatch.setenv("VMLX_GLM5_KDA_ROW_BLOCK", flag)
    data = inputs()
    monkeypatch.setattr(blocked, "_row_kernel", lambda: pytest.fail("unexpected dispatch"))
    assert blocked.glm5_kda_row_block(*data) is None
    actual = glm5_kda_step_decode(*data, enabled=True)
    expected = incumbent(data)
    mx.eval(*actual, *expected)
    assert all(mx.array_equal(a, b) for a, b in zip(actual, expected))


@pytest.mark.parametrize("field", range(6))
def test_unsupported_shape_declines_without_dispatch(field, monkeypatch):
    data = list(inputs())
    data[field] = mx.concatenate([data[field], data[field]], axis=0)
    monkeypatch.setattr(blocked, "_row_kernel", lambda: pytest.fail("unexpected dispatch"))
    assert blocked.glm5_kda_row_block(*data, enabled=True) is None


def test_unsupported_backend_or_state_and_parent_off(monkeypatch):
    data = list(inputs())
    monkeypatch.setenv("VMLX_GLM5_KDA_ROW_BLOCK", "1")
    monkeypatch.setattr(blocked, "_row_kernel", lambda: pytest.fail("unexpected dispatch"))
    assert glm5_kda_step_decode(*data, enabled=False) is None
    data[-1] = data[-1].astype(mx.float16)
    assert blocked.glm5_kda_row_block(*data, enabled=True) is None
    data[-1] = data[-1].astype(mx.float32)
    monkeypatch.setattr(blocked, "_hardware_allowed", lambda: False)
    assert blocked.glm5_kda_row_block(*data, enabled=True) is None


@qualified_gpu
def test_submission_failure_propagates(monkeypatch):
    data = inputs()
    def fail(*args, **kwargs):
        raise RuntimeError("injected submission failure")
    monkeypatch.setattr(blocked, "_row_kernel", lambda: fail)
    with pytest.raises(RuntimeError, match="injected submission"):
        blocked.glm5_kda_row_block(*data, enabled=True)


def test_candidate_cache_namespace_isolated_and_other_family_unchanged(monkeypatch):
    from vmlx_engine.prefix_cache import compute_model_cache_key
    glm = SimpleNamespace(args=SimpleNamespace(model_type="glm5_next"))
    qwen = SimpleNamespace(args=SimpleNamespace(model_type="qwen4_exp"))
    monkeypatch.setenv("VMLX_GLM5_KDA_ROW_BLOCK", "0")
    old_glm = compute_model_cache_key(glm)
    old_qwen = compute_model_cache_key(qwen)
    monkeypatch.setenv("VMLX_GLM5_KDA_ROW_BLOCK", "1")
    assert compute_model_cache_key(glm) != old_glm
    assert compute_model_cache_key(qwen) == old_qwen
