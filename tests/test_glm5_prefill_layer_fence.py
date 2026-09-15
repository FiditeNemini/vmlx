# SPDX-License-Identifier: Apache-2.0
"""Same-shape GLM prefill graph lifetime; never relax rechunking exactness."""
import copy

import mlx.core as mx
import pytest

from tests.test_glm5_next_runtime import TINY_CFG
from vmlx_engine.models.glm5_next.glm5_next import Model, ModelArgs


def exact(left, right):
    if left is None or right is None:
        assert left is None and right is None
        return
    assert left.shape == right.shape and left.dtype == right.dtype
    bits = mx.uint32 if left.dtype == mx.float32 else mx.uint16
    assert bool(mx.array_equal(left.view(bits), right.view(bits)))


def exact_cache(left, right):
    for a, b in zip(left, right, strict=True):
        assert type(a) is type(b) and a.meta_state == b.meta_state
        assert getattr(a, "offset", None) == getattr(b, "offset", None)
        for x, y in zip(a.state, b.state, strict=True):
            exact(x, y)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("length", [129, 257])
@pytest.mark.parametrize("embedded", [False, True])
def test_prefill_and_connected_decode_exact(monkeypatch, dtype, length, embedded):
    mx.random.seed(73)
    config = copy.deepcopy(TINY_CFG)
    config["text_config"]["num_nextn_predict_layers"] = 0
    model = Model(ModelArgs.from_dict(config))
    model.set_dtype(dtype)
    # Default tiny convolution weights are zero: they are not a state oracle.
    for layer in model.model.layers:
        if layer.is_linear:
            for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                weight = getattr(layer.self_attn, name)
                setattr(layer.self_attn, name,
                        (mx.random.normal(weight.shape) * 0.05).astype(dtype))
    mx.eval(model.parameters())
    ids = (mx.arange(length) % config["text_config"]["vocab_size"])[None, :]
    kwargs = {"inputs_embeds": model.model.embed_tokens(ids)} if embedded else {}
    left, right = model.make_cache(), model.make_cache()
    monkeypatch.setenv("VMLX_GLM5_PREFILL_LAYER_FENCE", "0")
    expected = model(ids, cache=left, **kwargs)
    mx.eval(expected, *(v for c in left for v in c.state if v is not None))
    assert model.model._prefill_layer_fence_calls == 0
    monkeypatch.delenv("VMLX_GLM5_PREFILL_LAYER_FENCE", raising=False)
    actual = model(ids, cache=right, **kwargs)
    mx.eval(actual, *(v for c in right for v in c.state if v is not None))
    assert model.model._prefill_layer_fence_calls == len(model.model.layers)
    exact(expected, actual)
    exact_cache(left, right)
    assert bool(mx.any(right[0].state[3] != 0))
    for token in (31, 32, 33):
        continuation = mx.array([[token]])
        expected, actual = model(continuation, cache=left), model(continuation, cache=right)
        mx.eval(expected, actual)
        exact(expected, actual)
        exact_cache(left, right)
    # The enabled control must not inject layer barriers into ordinary AR decode.
    assert model.model._prefill_layer_fence_calls == len(model.model.layers)


def test_no_cache_and_short_forwards_keep_lazy_path(monkeypatch):
    config = copy.deepcopy(TINY_CFG)
    config["text_config"]["num_nextn_predict_layers"] = 0
    model = Model(ModelArgs.from_dict(config))
    monkeypatch.delenv("VMLX_GLM5_PREFILL_LAYER_FENCE", raising=False)
    mx.eval(model(mx.array([[1, 2, 3]]), cache=model.make_cache()))
    mx.eval(model(mx.arange(129)[None, :]))
    assert model.model._prefill_layer_fence_calls == 0


def test_default_and_explicit_namespace_policy_is_glm_only(monkeypatch):
    from types import SimpleNamespace
    from vmlx_engine.prefix_cache import compute_model_cache_key

    for family in ("glm5_next", "glm5_next_text", "qwen4_exp", "qwen3_5"):
        model = SimpleNamespace(args=SimpleNamespace(model_type=family))
        monkeypatch.setenv("VMLX_GLM5_PREFILL_LAYER_FENCE", "0")
        control = compute_model_cache_key(model)
        monkeypatch.delenv("VMLX_GLM5_PREFILL_LAYER_FENCE", raising=False)
        default = compute_model_cache_key(model)
        monkeypatch.setenv("VMLX_GLM5_PREFILL_LAYER_FENCE", "1")
        candidate = compute_model_cache_key(model)
        assert default == candidate
        assert (control != candidate) == family.startswith("glm5_next")


@pytest.mark.parametrize("value,expected", [(None, True), ("1", True), ("0", False)])
def test_shared_policy_default_and_override(monkeypatch, value, expected):
    from vmlx_engine.glm5_prefill_policy import glm5_prefill_layer_fence_enabled

    if value is None:
        monkeypatch.delenv("VMLX_GLM5_PREFILL_LAYER_FENCE", raising=False)
    else:
        monkeypatch.setenv("VMLX_GLM5_PREFILL_LAYER_FENCE", value)
    assert glm5_prefill_layer_fence_enabled() is expected
