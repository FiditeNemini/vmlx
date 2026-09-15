# SPDX-License-Identifier: Apache-2.0
"""Convolution history owns only its logical tail, not a projected prompt."""
import copy
import gc

import mlx.core as mx
import pytest

from vmlx_engine.models.glm5_next.kda import short_conv, short_conv_with_states
from tests.test_glm5_prefill_layer_fence import exact, exact_cache


def view_reference(x, weight, state=None):
    """Pre-fix arithmetic and history view; never a production fallback."""
    if weight.ndim == 3:
        weight = weight.reshape(weight.shape[0], -1)
    channels, width = weight.shape
    batch, tokens, _ = x.shape
    if state is None:
        state = mx.zeros((batch, width - 1, channels), dtype=x.dtype)
    padded = mx.concatenate([state.astype(x.dtype), x], axis=1)
    y = mx.zeros((batch, tokens, channels), dtype=mx.float32)
    for tap in range(width):
        y = y + padded[:, tap:tap + tokens].astype(mx.float32) * weight[:, tap].astype(mx.float32)
    return (y * mx.sigmoid(y)).astype(x.dtype), padded[:, padded.shape[1] - (width - 1):]


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("batch,tokens,width", [(1, 1, 1), (2, 2, 8), (1, 257, 4)])
def test_stock_output_tail_and_continuation_exact(dtype, batch, tokens, width):
    mx.random.seed(73)
    # Strided projections, mixed weight/state dtypes, and T < W-1 matter too.
    x = mx.random.normal((batch, tokens, 256)).astype(dtype)[:, :, ::2]
    weight = mx.random.normal((128, 1, width)).astype(mx.bfloat16)
    state = mx.random.normal((batch, width - 1, 128)).astype(mx.float32)
    snapshot = mx.array(state)
    expected, previous = view_reference(x, weight, state)
    actual, owned = short_conv(x, weight, state)
    exact(expected, actual)
    exact(previous, owned)
    exact(state, snapshot)
    follow = mx.random.normal((batch, 3, 128)).astype(dtype)
    expected, previous = view_reference(follow, weight, previous)
    actual, owned = short_conv(follow, weight, owned)
    exact(expected, actual)
    exact(previous, owned)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_each_verify_position_keeps_exact_raw_tail(dtype):
    mx.random.seed(74)
    x = mx.random.normal((1, 5, 64)).astype(dtype)
    weight = mx.random.normal((64, 4)).astype(dtype)
    state = mx.random.normal((1, 3, 64)).astype(dtype)
    result, tail, positions = short_conv_with_states(x, weight, state)
    expected, last = view_reference(x, weight, state)
    exact(result, expected)
    exact(tail, last)
    assert len(positions) == 5
    for i, position in enumerate(positions):
        _, reference = view_reference(x[:, :i + 1], weight, state)
        exact(position, reference)


def settled_active():
    mx.synchronize()
    gc.collect()
    mx.clear_cache()
    return mx.get_active_memory()


def retained_tails(kind):
    from vmlx_engine.metal.glm5_kda_conv_prefill import kda_conv_prefill
    from vmlx_engine.metal.glm5_kda_qkv_prefill import kda_qkv_prefill

    # This exceeds the logical tail by >600x and exercises a shared QKV base.
    xs = tuple(mx.split(mx.random.normal((1, 2048, 3 * 256)).astype(mx.bfloat16), 3, axis=-1))
    weights = tuple(mx.random.normal((256, 4)).astype(mx.bfloat16) for _ in xs)
    if kind == "qkv":
        pairs = kda_qkv_prefill(xs, weights, (None, None, None), enabled=True)
        assert pairs is not None
    else:
        fn = kda_conv_prefill if kind == "fused" else short_conv
        pairs = tuple(fn(x, w) for x, w in zip(xs, weights))
    mx.eval(*(value for pair in pairs for value in pair))
    return tuple(pair[1] for pair in pairs)


@pytest.mark.parametrize("kind", ["stock", "fused", "qkv"])
def test_evaluated_tails_release_full_projection_allocation(kind):
    if not mx.metal.is_available():
        pytest.skip("Metal allocator ownership proof needs Metal")
    mx.random.seed(75)
    before = settled_active()
    tails = retained_tails(kind)
    retained = settled_active() - before
    logical = sum(t.nbytes for t in tails)
    assert retained <= logical + 4096, (kind, retained, logical)
    del tails
    assert settled_active() <= before + 4096


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_model_logits_native_state_and_decode_exact(monkeypatch, dtype):
    from tests.test_glm5_next_runtime import TINY_CFG
    from vmlx_engine.models.glm5_next import glm5_next as runtime

    mx.random.seed(76)
    config = copy.deepcopy(TINY_CFG)
    config["text_config"]["num_nextn_predict_layers"] = 0
    model = runtime.Model(runtime.ModelArgs.from_dict(config))
    model.set_dtype(dtype)
    for layer in model.layers:
        if layer.is_linear:
            for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                setattr(layer.self_attn, name, (mx.random.normal(getattr(layer.self_attn, name).shape) * 0.05).astype(dtype))
            layer.self_attn._fused_kda_prefill = False
            layer.self_attn._fused_kda_qkv_prefill = False
    mx.eval(model.parameters())
    left, right = model.make_cache(), model.make_cache()
    for tokens in (257, 1, 3):
        ids = (mx.arange(tokens) % config["text_config"]["vocab_size"])[None, :]
        monkeypatch.setattr(runtime, "short_conv", view_reference)
        expected = model(ids, cache=left)
        mx.eval(expected, *(v for c in left for v in c.state if v is not None))
        monkeypatch.setattr(runtime, "short_conv", short_conv)
        actual = model(ids, cache=right)
        mx.eval(actual, *(v for c in right for v in c.state if v is not None))
        exact(expected, actual)
        exact_cache(left, right)
