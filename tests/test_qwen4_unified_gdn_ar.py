"""New single-token AR boundary; existing verify-width rows remain separate."""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache
import numpy as np
import pytest

from vmlx_engine.metal import qwen4_unified_gdn_ar as impl
from vmlx_engine.models.qwen4_exp import language


def _exact(want, got):
    mx.eval(want, got)
    dtype = np.uint32 if want.dtype == mx.float32 else np.uint16
    np.testing.assert_array_equal(np.asarray(want).view(dtype), np.asarray(got).view(dtype))


def _case(steps=1, coefficient_dtype=mx.bfloat16):
    mx.random.seed(1515)
    layer = language.GatedDeltaNet(language.Qwen4ExpTextArgs())
    layer.eval()
    layer._unified_gdn_verify = False
    layer._fused_conv_decode = False
    layer.norm._fused_decode = False

    def random(shape, scale=.25, dtype=mx.float16):
        return (mx.random.normal(shape) * scale).astype(dtype)

    class Projection(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value

        def __call__(self, x):
            return self.value[:, :x.shape[1]]

    layer.in_proj_qkv = Projection(random((1, steps, 10240)))
    layer.in_proj_z = Projection(random((1, steps, 6144), 2))
    layer.in_proj_b = Projection(random((1, steps, 48), 2))
    layer.in_proj_a = Projection(random((1, steps, 48)))
    layer.out_proj = nn.Identity()
    layer.conv1d.weight = random((10240, 4, 1))
    layer.A_log = random((48,), dtype=coefficient_dtype)
    layer.dt_bias = random((48,), dtype=coefficient_dtype)
    layer.norm.weight = (1 + random((128,), .1)).astype(mx.float16)
    conv = random((1, 3, 10240))
    state = random((1, 48, 128, 128), .05, mx.float32)
    auxiliary = (mx.array([[11, 17, 23]]), mx.array([[[.5, -.25]]]))

    def cache():
        c = ArraysCache(size=4)
        c[0], c[1], c[2], c[3] = conv, state, *auxiliary
        c.ar_test_advanced = 0
        def advance(n):
            c.ar_test_advanced += n
        c.advance = advance
        return c

    return layer, mx.zeros((1, steps, 2560), dtype=mx.float16), cache


def _args(layer, c):
    return [layer.in_proj_qkv.value, layer.in_proj_z.value, layer.in_proj_b.value,
            layer.in_proj_a.value, c[0], layer.conv1d.weight, layer.A_log,
            layer.dt_bias, c[1], layer.norm.weight, layer.norm.eps]


def test_ar_default_off_and_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("VMLX_QWEN4_UNIFIED_GDN_AR", raising=False)
    assert not impl.unified_gdn_ar_requested()
    layer, _, _ = _case()
    assert not layer._unified_gdn_ar
    monkeypatch.setenv("VMLX_QWEN4_UNIFIED_GDN_AR", "1")
    assert impl.unified_gdn_ar_requested()
    layer, _, _ = _case()
    assert layer._unified_gdn_ar


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_ar_output_and_continued_cache_exact(dtype, caplog):
    layer, x, cache = _case(coefficient_dtype=dtype)
    want_cache, got_cache = cache(), cache()
    for _ in range(4):
        layer._unified_gdn_ar = False
        want = layer(x, cache=want_cache)
        layer._unified_gdn_ar = True
        with caplog.at_level("INFO"):
            got = layer(x, cache=got_cache)
        _exact(want, got)
        for w, g in zip(want_cache.state, got_cache.state):
            # PLE's integer context slots retain their dtype/value too.
            mx.eval(w, g)
            np.testing.assert_array_equal(np.asarray(w), np.asarray(g))
        _exact(want_cache[1], got_cache[1])
    assert layer._unified_gdn_ar_graph_calls == 4
    assert got_cache.ar_test_advanced == want_cache.ar_test_advanced == 4
    assert caplog.text.count("QWEN4_UNIFIED_GDN_AR graph_built") == 1


@pytest.mark.parametrize("bits,group", [(2, 32), (4, 64), (6, 32), (8, 64)])
def test_ar_preserves_quantized_output_consumer(bits, group):
    layer, x, cache = _case()
    layer.out_proj = nn.QuantizedLinear(6144, 2560, bias=False, group_size=group, bits=bits)
    layer._unified_gdn_ar = False
    want = layer(x, cache=cache())
    layer._unified_gdn_ar = True
    got = layer(x, cache=cache())
    _exact(want, got)
    assert (layer.out_proj.bits, layer.out_proj.group_size) == (bits, group)


@pytest.mark.parametrize("index,mutation", [
    (0, lambda v: mx.concatenate([v, v], axis=1)),
    (0, lambda v: mx.concatenate([v, v], axis=0)),
    (0, lambda v: v.astype(mx.bfloat16)),
    (1, lambda v: v[..., :-1]),
    (4, lambda v: None),
    (5, lambda v: v[:, :3]),
    (6, lambda v: v.astype(mx.int32)),
    (7, lambda v: v[:-1]),
    (8, lambda v: None),
    (8, lambda v: v.astype(mx.float16)),
    (9, lambda v: v.astype(mx.float32)),
    (10, lambda v: 1e-5),
])
def test_ar_unsupported_declines_before_dispatch(monkeypatch, index, mutation):
    layer, _, cache = _case()
    values = _args(layer, cache())
    values[index] = mutation(values[index])
    def forbidden():
        raise AssertionError("unsupported shape reached kernel")
    monkeypatch.setattr(impl, "_kernel", forbidden)
    assert impl.qwen4_unified_gdn_ar(*values, enabled=True) is None


def test_ar_disabled_hardware_and_cpu_fallback(monkeypatch):
    layer, _, cache = _case()
    values = _args(layer, cache())
    assert impl.qwen4_unified_gdn_ar(*values, enabled=False) is None
    monkeypatch.setattr(impl, "_hardware_supported", lambda: False)
    assert impl.qwen4_unified_gdn_ar(*values, enabled=True) is None
    monkeypatch.setattr(impl, "_hardware_supported", lambda: True)
    old = mx.default_device()
    try:
        mx.set_default_device(mx.cpu)
        assert impl.qwen4_unified_gdn_ar(*values, enabled=True) is None
    finally:
        mx.set_default_device(old)


@pytest.mark.parametrize("variant", ["off", "mask", "lengths", "confirmed", "prefill", "training", "conv_fused", "norm_fused"])
def test_ar_caller_excludes_other_paths(monkeypatch, variant):
    layer, x, cache = _case(steps=2 if variant == "prefill" else 1)
    layer._unified_gdn_ar = variant != "off"
    c = cache()
    kwargs = {}
    if variant == "mask":
        kwargs['mask'] = mx.ones((1, 1), dtype=mx.bool_)
    elif variant == "lengths":
        c.lengths = mx.array([1])
    elif variant == "confirmed":
        kwargs['n_confirmed'] = 1
    elif variant == "training":
        layer.train()
    elif variant == "conv_fused":
        layer._fused_conv_decode = True
    elif variant == "norm_fused":
        layer.norm._fused_decode = True
    def forbidden(*args, **kwargs):
        raise AssertionError("excluded path called AR helper")
    monkeypatch.setattr(language, "qwen4_unified_gdn_ar", forbidden)
    mx.eval(layer(x, cache=c, **kwargs))
    assert layer._unified_gdn_ar_graph_calls == 0


def test_ar_failure_does_not_retry_or_publish_state(monkeypatch):
    layer, x, cache = _case()
    layer._unified_gdn_ar = True
    c = cache()
    original = list(c.state)
    def failed(*args, **kwargs):
        raise RuntimeError("AR kernel build failure")
    monkeypatch.setattr(language, "qwen4_unified_gdn_ar", failed)
    with pytest.raises(RuntimeError, match="AR kernel build failure"):
        layer(x, cache=c)
    assert all(w is g for w, g in zip(original, c.state))
    assert c.ar_test_advanced == 0
