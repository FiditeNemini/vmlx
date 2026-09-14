"""Bulk QSA prefill must not silently allocate dense attention scores."""

import importlib
from types import SimpleNamespace

import mlx.core as mx
import pytest

module = importlib.import_module("vmlx_engine.metal.qwen4_prefill_sdpa")


def shapes(rows=256, context=4096, batch=1, dim=256, dtype=mx.bfloat16):
    def tensor(shape):
        return SimpleNamespace(shape=shape, ndim=4, dtype=dtype)
    return (
        tensor((batch, 24, rows, dim)),
        tensor((batch, 2, context, dim)),
        tensor((batch, 2, context, dim)),
    )


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_bulk_prefill_requests_fused_without_changing_inputs(batch, dtype, monkeypatch):
    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module.mx, "default_device", lambda: mx.gpu)
    monkeypatch.delenv("VMLX_QWEN4_PREFILL_FUSED_SDPA", raising=False)
    q, k, v = shapes(batch=batch, dtype=dtype)
    mask, output = object(), object()
    calls = []
    def sdpa(*args, **kwargs):
        calls.append((args, kwargs))
        return output
    monkeypatch.setattr(module.mx.fast, "scaled_dot_product_attention", sdpa)
    assert module.qwen4_prefill_sdpa(q, k, v, mask, scale=0.0625) is output
    assert calls == [((q, k, v), {"mask":mask, "scale":0.0625, "force_fused":True})]


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 8, 16, 255])
def test_decode_mtp_and_small_suffix_dispatch_stays_unchanged(rows, monkeypatch):
    monkeypatch.setattr(module.sys, "platform", "darwin")
    assert module.qwen4_prefill_sdpa(*shapes(rows=rows), None, scale=0.0625) is None


@pytest.mark.parametrize("kwargs", [{"dim":128}, {"dtype":mx.float32}, {"context":128}])
def test_unqualified_geometry_or_dtype_falls_through(kwargs, monkeypatch):
    monkeypatch.setattr(module.sys, "platform", "darwin")
    assert module.qwen4_prefill_sdpa(*shapes(**kwargs), None, scale=0.0625) is None


def test_training_cpu_other_platform_and_kill_switch_fall_through(monkeypatch):
    monkeypatch.setattr(module.sys, "platform", "darwin")
    q, k, v = shapes()
    assert module.qwen4_prefill_sdpa(q,k,v,None,scale=0.0625,training=True) is None
    monkeypatch.setattr(module.mx, "default_device", lambda: mx.cpu)
    assert module.qwen4_prefill_sdpa(q,k,v,None,scale=0.0625) is None
    monkeypatch.setattr(module.mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(module.sys, "platform", "linux")
    assert module.qwen4_prefill_sdpa(q,k,v,None,scale=0.0625) is None
    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setenv("VMLX_QWEN4_PREFILL_FUSED_SDPA", "0")
    assert module.qwen4_prefill_sdpa(q,k,v,None,scale=0.0625) is None


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_real_sparse_mask_gqa_stays_close_to_float32_reference(dtype):
    if module.sys.platform != "darwin" or mx.default_device() != mx.gpu:
        pytest.skip("requires the qualified Metal backend")
    mx.random.seed(9856)
    q = mx.random.normal((1, 4, 256, 256)).astype(dtype)
    k = mx.random.normal((1, 1, 1024, 256)).astype(dtype)
    v = mx.random.normal((1, 1, 1024, 256)).astype(dtype)
    qi, ki = mx.arange(768,1024)[:,None], mx.arange(1024)[None,:]
    keep = (ki <= qi) & (((ki//4)%3 == 0) | (ki//4 == qi//4))
    mask = mx.where(keep, 0., -float("inf"))[None,None].astype(dtype)
    candidate = module.qwen4_prefill_sdpa(q,k,v,mask,scale=0.0625)
    baseline = mx.fast.scaled_dot_product_attention(q,k,v,mask=mask,scale=0.0625)
    logits = (q.astype(mx.float32)/16) @ mx.swapaxes(k.astype(mx.float32),-1,-2)
    reference = mx.softmax(logits+mask.astype(mx.float32),axis=-1) @ v.astype(mx.float32)
    assert candidate is not None and candidate.dtype == dtype
    assert bool(mx.all(mx.isfinite(candidate)))
    candidate_rmse = float(mx.sqrt(mx.mean((candidate.astype(mx.float32)-reference)**2)))
    baseline_rmse = float(mx.sqrt(mx.mean((baseline.astype(mx.float32)-reference)**2)))
    assert candidate_rmse <= baseline_rmse * 1.5 + 1e-6
    assert bool(mx.allclose(candidate,baseline,atol=0.005,rtol=0.02))
