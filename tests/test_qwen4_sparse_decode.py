"""Exact AR partition order, rounded partials and strided logical KV views."""
import mlx.core as mx
import numpy as np
import pytest

from vmlx_engine.metal.qwen4_sparse_decode import attention


@pytest.mark.parametrize("tokens,tail", [(76800,0), (76801,1), (76802,2), (76803,3)])
def test_sparse_ar_matches_stock_partitions_and_incomplete_tail(tokens, tail):
    q = mx.random.normal((1,24,1,256), key=mx.random.key(10 + tail)).astype(mx.float16)
    k = mx.random.normal((1,2,tokens+19,256), key=mx.random.key(20 + tail)).astype(mx.float16)[:,:,:tokens]
    v = mx.random.normal((1,2,tokens+29,256), key=mx.random.key(30 + tail)).astype(mx.float16)[:,:,:tokens]
    block_ids = (mx.arange(512) * (tokens//4 - 1)//512).astype(mx.int32)
    keep = mx.put_along_axis(mx.zeros((tokens//4,), dtype=mx.bool_), block_ids,
                             mx.array(True), axis=0)
    keep = mx.concatenate([mx.repeat(keep,4),mx.ones((tail,),dtype=mx.bool_)])
    mask = mx.where(keep,0,-mx.inf).astype(mx.float16)[None,None,None,:]
    expected = mx.fast.scaled_dot_product_attention(q,k,v,scale=0.0625,mask=mask)
    actual = attention(q,k,v,mask,scale=0.0625,enabled=True)
    assert actual is not None
    mx.eval(expected,actual)
    np.testing.assert_array_equal(np.asarray(actual),np.asarray(expected))


@pytest.mark.parametrize("change", ["off", "dtype", "rows", "batch", "context", "override", "mask", "scale"])
def test_unqualified_ar_retains_stock(monkeypatch, change):
    q,k,v = mx.zeros((1,24,1,256),dtype=mx.float16),mx.zeros((1,2,76800,256),dtype=mx.float16),mx.zeros((1,2,76800,256),dtype=mx.float16)
    mask = mx.zeros((1,1,1,76800),dtype=mx.float16)
    enabled,scale = True,0.0625
    if change == "off": enabled=False
    if change == "dtype": q=q.astype(mx.bfloat16)
    if change == "rows": q=mx.zeros((1,24,4,256),dtype=mx.float16)
    if change == "batch": q=mx.zeros((2,24,1,256),dtype=mx.float16)
    if change == "context": k=k[:,:,:16384]
    if change == "override": monkeypatch.setenv("MLX_SDPA_BLOCKS","128")
    if change == "mask": mask=None
    if change == "scale": scale=0.125
    assert attention(q,k,v,mask,scale=scale,enabled=enabled) is None


@pytest.mark.parametrize("tokens,pattern", [(65537,"single"),(131072,"sparse"),(76803,"empty"),(76801,"finite")])
def test_dispatch_edges_and_strided_dimensions(tokens, pattern):
    # Last-dimension strides and logical capacity slices must not trigger a
    # hidden full-KV copy or treat backing capacity as the live frontier.
    q=mx.random.normal((1,24,1,512),key=mx.random.key(111)).astype(mx.float16)[:,:,:,::2]
    k=mx.random.normal((1,2,tokens+9,512),key=mx.random.key(112)).astype(mx.float16)[:,:,:tokens,::2]
    v=mx.random.normal((1,2,tokens+13,512),key=mx.random.key(113)).astype(mx.float16)[:,:,:tokens,::2]
    positions=mx.arange(tokens)
    if pattern == "single": keep=positions==tokens-1
    elif pattern == "empty": keep=positions<0
    else: keep=positions%41<4
    values=-(positions%5).astype(mx.float16) if pattern == "finite" else mx.zeros((tokens,),dtype=mx.float16)
    mask=mx.where(keep,values,-mx.inf)[None,None,None,:]
    expected=mx.fast.scaled_dot_product_attention(q,k,v,scale=0.0625,mask=mask)
    actual=attention(q,k,v,mask,scale=0.0625,enabled=True)
    mx.eval(expected,actual)
    np.testing.assert_array_equal(np.asarray(actual),np.asarray(expected))


def test_attention_runtime_default_and_opt_in(monkeypatch):
    from vmlx_engine.models.qwen4_exp.language import Qwen4ExpTextArgs, QSAAttention
    monkeypatch.delenv("VMLX_QWEN4_SPARSE_AR",raising=False)
    args=Qwen4ExpTextArgs(hidden_size=32)
    assert not QSAAttention(args)._sparse_ar_decode
    monkeypatch.setenv("VMLX_QWEN4_SPARSE_AR","1")
    assert QSAAttention(args)._sparse_ar_decode
