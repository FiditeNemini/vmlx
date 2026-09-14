import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
import numpy as np
import pytest

from vmlx_engine.metal.qwen4_qsa_score_reduce import qsa_score_reduce


@pytest.mark.parametrize("rows,pools", [(1,513), (1,4097), (7,2061), (64,8192), (512,16384)])
def test_qsa_score_reduce_matches_stock_fp32_and_selection(rows, pools):
    mx.random.seed(rows + pools)
    query = mx.random.normal((1,rows,4,128))
    keys = mx.random.normal((1,pools,128))
    dots = mx.einsum("bshd,bnd->bshn", query, keys)
    expected = mx.maximum(dots, 0.0).sum(axis=2) / (128**0.5)
    result = qsa_score_reduce(dots, head_dim=128, enabled=True)
    assert result is not None
    mx.eval(result, expected)
    np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))
    if rows <= 7:
        want = mx.argpartition(-expected, kth=511, axis=-1)[..., :512]
        got = mx.argpartition(-result, kth=511, axis=-1)[..., :512]
        mx.eval(want, got)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_qsa_score_reduce_preserves_dynamic_range_and_nonfinite_semantics():
    values = mx.array([0., -0., -1., 1., 1e-20, 1e20, float("inf"), -float("inf"), float("nan")])
    dots = mx.stack([values, values[::-1], values, values[::-1]])[None,None]
    expected = mx.maximum(dots, 0.0).sum(axis=2) / (128**0.5)
    result = qsa_score_reduce(dots, head_dim=128, enabled=True)
    mx.eval(result, expected)
    np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))


@pytest.mark.parametrize("shape,dtype,enabled", [
    ((1,1,4,512), mx.float32, False),
    ((1,1,8,512), mx.float32, True),
    ((2,1,4,512), mx.float32, True),
    ((1,1,4,512), mx.float16, True),
])
def test_qsa_score_reduce_retains_stock_for_unqualified_shapes(shape,dtype,enabled):
    assert qsa_score_reduce(mx.zeros(shape, dtype=dtype), head_dim=128, enabled=enabled) is None


def test_qsa_score_reduce_preserves_noncontiguous_input():
    dots = mx.random.normal((1, 5, 8193, 4)).transpose(0, 1, 3, 2)
    expected = mx.maximum(dots, 0.0).sum(axis=2) / (128**0.5)
    result = qsa_score_reduce(dots, head_dim=128, enabled=True)
    mx.eval(result, expected)
    np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))


@pytest.mark.parametrize("bits,group_size,dtype", [
    (2,32,mx.float16), (4,64,mx.bfloat16), (6,32,mx.bfloat16), (8,64,mx.float32)])
def test_score_reduce_indexer_preserves_selection_and_native_cache(
        monkeypatch, bits, group_size, dtype):
    from vmlx_engine.models.qwen4_exp.language import QSAIndexer, Qwen4ExpTextArgs
    from vmlx_engine.models.minimax_m3.cache import (
        MiniMaxM3SparseCache, clone_minimax_m3_sparse, restore_minimax_m3_sparse)

    monkeypatch.delenv("VMLX_QWEN4_QSA_SCORE_REDUCE", raising=False)
    mx.random.seed(14914 + bits)
    args = Qwen4ExpTextArgs(hidden_size=128, indexer_n_heads=4,
        indexer_head_dim=128, indexer_budget=2048, indexer_compress_ratio=4,
        head_dim=128, partial_rotary_factor=0.25, mrope_section=[8,4,4])
    indexer = QSAIndexer(args)
    assert not indexer._score_reduce
    monkeypatch.setenv("VMLX_QWEN4_QSA_SCORE_REDUCE", "1")
    assert QSAIndexer(args)._score_reduce
    indexer._merge_select = indexer._fused_score_decode = False
    indexer.update(tree_map(lambda v: v.astype(dtype), indexer.parameters()))
    nn.quantize(indexer, bits=bits, group_size=group_size)
    mx.eval(indexer.parameters())

    length = 8195
    initial = MiniMaxM3SparseCache()
    kv = mx.random.normal((1,1,length,4))
    initial.update_and_fetch(kv, kv)
    p = mx.arange(length).astype(mx.float32)
    pos = mx.stack([p,p+17,p+5],axis=-1)[None,None]
    initial.update_index(mx.concatenate([mx.random.normal((1,1,length,128)),pos],axis=-1))
    mx.eval(initial.state)
    base = restore_minimax_m3_sparse(*initial.state)
    candidate = restore_minimax_m3_sparse(*initial.state)
    for rows,blocks in [(1,True),(2,False),(4,True),(65,True)]:
        x = mx.random.normal((1,rows,128)).astype(dtype)
        p = mx.arange(base.offset,base.offset+rows)
        pos = mx.stack([p,p+17,p+5])[:,None,:]
        outputs = []
        for enabled,cache in [(False,base),(True,candidate)]:
            offset = cache.offset
            tail = mx.zeros((1,1,rows,4))
            cache.update_and_fetch(tail,tail)
            indexer._score_reduce = enabled
            outputs.append(indexer(x,cache,offset=offset,position_ids=pos,return_blocks=blocks))
        mx.eval(outputs,base.state,candidate.state)
        for left,right in (zip(*outputs) if blocks else [(outputs[0],outputs[1])]):
            np.testing.assert_array_equal(np.asarray(left),np.asarray(right))
        for left,right in zip(base.state,candidate.state):
            np.testing.assert_array_equal(np.asarray(left),np.asarray(right))
        assert base.offset == candidate.offset == base._idx_offset == candidate._idx_offset
        if rows == 4:
            base = clone_minimax_m3_sparse(base,base.offset-2)
            candidate = clone_minimax_m3_sparse(candidate,candidate.offset-2)
            base = restore_minimax_m3_sparse(*base.state)
            candidate = restore_minimax_m3_sparse(*candidate.state)


def test_score_reduce_falls_back_after_launch_error(monkeypatch, caplog):
    from vmlx_engine.metal import qwen4_qsa_score_reduce as module
    monkeypatch.setattr(module, "_observed", False)
    monkeypatch.setattr(module, "_failed", False)
    dots = mx.ones((1,1,4,512))
    def fail(*args, **kwargs):
        raise RuntimeError("controlled kernel launch error")
    monkeypatch.setattr(module, "_kernel", fail)
    assert qsa_score_reduce(dots, head_dim=128, enabled=True) is None
    assert module._failed and not module._observed
    assert "using stock scoring" in caplog.text
