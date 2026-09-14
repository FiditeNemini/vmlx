"""Exact ordering and bounded dispatch for experimental QSA pruned merges."""

import logging

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
import numpy as np
import pytest

from vmlx_engine.metal import sparse_merge_topk as selection


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    monkeypatch.setattr(selection, "_FAILED", False)
    monkeypatch.setattr(selection, "_OBSERVED", False)
    monkeypatch.setattr(selection, "_VERIFIED_VARIANTS", set())


def exact_selection(scores):
    before = mx.array(scores)
    expected = mx.argpartition(scores, kth=511, axis=-1)[..., :512]
    actual = selection.sparse_merge_topk(scores, k=512, enabled=True)
    assert actual is not None
    mx.eval(before, scores, expected, actual)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    np.testing.assert_array_equal(np.asarray(scores), np.asarray(before))


@pytest.mark.parametrize("rows,n", [(1,2049),(1,5003),(4,8192),(64,16387),(1,65536)])
def test_exact_random_odd_runs_and_context_growth(rows,n):
    rng = np.random.default_rng(914+n)
    exact_selection(mx.array(rng.normal(size=(1,rows,n)).astype(np.float32)))
    assert selection._VERIFIED_VARIANTS == {"tile", "merge"}


@pytest.mark.parametrize("kind", ["ties","zeros","masked","nonfinite","strided"])
def test_exact_ties_causal_masks_nonfinite_and_layout(kind):
    rng = np.random.default_rng(914)
    raw = rng.integers(-8,9,size=(1,4,5003)).astype(np.float32)
    if kind == "zeros":
        raw.fill(0)
        raw[...,1::2] = -0.0
    elif kind == "masked":
        for row,valid in enumerate((0,3,2048,5003)):
            raw[:,row,valid:] = np.inf
    elif kind == "nonfinite":
        raw[...,::3] = np.nan
        raw[...,1::13] = np.inf
        raw[...,2::19] = -np.inf
    scores = mx.array(raw)
    if kind == "strided":
        scores = mx.stack([scores,scores],axis=-1).reshape(1,4,-1)[...,::2]
    exact_selection(scores)


def test_ineligible_and_disabled_inputs_do_not_launch(monkeypatch):
    monkeypatch.setattr(selection,"_kernel",lambda _:pytest.fail("unexpected launch"))
    scores = mx.zeros((1,4,4097))
    assert selection.sparse_merge_topk(scores,k=512,enabled=False) is None
    for shape in [(2,4,4097),(1,4,2048),(1,0,4097),(4,4097)]:
        assert selection.sparse_merge_topk(mx.zeros(shape),k=512,enabled=True) is None
    for dtype in [mx.float16,mx.bfloat16,mx.int32]:
        assert selection.sparse_merge_topk(scores.astype(dtype),k=512,enabled=True) is None
    for k in [0,511,513,True,512.0]:
        assert selection.sparse_merge_topk(scores,k=k,enabled=True) is None
    monkeypatch.setattr(selection,"_compatible_sort_version",lambda:False)
    assert selection.sparse_merge_topk(scores,k=512,enabled=True) is None
    monkeypatch.setattr(selection,"_compatible_sort_version",lambda:True)
    monkeypatch.setattr(mx,"default_device",lambda:mx.cpu)
    assert selection.sparse_merge_topk(scores,k=512,enabled=True) is None


def test_first_lazy_failure_keeps_stock_and_does_not_retry(monkeypatch,caplog):
    scores = mx.zeros((1,4,4097))
    mx.eval(scores)
    def fail(*arrays):
        raise RuntimeError("controlled lazy dispatch failure")
    monkeypatch.setattr(mx,"eval",fail)
    with caplog.at_level(logging.WARNING):
        assert selection.sparse_merge_topk(scores,k=512,enabled=True) is None
    assert selection._FAILED and not selection._OBSERVED
    assert not selection._VERIFIED_VARIANTS
    assert "disabled after launch failure" in caplog.text
    monkeypatch.setattr(selection,"_kernel",lambda _:pytest.fail("unexpected retry"))
    assert selection.sparse_merge_topk(scores,k=512,enabled=True) is None


@pytest.mark.parametrize("bits,group_size,dtype", [
    (2,32,mx.float16),(4,64,mx.bfloat16),(6,32,mx.bfloat16),(8,64,mx.float32)])
def test_qsa_indexer_preserves_mask_selected_blocks_and_restored_state(
        monkeypatch,bits,group_size,dtype):
    from vmlx_engine.models.qwen4_exp.language import QSAIndexer, Qwen4ExpTextArgs
    from vmlx_engine.models.minimax_m3.cache import (
        MiniMaxM3SparseCache, clone_minimax_m3_sparse, restore_minimax_m3_sparse)

    monkeypatch.delenv("VMLX_QWEN4_QSA_MERGE_SELECT",raising=False)
    mx.random.seed(914+bits)
    args = Qwen4ExpTextArgs(hidden_size=128,indexer_n_heads=2,
        indexer_head_dim=32,indexer_budget=2048,indexer_compress_ratio=4,
        head_dim=32,partial_rotary_factor=0.25,mrope_section=[2,1,1])
    indexer = QSAIndexer(args)
    assert not indexer._merge_select
    monkeypatch.setenv("VMLX_QWEN4_QSA_MERGE_SELECT","1")
    assert QSAIndexer(args)._merge_select
    indexer.update(tree_map(lambda v:v.astype(dtype),indexer.parameters()))
    nn.quantize(indexer,bits=bits,group_size=group_size)
    mx.eval(indexer.parameters())

    length = 8195
    initial = MiniMaxM3SparseCache()
    kv = mx.random.normal((1,1,length,4))
    initial.update_and_fetch(kv,kv)
    pos = mx.arange(length).astype(mx.float32)
    positions = mx.stack([pos,pos+17,pos+5],axis=-1)[None,None,:,:]
    initial.update_index(mx.concatenate([
        mx.random.normal((1,1,length,32)),positions],axis=-1))
    mx.eval(initial.state)
    base = restore_minimax_m3_sparse(*initial.state)
    candidate = restore_minimax_m3_sparse(*initial.state)
    for rows,blocks in [(1,False),(2,True),(3,False),(4,True),(65,False)]:
        x = mx.random.normal((1,rows,128)).astype(dtype)
        p = mx.arange(base.offset,base.offset+rows)
        media_pos = mx.stack([p,p+17,p+5])[:,None,:]
        outputs = []
        for enabled,cache in [(False,base),(True,candidate)]:
            offset = cache.offset
            tail = mx.zeros((1,1,rows,4))
            cache.update_and_fetch(tail,tail)
            indexer._merge_select = enabled
            outputs.append(indexer(x,cache,offset=offset,position_ids=media_pos,
                                   return_blocks=blocks))
        mx.eval(outputs,base.state,candidate.state)
        pairs = zip(*outputs) if blocks else [(outputs[0],outputs[1])]
        for left,right in pairs:
            np.testing.assert_array_equal(np.asarray(left),np.asarray(right))
        for left,right in zip(base.state,candidate.state):
            np.testing.assert_array_equal(np.asarray(left),np.asarray(right))
        assert base.offset==candidate.offset==base._idx_offset==candidate._idx_offset
        if rows==4:
            # Model rollback followed by disk-format reconstruction; no
            # derived frontier may conceal a differing restored selection.
            base = clone_minimax_m3_sparse(base,base.offset-2)
            candidate = clone_minimax_m3_sparse(candidate,candidate.offset-2)
            base = restore_minimax_m3_sparse(*base.state)
            candidate = restore_minimax_m3_sparse(*candidate.state)
    assert selection._OBSERVED
