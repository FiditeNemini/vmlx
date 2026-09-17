"""Native cache isolation and first-use readiness for deferred packed PLE."""
import copy
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from vmlx_engine.models.qwen4_exp.deferred_ple import (
    Declined, UnsafeDeferredEvaluation, _active, guard_consumer_eval,
)
from vmlx_engine.models.qwen4_exp.language import _QSAPooledFrontier
from vmlx_engine.models.qwen4_exp.deferred_ple_runtime import (
    _clone_value, _helpers_ready, _make_shadows, _run_shadow_transaction,
)


def test_shadow_array_descriptor_is_not_original_setitem_target():
    source = mx.array([[[1, 2], [3, 4]]], dtype=mx.float16)
    shadow = _clone_value({"keys": source}, _QSAPooledFrontier)
    shadow["keys"][..., 0, :] = mx.array([[9, 9]], dtype=mx.float16)
    assert np.array_equal(np.asarray(source), [[[1, 2], [3, 4]]])
    assert np.array_equal(np.asarray(shadow["keys"]), [[[9, 9], [3, 4]]])


def test_derived_frontier_is_not_shared_mutable_state():
    frontier = _QSAPooledFrontier(4, 1)
    frontier.blocks = 2
    frontier.pooled = mx.ones((1, 2, 4))
    shadow = _clone_value({"qsa_pooled": frontier}, _QSAPooledFrontier)
    shadow["qsa_pooled"].truncate_to_tokens(4)
    assert frontier.blocks == 2 and frontier.pooled.shape == (1, 2, 4)
    assert shadow["qsa_pooled"].blocks == 1


def test_unknown_cache_mutable_declines_before_forward():
    with pytest.raises(Declined):
        _clone_value({"unknown": SimpleNamespace(a=1)}, _QSAPooledFrontier)


def test_early_read_guard_bypasses_kernel_fallback_except_exception():
    token = _active.set(object())
    try:
        with pytest.raises(UnsafeDeferredEvaluation):
            try:
                guard_consumer_eval()
            except Exception:
                pytest.fail("A kernel fallback swallowed unsafe pending evaluation")
    finally:
        _active.reset(token)
    guard_consumer_eval()  # unchanged baseline first-use eval is allowed


def test_q30_first_use_readiness_waits_without_mutating_receipt(monkeypatch):
    from vmlx_engine.metal import qwen4_exact_down as down
    from vmlx_engine.metal import affine_moe_pair_decode as pair
    monkeypatch.setattr(down, "_ENABLED", True)
    monkeypatch.setattr(down, "_OBSERVED", False)
    monkeypatch.setattr(down, "_FAILED", False)
    monkeypatch.setattr(down, "_projection_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(pair, "_FIRST_FAST_CALL", {"qwen4_exp"})
    switch = SimpleNamespace(gate_proj=SimpleNamespace(input_dims=2560),
        down_proj=SimpleNamespace(group_size=64, scales=mx.array([1], dtype=mx.float16)))
    setattr(switch, pair._CONFIG_ATTR, SimpleNamespace(
        hidden=2560, intermediate=640, top_k=10, family="qwen4_exp"))
    layer = SimpleNamespace(is_linear=True, mlp=SimpleNamespace(switch_mlp=switch, top_k=10))
    assert not _helpers_ready([layer], activation_dtype=mx.float16)
    assert down._OBSERVED is False
    monkeypatch.setattr(down, "_OBSERVED", True)
    assert _helpers_ready([layer], activation_dtype=mx.float16)


def test_unreachable_qsa_first_use_flags_do_not_block_short_context(monkeypatch):
    from vmlx_engine.metal import qwen4_exact_down as down
    from vmlx_engine.metal import qwen4_qsa_mask as mask
    from vmlx_engine.metal import sparse_merge_topk as merge
    monkeypatch.setattr(down, "_ENABLED", False)
    monkeypatch.setattr(mask, "_OBSERVED", set())
    monkeypatch.setattr(mask, "_FAILED", False)
    monkeypatch.setattr(merge, "_OBSERVED", False)
    monkeypatch.setattr(merge, "_FAILED", False)
    switch = SimpleNamespace()
    index = SimpleNamespace(compress_ratio=4, block_topk=512, n_heads=4, head_dim=128,
                            _fused_block_mask=True,
                            _merge_select=True, _score_reduce=False,
                            _fused_score_decode=False)
    attn = SimpleNamespace(indexer=index, _sparse_ar_decode=True, _sparse_ar_observed=False,
                          num_heads=24, num_kv_heads=2, head_dim=256, scale=0.0625)
    layer = SimpleNamespace(is_linear=False, self_attn=attn,
                            mlp=SimpleNamespace(switch_mlp=switch, top_k=10))
    assert _helpers_ready([layer], [SimpleNamespace(offset=1023)], mx.float16)
    assert _helpers_ready([layer], [SimpleNamespace(offset=2048)], mx.float16)
    assert not _helpers_ready([layer], [SimpleNamespace(offset=2051)], mx.float16)


@pytest.mark.parametrize("helper,field,value,ready", [
    ("mask", "compress_ratio", 4, False),
    ("mask", "compress_ratio", 8, True),
    ("score", "n_heads", 4, False),
    ("score", "n_heads", 8, True),
    ("merge", "block_topk", 512, False),
    ("merge", "block_topk", 256, True),
    ("score", "block_topk", 4096, True),  # all blocks visible: no score dispatch
])
def test_readiness_matches_same_family_helper_metadata(monkeypatch, helper, field, value, ready):
    from vmlx_engine.metal import qwen4_qsa_mask as mask
    from vmlx_engine.metal import qwen4_qsa_score_reduce as score
    from vmlx_engine.metal import sparse_merge_topk as merge
    monkeypatch.setattr(mask, "_OBSERVED", set())
    monkeypatch.setattr(mask, "_FAILED", False)
    monkeypatch.setattr(score, "_observed", False)
    monkeypatch.setattr(score, "_failed", False)
    monkeypatch.setattr(merge, "_OBSERVED", False)
    monkeypatch.setattr(merge, "_FAILED", False)
    monkeypatch.setattr(merge, "_compatible_sort_version", lambda: True)
    index = SimpleNamespace(compress_ratio=4, block_topk=512, n_heads=4, head_dim=128,
                            _fused_block_mask=helper == "mask", _score_reduce=helper == "score",
                            _merge_select=helper == "merge", _fused_score_decode=False)
    setattr(index, field, value)
    attn = SimpleNamespace(indexer=index, _sparse_ar_decode=False,
                          num_heads=24, num_kv_heads=2, head_dim=256, scale=0.0625)
    layer = SimpleNamespace(is_linear=False, self_attn=attn,
                            mlp=SimpleNamespace(switch_mlp=SimpleNamespace(), top_k=10))
    assert _helpers_ready([layer], [SimpleNamespace(offset=12287)], mx.float16) is ready


@pytest.mark.parametrize("heads,ready", [(24, False), (16, True)])
def test_sparse_readiness_requires_actual_attention_shape(monkeypatch, heads, ready):
    from vmlx_engine.metal import qwen4_sparse_decode as sparse
    monkeypatch.setattr(sparse, "_hardware_allowed", lambda: True)
    monkeypatch.delenv("MLX_SDPA_BLOCKS", raising=False)
    index = SimpleNamespace(compress_ratio=4, block_topk=512, n_heads=4, head_dim=128,
                            _fused_block_mask=False, _score_reduce=False,
                            _merge_select=False, _fused_score_decode=False)
    attn = SimpleNamespace(indexer=index, _sparse_ar_decode=True, _sparse_ar_observed=False,
                          num_heads=heads, num_kv_heads=2, head_dim=256, scale=0.0625)
    layer = SimpleNamespace(is_linear=False, self_attn=attn,
                            mlp=SimpleNamespace(switch_mlp=SimpleNamespace(), top_k=10))
    assert _helpers_ready([layer], [SimpleNamespace(offset=65536)], mx.float16) is ready


@pytest.mark.parametrize("fail_at", [None, "forward", "fill"])
def test_whole_native_cache_transaction_commits_all_or_preserves_all(monkeypatch, fail_at):
    from mlx_lm.models.cache import ArraysCache
    from vmlx_engine.models.minimax_m3.cache import MiniMaxM3SparseCache
    from vmlx_engine.models.qwen4_exp.packed_ple import PackedDeferredPLE
    from tests.test_qwen4_packed_ple import fixture

    ple_a, _, *_ = fixture()
    ple_b, _, *_ = fixture()
    a, b, sparse = ArraysCache(4), ArraysCache(4), MiniMaxM3SparseCache()
    for cache in (a, b):
        cache[0] = mx.array([1.0])
        cache[1] = mx.array([2.0])
        cache[2] = mx.array([[4, 5]], dtype=mx.int32)
        cache[3] = mx.array([3.0])
    sparse.keys = mx.zeros((1, 1, 2, 2))
    sparse.values = mx.ones((1, 1, 2, 2))
    sparse.idx_keys = mx.full((1, 1, 2, 2), 2.0)
    sparse.offset = sparse._idx_offset = 2
    sparse.derived["qsa_pooled"] = _QSAPooledFrontier(4, 1)
    originals = [a, b, sparse]
    layers = [SimpleNamespace(is_linear=True), SimpleNamespace(is_linear=True),
              SimpleNamespace(is_linear=False)]
    old_dicts = [cache.__dict__ for cache in originals]
    shadows = _make_shadows(layers, originals, ArraysCache, MiniMaxM3SparseCache,
                            _QSAPooledFrontier)
    scope = PackedDeferredPLE([(ple_a, shadows[0]), (ple_b, shadows[1])])
    calls = []

    if fail_at == "fill":
        def fail(_host):
            raise IOError("injected after first layer packed fill")
        monkeypatch.setattr(scope.plans[id(ple_b)].leaves[0], "fill", fail)

    def forward(work):
        calls.append(1)
        for cache in work[:2]:
            cache[0] = cache[0] + 10
            cache[1] = cache[1] + 10
            cache[3] = cache[3] + 10
        work[2].keys[..., 0, :] = 7
        work[2].values[..., 0, :] = 8
        work[2].idx_keys[..., 0, :] = 9
        work[2].offset = work[2]._idx_offset = 3
        work[2].derived["qsa_pooled"].blocks = 1
        token = mx.array([[17]])
        out_a = scope.bind_packed(ple_a, token, work[0])
        out_b = scope.bind_packed(ple_b, token, work[1])
        if fail_at == "forward":
            raise IOError("injected model build failure")
        return out_a + out_b

    if fail_at:
        with pytest.raises(IOError):
            _run_shadow_transaction(forward, originals, shadows, scope)
        assert all(cache.__dict__ is before for cache, before in zip(originals, old_dicts))
        for cache in originals[:2]:
            assert np.array_equal(np.asarray(cache[2]), [[4, 5]])
            assert cache[0].item() == 1 and cache[1].item() == 2 and cache[3].item() == 3
        assert sparse.offset == 2 and sparse.derived["qsa_pooled"].blocks == 0
        assert mx.array_equal(sparse.keys, mx.zeros((1, 1, 2, 2))).item()
        assert mx.array_equal(sparse.values, mx.ones((1, 1, 2, 2))).item()
        assert mx.array_equal(sparse.idx_keys, mx.full((1, 1, 2, 2), 2.0)).item()
    else:
        output = _run_shadow_transaction(forward, originals, shadows, scope)
        mx.eval(output)
        for cache in originals[:2]:
            assert np.array_equal(np.asarray(cache[2]), [[5, 17]])
            assert cache[0].item() == 11 and cache[1].item() == 12 and cache[3].item() == 13
        assert sparse.offset == 3 and sparse.derived["qsa_pooled"].blocks == 1
        assert np.array_equal(np.asarray(sparse.keys)[..., 0, :], [[[7, 7]]])
        assert np.array_equal(np.asarray(sparse.values)[..., 0, :], [[[8, 8]]])
        assert np.array_equal(np.asarray(sparse.idx_keys)[..., 0, :], [[[9, 9]]])
    assert calls == [1]  # never retry an already-mutated shadow forward
