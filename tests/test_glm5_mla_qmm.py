"""Absorbed-MLA decode/verify matmuls from kv_b_proj's affine codes (VMLX_GLM5_MLA_QMM=1) == the bf16-factor path.

bf16 MLA layer with the real absorbed geometry (rank 512, head dims 64 here), kv_b_proj affine-quantized (8 bit,
group 64) as in GLM-5.3-Flash bundles. 50-token cache, then a 1-row decode and a 4-row verify slab, dense (total below
index_topk) and sparse (index_topk 16). Both paths start from identical cache copies; the qmm path must actually run.
"""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from vmlx_engine.models.glm5_next.glm5_next import Glm5MLACache, MLAAttention, ModelArgs, clone_glm5_next_layer_cache


def _layer(topk):
    mx.random.seed(3)
    args = ModelArgs(hidden_size=512, num_attention_heads=8, q_lora_rank=128, kv_lora_rank=512, qk_nope_head_dim=64,
                     v_head_dim=64, index_n_heads=2, index_head_dim=64, num_hidden_layers=1, vocab_size=64,
                     index_topk=topk, layer_types=["deepseek_sparse_attention"])
    a = MLAAttention(args)
    from mlx.utils import tree_map
    a.update(tree_map(lambda p: p.astype(mx.bfloat16) if p.dtype == mx.float32 else p, a.parameters()))
    for lin in (a.q_a_proj, a.q_b_proj, a.kv_a_proj_with_mqa, a.kv_b_proj, a.o_proj):
        lin.weight = (mx.random.normal(lin.weight.shape) * lin.weight.shape[1] ** -0.5).astype(mx.bfloat16)
    nn.quantize(a, group_size=64, bits=8, class_predicate=lambda p, m: p == "kv_b_proj")
    mx.eval(a.parameters())
    assert hasattr(a.kv_b_proj, "scales")
    return a


def _copy(c):
    return clone_glm5_next_layer_cache(c, copy_fn=lambda x: mx.array(x), copy_is_detached=True)


@pytest.mark.parametrize("topk", [2048, 16])
@pytest.mark.parametrize("T", [1, 4])
def test_qmm_matches_bf16_factors(monkeypatch, topk, T):
    a = _layer(topk)
    mx.random.seed(7)
    cache = Glm5MLACache(absorbed=True)
    a(mx.random.normal((1, 50, 512)).astype(mx.bfloat16), cache=cache)
    mx.eval(cache.state)
    x = mx.random.normal((1, T, 512)).astype(mx.bfloat16)
    calls = []
    orig = MLAAttention._qmm_q_eff
    monkeypatch.setattr(MLAAttention, "_qmm_q_eff", lambda self, q, kbq: calls.append(1) or orig(self, q, kbq))
    monkeypatch.setenv("VMLX_GLM5_MLA_QMM", "0"); ref = a(x, cache=_copy(cache))
    monkeypatch.setenv("VMLX_GLM5_MLA_QMM", "1"); got = a(x, cache=_copy(cache))
    mx.eval(ref, got)
    assert calls, "the quantized path did not run"
    r, g = np.array(ref.astype(mx.float32)), np.array(got.astype(mx.float32))
    assert np.isfinite(g).all()
    # bf16 factors round the dequantized weights to bf16; the codes path accumulates them exactly in fp32
    assert np.max(np.abs(g - r)) <= 2e-2 * np.max(np.abs(r)), np.max(np.abs(g - r)) / np.max(np.abs(r))
