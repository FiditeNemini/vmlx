# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
import pytest

from vmlx_engine.glm5_decode_policy import glm5_kda_lowrank_requested
from vmlx_engine.metal.glm5_kda_lowrank import Glm5KDALowRankGroup
from vmlx_engine.models.glm5_next.glm5_next import (
    Glm5KDACache, KDAAttention, ModelArgs,
)


def exact(a, b):
    mx.eval(a, b)
    assert a.shape == b.shape and a.dtype == b.dtype
    word = mx.uint16 if a.dtype == mx.bfloat16 else mx.uint32
    assert bool(mx.all(a.view(word) == b.view(word)).item())


def projections(hidden=128, rank=128, heads=2):
    mx.random.seed(17)
    shapes = ((hidden,rank),(hidden,rank),(hidden,heads),
              (rank,heads*rank),(rank,heads*rank))
    linears = tuple(nn.Linear(a,b,bias=False) for a,b in shapes)
    for l in linears:
        l.weight = l.weight.astype(mx.bfloat16)
    return linears


@pytest.mark.parametrize("seed", [0, 31, 93])
def test_equal_shape_decode_and_original_size_fallback(seed):
    fa,ga,b,fb,gb = projections()
    group = Glm5KDALowRankGroup(fa,ga,b,fb,gb)
    mx.random.seed(seed)
    for batch,tokens in ((1,1),(1,7),(1,67),(2,1)):
        x = mx.random.normal((batch,tokens,128)).astype(mx.bfloat16)
        reference = (fb(fa(x)),gb(ga(x)),b(x))
        separate = (group.decay(x),group.output_gate(x),group.beta(x))
        for a,c in zip(reference,separate): exact(a,c)
        actual = group.decode(x)
        if (batch,tokens) == (1,1):
            for a,c in zip(reference,actual): exact(a,c)
        else:
            assert actual is None
    assert group.observed_calls == 1
    group.enabled = False
    assert group.decode(mx.zeros((1,1,128),dtype=mx.bfloat16)) is None


@pytest.mark.parametrize("defect", ["bias", "quantized", "dtype", "input", "output"])
def test_rejects_incompatible_weight_contract(defect):
    linears = list(projections())
    if defect == "bias": linears[0].bias = mx.zeros((128,),dtype=mx.bfloat16)
    elif defect == "quantized": linears[0] = nn.QuantizedLinear.from_linear(linears[0],bits=6,group_size=64)
    elif defect == "dtype": linears[0].weight = linears[0].weight.astype(mx.float32)
    elif defect == "input": linears[1].weight = linears[1].weight[:,:64]
    else: linears[4].weight = linears[4].weight[:128]
    assert not Glm5KDALowRankGroup.compatible(*linears)
    with pytest.raises(ValueError,match="matching unbiased BF16"):
        Glm5KDALowRankGroup(*linears)


def test_ineligible_activation_never_enters_grouped_matmul(monkeypatch):
    group = Glm5KDALowRankGroup(*projections())
    assert group.decode(mx.zeros((1,1,128),dtype=mx.float32)) is None
    assert group.decode(mx.zeros((1,128),dtype=mx.bfloat16)) is None
    monkeypatch.setattr(mx,"default_device",lambda:mx.cpu)
    assert group.decode(mx.zeros((1,1,128),dtype=mx.bfloat16)) is None
    assert group.observed_calls == 0


def test_default_off_and_only_glm_cache_identity(monkeypatch):
    from vmlx_engine.prefix_cache import compute_model_cache_key
    for family in ("glm5_next","glm5_next_text","qwen4_exp","qwen3_5"):
        model = SimpleNamespace(args=SimpleNamespace(model_type=family))
        monkeypatch.delenv("VMLX_GLM5_KDA_LOWRANK_GROUP",raising=False)
        assert not glm5_kda_lowrank_requested()
        off = compute_model_cache_key(model)
        monkeypatch.setenv("VMLX_GLM5_KDA_LOWRANK_GROUP","1")
        assert glm5_kda_lowrank_requested()
        assert (off != compute_model_cache_key(model)) == family.startswith("glm5_next")


def test_native_state_segments_and_packed_ownership(monkeypatch):
    args = ModelArgs(hidden_size=128,linear_num_heads=2)
    mx.random.seed(71)
    left = KDAAttention(args)
    left.update(tree_map(lambda a:a.astype(mx.bfloat16),left.parameters()))
    for name in ("q_conv1d","k_conv1d","v_conv1d"):
        setattr(left,name,(mx.random.normal((256,4))*.1).astype(mx.bfloat16))
    right = KDAAttention(args); right.update(left.parameters())
    monkeypatch.delenv("VMLX_GLM5_KDA_LOWRANK_GROUP",raising=False)
    assert not right.prepare_lowrank_runtime()
    old_bytes = sum(getattr(right,k).weight.nbytes for k in
                    ("f_a_proj","g_a_proj","b_proj","f_b_proj","g_b_proj"))
    monkeypatch.setenv("VMLX_GLM5_KDA_LOWRANK_GROUP","1")
    assert right.prepare_lowrank_runtime()
    group = right.lowrank_group
    assert right.prepare_lowrank_runtime() and right.lowrank_group is group
    assert sum(w.nbytes for w in (group.input_weights,group.output_weights,group.beta_weight)) == old_bytes
    assert all(getattr(right,k) is None for k in
               ("f_a_proj","g_a_proj","b_proj","f_b_proj","g_b_proj"))
    ca,cb = Glm5KDACache(),Glm5KDACache()
    for tokens in (67,1,1,7,1):
        x = (mx.random.normal((1,tokens,128))*.1).astype(mx.bfloat16)
        a=left(x,cache=ca);c=right(x,cache=cb)
        exact(a,c)
        for sa,sb in zip(ca.state,cb.state):exact(sa,sb)
    assert group.observed_calls == 3
