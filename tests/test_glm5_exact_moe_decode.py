"""Opt-in native-arithmetic GLM expert path; no other-family qualification."""
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.switch_layers import SwitchGLU

from vmlx_engine.metal import glm5_exact_moe_decode as fused
from vmlx_engine.metal.affine_moe_pair_decode import (
    affine_moe_ar_scope, affine_moe_ar_scope_active,
)
from vmlx_engine.models.glm5_next.glm5_next import ClampedSwiGLU


def exact(a, b):
    mx.eval(a, b)
    assert a.shape == b.shape and a.dtype == b.dtype
    assert bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)))


@pytest.fixture(scope="module")
def switch():
    if not mx.metal.is_available() or not fused._compatible_runtime():
        pytest.skip("exact q2 arithmetic requires MLX0.32.2/M5")
    mx.random.seed(2910)
    obj = SwitchGLU(4096, 2048, 8, activation=ClampedSwiGLU(10.), bias=False)
    nn.quantize(obj, bits=2, group_size=128, mode="affine")
    for p in (obj.gate_proj, obj.up_proj, obj.down_proj):
        p.scales = p.scales.astype(mx.bfloat16)
        p.biases = p.biases.astype(mx.bfloat16)
    obj.eval()
    mx.eval(obj.parameters())
    return obj


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    monkeypatch.setattr(fused, "_FAILED", False)
    monkeypatch.setattr(fused, "_OBSERVED", False)
    monkeypatch.setattr(fused, "_CALL_COUNT", 0)


def operands():
    x = mx.random.normal((1, 1, 4096)).astype(mx.bfloat16)
    indices = mx.array([[[7, 0, 3, 6, 2, 1, 5, 4]]], dtype=mx.uint32)
    weights = mx.array([[[.07, .31, .23, .19, .17, .33, .41, .29]]])
    return x, indices, weights


@pytest.mark.parametrize("case", ["random", "strided", "tiny", "signed_zero", "large"])
def test_native_weighted_output_exact(switch, case):
    x, indices, weights = operands()
    if case == "strided":
        x = mx.stack((x, -x), axis=-1)[..., 0]
        weights = mx.stack((weights, -weights), axis=-1)[..., 0]
    elif case == "tiny":
        x = (x * .0001).astype(mx.bfloat16)
    elif case == "signed_zero":
        x = mx.broadcast_to(mx.array([-0.0], dtype=mx.bfloat16), x.shape)
    elif case == "large":
        x = (x * 40).astype(mx.bfloat16)
    expected = switch(x, indices)
    expected = (expected * weights[..., None].astype(expected.dtype)).sum(axis=-2)
    with affine_moe_ar_scope():
        for _ in range(2):
            actual = fused.glm5_exact_moe_output(switch, x, indices, weights, enabled=True)
            assert actual is not None, "the candidate must execute, not fallback"
            exact(actual, expected)
    assert fused._CALL_COUNT == 2 and fused._OBSERVED


def test_disabled_and_non_ar_never_construct_kernels(switch, monkeypatch):
    def forbidden():
        raise AssertionError("unqualified call reached a kernel")
    monkeypatch.setattr(fused, "_kernels", forbidden)
    x, i, w = operands()
    assert not affine_moe_ar_scope_active()
    assert fused.glm5_exact_moe_output(switch, x, i, w, enabled=True) is None
    with affine_moe_ar_scope():
        assert affine_moe_ar_scope_active()
        assert fused.glm5_exact_moe_output(switch, x, i, w, enabled=False) is None
        with affine_moe_ar_scope():
            assert affine_moe_ar_scope_active()
        assert affine_moe_ar_scope_active()
    assert not affine_moe_ar_scope_active()


@pytest.mark.parametrize("case", ["prefill", "batch2", "half", "float", "routes", "scores",
                                  "q4", "gs64", "mixed_down", "wrong_activation", "training"])
def test_ineligible_layouts_retain_stock(switch, monkeypatch, case):
    x, i, w = operands()
    if case == "prefill":
        x = mx.broadcast_to(x, (1, 2, 4096))
    elif case == "batch2":
        x = mx.broadcast_to(x, (2, 1, 4096))
    elif case in ("half", "float"):
        x = x.astype(mx.float16 if case == "half" else mx.float32)
    elif case == "routes":
        i = i[..., :4]
    elif case == "scores":
        w = w[..., :4]
    elif case == "q4":
        monkeypatch.setattr(switch.gate_proj, "bits", 4)
    elif case == "gs64":
        monkeypatch.setattr(switch.up_proj, "group_size", 64)
    elif case == "mixed_down":
        monkeypatch.setattr(switch.down_proj, "mode", "mxfp4")
    elif case == "wrong_activation":
        monkeypatch.setattr(switch.activation, "_limit", 9.)
    else:
        switch.train()
    def forbidden():
        raise AssertionError("ineligible layout reached kernel")
    monkeypatch.setattr(fused, "_kernels", forbidden)
    try:
        with affine_moe_ar_scope():
            assert fused.glm5_exact_moe_output(switch, x, i, w, enabled=True) is None
    finally:
        switch.eval()


def test_first_launch_failure_is_fail_closed(switch, monkeypatch, caplog):
    def broken():
        raise RuntimeError("controlled compiler failure")
    monkeypatch.setattr(fused, "_kernels", broken)
    x, i, w = operands()
    with affine_moe_ar_scope():
        assert fused.glm5_exact_moe_output(switch, x, i, w, enabled=True) is None
    assert fused._FAILED and not fused._OBSERVED
    assert "retaining stock expert path" in caplog.text
    assert fused._CALL_COUNT == 0


def test_unqualified_runtime_never_launches(switch, monkeypatch):
    monkeypatch.setattr(fused, "_compatible_runtime", lambda: False)
    def forbidden():
        raise AssertionError("wrong runtime reached kernel")
    monkeypatch.setattr(fused, "_kernels", forbidden)
    with affine_moe_ar_scope():
        assert fused.glm5_exact_moe_output(switch, *operands(), enabled=True) is None


def test_default_off_and_glm_only_experimental_namespace(monkeypatch):
    from vmlx_engine.glm5_decode_policy import glm5_exact_moe_requested
    from vmlx_engine.prefix_cache import compute_model_cache_key
    for family in ("glm5_next", "glm5_next_text", "qwen4_exp", "qwen3_5"):
        model = SimpleNamespace(args=SimpleNamespace(model_type=family))
        monkeypatch.delenv("VMLX_GLM5_EXACT_MOE_DECODE", raising=False)
        assert not glm5_exact_moe_requested()
        default = compute_model_cache_key(model)
        monkeypatch.setenv("VMLX_GLM5_EXACT_MOE_DECODE", "0")
        assert compute_model_cache_key(model) == default
        monkeypatch.setenv("VMLX_GLM5_EXACT_MOE_DECODE", "1")
        assert glm5_exact_moe_requested()
        assert (compute_model_cache_key(model) != default) == family.startswith("glm5_next")


def test_glm_layer_snapshots_flag_and_falls_back_for_prefill(monkeypatch):
    from vmlx_engine.models.glm5_next.glm5_next import MoEBlock
    args = SimpleNamespace(hidden_size=128, n_routed_experts=8,
                           num_experts_per_tok=8, norm_topk_prob=True,
                           routed_scaling_factor=2.5, moe_intermediate_size=64,
                           n_shared_experts=1, swiglu_limit=10.)
    monkeypatch.delenv("VMLX_GLM5_EXACT_MOE_DECODE", raising=False)
    default = MoEBlock(args)
    assert not default._exact_moe_decode
    monkeypatch.setenv("VMLX_GLM5_EXACT_MOE_DECODE", "1")
    assert not default._exact_moe_decode
    opted = MoEBlock(args)
    assert opted._exact_moe_decode
    opted.eval()
    x = mx.random.normal((1, 2, 128)).astype(mx.bfloat16)
    opted._exact_moe_decode = False
    expected = opted(x)
    opted._exact_moe_decode = True
    with affine_moe_ar_scope():
        actual = opted(x)
    mx.eval(expected, actual)
    assert bool(mx.array_equal(expected, actual))
    assert fused._CALL_COUNT == 0
