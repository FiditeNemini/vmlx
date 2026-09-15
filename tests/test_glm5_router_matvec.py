"""Exact opt-in router cast fusion, not whole-model/sustained qualification."""
from types import SimpleNamespace

import mlx.core as mx
import pytest

from vmlx_engine.metal import glm5_router_matvec as fused
from vmlx_engine.metal.affine_moe_pair_decode import affine_moe_ar_scope


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    monkeypatch.setattr(fused, "_FAILED", False)
    monkeypatch.setattr(fused, "_OBSERVED", False)
    monkeypatch.setattr(fused, "_CALL_COUNT", 0)


@pytest.fixture
def operands():
    if not mx.metal.is_available() or not fused._compatible_runtime():
        pytest.skip("native FP32 tree is qualified only on MLX0.32.2/M5")
    mx.random.seed(2917)
    return (mx.random.normal((1, 1, 4096)).astype(mx.bfloat16),
            mx.random.normal((288, 4096)).astype(mx.bfloat16))


@pytest.mark.parametrize("case", ["random", "strided", "tiny", "large", "signed_zero"])
def test_full_fp32_words_match_native(operands, case):
    x, weight = operands
    if case == "strided":
        x = mx.stack((x, -x), axis=-1)[..., 0]
        weight = mx.stack((weight, -weight), axis=-1)[..., 0]
    elif case in ("tiny", "large"):
        x = (x * (0.0001 if case == "tiny" else 1000)).astype(mx.bfloat16)
    elif case == "signed_zero":
        x = mx.broadcast_to(mx.array([-0.0], dtype=mx.bfloat16), x.shape)
    expected = x.astype(mx.float32) @ weight.astype(mx.float32).T
    with affine_moe_ar_scope():
        for _ in range(2):
            actual = fused.glm5_router_logits(x, weight, enabled=True)
            assert actual is not None and actual.dtype == mx.float32
            mx.eval(actual, expected)
            assert bool(mx.array_equal(actual.view(mx.uint32), expected.view(mx.uint32)))
    assert fused._CALL_COUNT == 2 and fused._OBSERVED


@pytest.mark.parametrize("case", ["off", "non_ar", "training", "prefill", "batch2",
                                  "hidden", "experts", "float", "half", "weight_float", "runtime"])
def test_unqualified_shapes_keep_stock(operands, monkeypatch, case):
    x, w = operands
    if case == "prefill": x = mx.broadcast_to(x, (1, 2, 4096))
    elif case == "batch2": x = mx.broadcast_to(x, (2, 1, 4096))
    elif case == "hidden": x = x[..., :2048]
    elif case == "experts": w = w[:256]
    elif case in ("float", "half"): x = x.astype(mx.float32 if case == "float" else mx.float16)
    elif case == "weight_float": w = w.astype(mx.float32)
    elif case == "runtime": monkeypatch.setattr(fused, "_compatible_runtime", lambda: False)
    def forbidden(): raise AssertionError("unqualified shape reached kernel")
    monkeypatch.setattr(fused, "_kernel", forbidden)
    if case == "non_ar":
        assert fused.glm5_router_logits(x, w, enabled=True) is None
    else:
        with affine_moe_ar_scope():
            assert fused.glm5_router_logits(x, w, enabled=case != "off", training=case == "training") is None
    assert fused._CALL_COUNT == 0


def test_first_launch_failure_retains_stock(operands, monkeypatch, caplog):
    def broken(): raise RuntimeError("controlled compiler failure")
    monkeypatch.setattr(fused, "_kernel", broken)
    with affine_moe_ar_scope():
        assert fused.glm5_router_logits(*operands, enabled=True) is None
    assert fused._FAILED and not fused._OBSERVED and fused._CALL_COUNT == 0
    assert "retaining stock FP32 matmul" in caplog.text


def test_default_off_and_glm_only_namespace(monkeypatch):
    from vmlx_engine.glm5_decode_policy import glm5_router_matvec_requested
    from vmlx_engine.prefix_cache import compute_model_cache_key
    for family in ("glm5_next", "glm5_next_text", "qwen4_exp", "qwen3_5"):
        model = SimpleNamespace(args=SimpleNamespace(model_type=family))
        monkeypatch.delenv("VMLX_GLM5_ROUTER_MATVEC", raising=False)
        assert not glm5_router_matvec_requested()
        default = compute_model_cache_key(model)
        monkeypatch.setenv("VMLX_GLM5_ROUTER_MATVEC", "0")
        assert compute_model_cache_key(model) == default
        monkeypatch.setenv("VMLX_GLM5_ROUTER_MATVEC", "1")
        assert glm5_router_matvec_requested()
        assert (compute_model_cache_key(model) != default) == family.startswith("glm5_next")


def test_layer_snapshots_flag_and_prefill_falls_back(monkeypatch):
    from vmlx_engine.models.glm5_next.glm5_next import MoEBlock
    args = SimpleNamespace(hidden_size=128, n_routed_experts=8, num_experts_per_tok=8,
                           norm_topk_prob=True, routed_scaling_factor=2.5,
                           moe_intermediate_size=64, n_shared_experts=1, swiglu_limit=10.)
    monkeypatch.delenv("VMLX_GLM5_ROUTER_MATVEC", raising=False)
    default = MoEBlock(args)
    assert not default._router_matvec
    monkeypatch.setenv("VMLX_GLM5_ROUTER_MATVEC", "1")
    assert not default._router_matvec
    opted = MoEBlock(args); opted.eval()
    assert opted._router_matvec
    x = mx.random.normal((1, 2, 128)).astype(mx.bfloat16)
    opted._router_matvec = False; expected = opted(x)
    opted._router_matvec = True
    with affine_moe_ar_scope(): actual = opted(x)
    mx.eval(expected, actual)
    assert bool(mx.array_equal(expected, actual)) and fused._CALL_COUNT == 0
