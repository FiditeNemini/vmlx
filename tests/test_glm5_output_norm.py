"""GLM-only exact norm boundary and fallbacks; not serving-speed evidence."""
from types import SimpleNamespace

import mlx.core as mx
import pytest

from vmlx_engine.metal import glm5_output_norm as fused
from vmlx_engine.metal.affine_moe_pair_decode import affine_moe_ar_scope


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    monkeypatch.setattr(fused, "_FAILED", False)
    monkeypatch.setattr(fused, "_OBSERVED", False)
    monkeypatch.setattr(fused, "_CALL_COUNT", 0)


@pytest.fixture
def operands():
    if not mx.metal.is_available() or not fused._compatible_runtime():
        pytest.skip("native sum tree qualified only on MLX0.32.2/M5")
    mx.random.seed(2121)
    return (mx.random.normal((1, 1, 64, 128)),
            (mx.random.normal((1, 1, 64, 128)) * 8).astype(mx.bfloat16),
            mx.random.normal((128,)).astype(mx.bfloat16))


def reference(x, gate, weight, eps):
    o = x.astype(mx.float32)
    o = o * mx.rsqrt(mx.mean(o * o, axis=-1, keepdims=True) + eps)
    o = weight.astype(mx.float32) * o
    return (o * mx.sigmoid(gate.astype(mx.float32))).astype(mx.bfloat16)


def assert_exact(x, gate, weight, eps=1e-6):
    before = [mx.array(v) for v in (x, gate, weight)]
    mx.eval(before)
    expected = reference(x, gate, weight, eps)
    with affine_moe_ar_scope():
        actual = fused.glm5_output_norm(x, gate, weight, eps,
            output_dtype=mx.bfloat16, enabled=True)
    assert actual is not None
    mx.eval(expected, actual)
    assert bool(mx.array_equal(expected.view(mx.uint16), actual.view(mx.uint16)))
    for old, new in zip(before, (x, gate, weight)):
        dtype = mx.uint32 if new.dtype == mx.float32 else mx.uint16
        assert bool(mx.array_equal(old.view(dtype), new.view(dtype)))


@pytest.mark.parametrize("case", ["random", "strided", "tiny", "large", "zero", "signed_zero"])
def test_matches_native_words(operands, case):
    x, gate, weight = operands
    if case == "strided":
        x = mx.stack((x, -x), axis=-1)[..., 0]
        gate = mx.stack((gate, -gate), axis=-1)[..., 0]
        weight = mx.stack((weight, -weight), axis=-1)[..., 0]
    elif case in ("tiny", "large"):
        x = x * (1e-15 if case == "tiny" else 1e15)
    elif case in ("zero", "signed_zero"):
        x = mx.broadcast_to(mx.array(-0.0 if case == "signed_zero" else 0.0), x.shape)
    assert_exact(x, gate, weight)
    assert fused._OBSERVED and fused._CALL_COUNT == 1


def test_all_finite_bfloat_gate_encodings(operands):
    x, _, weight = operands
    # Two finite ranges exclude the positive and negative inf/NaN encodings.
    bits = mx.concatenate((mx.arange(0, 0x7f80), mx.arange(0x8000, 0xff80))).astype(mx.uint16)
    bits = mx.concatenate((bits, mx.zeros((256,), dtype=mx.uint16)))
    assert bits.size == 65536
    for i in range(8):
        gate = bits[i * 8192:(i + 1) * 8192].view(mx.bfloat16).reshape(x.shape)
        assert_exact(x, gate, weight)
    assert fused._CALL_COUNT == 8


@pytest.mark.parametrize("eps", [1e-6, 1e-5, 1e-4])
def test_epsilon_is_request_value_not_hardcoded(operands, eps):
    assert_exact(*operands, eps=eps)


@pytest.mark.parametrize("case", ["off", "non_ar", "training", "prefill", "batch2",
    "heads", "width", "input_bf16", "gate_fp32", "weight_fp32", "output_fp32",
    "runtime", "eps_zero", "eps_negative", "eps_nan", "eps_inf"])
def test_unqualified_shapes_retain_stock(operands, monkeypatch, case):
    x, g, w = operands
    eps, dtype = 1e-6, mx.bfloat16
    if case == "prefill": x = mx.broadcast_to(x, (1, 2, 64, 128)); g = mx.broadcast_to(g, x.shape)
    elif case == "batch2": x = mx.broadcast_to(x, (2, 1, 64, 128)); g = mx.broadcast_to(g, x.shape)
    elif case == "heads": x, g = x[:, :, :32], g[:, :, :32]
    elif case == "width": x, g, w = x[..., :64], g[..., :64], w[:64]
    elif case == "input_bf16": x = x.astype(mx.bfloat16)
    elif case == "gate_fp32": g = g.astype(mx.float32)
    elif case == "weight_fp32": w = w.astype(mx.float32)
    elif case == "output_fp32": dtype = mx.float32
    elif case == "runtime": monkeypatch.setattr(fused, "_compatible_runtime", lambda: False)
    elif case.startswith("eps_"):
        eps = {"eps_zero": 0., "eps_negative": -1., "eps_nan": float("nan"), "eps_inf": float("inf")}[case]
    def forbidden(): raise AssertionError("unqualified request reached kernel")
    monkeypatch.setattr(fused, "_kernel", forbidden)
    def call():
        return fused.glm5_output_norm(x, g, w, eps, output_dtype=dtype,
            enabled=case != "off", training=case == "training")
    if case == "non_ar":
        assert call() is None
    else:
        with affine_moe_ar_scope(): assert call() is None
    assert fused._CALL_COUNT == 0


def test_first_launch_failure_is_logged_once_and_keeps_stock(operands, monkeypatch, caplog):
    calls = []
    def broken():
        calls.append(1)
        raise RuntimeError("controlled compiler failure")
    monkeypatch.setattr(fused, "_kernel", broken)
    with affine_moe_ar_scope():
        for _ in range(2):
            assert fused.glm5_output_norm(*operands, 1e-6, output_dtype=mx.bfloat16, enabled=True) is None
    assert calls == [1] and fused._FAILED and not fused._OBSERVED
    assert fused._CALL_COUNT == 0
    assert "retaining stock FP32 expression" in caplog.text


def test_first_active_log_is_once(operands, caplog):
    with caplog.at_level("INFO", logger=fused.__name__):
        for _ in range(2): assert_exact(*operands)
    assert caplog.text.count("GLM exact output norm active") == 1
    assert fused._CALL_COUNT == 2


def test_default_off_and_glm_only_namespace(monkeypatch):
    from vmlx_engine.glm5_decode_policy import glm5_output_norm_requested
    from vmlx_engine.prefix_cache import compute_model_cache_key
    for family in ("glm5_next", "glm5_next_text", "qwen4_exp", "qwen3_5"):
        model = SimpleNamespace(args=SimpleNamespace(model_type=family))
        monkeypatch.delenv("VMLX_GLM5_EXACT_OUTPUT_NORM", raising=False)
        assert not glm5_output_norm_requested()
        default = compute_model_cache_key(model)
        monkeypatch.setenv("VMLX_GLM5_EXACT_OUTPUT_NORM", "0")
        assert compute_model_cache_key(model) == default
        monkeypatch.setenv("VMLX_GLM5_EXACT_OUTPUT_NORM", "1")
        assert glm5_output_norm_requested()
        assert (compute_model_cache_key(model) != default) == family.startswith("glm5_next")


def test_layer_snapshots_flag_and_preserves_prefill(monkeypatch):
    from vmlx_engine.models.glm5_next import glm5_next as model
    args = SimpleNamespace(hidden_size=16, linear_num_heads=2, linear_head_dim=4,
                           linear_lower_bound=-5., linear_conv_kernel=4, rms_norm_eps=1e-6)
    monkeypatch.delenv("VMLX_GLM5_EXACT_OUTPUT_NORM", raising=False)
    default = model.KDAAttention(args)
    assert not default._exact_output_norm
    monkeypatch.setenv("VMLX_GLM5_EXACT_OUTPUT_NORM", "1")
    assert not default._exact_output_norm
    layer = model.KDAAttention(args); layer.eval()
    assert layer._exact_output_norm
    x = mx.random.normal((1, 2, 16)).astype(mx.bfloat16)
    layer._exact_output_norm = False; expected = layer(x)
    layer._exact_output_norm = True
    with affine_moe_ar_scope(): actual = layer(x)
    mx.eval(expected, actual)
    assert bool(mx.array_equal(expected, actual)) and fused._CALL_COUNT == 0
