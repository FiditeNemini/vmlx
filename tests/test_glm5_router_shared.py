"""GLM mixed router/shared projection numerical and admission contracts."""
import mlx.core as mx
import mlx.nn as nn
import pytest

from vmlx_engine.metal import glm5_router_shared as fused
from vmlx_engine.metal.affine_moe_pair_decode import affine_moe_ar_scope
from vmlx_engine.metal.quantized_projection_group import QuantizedProjectionGroup


@pytest.fixture
def operands():
    if not mx.metal.is_available() or not fused._compatible_runtime():
        pytest.skip("requires sole-owner M5 with MLX0.32.2")
    mx.random.seed(2917)
    x = mx.random.normal((1, 1, 4096)).astype(mx.bfloat16)
    router = mx.random.normal((288, 4096)).astype(mx.bfloat16)
    gate = nn.QuantizedLinear(4096, 2048, bias=False, group_size=64, bits=8)
    up = nn.QuantizedLinear(4096, 2048, bias=False, group_size=64, bits=8)
    # Match hydrated storage, not F16 safetensors header storage.
    gate.set_dtype(mx.bfloat16)
    up.set_dtype(mx.bfloat16)
    group = QuantizedProjectionGroup((gate, up))
    mx.eval(x, router, group.weight, group.scales, group.biases)
    return x, router, group


def words_equal(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    dtype = mx.uint32 if actual.dtype == mx.float32 else mx.uint16
    assert bool(mx.array_equal(actual.view(dtype), expected.view(dtype)).item())


@pytest.mark.parametrize("case", ["random", "tiny", "large", "signed_zero", "strided"])
def test_all_outputs_word_exact(operands, case):
    x, router, group = operands
    if case == "tiny":
        x = (x * 0.0001).astype(mx.bfloat16)
    elif case == "large":
        x = (x * 1000).astype(mx.bfloat16)
    elif case == "signed_zero":
        x = mx.broadcast_to(mx.array([-0.0], dtype=mx.bfloat16), x.shape)
    elif case == "strided":
        x = mx.stack((x, -x), axis=-1)[..., 0]
        router = mx.stack((router, -router), axis=-1)[..., 0]
    expected = (x.astype(mx.float32) @ router.astype(mx.float32).T, *group(x))
    before = fused._GRAPH_CALLS
    with affine_moe_ar_scope():
        for _ in range(2):
            actual = fused.try_glm5_router_shared(x, router, group, enabled=True)
            assert actual is not None
            mx.eval(*actual, *expected)
            for got, wanted in zip(actual, expected):
                words_equal(got, wanted)
    assert fused._GRAPH_CALLS - before == 2


@pytest.mark.parametrize("case", ["off", "outside_ar", "training", "batch", "prefill",
                                 "activation_f32", "router_f32", "unprepared", "q4",
                                 "g32", "format", "split", "coeff_f16", "runtime"])
def test_decline_before_kernel(operands, monkeypatch, case):
    x, router, group = operands
    if case == "batch": x = mx.broadcast_to(x, (2, 1, 4096))
    elif case == "prefill": x = mx.broadcast_to(x, (1, 2, 4096))
    elif case == "activation_f32": x = x.astype(mx.float32)
    elif case == "router_f32": router = router.astype(mx.float32)
    elif case == "unprepared": group = None
    elif case == "q4": group.bits = 4
    elif case == "g32": group.group_size = 32
    elif case == "format": group.mode = "mxfp8"
    elif case == "split": group.split_indices = (1024,)
    elif case == "coeff_f16": group.scales = group.scales.astype(mx.float16)
    elif case == "runtime": monkeypatch.setattr(fused, "_compatible_runtime", lambda: False)
    def forbidden():
        raise AssertionError("declined metadata reached kernel")
    monkeypatch.setattr(fused, "_kernel", forbidden)
    def attempt():
        return fused.try_glm5_router_shared(x, router, group, enabled=case != "off",
                                           training=case == "training")
    before = fused._GRAPH_CALLS
    if case == "outside_ar":
        assert attempt() is None
    else:
        with affine_moe_ar_scope():
            assert attempt() is None
    assert fused._GRAPH_CALLS == before


def test_failure_propagates_without_replay(operands, monkeypatch):
    def broken():
        raise RuntimeError("controlled compile failure")
    monkeypatch.setattr(fused, "_kernel", broken)
    with affine_moe_ar_scope(), pytest.raises(RuntimeError, match="controlled compile"):
        fused.try_glm5_router_shared(*operands, enabled=True)


def test_default_off(monkeypatch):
    monkeypatch.delenv("VMLX_GLM5_ROUTER_SHARED", raising=False)
    assert not fused.glm5_router_shared_requested()
    monkeypatch.setenv("VMLX_GLM5_ROUTER_SHARED", "1")
    assert fused.glm5_router_shared_requested()


def test_health_exposes_graph_construction_not_gpu_completion(monkeypatch):
    from vmlx_engine import server

    monkeypatch.setattr(server, "_read_bundle_json", lambda *args: {"model_type": "glm5_next"})
    monkeypatch.setattr(server, "_loaded_acceleration_attestation", lambda: None)
    monkeypatch.setattr(fused, "_GRAPH_CALLS", 42)
    monkeypatch.setenv("VMLX_GLM5_ROUTER_SHARED", "1")
    contract = server._family_acceleration_contract(None)
    feature = next(row for row in contract["features"] if row["id"] == "affine_moe_pair")
    assert feature["runtime"]["router_shared_projection"] == {
        "graph_calls": 42, "requested": True,
    }
