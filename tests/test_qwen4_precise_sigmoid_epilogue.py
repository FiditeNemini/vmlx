"""Q01 owning integration only; no model weights, timing or U14/U15 replay.

Numerical rows use the real RMS/consumer and GDN recurrence. Admission spies
are explicitly control-flow-only. Root runs this file on the qualified host.
"""
from contextlib import nullcontext
import importlib.metadata
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm.models.cache import ArraysCache

from vmlx_engine import qwen4_decode_policy as policy
from vmlx_engine.metal import qwen4_precise_sigmoid_epilogue as precise
from vmlx_engine.metal.affine_moe_pair_decode import affine_moe_ar_scope
from vmlx_engine.models.qwen4_exp import language


@pytest.fixture(autouse=True)
def isolated_policy(monkeypatch):
    monkeypatch.delenv("VMLX_QWEN4_PRECISE_GDN_EPILOGUE", raising=False)
    monkeypatch.setenv("VMLX_FUSED_GATED_RMSNORM", "0")
    runtime_predicate = precise._compatible_runtime
    runtime_predicate.cache_clear()
    yield
    runtime_predicate.cache_clear()


@pytest.fixture
def qualified_gpu():
    if not (mx.metal.is_available() and mx.default_device() == mx.gpu
            and precise._compatible_runtime()):
        pytest.skip("Q01 numerical integration requires Apple M5 Max / MLX0.32.2 GPU")


def words_equal(left, right):
    assert left.shape == right.shape and left.dtype == right.dtype
    word = mx.uint16 if left.dtype in (mx.float16, mx.bfloat16) else mx.uint32
    mx.eval(left, right)
    np.testing.assert_array_equal(np.asarray(left.view(word)), np.asarray(right.view(word)))


def cache_equal(left, right):
    assert type(left) is type(right) and len(left.state) == len(right.state) == 4
    for a, b in zip(left.state, right.state):
        if a is None or b is None:
            assert a is b
        else:
            words_equal(a, b)
    # ArraysCache has no invented decode offset. Compare metadata only where
    # its actual installed implementation exposes it.
    for name in ("meta_state", "offset"):
        if hasattr(left, name):
            assert getattr(left, name) == getattr(right, name)
    for name in ("lengths",):
        a, b = getattr(left, name, None), getattr(right, name, None)
        if a is not None or b is not None:
            assert a is not None and b is not None
            words_equal(a, b)
    ca = getattr(left, "prefill_checkpoint_states", {})
    cb = getattr(right, "prefill_checkpoint_states", {})
    assert ca.keys() == cb.keys()
    for key in ca:
        for a, b in zip(ca[key], cb[key]):
            words_equal(a, b)


@pytest.mark.parametrize("case", [
    "disabled", "cpu", "no_metal", "runtime", "non_ar", "norm_bf16",
    "norm_f32", "gate_bf16", "gate_f32", "out_bf16", "out_f32",
    "batch2", "sequence2", "heads", "width", "gate_shape",
])
def test_helper_refusal_never_constructs_kernel(monkeypatch, case):
    # CPU arrays and a forbidden constructor prove only admission, not math.
    with mx.stream(mx.cpu):
        shape = {"batch2": (2, 1, 48, 128), "sequence2": (1, 2, 48, 128),
                 "heads": (1, 1, 24, 128), "width": (1, 1, 48, 64)}.get(case, (1, 1, 48, 128))
        normed = mx.zeros(shape, dtype={"norm_bf16": mx.bfloat16, "norm_f32": mx.float32}.get(case, mx.float16))
        gate = mx.zeros((1, 1, 48, 127) if case == "gate_shape" else shape,
                        dtype={"gate_bf16": mx.bfloat16, "gate_f32": mx.float32}.get(case, mx.float16))
    monkeypatch.setattr(mx, "default_device", lambda: mx.cpu if case == "cpu" else mx.gpu)
    monkeypatch.setattr(mx.metal, "is_available", lambda: case != "no_metal")
    monkeypatch.setattr(precise, "_compatible_runtime", lambda: case != "runtime")
    def forbidden():
        raise AssertionError("refused input constructed a Metal kernel")
    monkeypatch.setattr(precise, "_kernel", forbidden)
    with nullcontext() if case == "non_ar" else affine_moe_ar_scope():
        result = precise.precise_sigmoid_epilogue(
            normed, gate, output_dtype={"out_bf16": mx.bfloat16, "out_f32": mx.float32}.get(case, mx.float16),
            enabled=case != "disabled")
    assert result is None


@pytest.mark.parametrize("device,version,expected", [
    ("Apple M5 Max", "0.32.2", True),
    ("Apple M5 Pro", "0.32.2", False),
    ("Apple M5", "0.32.2", False),
    ("Apple M3 Ultra", "0.32.2", False),
    ("Apple M5 Max", "0.32.3", False),
])
def test_runtime_identity_is_exact(monkeypatch, device, version, expected):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: version)
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": device})
    # Metal availability/default device belong to helper admission, above.
    assert precise._compatible_runtime() is expected


def test_runtime_identity_error_fails_closed(monkeypatch):
    def unavailable(name):
        raise importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(importlib.metadata, "version", unavailable)
    assert precise._compatible_runtime() is False


@pytest.mark.parametrize("value,expected", [(None, False), ("0", False), ("true", False),
                                          ("", False), ("2", False), ("1", True)])
def test_policy_exact_opt_in(monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv("VMLX_QWEN4_PRECISE_GDN_EPILOGUE", value)
    assert policy.precise_gdn_epilogue_requested() is expected
    assert isinstance(policy.QWEN4_PRECISE_GDN_EPILOGUE_MATH_ABI, str)
    assert policy.QWEN4_PRECISE_GDN_EPILOGUE_MATH_ABI


@pytest.mark.parametrize("family", [
    "qwen4_exp", "qwen4_exp_text", "qwen3_5", "glm5_next", "llama",
])
def test_q01_requested_math_isolates_only_flash_identity(monkeypatch, family):
    from vmlx_engine import prefix_cache as keys
    from vmlx_engine.utils import ssm_companion_disk_store as disk_module
    from vmlx_engine.utils.ssm_companion_cache import SSMCompanionCache

    flag = "VMLX_QWEN4_PRECISE_GDN_EPILOGUE"
    model = SimpleNamespace(args=SimpleNamespace(model_type=family))
    monkeypatch.setattr(keys, "runtime_cache_fingerprint", lambda: "q01-runtime-fixed")
    # Identity only: never open an env-selected legacy/global disk store.
    monkeypatch.setattr(disk_module, "get_disk_store", lambda: None)

    def identities():
        key = keys.compute_model_cache_key(model)
        namespace = keys.build_block_cache_namespace(
            model=model, model_path="/q01-identity-fixture", quant_tag="none",
            tq_native_tag="off",
        )
        companion = SSMCompanionCache(model_key=key)
        return key, namespace, companion._key([1, 2, 3], 3)

    default = identities()
    for disabled in ("0", "true"):
        monkeypatch.setenv(flag, disabled)
        assert not policy.precise_gdn_epilogue_requested()
        assert identities() == default
    monkeypatch.setenv(flag, "1")
    enabled = identities()
    is_flash = family in {"qwen4_exp", "qwen4_exp_text"}
    assert tuple(a != b for a, b in zip(default, enabled)) == (is_flash,) * 3
    assert identities() == enabled


@pytest.mark.parametrize("family", ["qwen4_exp", "qwen4_exp_text", "qwen3_5"])
@pytest.mark.parametrize("enabled", [False, True])
def test_q01_math_revision_changes_only_enabled_flash_identity(monkeypatch, family, enabled):
    from vmlx_engine import prefix_cache as keys

    monkeypatch.setattr(keys, "runtime_cache_fingerprint", lambda: "q01-runtime-fixed")
    monkeypatch.setenv("VMLX_QWEN4_PRECISE_GDN_EPILOGUE", str(int(enabled)))
    model = SimpleNamespace(args=SimpleNamespace(model_type=family))
    before = keys.compute_model_cache_key(model)
    monkeypatch.setattr(policy, "QWEN4_PRECISE_GDN_EPILOGUE_MATH_ABI", "future-q01")
    after = keys.compute_model_cache_key(model)
    assert (before != after) == (enabled and family in {"qwen4_exp", "qwen4_exp_text"})


def test_constructor_snapshots_policy(monkeypatch):
    args = language.Qwen4ExpTextArgs(hidden_size=8, linear_num_key_heads=1,
                                   linear_num_value_heads=1, linear_key_head_dim=8,
                                   linear_value_head_dim=8)
    with mx.stream(mx.cpu):
        off = language.GatedDeltaNet(args)
        monkeypatch.setenv("VMLX_QWEN4_PRECISE_GDN_EPILOGUE", "1")
        on = language.GatedDeltaNet(args)
    assert not off._precise_gdn_epilogue and on._precise_gdn_epilogue
    monkeypatch.setenv("VMLX_QWEN4_PRECISE_GDN_EPILOGUE", "0")
    assert not off._precise_gdn_epilogue and on._precise_gdn_epilogue


@pytest.mark.parametrize("strided", [False, True])
def test_fp16_norm_boundary_and_real_consumer_are_word_exact(qualified_gpu, strided):
    rng = np.random.default_rng(166101)
    shape = (1, 1, 48, 256 if strided else 128)
    x = mx.array(rng.normal(0, 0.25, shape).astype(np.float16))
    # Both zeros, half subnormals, finite extrema and ordinary sigmoid inputs.
    gate_bits = np.array([0, 0x8000, 1, 0x8001, 0x3C00, 0xBC00,
                          0x4800, 0xC800, 0x7BFF, 0xFBFF], dtype=np.uint16)
    gate = mx.array(np.resize(gate_bits.view(np.float16), shape))
    if strided:
        x, gate = x[..., ::2], gate[..., 1::2]
    norm = language.RMSNormGatedSigmoid(128, eps=1e-6)
    norm.weight = mx.array(rng.uniform(0.8, 1.2, 128).astype(np.float16))
    norm._fused_decode = False
    norm.eval()
    original_weight = mx.array(norm.weight)
    consumer = nn.Linear(6144, 64, bias=False)
    consumer.weight = mx.array(rng.normal(0, 0.002, (64, 6144)).astype(np.float16))
    precise._kernel.cache_clear()
    want = norm(x, gate)
    with affine_moe_ar_scope():
        got = norm(x, gate, precise_epilogue=True)
    words_equal(want, got)
    a, b = consumer(want.reshape(1, 1, -1)), consumer(got.reshape(1, 1, -1))
    words_equal(a, b)
    assert bool(mx.all(mx.isfinite(a)).item())
    assert got.dtype == mx.float16 and got.shape == (1, 1, 48, 128)
    words_equal(norm.weight, original_weight)
    assert norm.eps == 1e-6 and norm._precise_epilogue_graph_calls == 1
    mx.eval(got)
    assert norm._precise_epilogue_graph_calls == 1
    with affine_moe_ar_scope():
        again = norm(x, gate, precise_epilogue=True)
    words_equal(again, want)
    assert norm._precise_epilogue_graph_calls == 2
    assert precise._kernel.cache_info().misses == 1
    assert precise._kernel.cache_info().hits >= 1


def layer_case(batch=1, steps=1):
    """Bounded fixed projections; real conv, GDN update, RMS and Linear."""
    rng = np.random.default_rng(166102)
    layer = language.GatedDeltaNet(language.Qwen4ExpTextArgs(hidden_size=64))
    def half(shape, scale=0.2):
        return mx.array(rng.normal(0, scale, shape).astype(np.float16))
    class Projection(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value
        def __call__(self, inputs):
            return self.value[:inputs.shape[0], :inputs.shape[1]]
    for name, width in (("qkv", layer.conv_dim), ("z", layer.value_dim),
                        ("a", layer.num_v_heads), ("b", layer.num_v_heads)):
        setattr(layer, "in_proj_" + name, Projection(half((batch, steps, width))))
    layer.conv1d.weight = half((layer.conv_dim, 4, 1))
    layer.A_log = half((48,)).astype(mx.bfloat16)
    layer.dt_bias = half((48,)).astype(mx.bfloat16)
    layer.norm.weight = mx.array(rng.uniform(0.8, 1.2, 128).astype(np.float16))
    layer.out_proj = nn.Linear(6144, 64, bias=False)
    layer.out_proj.weight = half((64, 6144), 0.002)
    layer._fused_conv_decode = False
    layer._unified_gdn_verify = False
    layer.norm._fused_decode = False
    layer.eval()
    initial = [half((batch, 3, layer.conv_dim)),
               mx.array(rng.normal(0, 0.025, (batch, 48, 128, 128)).astype(np.float32)),
               mx.array(np.tile(np.array([[11, 17, 23]], dtype=np.int32), (batch, 1))),
               half((batch, 2, 8))]
    def cache():
        value = ArraysCache(size=4)
        for i, array in enumerate(initial):
            value[i] = mx.array(np.array(array, copy=True))
        mx.eval(*value.state)
        return value
    inputs = half((batch, steps, 64))
    return layer, inputs, cache, initial


def test_real_recurrence_two_connected_steps_preserve_native_and_aux(qualified_gpu):
    layer, inputs, new_cache, initial = layer_case()
    off, on = new_cache(), new_cache()
    cache_equal(off, on)
    assert all(off[i] is not on[i] for i in range(4))
    for step in range(2):
        # Change projection inputs for the connected step without replacing
        # recurrence or copying the resulting state between arms.
        if step:
            for name in ("qkv", "z", "a", "b"):
                projection = getattr(layer, "in_proj_" + name)
                projection.value = (projection.value * 0.75 + 0.01).astype(mx.float16)
        layer._precise_gdn_epilogue = False
        with affine_moe_ar_scope():
            want = layer(inputs, cache=off)
        before = layer.norm._precise_epilogue_graph_calls
        assert before == step
        layer._precise_gdn_epilogue = True
        with affine_moe_ar_scope():
            got = layer(inputs, cache=on)
        words_equal(want, got)
        cache_equal(off, on)
        assert on[0].dtype == mx.float16 and on[1].dtype == mx.float32
        assert bool(mx.all(mx.isfinite(got)).item())
        for index in (2, 3):
            words_equal(on[index], initial[index])
        assert layer.norm._precise_epilogue_graph_calls == step + 1


@pytest.mark.parametrize("case", [
    "off", "training", "norm_training", "confirmed", "checkpoints", "batch2",
    "sequence2", "mask", "mismatched_mask", "lengths", "no_cache",
    "approx_norm", "approx_conv", "norm_subclass", "non_ar",
])
def test_callsite_guards_retain_same_real_stock_path(qualified_gpu, case):
    batch = 2 if case == "batch2" else 1
    steps = 3 if case == "checkpoints" else 2 if case == "sequence2" else 1
    layer, inputs, new_cache, _ = layer_case(batch, steps)
    if case == "training":
        layer.train()
    elif case == "norm_training":
        layer.norm.train()
    elif case == "approx_norm":
        layer.norm._fused_decode = True
    elif case == "approx_conv":
        layer._fused_conv_decode = True
    elif case == "norm_subclass":
        class LegacyNorm(language.RMSNormGatedSigmoid):
            def __call__(self, x, gate):
                return super().__call__(x, gate)
        norm = LegacyNorm(128, eps=layer.norm.eps)
        norm.weight = layer.norm.weight
        norm._fused_decode = False
        norm.eval()
        layer.norm = norm
    kwargs = {}
    if case == "confirmed":
        kwargs["n_confirmed"] = 1
    elif case == "checkpoints":
        kwargs["prefill_checkpoint_steps"] = (1, 2)
    elif case in ("mask", "mismatched_mask"):
        kwargs["mask"] = mx.ones((2 if case == "mismatched_mask" else batch, steps), dtype=mx.bool_)
    caches, outputs = [], []
    for enabled in (False, case != "off"):
        cache = None if case == "no_cache" else new_cache()
        if case == "lengths":
            cache.lengths = mx.array([steps] * batch)
        layer._precise_gdn_epilogue = enabled
        with nullcontext() if case == "non_ar" else affine_moe_ar_scope():
            outputs.append(layer(inputs, cache=cache, **kwargs))
        caches.append(cache)
    words_equal(*outputs)
    if caches[0] is not None:
        cache_equal(*caches)
    assert layer.norm._precise_epilogue_graph_calls == 0


def test_invalid_checkpoint_keeps_error_and_cache(qualified_gpu):
    layer, inputs, new_cache, _ = layer_case()
    for enabled in (False, True):
        cache, original = new_cache(), new_cache()
        layer._precise_gdn_epilogue = enabled
        with affine_moe_ar_scope(), pytest.raises(ValueError, match="invalid Qwen4 prefill checkpoint"):
            layer(inputs, cache=cache, prefill_checkpoint_steps=(1,))
        cache_equal(cache, original)
    assert layer.norm._precise_epilogue_graph_calls == 0


def test_helper_is_lazy_and_propagates_launch_error(qualified_gpu, monkeypatch):
    normed = mx.ones((1, 1, 48, 128), dtype=mx.float16)
    gate = mx.zeros_like(normed)
    def no_eval(*args):
        raise AssertionError("helper must not evaluate")
    monkeypatch.setattr(mx, "eval", no_eval)
    with affine_moe_ar_scope():
        got = precise.precise_sigmoid_epilogue(normed, gate, output_dtype=mx.float16, enabled=True)
    assert got is not None  # Graph only; deliberate no numerical claim.
    def broken():
        raise RuntimeError("controlled Q01 launch failure")
    monkeypatch.setattr(precise, "_kernel", broken)
    with pytest.raises(RuntimeError, match="controlled Q01 launch failure"):
        with affine_moe_ar_scope():
            precise.precise_sigmoid_epilogue(normed, gate, output_dtype=mx.float16, enabled=True)
    # Scope must unwind on the propagated exception, with no fallback retry.
    assert precise.precise_sigmoid_epilogue(normed, gate, output_dtype=mx.float16, enabled=True) is None


def test_failure_after_cache_advance_is_not_retried(qualified_gpu, monkeypatch):
    layer, inputs, new_cache, _ = layer_case()
    cache = new_cache()
    old_conv, old_state = cache[0], cache[1]
    calls = []
    original = layer._process_chunk
    def process(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    def broken():
        raise RuntimeError("controlled post-state epilogue failure")
    monkeypatch.setattr(layer, "_process_chunk", process)
    monkeypatch.setattr(precise, "_kernel", broken)
    layer._precise_gdn_epilogue = True
    with affine_moe_ar_scope(), pytest.raises(RuntimeError, match="controlled post-state epilogue failure"):
        layer(inputs, cache=cache)
    assert calls == [1]
    assert cache[0] is not old_conv and cache[1] is not old_state
    assert layer.norm._precise_epilogue_graph_calls == 0


def test_sharded_owner_returns_before_candidate(monkeypatch):
    # An inherited-call spy proves only routing; no distributed numerical claim.
    args = language.Qwen4ExpTextArgs(hidden_size=8, linear_num_key_heads=1,
                                   linear_num_value_heads=1, linear_key_head_dim=8,
                                   linear_value_head_dim=8)
    with mx.stream(mx.cpu):
        layer = language.GatedDeltaNet(args)
        inputs = mx.zeros((1, 1, 8), dtype=mx.float16)
    seen = []
    def inherited(self, x, **kwargs):
        seen.append(kwargs)
        return x
    monkeypatch.setattr(language._Qwen35GatedDeltaNet, "__call__", inherited)
    layer.sharding_group = object()
    layer._precise_gdn_epilogue = True
    with affine_moe_ar_scope():
        assert layer(inputs) is inputs
    assert seen == [{"mask": None, "cache": None}]
    assert layer.norm._precise_epilogue_graph_calls == 0
