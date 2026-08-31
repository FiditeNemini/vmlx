from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.switch_layers import SwiGLU, SwitchGLU

from vmlx_engine.metal.affine_moe_pair_decode import (
    _CONFIG_ATTR,
    _PairConfig,
    _projection_reason,
    _requested,
    _run_pair,
    _run_weighted_down,
    affine_moe_pair_activation,
    affine_moe_routed_output,
)
from vmlx_engine.models.glm5_next.glm5_next import ClampedSwiGLU


def _quantized_switch(
    *,
    bits: int,
    activation: nn.Module,
    metadata_dtype=mx.float16,
) -> SwitchGLU:
    switch = SwitchGLU(128, 64, 8, activation=activation, bias=False)
    nn.quantize(switch, bits=bits, group_size=32, mode="affine")
    for projection in (switch.gate_proj, switch.up_proj, switch.down_proj):
        projection.scales = projection.scales.astype(metadata_dtype)
        projection.biases = projection.biases.astype(metadata_dtype)
    switch.eval()
    return switch


def test_affine_moe_pair_family_defaults_and_explicit_overrides(monkeypatch):
    monkeypatch.delenv("VMLX_QWEN4_FUSED_MOE_PAIR", raising=False)
    monkeypatch.delenv("VMLX_GLM5_FUSED_MOE_PAIR", raising=False)
    assert _requested("qwen4_exp") is True
    assert _requested("glm5_next") is False

    monkeypatch.setenv("VMLX_QWEN4_FUSED_MOE_PAIR", "0")
    monkeypatch.setenv("VMLX_GLM5_FUSED_MOE_PAIR", "1")
    assert _requested("qwen4_exp") is False
    assert _requested("glm5_next") is True


@pytest.mark.parametrize(
    (
        "bits",
        "activation",
        "clamp_limit",
        "metadata_dtype",
        "activation_dtype",
    ),
    [
        (2, SwiGLU(), None, mx.float16, mx.float16),
        (4, SwiGLU(), None, mx.float16, mx.float16),
        (2, ClampedSwiGLU(10.0), 10.0, mx.float16, mx.float16),
        (2, ClampedSwiGLU(10.0), 10.0, mx.bfloat16, mx.bfloat16),
    ],
)
def test_affine_moe_pair_kernel_matches_stock_activation(
    bits,
    activation,
    clamp_limit,
    metadata_dtype,
    activation_dtype,
):
    switch = _quantized_switch(
        bits=bits,
        activation=activation,
        metadata_dtype=metadata_dtype,
    )
    x = (mx.random.normal((1, 1, 128)) * 0.2).astype(activation_dtype)
    indices = mx.array([0, 2, 5, 7], dtype=mx.uint32).reshape(1, 1, 4)
    expanded = mx.expand_dims(x, (-2, -3))
    reference = switch.activation(
        switch.up_proj(expanded, indices),
        switch.gate_proj(expanded, indices),
    )
    config = _PairConfig(
        family=f"test_q{bits}_{'clamped' if clamp_limit else 'plain'}",
        hidden=128,
        intermediate=64,
        top_k=4,
        bits=bits,
        group_size=32,
        clamp_limit=clamp_limit,
    )
    candidate = _run_pair(switch, config, x, indices)
    mx.eval(reference, candidate)

    assert candidate.shape == reference.shape
    uses_bf16 = (
        metadata_dtype == mx.bfloat16 or activation_dtype == mx.bfloat16
    )
    atol = 2.5e-4 if uses_bf16 else 3.1e-5
    rtol = 6e-3 if uses_bf16 else 2e-3
    assert mx.allclose(candidate, reference, atol=atol, rtol=rtol)


@pytest.mark.parametrize("metadata_dtype", [mx.float16, mx.bfloat16])
def test_affine_moe_pair_registration_accepts_supported_metadata_dtypes(
    metadata_dtype,
):
    switch = _quantized_switch(
        bits=2,
        activation=ClampedSwiGLU(10.0),
        metadata_dtype=metadata_dtype,
    )
    assert (
        _projection_reason(
            switch.gate_proj,
            hidden=128,
            intermediate=64,
        )
        is None
    )


@pytest.mark.parametrize("activation_dtype", [mx.float16, mx.bfloat16])
def test_affine_moe_pair_registration_owns_decode_and_falls_back_for_prefill(
    activation_dtype,
):
    switch = _quantized_switch(
        bits=2,
        activation=SwiGLU(),
        metadata_dtype=activation_dtype,
    )
    config = _PairConfig(
        family="test_dispatch",
        hidden=128,
        intermediate=64,
        top_k=4,
        bits=2,
        group_size=32,
        clamp_limit=None,
    )
    setattr(switch, _CONFIG_ATTR, config)
    try:
        decode = mx.ones((1, 1, 128), dtype=activation_dtype)
        decode_indices = mx.array([0, 1, 2, 3], dtype=mx.uint32).reshape(
            1, 1, 4
        )
        output, used = affine_moe_pair_activation(
            switch, decode, decode_indices
        )
        assert used is True
        from vmlx_engine.metal.affine_moe_pair_decode import (
            affine_moe_pair_status,
        )

        assert affine_moe_pair_status("test_dispatch")["observed_calls"] == 1
        assert output.shape == (1, 1, 4, 1, 64)

        prefill = mx.ones((1, 2, 128), dtype=activation_dtype)
        prefill_indices = mx.zeros((1, 2, 4), dtype=mx.uint32)
        output, used = affine_moe_pair_activation(
            switch, prefill, prefill_indices
        )
        assert used is False
        assert output is None
    finally:
        delattr(switch, _CONFIG_ATTR)


@pytest.mark.parametrize("activation_dtype", [mx.float16, mx.bfloat16])
def test_affine_moe_weighted_down_matches_stock_route_reduction(
    activation_dtype,
):
    switch = _quantized_switch(
        bits=2,
        activation=ClampedSwiGLU(10.0),
        metadata_dtype=activation_dtype,
    )
    config = _PairConfig(
        family="test_full_glm",
        hidden=128,
        intermediate=64,
        top_k=4,
        bits=2,
        group_size=32,
        clamp_limit=10.0,
        fuse_down=True,
    )
    x = (mx.random.normal((1, 1, 128)) * 0.2).astype(activation_dtype)
    indices = mx.array([0, 2, 5, 7], dtype=mx.uint32).reshape(1, 1, 4)
    scores = mx.array([0.1, 0.2, 0.3, 0.4], dtype=mx.float32).reshape(
        1, 1, 4
    )
    activated = _run_pair(switch, config, x, indices)
    selected = switch.down_proj(activated, indices).squeeze(-2)
    reference = (selected * scores[..., None].astype(selected.dtype)).sum(
        axis=-2
    )
    candidate = _run_weighted_down(
        switch, config, activated, indices, scores
    )
    mx.eval(reference, candidate)
    assert candidate.shape == reference.shape
    atol = 5e-4 if activation_dtype == mx.bfloat16 else 6.2e-5
    rtol = 8e-3 if activation_dtype == mx.bfloat16 else 3e-3
    assert mx.allclose(candidate, reference, atol=atol, rtol=rtol)

    setattr(switch, _CONFIG_ATTR, config)
    try:
        routed, used = affine_moe_routed_output(
            switch, x, indices, scores
        )
        assert used is True
        mx.eval(routed)
        assert mx.allclose(routed, reference, atol=atol, rtol=rtol)
    finally:
        delattr(switch, _CONFIG_ATTR)


def test_qwen_full_affine_moe_precedes_pair_only_candidate(monkeypatch):
    from vmlx_engine.metal import qwen4_affine_moe_decode as full_moe

    switch = SimpleNamespace()
    setattr(switch, full_moe._OK_ATTR, True)
    expected = mx.ones((1, 1, 2560), dtype=mx.float16)
    monkeypatch.setattr(full_moe, "_fused", lambda *_args: expected)

    def pair_must_not_run(*_args):
        raise AssertionError("gate/up-only fusion shadowed the full MoE kernel")

    monkeypatch.setattr(full_moe, "affine_moe_pair_activation", pair_must_not_run)
    x = mx.zeros((1, 1, 2560), dtype=mx.float16)
    indices = mx.zeros((1, 1, 10), dtype=mx.uint32)
    scores = mx.zeros((1, 1, 10), dtype=mx.float16)

    output, used = full_moe.qwen4_affine_switchglu(
        switch, x, indices, scores
    )
    assert used is True
    assert output is expected
    assert full_moe.qwen4_affine_moe_status()["observed_calls"] == 1


def test_qwen_full_affine_moe_accepts_standard_and_legacy_env(monkeypatch):
    from vmlx_engine.metal import qwen4_affine_moe_decode as full_moe

    monkeypatch.delenv("VMLX_QWEN4_AFFINE_MOE", raising=False)
    monkeypatch.setenv("VMLINUX_QWEN4_AFFINE_MOE", "1")
    assert full_moe._enabled() is True
    monkeypatch.setenv("VMLX_QWEN4_AFFINE_MOE", "0")
    assert full_moe._enabled() is False
    monkeypatch.setenv("VMLX_QWEN4_AFFINE_MOE", "1")
    monkeypatch.setenv("VMLINUX_QWEN4_AFFINE_MOE", "0")
    assert full_moe._enabled() is True
