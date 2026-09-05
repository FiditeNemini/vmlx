from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.switch_layers import SwiGLU, SwitchGLU

from vmlx_engine.metal.affine_moe_pair_decode import (
    _CONFIG_ATTR,
    _FAMILY_CONTRACTS,
    _PairConfig,
    _projection_reason,
    _requested,
    _run_pair,
    _run_weighted_down,
    affine_moe_pair_activation,
    affine_moe_pair_status,
    affine_moe_routed_output,
    install_affine_moe_pair_decode,
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


def test_mixed_layout_registration_keeps_entire_stack_on_stock_path(
    monkeypatch,
):
    compatible = _quantized_switch(bits=2, activation=SwiGLU())
    fallback = _quantized_switch(bits=3, activation=SwiGLU())

    class _MixedModel:
        def named_modules(self):
            return [("compatible", compatible), ("fallback", fallback)]

    monkeypatch.setitem(
        _FAMILY_CONTRACTS,
        "qwen4_exp",
        {
            "hidden": 128,
            "intermediate": 64,
            "top_k": 4,
            "layouts": {(2, 32), (4, 32)},
            "clamp_limit": None,
        },
    )
    monkeypatch.delenv("VMLX_QWEN4_FUSED_MOE_PAIR", raising=False)

    installed = install_affine_moe_pair_decode(
        _MixedModel(), family="qwen4_exp"
    )
    status = affine_moe_pair_status("qwen4_exp")

    assert installed == 0
    assert not hasattr(compatible, _CONFIG_ATTR)
    assert not hasattr(fallback, _CONFIG_ATTR)
    assert status["eligible_modules"] == 1
    assert status["fallback_modules"] == 1
    assert status["total_modules"] == 2
    assert status["reason"] == "mixed_layout_atomic_fallback"
    assert status["layouts"] == []


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
    mx.random.seed(2026)
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


def test_switch_config_derives_intermediate_from_sharded_projections(
    monkeypatch,
):
    """An OUT-sharded export halves gate/up output_dims; the registration
    contract is the cross-projection invariant, not a hardcoded width."""

    monkeypatch.setitem(
        _FAMILY_CONTRACTS,
        "glm5_next",
        {
            "hidden": 128,
            "top_k": 4,
            "layouts": {(2, 32)},
            "clamp_limit": 10.0,
        },
    )
    monkeypatch.setenv("VMLX_GLM5_FUSED_MOE_PAIR", "1")
    switch = SwitchGLU(128, 32, 8, activation=ClampedSwiGLU(10.0), bias=False)
    nn.quantize(switch, bits=2, group_size=32, mode="affine")
    for projection in (switch.gate_proj, switch.up_proj, switch.down_proj):
        projection.scales = projection.scales.astype(mx.float16)
        projection.biases = projection.biases.astype(mx.float16)
    switch.eval()

    class _Model:
        def named_modules(self):
            return [("layers.0.mlp.switch_mlp", switch)]

    installed = install_affine_moe_pair_decode(_Model(), family="glm5_next")
    assert installed == 1
    config = getattr(switch, _CONFIG_ATTR)
    try:
        assert config.hidden == 128
        assert config.intermediate == 32
        x = (mx.random.normal((1, 1, 128)) * 0.2).astype(mx.float16)
        indices = mx.array([0, 2, 5, 7], dtype=mx.uint32).reshape(1, 1, 4)
        output, used = affine_moe_pair_activation(switch, x, indices)
        assert used is True
        assert output.shape == (1, 1, 4, 1, 32)
        expanded = mx.expand_dims(x, (-2, -3))
        reference = switch.activation(
            switch.up_proj(expanded, indices),
            switch.gate_proj(expanded, indices),
        )
        mx.eval(output, reference)
        assert mx.allclose(output, reference, atol=2.5e-4, rtol=6e-3)
    finally:
        delattr(switch, _CONFIG_ATTR)


def test_switch_config_rejects_cross_projection_geometry_mismatch(
    monkeypatch,
):
    monkeypatch.setitem(
        _FAMILY_CONTRACTS,
        "glm5_next",
        {
            "hidden": 128,
            "top_k": 4,
            "layouts": {(2, 32)},
            "clamp_limit": 10.0,
        },
    )
    monkeypatch.setenv("VMLX_GLM5_FUSED_MOE_PAIR", "1")
    switch = SwitchGLU(128, 32, 8, activation=ClampedSwiGLU(10.0), bias=False)
    other = SwitchGLU(128, 64, 8, activation=ClampedSwiGLU(10.0), bias=False)
    nn.quantize(switch, bits=2, group_size=32, mode="affine")
    nn.quantize(other, bits=2, group_size=32, mode="affine")
    switch.up_proj = other.up_proj
    for projection in (switch.gate_proj, switch.up_proj, switch.down_proj):
        projection.scales = projection.scales.astype(mx.float16)
        projection.biases = projection.biases.astype(mx.float16)
    switch.eval()

    class _Model:
        def named_modules(self):
            return [("layers.0.mlp.switch_mlp", switch)]

    installed = install_affine_moe_pair_decode(_Model(), family="glm5_next")
    assert installed == 0
    assert not hasattr(switch, _CONFIG_ATTR)
    status = affine_moe_pair_status("glm5_next")
    assert any(
        "intermediate geometry mismatch" in reason
        for reason in status["fallback_reasons"]
    )


# ---- mixed-layout / 3-6 bit pair kernel (Flash-Next 4S: q2 gate + q3 up; 6S: q4/q6) ----

def _mixed_switch(gate_bits, gate_gs, up_bits, up_gs, down_bits=4, metadata_dtype=mx.float16):
    from mlx_lm.models.switch_layers import SwitchLinear

    mx.random.seed(77)
    switch = SwitchGLU(128, 64, 8, activation=SwiGLU(), bias=False)
    switch.gate_proj = switch.gate_proj.to_quantized(group_size=gate_gs, bits=gate_bits)
    switch.up_proj = switch.up_proj.to_quantized(group_size=up_gs, bits=up_bits)
    switch.down_proj = switch.down_proj.to_quantized(group_size=32, bits=down_bits)
    for projection in (switch.gate_proj, switch.up_proj, switch.down_proj):
        projection.scales = projection.scales.astype(metadata_dtype)
        projection.biases = projection.biases.astype(metadata_dtype)
    switch.eval()
    mx.eval(switch.parameters())
    return switch


@pytest.mark.parametrize(
    ("gate_bits", "gate_gs", "up_bits", "up_gs", "activation_dtype"),
    [
        (2, 64, 3, 64, mx.float16),   # 4S majority layer
        (3, 64, 3, 64, mx.float16),   # 4S q3 gate layers
        (2, 32, 3, 64, mx.float16),   # 4S single q2/g32 gate layer
        (4, 64, 6, 64, mx.float16),   # 6S layers with q6 up
        (6, 64, 6, 64, mx.bfloat16),
        (8, 64, 2, 64, mx.float16),
        (4, 64, 4, 64, mx.float16),   # uniform layout must still match through the generic path
    ],
)
def test_mixed_layout_pair_kernel_matches_stock(gate_bits, gate_gs, up_bits, up_gs, activation_dtype):
    switch = _mixed_switch(gate_bits, gate_gs, up_bits, up_gs)
    x = (mx.random.normal((1, 1, 128)) * 0.2).astype(activation_dtype)
    indices = mx.array([0, 2, 5, 7], dtype=mx.uint32).reshape(1, 1, 4)
    expanded = mx.expand_dims(x, (-2, -3))
    reference = switch.activation(
        switch.up_proj(expanded, indices), switch.gate_proj(expanded, indices)
    )
    config = _PairConfig(
        family=f"test_mixed_g{gate_bits}_{gate_gs}_u{up_bits}_{up_gs}",
        hidden=128, intermediate=64, top_k=4,
        bits=gate_bits, group_size=gate_gs, clamp_limit=None,
        up_bits=up_bits if (up_bits, up_gs) != (gate_bits, gate_gs) else None,
        up_group_size=up_gs if (up_bits, up_gs) != (gate_bits, gate_gs) else None,
    )
    if (gate_bits, gate_gs) == (up_bits, up_gs) and gate_bits not in (3, 6):
        # force the generic path for the uniform control case
        config = _PairConfig(**{**config.__dict__, "up_bits": up_bits, "up_group_size": up_gs, "bits": gate_bits})
        object.__setattr__(config, "up_group_size", up_gs)
    candidate = _run_pair(switch, config, x, indices)
    mx.eval(reference, candidate)
    assert candidate.shape == reference.shape
    uses_bf16 = activation_dtype == mx.bfloat16
    atol = 2.5e-4 if uses_bf16 else 3.1e-5
    rtol = 6e-3 if uses_bf16 else 2e-3
    assert mx.allclose(candidate, reference, atol=atol, rtol=rtol), (
        f"max abs {float(mx.abs(candidate - reference).max()):.3e}"
    )


def test_mixed_layout_registration_is_opt_in(monkeypatch):
    mixed = _mixed_switch(2, 64, 3, 64)
    uniform = _mixed_switch(2, 64, 2, 64)

    class _Model:
        def named_modules(self):
            return [("a", mixed), ("b", uniform)]

    monkeypatch.setitem(
        _FAMILY_CONTRACTS, "qwen4_exp",
        {"hidden": 128, "intermediate": 64, "top_k": 4,
         "layouts": {(2, 64), (4, 64)},
         "mixed_layouts": {(2, 32), (2, 64), (3, 64), (4, 64), (6, 64), (8, 64)},
         "clamp_limit": None},
    )
    monkeypatch.delenv("VMLX_QWEN4_FUSED_MOE_PAIR", raising=False)
    monkeypatch.delenv("VMLX_QWEN4_FUSED_MOE_PAIR_MIXED", raising=False)
    assert install_affine_moe_pair_decode(_Model(), family="qwen4_exp") == 0
    assert affine_moe_pair_status("qwen4_exp")["reason"] == "mixed_layout_atomic_fallback"

    monkeypatch.setenv("VMLX_QWEN4_FUSED_MOE_PAIR_MIXED", "1")
    assert install_affine_moe_pair_decode(_Model(), family="qwen4_exp") == 2
    status = affine_moe_pair_status("qwen4_exp")
    assert status["installed"] == 2 and status["fallback_modules"] == 0
    assert (3, 64) in status["layouts"] and (2, 64) in status["layouts"]
    cfg = getattr(mixed, _CONFIG_ATTR)
    assert cfg.mixed_layout and cfg.up_bits_eff == 3 and cfg.bits == 2
    x = (mx.random.normal((1, 1, 128)) * 0.2).astype(mx.float16)
    idx = mx.array([1, 3, 4, 6], dtype=mx.uint32).reshape(1, 1, 4)
    out, fused = affine_moe_pair_activation(mixed, x, idx)
    assert fused and out is not None
    ref = mixed.activation(mixed.up_proj(mx.expand_dims(x, (-2, -3)), idx), mixed.gate_proj(mx.expand_dims(x, (-2, -3)), idx))
    mx.eval(out, ref)
    assert mx.allclose(out, ref, atol=3.1e-5, rtol=2e-3)
