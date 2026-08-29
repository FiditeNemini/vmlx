from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.switch_layers import SwiGLU, SwitchGLU

from vmlx_engine.metal.affine_moe_pair_decode import (
    _CONFIG_ATTR,
    _PairConfig,
    _run_pair,
    affine_moe_pair_activation,
)
from vmlx_engine.models.glm5_next.glm5_next import ClampedSwiGLU


def _quantized_switch(*, bits: int, activation: nn.Module) -> SwitchGLU:
    switch = SwitchGLU(128, 64, 8, activation=activation, bias=False)
    nn.quantize(switch, bits=bits, group_size=32, mode="affine")
    for projection in (switch.gate_proj, switch.up_proj, switch.down_proj):
        projection.scales = projection.scales.astype(mx.float16)
        projection.biases = projection.biases.astype(mx.float16)
    switch.eval()
    return switch


@pytest.mark.parametrize(
    ("bits", "activation", "clamp_limit"),
    [
        (2, SwiGLU(), None),
        (4, SwiGLU(), None),
        (2, ClampedSwiGLU(10.0), 10.0),
    ],
)
def test_affine_moe_pair_kernel_matches_stock_activation(
    bits,
    activation,
    clamp_limit,
):
    switch = _quantized_switch(bits=bits, activation=activation)
    x = (mx.random.normal((1, 1, 128)) * 0.2).astype(mx.float16)
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
    assert mx.allclose(candidate, reference, atol=3.1e-5, rtol=2e-3)


def test_affine_moe_pair_registration_owns_decode_and_falls_back_for_prefill():
    switch = _quantized_switch(bits=2, activation=SwiGLU())
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
        decode = mx.ones((1, 1, 128), dtype=mx.float16)
        decode_indices = mx.array([0, 1, 2, 3], dtype=mx.uint32).reshape(
            1, 1, 4
        )
        output, used = affine_moe_pair_activation(
            switch, decode, decode_indices
        )
        assert used is True
        assert output.shape == (1, 1, 4, 1, 64)

        prefill = mx.ones((1, 2, 128), dtype=mx.float16)
        prefill_indices = mx.zeros((1, 2, 4), dtype=mx.uint32)
        output, used = affine_moe_pair_activation(
            switch, prefill, prefill_indices
        )
        assert used is False
        assert output is None
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
