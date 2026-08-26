"""Tests for the fused pair-SwiGLU routed-MoE decode fastpath.

Uses real DSV4 projection shapes (gate/up b2gs64 (E,2048,256), down b2gs32
(E,4096,128)) with a small expert count so the Metal kernels execute for real
against the stock gather_qmm op chain.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from vmlx_engine.metal import fused_pair_moe_decode as fpm

D = 4096
M = 2048
K = 6
E = 8
LIMIT = 10.0


class _Args:
    num_experts_per_tok = K
    swiglu_limit = LIMIT


class _Proj(nn.Module):
    def __init__(self, out_dim, in_dim, group_size):
        super().__init__()
        w = (mx.random.normal((E, out_dim, in_dim)) * 0.02).astype(mx.float16)
        self.weight, self.scales, self.biases = mx.quantize(
            w, group_size=group_size, bits=2
        )
        self.bits = 2
        self.group_size = group_size
        self.mode = "affine"


class _Switch(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = _Proj(M, D, 64)
        self.up_proj = _Proj(M, D, 64)
        self.down_proj = _Proj(D, M, 32)


def _swiglu_fp32(gate, up, limit):
    gate = gate.astype(mx.float32)
    up = up.astype(mx.float32)
    up = mx.clip(up, a_min=-limit, a_max=limit)
    gate = mx.clip(gate, a_min=None, a_max=limit)
    return nn.silu(gate) * up


def _make_moe_cls():
    class _MoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.args = _Args()
            self.switch_mlp = _Switch()

        def _weighted_routed_experts(self, x, inds, scores):
            s = self.switch_mlp
            xe = mx.expand_dims(mx.expand_dims(x, -2), -3)
            xg = mx.gather_qmm(
                xe, s.gate_proj.weight, s.gate_proj.scales, s.gate_proj.biases,
                rhs_indices=inds, transpose=True, group_size=64, bits=2,
                mode="affine", sorted_indices=False,
            )
            xu = mx.gather_qmm(
                xe, s.up_proj.weight, s.up_proj.scales, s.up_proj.biases,
                rhs_indices=inds, transpose=True, group_size=64, bits=2,
                mode="affine", sorted_indices=False,
            )
            act = _swiglu_fp32(xg, xu, LIMIT) * scores[..., None, None]
            act = act.astype(x.dtype)
            dn = mx.gather_qmm(
                act, s.down_proj.weight, s.down_proj.scales, s.down_proj.biases,
                rhs_indices=inds, transpose=True, group_size=32, bits=2,
                mode="affine", sorted_indices=False,
            )
            return dn.squeeze(-2)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = _MoE()

    return _MoE, _Model


def _decode_inputs():
    mx.random.seed(11)
    x = mx.random.normal((1, 1, D)).astype(mx.float16)
    inds = mx.array([0, 3, 1, 7, 5, 2], dtype=mx.uint32).reshape(1, 1, K)
    scores = mx.random.uniform(shape=(1, 1, K)).astype(mx.float32)
    scores = scores / scores.sum()
    return x, inds, scores


def test_installs_and_matches_stock(monkeypatch):
    monkeypatch.delenv("VMLX_DSV4_FUSED_MOE_PAIR", raising=False)
    moe_cls, model_cls = _make_moe_cls()
    stock = moe_cls._weighted_routed_experts
    model = model_cls()
    assert fpm.install_dsv4_fused_pair_moe(model) == 1
    status = fpm.dsv4_fused_pair_moe_status()
    assert status["installed"] == 1
    assert status["reason"] is None
    assert status["self_test_rel"] <= fpm._SELF_TEST_MAX_REL

    x, inds, scores = _decode_inputs()
    ref = stock(model.mlp, x, inds, scores).astype(mx.float32)
    got = moe_cls._weighted_routed_experts(model.mlp, x, inds, scores).astype(
        mx.float32
    )
    mx.eval(ref, got)
    rel = float(mx.abs(got - ref).max()) / max(float(mx.abs(ref).max()), 1e-9)
    assert got.shape == ref.shape == (1, 1, K, D)
    assert rel < 5e-3  # f16 reassociation class

    # re-install is idempotent: original stays the true stock method
    assert fpm.install_dsv4_fused_pair_moe(model) == 1
    assert getattr(moe_cls, fpm._ORIGINAL_ATTR) is stock


def test_prefill_shape_takes_stock_path(monkeypatch):
    monkeypatch.delenv("VMLX_DSV4_FUSED_MOE_PAIR", raising=False)
    moe_cls, model_cls = _make_moe_cls()
    stock = moe_cls._weighted_routed_experts
    model = model_cls()
    assert fpm.install_dsv4_fused_pair_moe(model) == 1
    mx.random.seed(12)
    x = mx.random.normal((1, 4, D)).astype(mx.float16)
    inds = mx.array([[[0, 3, 1, 7, 5, 2]] * 4], dtype=mx.uint32)
    scores = mx.random.uniform(shape=(1, 4, K)).astype(mx.float32)
    ref = stock(model.mlp, x, inds, scores)
    got = moe_cls._weighted_routed_experts(model.mlp, x, inds, scores)
    mx.eval(ref, got)
    assert bool((ref == got).all())  # exact: same code path


def test_env_off_disables(monkeypatch):
    monkeypatch.setenv("VMLX_DSV4_FUSED_MOE_PAIR", "0")
    moe_cls, model_cls = _make_moe_cls()
    stock = moe_cls._weighted_routed_experts
    model = model_cls()
    assert fpm.install_dsv4_fused_pair_moe(model) == 0
    assert moe_cls._weighted_routed_experts is stock
    assert fpm.dsv4_fused_pair_moe_status()["reason"] == "disabled via env"


def test_wrong_layout_rejected(monkeypatch):
    monkeypatch.delenv("VMLX_DSV4_FUSED_MOE_PAIR", raising=False)
    moe_cls, model_cls = _make_moe_cls()
    stock = moe_cls._weighted_routed_experts
    model = model_cls()
    model.mlp.switch_mlp.down_proj.group_size = 64  # not the validated layout
    assert fpm.install_dsv4_fused_pair_moe(model) == 0
    assert moe_cls._weighted_routed_experts is stock


def test_threadgroup_width_is_validated_and_reported(monkeypatch):
    monkeypatch.delenv("VMLX_DSV4_FUSED_MOE_PAIR", raising=False)
    monkeypatch.setenv("VMLX_DSV4_FUSED_MOE_THREADGROUP", "256")
    moe_cls, model_cls = _make_moe_cls()
    stock = moe_cls._weighted_routed_experts
    model = model_cls()
    assert fpm.install_dsv4_fused_pair_moe(model) == 1
    assert fpm.dsv4_fused_pair_moe_status()["threadgroup_width"] == 256

    x, inds, scores = _decode_inputs()
    ref = stock(model.mlp, x, inds, scores).astype(mx.float32)
    got = moe_cls._weighted_routed_experts(model.mlp, x, inds, scores).astype(
        mx.float32
    )
    mx.eval(ref, got)
    rel = float(mx.abs(got - ref).max()) / max(float(mx.abs(ref).max()), 1e-9)
    assert rel < 5e-3


def test_invalid_threadgroup_width_fails_closed(monkeypatch):
    monkeypatch.delenv("VMLX_DSV4_FUSED_MOE_PAIR", raising=False)
    monkeypatch.setenv("VMLX_DSV4_FUSED_MOE_THREADGROUP", "48")
    moe_cls, model_cls = _make_moe_cls()
    stock = moe_cls._weighted_routed_experts
    model = model_cls()
    assert fpm.install_dsv4_fused_pair_moe(model) == 0
    assert moe_cls._weighted_routed_experts is stock
    assert "not in" in fpm.dsv4_fused_pair_moe_status()["reason"]
