"""One-dispatch Qwen4 hyper-connection mix: close to the composed graph, row-invariant, and safely declines ineligible inputs."""
import mlx.core as mx
import mlx.nn as nn
import pytest

from vmlx_engine.metal import qwen4_hc_mix_fused as F

H, R, C = 2560, 320, 4


def _stock(combined, normed, w_up):
    """The composed MLX graph the kernel replaces (GatedResidual._forward_normed tail)."""
    mix = combined[..., :R]
    injection = combined[..., R:]
    s = nn.silu(mix / C)
    up = mx.sigmoid(s @ w_up.T).astype(normed.dtype)
    up = up.reshape(*up.shape[:-1], C, H)
    mixed = (up * normed.reshape(*normed.shape[:-1], C, H)).mean(-2).astype(normed.dtype)
    inject = (2.0 * mx.sigmoid(injection / C)).astype(normed.dtype)
    return mixed, inject


def _inputs(rows, dtype=mx.float16, seed=0):
    mx.random.seed(seed)
    normed = (mx.random.normal((1, rows, C * H)) * 0.5).astype(dtype)
    combined = (mx.random.normal((1, rows, R + C)) * 2.0).astype(dtype)
    w_up = (mx.random.normal((C * H, R)) * 0.05).astype(dtype)
    return combined, normed, w_up


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 8])
def test_matches_composed_graph_within_fp16_rounding(rows):
    combined, normed, w_up = _inputs(rows)
    got = F.hc_mix_fused(combined, normed, w_up, hc_count=C, lowrank=R, hidden=H)
    assert got is not None
    want = _stock(combined, normed, w_up)
    mx.eval(*got, *want)
    # inject_w is pure elementwise with identical rounding points -> exact
    assert mx.array_equal(got[1], want[1]).item()
    # mixed: the up-GEMV reduction order differs from MLX's GEMV -> a few fp16 ULP at most
    assert float(mx.max(mx.abs(got[0].astype(mx.float32) - want[0].astype(mx.float32))).item()) < 4e-3


def test_rows_are_independent_of_how_many_rows_share_the_call():
    combined, normed, w_up = _inputs(4, seed=3)
    many = F.hc_mix_fused(combined, normed, w_up, hc_count=C, lowrank=R, hidden=H)
    mx.eval(*many)
    for r in range(4):
        one = F.hc_mix_fused(combined[:, r:r + 1], normed[:, r:r + 1], w_up, hc_count=C, lowrank=R, hidden=H)
        mx.eval(*one)
        assert mx.array_equal(one[0], many[0][:, r:r + 1]).item()
        assert mx.array_equal(one[1], many[1][:, r:r + 1]).item()


def test_bfloat16_supported():
    combined, normed, w_up = _inputs(2, dtype=mx.bfloat16)
    got = F.hc_mix_fused(combined, normed, w_up, hc_count=C, lowrank=R, hidden=H)
    want = _stock(combined, normed, w_up)
    mx.eval(*got, *want)
    assert float(mx.max(mx.abs(got[0].astype(mx.float32) - want[0].astype(mx.float32))).item()) < 3e-2


@pytest.mark.parametrize("case", ["too_many_rows", "dtype_mismatch", "bad_geometry"])
def test_declines_ineligible_inputs(case):
    combined, normed, w_up = _inputs(9 if case == "too_many_rows" else 1)
    kwargs = dict(hc_count=C, lowrank=R, hidden=H)
    if case == "dtype_mismatch":
        w_up = w_up.astype(mx.float32)
    if case == "bad_geometry":
        kwargs["hc_count"] = 3
    assert F.hc_mix_fused(combined, normed, w_up, **kwargs) is None


def test_default_is_on_and_env_opts_out(monkeypatch):
    monkeypatch.delenv("VMLX_QWEN4_HC_FUSED_MIX", raising=False)
    assert F.hc_fused_mix_requested() is True
    monkeypatch.setenv("VMLX_QWEN4_HC_FUSED_MIX", "0")
    assert F.hc_fused_mix_requested() is False
