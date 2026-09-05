"""Item B opt-in gates: grouped GDN projections and compiled hyper-connections
at MTP verify widths (S=2/3/4) must be bitwise identical to the stock path."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from vmlx_engine.metal.quantized_projection_group import (
    QuantizedProjectionGroup,
    quantized_projection_group_reason,
)
from vmlx_engine.models.qwen4_exp import language as L


def _qlinears(in_dims, outs, bits, group_size, dtype):
    mx.random.seed(3)
    linears = []
    for o in outs:
        lin = nn.Linear(in_dims, o, bias=False)
        lin.weight = (mx.random.normal((o, in_dims)) * 0.05).astype(dtype)
        q = nn.QuantizedLinear.from_linear(lin, group_size=group_size, bits=bits)
        linears.append(q)
    mx.eval(*[l.parameters() for l in linears])
    return tuple(linears)


@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_grouped_projection_bitwise_at_verify_widths(bits, dtype, rows):
    # 4S/4M GDN geometry class: four same-input projections (qkv, z, b, a).
    linears = _qlinears(512, (1536, 768, 32, 32), bits, 64, dtype)
    assert quantized_projection_group_reason(linears, activation_dtype=dtype) is None
    group = QuantizedProjectionGroup(linears)
    mx.random.seed(11)
    x = (mx.random.normal((1, rows, 512)) * 0.5).astype(dtype)
    got = group(x)
    want = tuple(l(x) for l in linears)
    for g, w in zip(got, want):
        assert g.dtype == w.dtype and g.shape == w.shape
        assert bool(mx.array_equal(g, w)), f"rows={rows} bits={bits} {dtype}"


def test_gdn_group_gate_defaults_to_single_row(monkeypatch):
    monkeypatch.delenv("VMLX_QWEN4_GDN_GROUP_MAX_ROWS", raising=False)
    linears = _qlinears(256, (256, 128, 16, 16), 4, 64, mx.float16)
    x = (mx.random.normal((1, 2, 256))).astype(mx.float16)
    assert L._decode_quantized_linears_fused(linears, x) is None
    x1 = x[:, :1]
    assert L._decode_quantized_linears_fused(linears, x1) is not None
    monkeypatch.setenv("VMLX_QWEN4_GDN_GROUP_MAX_ROWS", "4")
    got = L._decode_quantized_linears_fused(linears, x)
    assert got is not None
    for g, w in zip(got, (l(x) for l in linears)):
        assert bool(mx.array_equal(g, w))
    x5 = mx.concatenate([x, x, x], axis=1)  # 6 rows > cap
    assert L._decode_quantized_linears_fused(linears, x5) is None


def test_env_rows_parsing(monkeypatch):
    monkeypatch.setenv("VMLX_QWEN4_HC_COMPILE_MAX_ROWS", "3")
    assert L._hc_compile_max_rows() == 3
    monkeypatch.setenv("VMLX_QWEN4_HC_COMPILE_MAX_ROWS", "garbage")
    assert L._hc_compile_max_rows() == 1
    monkeypatch.setenv("VMLX_QWEN4_HC_COMPILE_MAX_ROWS", "0")
    assert L._hc_compile_max_rows() == 1


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_compiled_hyper_connection_matches_eager(rows, monkeypatch):
    args = L.Qwen4ExpTextArgs(hidden_size=64, hc_count=4, hc_lowrank=16)
    mx.random.seed(5)
    module = L.GatedResidual(args, use_combine=True) if "use_combine" in L.GatedResidual.__init__.__code__.co_varnames else L.GatedResidual(args)
    mx.eval(module.parameters())
    x = mx.random.normal((1, rows, 64 * 4)).astype(mx.float32)
    monkeypatch.delenv("VMLX_QWEN4_HC_COMPILE_MAX_ROWS", raising=False)
    eager = module(x)
    L.compile_hyper_connections(module)
    monkeypatch.setenv("VMLX_QWEN4_HC_COMPILE_MAX_ROWS", "4")
    compiled = module(x)
    mx.eval(eager, compiled)
    e = eager if isinstance(eager, mx.array) else mx.concatenate([t.reshape(-1) for t in eager if t is not None])
    c = compiled if isinstance(compiled, mx.array) else mx.concatenate([t.reshape(-1) for t in compiled if t is not None])
    assert bool(mx.allclose(e, c, atol=1e-5, rtol=1e-5)), f"rows={rows}"
