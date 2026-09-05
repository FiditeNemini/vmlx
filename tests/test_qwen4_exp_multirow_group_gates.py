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


def test_gdn_group_gate_defaults_to_verify_width_and_can_be_narrowed(monkeypatch):
    monkeypatch.delenv("VMLX_QWEN4_GDN_GROUP_MAX_ROWS", raising=False)
    linears = _qlinears(256, (256, 128, 16, 16), 4, 64, mx.float16)
    x = (mx.random.normal((1, 4, 256))).astype(mx.float16)
    got = L._decode_quantized_linears_fused(linears, x)  # default: verify width admitted
    assert got is not None
    monkeypatch.setenv("VMLX_QWEN4_GDN_GROUP_MAX_ROWS", "1")
    assert L._decode_quantized_linears_fused(linears, x) is None
    assert L._decode_quantized_linears_fused(linears, x[:, :1]) is not None
    monkeypatch.setenv("VMLX_QWEN4_GDN_GROUP_MAX_ROWS", "4")
    got = L._decode_quantized_linears_fused(linears, x)
    assert got is not None
    for g, w in zip(got, (l(x) for l in linears)):
        assert bool(mx.array_equal(g, w))
    x5 = mx.concatenate([x, x[:, :1]], axis=1)  # 5 rows > verify width -> stock
    assert L._decode_quantized_linears_fused(linears, x5) is None
    monkeypatch.setenv("VMLX_QWEN4_GDN_GROUP_MAX_ROWS", "1")
    assert L._decode_quantized_linears_fused(linears, x) is None
    assert L._decode_quantized_linears_fused(linears, x[:, :1]) is not None


def test_env_rows_parsing(monkeypatch):
    monkeypatch.setenv("VMLX_QWEN4_HC_COMPILE_MAX_ROWS", "3")
    assert L._hc_compile_max_rows() == 3
    monkeypatch.setenv("VMLX_QWEN4_HC_COMPILE_MAX_ROWS", "garbage")
    assert L._hc_compile_max_rows() == 4
    monkeypatch.setenv("VMLX_QWEN4_HC_COMPILE_MAX_ROWS", "0")
    assert L._hc_compile_max_rows() == 1
    monkeypatch.delenv("VMLX_QWEN4_HC_COMPILE_MAX_ROWS")
    monkeypatch.delenv("VMLX_QWEN4_GDN_GROUP_MAX_ROWS", raising=False)
    assert L._hc_compile_max_rows() == 4 and L._gdn_group_max_rows() == 4


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


def test_route_overlap_probe_counts_distinct_experts():
    probe = L._RouteOverlapProbe()
    inds = mx.array([[[1, 2, 3], [2, 3, 4], [1, 2, 3], [9, 9, 9]]])  # S=4, top_k=3
    probe.observe(inds)  # call 1 sampled
    assert probe.sampled == 1 and probe.rows == 4 and probe.slots == 12
    assert probe.distinct == 5  # {1,2,3,4,9}
    for _ in range(15):
        probe.observe(inds)  # calls 2..16 skipped by the 1-in-16 sampler
    assert probe.sampled == 1
    probe.observe(inds)  # call 17 sampled again
    assert probe.sampled == 2 and probe.by_rows[4] == [10, 24]
