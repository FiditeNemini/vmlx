"""JANGT prefill outlier columns read the bundle's fp16 wout directly (2026-10-12).

The fused gate/up epilogue used resident fp32 copies (_wout32) and the expert-sorted projection a resident fp32
transposed copy (_wo) of every layer's wout: Naive N0.5 kept 0.66 GiB above its weights (2x its 0.35 GiB of fp16
wout). fp16 -> fp32 widening is exact, so reading fp16 and widening per element / per call must be bit-identical to
the fp32 copies, and nothing may be cached on the module."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from vmlx_engine.jangh import kernels as JK  # noqa: E402
from vmlx_engine.jangt import prefill as PF  # noqa: E402

if not JK.nax_available():
    pytest.skip("NAX prefill kernels need macOS 26.2+ on M5", allow_module_level=True)

from tests.test_jangt_expert_tiles import _inputs, _module  # noqa: E402


@pytest.mark.parametrize("dn_mode", ["jangt", "jangh"])
def test_fused_gate_up_epilogue_fp16_wout_equals_fp32_copy(dn_mode):
    m = _module(dn_mode, 3, seed=11)
    x, idx, _ = _inputs(200, seed=12)
    U = m._unit()
    g, u = U["gate"], U["up"]
    assert "_wout32" not in g and "_wout32" not in u
    flat = idx.reshape(-1); order = mx.argsort(flat); idx_s = flat[order].astype(mx.uint32); tok = order // idx.shape[1]
    from vmlx_engine.jangt import moe_kernels as MK
    xr = MK.rotate_rows(x, U["su_gu"]).astype(mx.bfloat16)[tok]
    xo = x[:, U["gu_cols"]].astype(mx.float32)[tok]
    plan = JK.expert_tile_plan(idx_s, m.E)
    a = PF.gather_gate_up_sorted_jt(xr, xo, g, u, idx_s, plan=plan)
    g32 = dict(g, wout=g["wout"].astype(mx.float32)); u32 = dict(u, wout=u["wout"].astype(mx.float32))
    b = PF.gather_gate_up_sorted_jt(xr, xo, g32, u32, idx_s, plan=plan)
    mx.eval(a, b)
    assert g["wout"].dtype == mx.float16
    assert mx.array_equal(a, b).item()


def test_expert_sorted_projection_keeps_no_fp32_copy():
    m = _module("jangt", 3, seed=13)
    x, idx, w = _inputs(64, seed=14)
    y = m._prefill(x, idx, w)
    mx.eval(y)
    assert not hasattr(m, "_wo")
    U = m._unit()
    assert all("_wout32" not in U[k] for k in ("gate", "up") if U.get(k))
