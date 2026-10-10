"""JANGT gate/up decode with the activations rotated in the epilogue (rot_out) == h32_rows of the plain kernel, and the
mixed JANGT-gate/up + JANGH-down routed output is unchanged by it (JANGT_ROT_OUT_DECODE)."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from vmlx_engine.jangh import kernels as JK  # noqa: E402
from vmlx_engine.jangt import moe_kernels as MK  # noqa: E402
from vmlx_engine.jangt import switch as SW  # noqa: E402

D, I, E = 1024, 256, 8


def _module(seed=0, limit=10.0):
    rng = np.random.default_rng(seed)
    m = SW.JTMixedSwitchGLU(D, I, E, {"mode": "jangt", "kbits": 2}, {"mode": "jangtq2", "bits": 2, "rotation": "hadamard32"},
                            0, 0, limit)
    for p in (m.gate_proj, m.up_proj):
        p.jt_packed = mx.array(rng.integers(0, 2 ** 32, size=p.jt_packed.shape, dtype=np.uint64).astype(np.uint32))
        p.jt_scale = mx.array(rng.uniform(0.02, 0.05, size=p.jt_scale.shape).astype(np.float32))
        p.jt_su = mx.array(rng.choice([-1.0, 1.0], size=p.jt_su.shape).astype(np.float32))
    m.up_proj.jt_su = m.gate_proj.jt_su
    d = m.down_proj
    d.tq2_packed = mx.array(rng.integers(0, 2 ** 32, size=d.tq2_packed.shape, dtype=np.uint64).astype(np.uint32))
    d.tq2_scales = mx.array(rng.uniform(0.01, 0.03, size=d.tq2_scales.shape).astype(np.float16))
    return m


def _inputs(T, seed=1):
    rng = np.random.default_rng(seed)
    x = mx.array((rng.standard_normal((T, D)) * 0.5).astype(np.float32)).astype(mx.bfloat16)
    idx = mx.array(np.stack([rng.permutation(E)[:4] for _ in range(T)]).astype(np.int32))
    return x, idx


@pytest.mark.parametrize("T", [1, 3])
def test_rot_out_equals_h32_of_plain(T):
    m = _module(T)
    x, idx = _inputs(T)
    U = m._unit(); xr = MK.rotate_rows(x, U["su_gu"])
    plain = MK.gather_gate_up_swiglu_v3(xr, x, idx, U, out_dtype=mx.float32, limit=10.0)
    rot = MK.gather_gate_up_swiglu_v3(xr, x, idx, U, out_dtype=mx.float32, limit=10.0, rot_out=True)
    ref = JK.h32_rows(plain, mx.float32)
    mx.eval(rot, ref)
    r, g = np.array(ref), np.array(rot)
    assert np.max(np.abs(r - g)) <= 1e-5 * np.max(np.abs(r)) + 1e-7


def test_routed_unchanged(monkeypatch):
    m = _module(7)
    x, idx = _inputs(2)
    w = mx.full(idx.shape, 0.25)
    monkeypatch.setenv("JANGT_ROT_OUT_DECODE", "0"); a = m.routed(x, idx, w)
    monkeypatch.setenv("JANGT_ROT_OUT_DECODE", "1"); b = m.routed(x, idx, w)
    mx.eval(a, b)
    a_, b_ = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    assert np.isfinite(a_).all() and np.max(np.abs(a_ - b_)) <= 8e-3 * np.max(np.abs(a_))
