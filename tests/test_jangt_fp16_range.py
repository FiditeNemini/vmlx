"""JANGT v3 decode kernels stay finite and exact for activations beyond the fp16 range of their half2 trellis dot.

The kernels convert each lane's 16 inputs to half and accumulate 8 code products per chunk in half2. Unclamped
SwiGLU experts (Naive-N0.5) feed the down projection hidden units of 1e5 and more; before the guard such rows became
inf/NaN (|h| ~ 3.3e3 was already enough to overflow a chunk: 8 x 2 x 3.3e3 > 65504). The guard scales a lane by a
power of two above |x| = 1024, so: (1) output(x * 2^16) == 2^16 * output(x) exactly up to fp32 rounding (the power of
two only moves exponents); (2) it matches a dense float32 reference built from ``moe_kernels.dequant_rot``.
"""
from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from vmlx_engine.jangt import moe_kernels as MK  # noqa: E402
from vmlx_engine.jangt import switch as SW  # noqa: E402

D, I, E, K = 1024, 512, 8, 2          # kernels and dequant_rot work in 512-column blocks
S = float(2 ** 16)


def _module(seed: int = 0):
    rng = np.random.default_rng(seed)
    m = SW.JTMixedSwitchGLU(D, I, E, {"mode": "jangt", "kbits": K}, {"mode": "jangt", "kbits": 2.5}, 0, 0, 0.0)
    for p in (m.gate_proj, m.up_proj, m.down_proj):
        p.jt_packed = mx.array(rng.integers(0, 2 ** 32, size=p.jt_packed.shape, dtype=np.uint64).astype(np.uint32))
        p.jt_scale = mx.array(rng.uniform(0.01, 0.03, size=p.jt_scale.shape).astype(np.float32))
        p.jt_su = mx.array(rng.choice([-1.0, 1.0], size=p.jt_su.shape).astype(np.float32))
    m.up_proj.jt_su = m.gate_proj.jt_su
    return m


def _rel(a, b):
    a, b = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    assert np.isfinite(a).all() and np.isfinite(b).all()
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


def _down(m, h, idx, w):
    U = m._unit()
    hr = MK.rotate_rows(h, U["sv_dn"], idx.reshape(-1))
    return MK.gather_down_weighted_v3(hr, h, idx, w, U)


def test_down_kernel_is_scale_exact_beyond_fp16():
    m = _module(1)
    rng = np.random.default_rng(2)
    T, k = 3, 4
    idx = mx.array(np.stack([rng.permutation(E)[:k] for _ in range(T)]).astype(np.int32))
    w = mx.full(idx.shape, 0.25)
    h = mx.array(rng.standard_normal((T * k, I)).astype(np.float32) * 4.0)           # max |h| ~ 15: no scaling
    small = _down(m, h, idx, w)
    big = _down(m, h * S, idx, w)                                                    # max |h| ~ 1e6
    mx.eval(small, big)
    assert _rel(big, small * S) < 1e-5


def test_down_kernel_matches_dense_reference_at_1e5():
    m = _module(3)
    rng = np.random.default_rng(4)
    T, k = 2, 4
    idx = mx.array(np.stack([rng.permutation(E)[:k] for _ in range(T)]).astype(np.int32))
    w = mx.array(rng.uniform(0.1, 0.4, size=(T, k)).astype(np.float32))
    h = mx.array(rng.standard_normal((T * k, I)).astype(np.float32) * 3e4)          # |h| up to ~1.2e5
    got = _down(m, h, idx, w)
    U = m._unit()
    hr = MK.rotate_rows(h, U["sv_dn"], idx.reshape(-1)).astype(mx.float32)
    Wd = MK.dequant_rot(U["down"], 0, E).astype(mx.float32)                        # (E, D, I), rotated basis
    y = mx.sum(hr[:, None, :] * Wd[idx.reshape(-1)], axis=-1).reshape(T, k, D)
    ref = mx.sum(y * w[..., None], axis=1)
    mx.eval(got, ref)
    assert _rel(got, ref) < 2e-3


def test_gate_up_kernel_matches_dense_reference_beyond_fp16():
    m = _module(5)
    rng = np.random.default_rng(6)
    T, k = 2, 4
    idx = mx.array(np.stack([rng.permutation(E)[:k] for _ in range(T)]).astype(np.int32))
    x = mx.array((rng.standard_normal((T, D)) * 2e4).astype(np.float32)).astype(mx.bfloat16)   # |x| up to ~8e4
    U = m._unit()
    xr = MK.rotate_rows(x, U["su_gu"])
    got = MK.gather_gate_up_swiglu_v3(xr, x, idx, U, out_dtype=mx.float32, limit=0.0)
    xf = xr.astype(mx.float32)
    rows = mx.repeat(xf, k, axis=0)
    e = idx.reshape(-1)
    g = mx.sum(rows[:, None, :] * MK.dequant_rot(U["gate"], 0, E).astype(mx.float32)[e], axis=-1)
    u = mx.sum(rows[:, None, :] * MK.dequant_rot(U["up"], 0, E).astype(mx.float32)[e], axis=-1)
    ref = g * mx.sigmoid(g) * u
    mx.eval(got, ref)
    assert _rel(got, ref) < 5e-3
