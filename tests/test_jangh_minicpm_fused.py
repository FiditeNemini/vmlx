"""MiniCPM-V-4.7 JANGH opt-in: per-module in-kernel decode rotation, fused prefill options, cache identity."""
import mlx.core as mx
import mlx.nn as nn
import pytest

from vmlx_engine.jangh import install as I
from vmlx_engine.jangh import kernels as K
from vmlx_engine.jangh import runtime_identity as R
from vmlx_engine.jangh.switch import TQSwitchGLU

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")

D, F, E, TOPK = 256, 128, 16, 4


def _bank(seed=0):
    mx.random.seed(seed)
    t = TQSwitchGLU(D, F, E, 2, 3, 0.0, rotation_gate_up="hadamard32", rotation_down="hadamard32")
    for i, p in enumerate((t.gate_proj, t.up_proj, t.down_proj)):
        p.tq2_packed = mx.random.randint(0, 2**31 - 1, p.tq2_packed.shape, key=mx.random.key(10 + i)).astype(mx.uint32)
        p.tq2_scales = mx.random.uniform(0.005, 0.02, p.tq2_scales.shape, key=mx.random.key(20 + i)).astype(mx.float16)
    mx.eval(t.parameters())
    return t


def _inputs(rows):
    x = (mx.random.normal((1, rows, D), key=mx.random.key(rows)) * 0.5).astype(mx.bfloat16)
    logits = mx.random.normal((1, rows, E), key=mx.random.key(rows + 1))
    inds = mx.argpartition(logits, kth=-TOPK, axis=-1)[..., -TOPK:]
    sc = mx.softmax(mx.take_along_axis(logits, inds, axis=-1), axis=-1)
    mx.eval(x, inds, sc)
    return x, inds, sc


@pytest.mark.parametrize("rows", [1, 3])
def test_per_module_decode_rotation_runs_in_kernel_and_matches_host(monkeypatch, rows):
    bank = _bank()
    x, inds, sc = _inputs(rows)
    host = bank.routed(x, inds, sc)
    seen = []
    real = K.gather_qmv

    def spy(*args, **kwargs):
        seen.append(bool(kwargs.get("rotate")))
        return real(*args, **kwargs)
    monkeypatch.setattr(K, "gather_qmv", spy)
    bank.decode_rot_kernel = True
    kern = bank.routed(x, inds, sc)
    mx.eval(host, kern)
    assert seen and all(seen), seen                       # gate/up and down kernels rotate in-register
    h, k = host.astype(mx.float32), kern.astype(mx.float32)
    assert (mx.abs(h - k).max() / (mx.abs(h).max() + 1e-6)).item() < 2e-2   # same math, different rounding order


def test_enable_minicpm_fused_sets_the_measured_options(monkeypatch):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [_bank(1), _bank(2)]
    m = M()
    monkeypatch.setattr(R, "MINICPM_FUSED", "1")
    assert I.enable_minicpm_fused(m) == 2
    for b in m.layers:
        assert b.use_weighted_unsort and b.decode_rot_kernel
        assert all(lin.use_h32_rows for lin in (b.gate_proj, b.up_proj, b.down_proj))
    m2 = M()
    monkeypatch.setattr(R, "MINICPM_FUSED", "0")
    assert I.enable_minicpm_fused(m2) == 0
    assert not getattr(m2.layers[0], "decode_rot_kernel", False)


def test_minicpm_fused_is_part_of_the_runtime_identity():
    assert ";minicpm_fused=" + R.MINICPM_FUSED in R.runtime_identity()


def test_jang_loader_opts_minicpm_in():
    import inspect
    from vmlx_engine.utils import jang_loader

    src = inspect.getsource(jang_loader)
    assert 'str(config.get("model_type", "")).lower() == "minicpmv4_7"' in src and "enable_minicpm_fused(model)" in src
