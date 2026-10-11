"""JANGT routed experts must be correct on EVERY GPU, not only NAX (M5) ones.

User report 2026-10-11 (M4 Max, macOS 26.6.2): jangt.switch routed every batch of >= SORT_THRESHOLD slots to the NAX-only
prefill kernels without checking NAX; on the M4 they computed garbage (prefill rel-L2 133.6 % vs decode 0.25 % against
an FP32 reference), Naive JANGHT answered "2 + 2" with gibberish. tests/test_jangt_expert_tiles.py skips its whole
module without NAX, so nothing exercised that dispatch on older GPUs. These tests run everywhere:
  * a prefill-size batch through routed() equals the decode kernels (on NAX GPUs: the NAX tiles vs decode; elsewhere:
    the route must BE the decode kernels);
  * with NAX unavailable the sorted prefill is never entered;
  * the effective route is part of the persisted-cache identity.
"""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from vmlx_engine.jangh import kernels as JK  # noqa: E402
from vmlx_engine.jangt import switch as SW  # noqa: E402

# Fixture builders copied from tests/test_jangt_expert_tiles.py: importing that module would pull in its NAX-only
# module-level skip and skip this file on exactly the GPUs it exists for.
D, I, E = 1024, 512, 16


def _module(dn_mode: str, n_out: int, seed: int):
    rng = np.random.default_rng(seed)
    dn = {"mode": "jangt", "kbits": 2.5} if dn_mode == "jangt" else {"mode": "jangtq2", "bits": 3, "rotation": "hadamard32"}
    m = SW.JTMixedSwitchGLU(D, I, E, {"mode": "jangt", "kbits": 2}, dn, n_out, 0, 0.0)
    for p in (m.gate_proj, m.up_proj):
        p.jt_packed = mx.array(rng.integers(0, 2 ** 32, size=p.jt_packed.shape, dtype=np.uint64).astype(np.uint32))
        p.jt_scale = mx.array(rng.uniform(0.02, 0.05, size=p.jt_scale.shape).astype(np.float32))
        p.jt_su = mx.array(rng.choice([-1.0, 1.0], size=p.jt_su.shape).astype(np.float32))
        if n_out:
            p.jt_ocol = mx.array(np.tile(np.arange(n_out, dtype=np.int32) * 7, (E, 1)))
            p.jt_wout = mx.array(rng.standard_normal(p.jt_wout.shape).astype(np.float16) * 0.05)
    m.up_proj.jt_su = m.gate_proj.jt_su
    d = m.down_proj
    if dn_mode == "jangt":
        d.jt_packed = mx.array(rng.integers(0, 2 ** 32, size=d.jt_packed.shape, dtype=np.uint64).astype(np.uint32))
        d.jt_scale = mx.array(rng.uniform(0.01, 0.03, size=d.jt_scale.shape).astype(np.float32))
        d.jt_su = mx.array(rng.choice([-1.0, 1.0], size=d.jt_su.shape).astype(np.float32))
    else:
        d.tq2_packed = mx.array(rng.integers(0, 2 ** 32, size=d.tq2_packed.shape, dtype=np.uint64).astype(np.uint32))
        d.tq2_scales = mx.array(rng.uniform(0.01, 0.03, size=d.tq2_scales.shape).astype(np.float16))
    return m


def _inputs(T, seed, k=4):
    rng = np.random.default_rng(seed)
    x = mx.array((rng.standard_normal((T, D)) * 0.5).astype(np.float32)).astype(mx.bfloat16)
    # experts 0..9 only: 6 of 16 are empty, and per-expert row counts are ragged
    idx = mx.array(np.stack([rng.choice(10, size=k, replace=False) for _ in range(T)]).astype(np.int32))
    w = mx.array(rng.uniform(0.1, 0.4, size=(T, k)).astype(np.float32))
    return x, idx, w


def _decode_reference(m, x, idx, w):
    y = m._decode_down(m._decode_h(x, idx, rot_out=False), idx, w, x.dtype, h_rotated=False)
    mx.eval(y)
    return np.array(y.astype(mx.float32))


def _routed(m, x, idx, w):
    y = m.routed(x, idx, w)
    mx.eval(y)
    return np.array(y.astype(mx.float32))


@pytest.mark.parametrize("dn_mode,n_out", [("jangt", 0), ("jangh", 0), ("jangh", 3)])
def test_prefill_size_batch_matches_the_decode_kernels(dn_mode, n_out):
    m = _module(dn_mode, n_out, seed=7)
    x, idx, w = _inputs(48, seed=11)                  # 48 tokens x 4 experts = 192 slots >= SORT_THRESHOLD
    assert idx.size >= SW.SORT_THRESHOLD
    ref = _decode_reference(m, x, idx, w)
    got = _routed(m, x, idx, w)
    assert np.isfinite(ref).all() and float(np.linalg.norm(ref)) > 0     # fail closed on an empty reference
    rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    assert rel < 2e-2, f"routed prefill disagrees with the decode kernels: rel-L2 {rel:.4f} (nax={JK.nax_available()})"


def test_without_nax_every_batch_takes_the_decode_kernels(monkeypatch):
    m = _module("jangh", 3, seed=3)
    x, idx, w = _inputs(48, seed=5)
    monkeypatch.setattr(JK, "nax_available", lambda: False)

    def boom(*args, **kwargs):
        raise AssertionError("NAX-only JANGT prefill entered on a GPU without NAX")

    monkeypatch.setattr(m, "_prefill", boom)
    got = _routed(m, x, idx, w)
    np.testing.assert_allclose(got, _decode_reference(m, x, idx, w), rtol=0, atol=0)


def test_prefill_route_is_part_of_the_cache_identity(monkeypatch):
    # Import BEFORE patching: prefix_cache computes the process fingerprint once at import, and a first import under
    # the patch would freeze a "decode" fingerprint for every later test in this process.
    from vmlx_engine import prefix_cache

    monkeypatch.setattr(JK, "nax_available", lambda: True)
    nax = SW.jangt_runtime_identity()
    monkeypatch.setattr(JK, "nax_available", lambda: False)
    decode = SW.jangt_runtime_identity()
    assert nax.endswith(";jangt_prefill=nax") and decode.endswith(";jangt_prefill=decode")
    assert nax.split(";")[0] == decode.split(";")[0] and nax != decode
    assert ";jangt_prefill=decode" in prefix_cache._resolve_runtime_cache_fingerprint()


def test_route_and_identity_do_not_depend_on_the_default_device():
    """A process whose default device is the CPU (test children, CPU-side tools) must compute the same NAX answer and
    cache identity as the serving process: mx.device_info() describes the default device ("arm64" on CPU), and a
    different identity there split the SSD namespace between processes (test_ssm_disk_candidate_index caught it)."""
    import subprocess
    import sys
    from pathlib import Path

    root = str(Path(__file__).resolve().parents[1])
    script = (
        f"import sys; sys.path.insert(0, {root!r})\n"
        "import mlx.core as mx\nmx.set_default_device(mx.cpu)\n"
        "from vmlx_engine.jangt import switch as SW\nprint(SW.jangt_runtime_identity())\n"
    )
    child = subprocess.run([sys.executable, "-I", "-B", "-c", script], capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip().splitlines()[-1] == SW.jangt_runtime_identity()
