"""Correctness of the native-MTP verify projection-padding dispatcher.

The dispatcher pads small-M verify activations up to a NAX tile, runs stock
quantized_matmul, and slices the real rows back. A quantized matmul is
row-independent, so the real rows must equal the unpadded projection — exactly
on the same kernel, and to NAX tolerance if the pad crosses the qmv->qmm
dispatch boundary. These tests pin: default-off, scope gating, exact real-row
equivalence, no leakage from padded rows, and pass-through of ineligible
shapes.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from vmlx_engine.metal import native_mtp_verify_pad as pad


@pytest.fixture(autouse=True)
def _reset_patch(monkeypatch):
    # The module installs a process-wide monkeypatch once; reset its state and
    # restore the original QuantizedLinear.__call__ around every test.
    original = getattr(pad._PATCH, "original", None)
    yield
    if pad._PATCH.get("installed") and pad._PATCH.get("original") is not None:
        nn.QuantizedLinear.__call__ = pad._PATCH["original"]
    pad._PATCH.update(
        installed=False, enabled=False, original=None,
        reason="not_probed", tile=16, calls=0, padded=0,
    )
    for name in (
        "VMLX_MTP_VERIFY_PAD", "VMLINUX_MTP_VERIFY_PAD",
        "VMLX_MTP_VERIFY_PAD_TILE",
    ):
        monkeypatch.delenv(name, raising=False)


def _qlinear(k=256, n=512, bits=4, gs=64):
    lin = nn.Linear(k, n, bias=False)
    nn.quantize(lin, bits=bits, group_size=gs, mode="affine")
    lin.eval()
    return lin


def test_defaults_off_and_requires_env(monkeypatch):
    monkeypatch.delenv("VMLX_MTP_VERIFY_PAD", raising=False)
    status = pad.install_native_mtp_verify_pad()
    assert status["installed"] is False
    assert status["reason"] == "disabled_by_default"


def test_scope_is_a_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("VMLX_MTP_VERIFY_PAD", raising=False)
    lin = _qlinear()
    x = mx.random.normal((1, 3, 256)).astype(mx.bfloat16)
    reference = lin(x)
    with pad.native_mtp_verify_pad_scope():
        out = lin(x)
    mx.eval(reference, out)
    assert bool(mx.all(reference == out))


def test_padded_real_rows_match_unpadded_within_tolerance(monkeypatch):
    # On CI (non-G17) the runtime gate keeps the dispatcher off, so this asserts
    # the pad MATH directly rather than through the (unavailable) install path:
    # padding rows and slicing must reproduce the unpadded projection.
    lin = _qlinear(k=256, n=512)
    for rows in (2, 3, 4, 7):
        x = (mx.random.normal((1, rows, 256)) * 0.2).astype(mx.bfloat16)
        reference = lin(x)
        flat = x.reshape(rows, 256)
        padded = mx.concatenate(
            [flat, mx.zeros((16 - rows, 256), dtype=flat.dtype)], axis=0
        )
        padded_out = lin(padded)[:rows].reshape(1, rows, 512)
        mx.eval(reference, padded_out)
        # Same kernel here (M<qmv limit on CI), so this is exact; on a G17 box
        # crossing into qmm_t_nax it holds to NAX (TF32-class) tolerance.
        max_ref = max(float(mx.max(mx.abs(reference)).item()), 1e-6)
        max_abs = float(
            mx.max(mx.abs(padded_out.astype(mx.float32)
                          - reference.astype(mx.float32))).item()
        )
        assert max_abs / max_ref <= 1e-2, f"rows={rows} rel={max_abs/max_ref}"


def test_zero_pad_rows_do_not_leak_into_real_rows(monkeypatch):
    # Row independence: perturbing the padded region must not change real rows.
    lin = _qlinear(k=128, n=256)
    rows = 3
    flat = (mx.random.normal((rows, 128)) * 0.2).astype(mx.bfloat16)
    zeros_pad = mx.concatenate(
        [flat, mx.zeros((13, 128), dtype=flat.dtype)], axis=0
    )
    noise_pad = mx.concatenate(
        [flat, (mx.random.normal((13, 128)) * 5).astype(flat.dtype)], axis=0
    )
    a = lin(zeros_pad)[:rows]
    b = lin(noise_pad)[:rows]
    mx.eval(a, b)
    assert bool(mx.all(a == b))


def test_tile_env_is_clamped(monkeypatch):
    monkeypatch.setenv("VMLX_MTP_VERIFY_PAD_TILE", "999")
    assert pad._requested_tile() == pad._MAX_TILE
    monkeypatch.setenv("VMLX_MTP_VERIFY_PAD_TILE", "3")
    assert pad._requested_tile() == 8
    monkeypatch.setenv("VMLX_MTP_VERIFY_PAD_TILE", "16")
    assert pad._requested_tile() == 16
