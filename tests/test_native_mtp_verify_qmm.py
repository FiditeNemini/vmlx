from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from vmlx_engine.metal import native_mtp_verify_qmm as verify_qmm


@pytest.fixture(autouse=True)
def _reset_dispatcher():
    original = nn.QuantizedLinear.__call__
    verify_qmm._PATCH.update(
        installed=False,
        kernel_enabled=False,
        original=None,
        reason="not_probed",
        calls=0,
    )
    yield
    nn.QuantizedLinear.__call__ = original
    verify_qmm._PATCH.update(
        installed=False,
        kernel_enabled=False,
        original=None,
        reason="not_probed",
        calls=0,
    )


def test_supported_runtime_keeps_dependency_kernel_off_by_default(monkeypatch):
    monkeypatch.setattr(verify_qmm, "_runtime_supported", lambda: (True, "supported"))
    monkeypatch.delenv("DFLASH_VERIFY_QMM", raising=False)

    status = verify_qmm.install_native_mtp_verify_qmm()

    assert status["installed"] is False
    assert status["kernel_enabled"] is False
    assert status["reason"] == "disabled_by_default"
    assert verify_qmm.native_mtp_verify_qmm_active() is False
    assert "DFLASH_VERIFY_QMM" not in verify_qmm.os.environ


def test_supported_runtime_honors_explicit_kernel_enable(monkeypatch):
    monkeypatch.setattr(verify_qmm, "_runtime_supported", lambda: (True, "supported"))
    monkeypatch.setenv("DFLASH_VERIFY_QMM", "1")

    status = verify_qmm.install_native_mtp_verify_qmm()

    assert status["installed"] is True
    assert status["kernel_enabled"] is True
    assert status["reason"] == "active"
    assert verify_qmm.native_mtp_verify_qmm_active() is True


def test_explicit_dependency_kernel_opt_out_is_honored(monkeypatch):
    monkeypatch.setattr(verify_qmm, "_runtime_supported", lambda: (True, "supported"))
    monkeypatch.setenv("DFLASH_VERIFY_QMM", "0")

    status = verify_qmm.install_native_mtp_verify_qmm()

    assert status["installed"] is False
    assert status["kernel_enabled"] is False
    assert status["reason"] == "disabled_by_env"
    assert verify_qmm.native_mtp_verify_qmm_active() is False


def test_scope_counts_q4_calls_but_not_ineligible_q6_fallback(monkeypatch):
    import dflash_mlx.verify_qmm as kernel

    monkeypatch.setattr(verify_qmm, "_runtime_supported", lambda: (True, "supported"))
    monkeypatch.setenv("DFLASH_VERIFY_QMM", "1")
    monkeypatch.setattr(kernel, "is_enabled", lambda: True)
    monkeypatch.setattr(
        kernel,
        "verify_matmul",
        lambda x, weight, _scales, _biases, **_kwargs: mx.zeros(
            (*x.shape[:-1], int(weight.shape[0])), dtype=x.dtype
        ),
    )
    verify_qmm.install_native_mtp_verify_qmm()
    q4 = nn.QuantizedLinear(64, 8, bias=False, group_size=32, bits=4)
    q6 = nn.QuantizedLinear(64, 8, bias=False, group_size=32, bits=6)
    x = mx.zeros((1, 4, 64), dtype=mx.bfloat16)

    with verify_qmm.native_mtp_verify_qmm_scope() as scope_stats:
        accelerated = q4(x)
        fallback = q6(x)
    mx.eval(accelerated, fallback)

    assert scope_stats == {"calls": 1}
    assert verify_qmm.native_mtp_verify_qmm_status()["calls"] == 1


def test_mtp_health_exposes_live_verifier_status(monkeypatch):
    from vmlx_engine import native_mtp, server

    monkeypatch.setattr(
        server,
        "_model_mtp_status",
        lambda _bundle: {"runtime_available": False},
    )
    monkeypatch.setattr(native_mtp, "native_mtp_disabled_by_env", lambda: False)
    monkeypatch.setattr(
        verify_qmm,
        "native_mtp_verify_qmm_status",
        lambda: {
            "installed": True,
            "kernel_enabled": True,
            "reason": "active",
            "calls": 37,
        },
    )

    status = server._model_mtp_status_with_loaded_runtime("/tmp/model")

    assert status["verify_qmm"] == {
        "installed": True,
        "kernel_enabled": True,
        "reason": "active",
        "calls": 37,
    }
