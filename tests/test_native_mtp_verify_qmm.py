from __future__ import annotations

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


def test_supported_runtime_enables_dependency_kernel_by_default(monkeypatch):
    monkeypatch.setattr(verify_qmm, "_runtime_supported", lambda: (True, "supported"))
    monkeypatch.delenv("DFLASH_VERIFY_QMM", raising=False)

    status = verify_qmm.install_native_mtp_verify_qmm()

    assert status["installed"] is True
    assert status["kernel_enabled"] is True
    assert status["reason"] == "active"
    assert verify_qmm.native_mtp_verify_qmm_active() is True
    assert verify_qmm.os.environ["DFLASH_VERIFY_QMM"] == "1"


def test_explicit_dependency_kernel_opt_out_is_honored(monkeypatch):
    monkeypatch.setattr(verify_qmm, "_runtime_supported", lambda: (True, "supported"))
    monkeypatch.setenv("DFLASH_VERIFY_QMM", "0")

    status = verify_qmm.install_native_mtp_verify_qmm()

    assert status["installed"] is False
    assert status["kernel_enabled"] is False
    assert status["reason"] == "disabled_by_env"
    assert verify_qmm.native_mtp_verify_qmm_active() is False


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
