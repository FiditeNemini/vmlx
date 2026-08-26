"""DFlash small-M affine QMM routing for native MTP verification.

Only the four-row target verification forward is eligible. Decode, prefill,
replay, MTP-head, and ordinary batched QuantizedLinear calls retain MLX's
stock implementation. The external kernel is enabled only on the M5/G17
runtime and MLX 0.32.1+, where it passed the real Qwen3.8-27B JANG_4D gate.

The kernel implementation is provided by dflash-mlx (Apache-2.0). Keeping the
dependency separate avoids maintaining a private fork of its Metal source.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
import platform
from collections.abc import Iterator
from importlib import metadata

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)
_VERIFY_SCOPE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "vmlx_native_mtp_verify_qmm", default=False
)
_PATCH = {
    "installed": False,
    "kernel_enabled": False,
    "original": None,
    "reason": "not_probed",
    "calls": 0,
}


def _version_tuple(value: str) -> tuple[int, ...]:
    parts = []
    for item in str(value).split("."):
        digits = "".join(ch for ch in item if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _runtime_supported() -> tuple[bool, str]:
    try:
        mlx_version = metadata.version("mlx")
    except metadata.PackageNotFoundError:
        return False, "mlx_distribution_missing"
    if _version_tuple(mlx_version) < (0, 32, 1):
        return False, f"mlx_{mlx_version}_below_0.32.1"

    architecture = str(mx.device_info().get("architecture", "")).lower()
    if not architecture.startswith("applegpu_g17"):
        return False, f"unsupported_gpu_{architecture or 'unknown'}"

    macos = _version_tuple(platform.mac_ver()[0])
    if macos < (26, 2):
        return False, "macos_below_26.2"
    return True, "supported"


def install_native_mtp_verify_qmm() -> dict[str, object]:
    """Install the narrowly scoped QuantizedLinear dispatcher once."""

    if _PATCH["installed"]:
        return native_mtp_verify_qmm_status()

    supported, reason = _runtime_supported()
    _PATCH["reason"] = reason
    if not supported:
        return native_mtp_verify_qmm_status()

    # dflash-mlx ships the verifier behind an environment gate.  vMLX already
    # guards this dispatcher by hardware, OS, MLX version, q4 affine shape and
    # the native-MTP verify scope, so an unset dependency flag must mean enabled
    # here.  Previously the dispatcher reported itself active while
    # verify_matmul silently fell back to stock MLX on every call.
    verify_flag = os.environ.get("DFLASH_VERIFY_QMM", "").strip()
    if verify_flag and verify_flag != "1":
        _PATCH["reason"] = "disabled_by_env"
        return native_mtp_verify_qmm_status()
    os.environ["DFLASH_VERIFY_QMM"] = "1"

    try:
        from dflash_mlx.verify_qmm import is_enabled, verify_matmul
    except Exception as exc:
        _PATCH["reason"] = f"dflash_import_failed:{type(exc).__name__}"
        return native_mtp_verify_qmm_status()
    if not is_enabled():
        _PATCH["reason"] = "dflash_kernel_disabled"
        return native_mtp_verify_qmm_status()

    original = nn.QuantizedLinear.__call__

    def patched(self, x):  # type: ignore[no-untyped-def]
        if not _VERIFY_SCOPE.get() or x.ndim < 2:
            return original(self, x)

        rows = 1
        for dim in x.shape[:-1]:
            rows *= int(dim)
        bits = int(getattr(self, "bits", 0) or 0)
        group_size = int(getattr(self, "group_size", 0) or 0)
        mode = str(getattr(self, "mode", "affine") or "affine")
        if (
            rows != 4
            or bits != 4
            or group_size not in (32, 64, 128)
            or mode != "affine"
            or x.dtype not in (mx.bfloat16, mx.float16)
        ):
            return original(self, x)

        _PATCH["calls"] = int(_PATCH["calls"]) + 1
        y = verify_matmul(
            x,
            self["weight"],
            self["scales"],
            self["biases"],
            transpose=True,
            group_size=group_size,
            bits=bits,
        )
        if "bias" in self:
            y = y + self["bias"]
        return y

    nn.QuantizedLinear.__call__ = patched
    _PATCH["installed"] = True
    _PATCH["kernel_enabled"] = True
    _PATCH["original"] = original
    _PATCH["reason"] = "active"
    logger.info(
        "Native MTP DFlash verifier active for four-row q4 affine projections "
        "(MLX %s, %s)",
        metadata.version("mlx"),
        mx.device_info().get("architecture", "unknown"),
    )
    return native_mtp_verify_qmm_status()


def native_mtp_verify_qmm_active() -> bool:
    return bool(_PATCH["installed"] and _PATCH["kernel_enabled"])


def native_mtp_verify_qmm_status() -> dict[str, object]:
    return {
        "installed": bool(_PATCH["installed"]),
        "kernel_enabled": bool(_PATCH["kernel_enabled"]),
        "reason": str(_PATCH["reason"]),
        "calls": int(_PATCH["calls"]),
    }


@contextlib.contextmanager
def native_mtp_verify_qmm_scope() -> Iterator[None]:
    """Enable the custom q4 route only while building one target verify."""

    install_native_mtp_verify_qmm()
    token = _VERIFY_SCOPE.set(native_mtp_verify_qmm_active())
    try:
        yield
    finally:
        _VERIFY_SCOPE.reset(token)
