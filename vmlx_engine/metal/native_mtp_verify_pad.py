"""Stock-MLX activation-row padding for native-MTP verification projections.

The native-MTP verify forward runs M = depth+1 activation rows (2..4). MLX
dispatches quantized matmuls below its ``qmv_batch_limit`` (10-18 on M5 Max)
to the per-row ``qmv`` loop, which never touches the M5 neural accelerators;
the NAX ``qmm_t_nax`` path engages only at M >= that limit with K % 64 == 0.

A quantized matmul is ROW-INDEPENDENT: output row i depends only on activation
row i, so padding the activation to a larger M and slicing the real rows back
is exact to the kernel's own tolerance -- and it changes NOTHING about the
sequence the attention/cache layers see, because the padding happens INSIDE the
projection call, not on the token sequence. This avoids the phantom-cache-token
problem of padding the verify sequence.

This dispatcher patches ``nn.QuantizedLinear.__call__`` only while a verify
scope is active, pads eligible small-M activations up to a tile, runs STOCK
``mx.quantized_matmul`` (no external dependency), and slices back. It is a
self-contained alternative to the dflash ``native_mtp_verify_qmm`` path.

Default OFF. History: the dflash rows==4 kernel won an isolated projection
microbenchmark but lost the complete lazy Qwen graph, so this ships gated
behind ``VMLX_MTP_VERIFY_PAD=1`` and stays off until a full-model A/B on the
exact bundle proves it out (house rule: isolated wins die on the full graph).

MEASURED RESULT (2026-09-03, Qwen3.8-Flash-Next-JANG_4S, fixed D3, G17s,
MLX 0.32.2, 4 probes median): this dispatcher LOSES on the full graph —
count 72.4 -> 45.2 tok/s, code 72.7 -> 43.8 (~-38%), AND acceptance drops
73.0% -> 65.1% because crossing qmv -> qmm_t_nax changes the verify logits
(TF32-class), which shifts which drafts are accepted. Zero-pad-to-tile +
stock quantized_matmul is therefore the WRONG mechanism; the verify-NAX-tile
idea needs a real fused kernel that applies scales on the cooperative
destination (the D-lane research item), not pad-and-stock-matmul. Kept
default-off as a documented dead end + the row-independence unit tests.
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

_SCOPE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "vmlx_native_mtp_verify_pad", default=False
)
_SCOPE_STATS: contextvars.ContextVar[dict[str, int] | None] = contextvars.ContextVar(
    "vmlx_native_mtp_verify_pad_stats", default=None
)
_PATCH: dict[str, object] = {
    "installed": False,
    "enabled": False,
    "original": None,
    "reason": "not_probed",
    "tile": 16,
    "calls": 0,
    "padded": 0,
}

# Default pad tile. 16 clears the M5 Max qmv->qmm limit for K,N <= 4096 (12)
# and is a multiple of 8 (the Metal-4 matmul2d floor). Overridable for A/B.
_DEFAULT_TILE = 16
_MAX_TILE = 32


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for item in str(value).split("."):
        digits = "".join(ch for ch in item if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _requested_tile() -> int:
    raw = os.environ.get("VMLX_MTP_VERIFY_PAD_TILE", "").strip()
    if not raw:
        return _DEFAULT_TILE
    try:
        tile = int(raw)
    except ValueError:
        return _DEFAULT_TILE
    return max(8, min(_MAX_TILE, tile))


def _requested() -> bool:
    value = os.environ.get(
        "VMLINUX_MTP_VERIFY_PAD",
        os.environ.get("VMLX_MTP_VERIFY_PAD", "0"),
    ).strip().lower()
    return value not in {"", "0", "false", "off", "no"}


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


def install_native_mtp_verify_pad() -> dict[str, object]:
    """Install the padding dispatcher once, if requested and supported."""

    if _PATCH["installed"]:
        return native_mtp_verify_pad_status()

    if not _requested():
        _PATCH["reason"] = "disabled_by_default"
        return native_mtp_verify_pad_status()

    supported, reason = _runtime_supported()
    if not supported:
        _PATCH["reason"] = reason
        return native_mtp_verify_pad_status()

    tile = _requested_tile()
    _PATCH["tile"] = tile
    original = nn.QuantizedLinear.__call__

    def _pad_call(self: nn.QuantizedLinear, x: mx.array) -> mx.array:
        if not _SCOPE.get() or x.ndim < 2:
            return original(self, x)

        leading = tuple(x.shape[:-1])
        k = int(x.shape[-1])
        rows = 1
        for dim in leading:
            rows *= int(dim)
        # Only small-M verify activations below the NAX limit benefit, and the
        # qmm NAX path requires K % 64 == 0. Everything else is stock MLX.
        if (
            rows <= 1
            or rows >= tile
            or k % 64 != 0
            or str(getattr(self, "mode", "affine") or "affine") != "affine"
            or x.dtype not in (mx.bfloat16, mx.float16)
        ):
            return original(self, x)

        _PATCH["calls"] = int(_PATCH["calls"]) + 1
        stats = _SCOPE_STATS.get()
        if stats is not None:
            stats["calls"] = int(stats.get("calls", 0)) + 1

        flat = x.reshape(rows, k)
        pad_rows = tile - rows
        # Zero-pad the extra rows. Row independence makes the real rows exact
        # to the kernel's tolerance; the padded rows are discarded.
        padded = mx.concatenate(
            [flat, mx.zeros((pad_rows, k), dtype=flat.dtype)], axis=0
        )
        out = original(self, padded)
        real = out[:rows]
        _PATCH["padded"] = int(_PATCH["padded"]) + 1
        if stats is not None:
            stats["padded"] = int(stats.get("padded", 0)) + 1
        return real.reshape(*leading, int(out.shape[-1]))

    nn.QuantizedLinear.__call__ = _pad_call
    _PATCH["installed"] = True
    _PATCH["enabled"] = True
    _PATCH["original"] = original
    _PATCH["reason"] = "active"
    logger.info(
        "Native MTP verify-pad dispatcher active (tile=%d, MLX %s, %s)",
        tile,
        metadata.version("mlx"),
        mx.device_info().get("architecture", "unknown"),
    )
    return native_mtp_verify_pad_status()


def native_mtp_verify_pad_active() -> bool:
    return bool(_PATCH["installed"] and _PATCH["enabled"])


def native_mtp_verify_pad_status() -> dict[str, object]:
    return {
        "installed": bool(_PATCH["installed"]),
        "enabled": bool(_PATCH["enabled"]),
        "reason": str(_PATCH["reason"]),
        "tile": int(_PATCH["tile"]),
        "calls": int(_PATCH["calls"]),
        "padded": int(_PATCH["padded"]),
    }


@contextlib.contextmanager
def native_mtp_verify_pad_scope() -> Iterator[dict[str, int]]:
    """Enable and count the padding route for one target verify forward."""

    install_native_mtp_verify_pad()
    stats = {"calls": 0, "padded": 0}
    active_token = _SCOPE.set(native_mtp_verify_pad_active())
    stats_token = _SCOPE_STATS.set(stats)
    try:
        yield stats
    finally:
        _SCOPE_STATS.reset(stats_token)
        _SCOPE.reset(active_token)


__all__ = [
    "install_native_mtp_verify_pad",
    "native_mtp_verify_pad_active",
    "native_mtp_verify_pad_scope",
    "native_mtp_verify_pad_status",
]
