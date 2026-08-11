# SPDX-License-Identifier: Apache-2.0
"""Family-agnostic prefill admission control and allocation-error triage.

DSV4 already had a prefill valve: it projects the next chunk's peak Metal
working set and aborts the request BEFORE submitting GPU work, because a Metal
allocation failure inside libc++ can take the process down. Every other family
had nothing, so a prompt the device cannot serve was discovered by failing
rather than by checking.

Two pure pieces live here so both are unit-testable without Metal:

* :func:`prefill_valve_check` — the projection, generalised from DSV4's.
* :func:`classify_allocation_error` — tells a PERMANENT allocation failure
  (one a retry can never satisfy) apart from transient exhaustion.

The distinction matters because the recovery for the two is opposite. Clearing
the cache and rescheduling fixes transient exhaustion; against a request for
more than the device's maximum buffer size it re-runs the identical doomed
allocation forever. The scheduler already learned this the hard way with
``DSV4PrefillMemoryError``, which it excludes from recovery BY NAME.
"""

from __future__ import annotations

import os

_GIB = 1024**3

# MLX reports an over-maximum buffer request like:
#   [metal::malloc] Attempting to allocate 4398046511104 bytes which is greater
#   than the maximum allowed buffer size of 86586540032 bytes.
# Measured on-device: this is a CATCHABLE Python RuntimeError, not a process
# abort — the failure mode is that nothing classifies it, not that it escapes.
_PERMANENT_SIGNATURES = (
    "greater than the maximum allowed buffer size",
    "maximum allowed buffer size",
)

# Exhaustion that a cache clear can genuinely relieve.
_TRANSIENT_SIGNATURES = (
    "out of memory",
    "cannot allocate memory",
    "allocation failed",
    "insufficient memory",
)


class PrefillAdmissionError(RuntimeError):
    """The request cannot be served on this device — rejected before GPU work.

    The message deliberately avoids every scheduler ``CACHE_CORRUPTION_PATTERNS``
    substring ("out of memory", "Allocation failed", ...). Matching one would
    route this into cache-clear + reschedule, which would re-run the doomed
    prefill in a loop. Same contract as ``DSV4PrefillMemoryError``.
    """


def prefill_valve_enabled() -> bool:
    """Admission control is ON by default; set VMLX_PREFILL_ADMISSION=0 to disable."""
    raw = os.environ.get("VMLX_PREFILL_ADMISSION", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def prefill_valve_min_margin_bytes(default_gb: float = 2.0) -> int:
    """Transient-headroom floor used before any per-chunk peak is observed.

    Mirrors the DSV4 valve's default: measured transient peak-over-active on
    DSV4-Flash reaches ~3.4GB at 430k context, and the valve adapts upward from
    observed peaks, so this only has to cover the first chunks.
    """
    try:
        gb = float(os.environ.get("VMLX_PREFILL_ADMISSION_MIN_MARGIN_GB", default_gb))
    except (TypeError, ValueError):
        gb = default_gb
    return max(0, int(gb * _GIB))


def project_chunk_peak_bytes(
    active_bytes: int,
    observed_transient_bytes: int,
    min_margin_bytes: int,
) -> int:
    """Projected peak for the next chunk: active + 1.25x the largest transient seen."""
    margin = max(int(observed_transient_bytes * 1.25), int(min_margin_bytes))
    return int(active_bytes) + margin


def prefill_valve_check(
    active_bytes: int,
    max_ws_bytes: int,
    observed_transient_bytes: int,
    min_margin_bytes: int,
    *,
    chunk_start: int,
    chunk_end: int,
    model_label: str = "model",
) -> None:
    """Raise :class:`PrefillAdmissionError` when the next chunk cannot fit.

    Pure: no MLX calls, so the threshold is unit-testable without a GPU. A
    non-positive limit or active reading means "unknown", and unknown must never
    reject a request that would have worked.
    """
    if max_ws_bytes <= 0 or active_bytes <= 0:
        return
    projected = project_chunk_peak_bytes(
        active_bytes, observed_transient_bytes, min_margin_bytes
    )
    if projected <= max_ws_bytes:
        return
    raise PrefillAdmissionError(
        f"{model_label}: prefill admission rejected chunk "
        f"[{chunk_start}:{chunk_end}) — active Metal working set "
        f"{active_bytes / _GIB:.2f}GB plus projected transient "
        f"{(projected - active_bytes) / _GIB:.2f}GB exceeds the device "
        f"working-set limit {max_ws_bytes / _GIB:.2f}GB. A context of this "
        f"length cannot be served on this hardware; reduce the prompt or "
        f"context size."
    )


def classify_allocation_error(error: BaseException | str) -> str | None:
    """Return ``"permanent"``, ``"transient"``, or ``None`` (not allocation-related).

    ``"permanent"`` means a retry cannot succeed — the request asked for more
    than the device's maximum buffer size. Recovering by clearing caches would
    re-run the identical allocation, so callers must fail the request instead.
    """
    text = str(error).lower()
    if not text:
        return None
    for signature in _PERMANENT_SIGNATURES:
        if signature in text:
            return "permanent"
    if isinstance(error, PrefillAdmissionError):
        return "permanent"
    for signature in _TRANSIENT_SIGNATURES:
        if signature in text:
            return "transient"
    if "[metal::malloc]" in text:
        # A metal::malloc failure with no clearer signal: treat as transient so
        # the existing cache-clear recovery gets one chance at it.
        return "transient"
    return None


def is_permanent_allocation_error(error: BaseException | str) -> bool:
    return classify_allocation_error(error) == "permanent"
