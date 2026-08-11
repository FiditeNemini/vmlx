# SPDX-License-Identifier: Apache-2.0
"""Prefill admission control and allocation-error triage.

DSV4 has long had a prefill valve that rejects a chunk BEFORE submitting GPU
work; every other family reached the prefill loop with no check, which is how a
model advertising ~69k tokens dies somewhere around 20-25k. Both pieces here are
PURE so the thresholds are testable without a GPU.

The triage half matters just as much: clearing the cache relieves transient
exhaustion, but against a request for more than the device's maximum buffer size
it re-runs the identical doomed allocation forever.
"""

import pytest

from vmlx_engine.utils.prefill_admission import (
    PrefillAdmissionError,
    classify_allocation_error,
    is_permanent_allocation_error,
    prefill_valve_check,
    project_chunk_peak_bytes,
)

GIB = 1024**3

# The exact string MLX produces on this hardware, measured (exit code 0 — it is
# a catchable RuntimeError, not a process abort).
REAL_METAL_OVERMAX = (
    "[metal::malloc] Attempting to allocate 4398046511104 bytes which is "
    "greater than the maximum allowed buffer size of 86586540032 bytes."
)


class TestValve:
    def test_rejects_when_projection_exceeds_the_limit(self):
        with pytest.raises(PrefillAdmissionError) as excinfo:
            prefill_valve_check(
                active_bytes=80 * GIB,
                max_ws_bytes=86 * GIB,
                observed_transient_bytes=8 * GIB,
                min_margin_bytes=2 * GIB,
                chunk_start=20000,
                chunk_end=22048,
            )
        message = str(excinfo.value)
        assert "cannot be served on this hardware" in message
        assert "[20000:22048)" in message

    def test_admits_when_the_chunk_fits(self):
        prefill_valve_check(
            active_bytes=10 * GIB,
            max_ws_bytes=86 * GIB,
            observed_transient_bytes=1 * GIB,
            min_margin_bytes=2 * GIB,
            chunk_start=0,
            chunk_end=2048,
        )

    @pytest.mark.parametrize(
        "active,limit", [(0, 86 * GIB), (10 * GIB, 0), (0, 0), (-1, 86 * GIB)]
    )
    def test_unknown_readings_never_reject(self, active, limit):
        """An unreadable meter must not reject a request that would have worked."""
        prefill_valve_check(
            active_bytes=active,
            max_ws_bytes=limit,
            observed_transient_bytes=99 * GIB,
            min_margin_bytes=99 * GIB,
            chunk_start=0,
            chunk_end=1,
        )

    def test_projection_uses_the_larger_of_observed_and_floor(self):
        assert project_chunk_peak_bytes(10 * GIB, 8 * GIB, 2 * GIB) == 10 * GIB + int(
            8 * GIB * 1.25
        )
        assert project_chunk_peak_bytes(10 * GIB, 0, 2 * GIB) == 12 * GIB

    def test_message_avoids_every_cache_corruption_pattern(self):
        """Matching one would route this into cache-clear + reschedule, forever."""
        from vmlx_engine.scheduler import CACHE_CORRUPTION_PATTERNS

        with pytest.raises(PrefillAdmissionError) as excinfo:
            prefill_valve_check(
                active_bytes=80 * GIB,
                max_ws_bytes=81 * GIB,
                observed_transient_bytes=8 * GIB,
                min_margin_bytes=2 * GIB,
                chunk_start=0,
                chunk_end=1,
            )
        message = str(excinfo.value)
        for pattern in CACHE_CORRUPTION_PATTERNS:
            assert pattern not in message, f"message contains {pattern!r}"


class TestTriage:
    def test_real_metal_overmax_is_permanent(self):
        assert classify_allocation_error(REAL_METAL_OVERMAX) == "permanent"
        assert is_permanent_allocation_error(RuntimeError(REAL_METAL_OVERMAX))

    def test_admission_error_is_permanent(self):
        assert is_permanent_allocation_error(PrefillAdmissionError("nope"))

    @pytest.mark.parametrize(
        "message",
        [
            "MTLCommandBuffer execution failed: out of memory",
            "Cannot allocate memory",
            "Allocation failed for buffer",
            "[metal::malloc] something else went wrong",
        ],
    )
    def test_transient_exhaustion_stays_recoverable(self, message):
        assert classify_allocation_error(message) == "transient"
        assert not is_permanent_allocation_error(message)

    @pytest.mark.parametrize(
        "message",
        ["shape mismatch", "'NoneType' object is not subscriptable", ""],
    )
    def test_unrelated_errors_are_not_classified(self, message):
        assert classify_allocation_error(message) is None


class TestSchedulerIntegration:
    def test_permanent_error_is_not_treated_as_cache_corruption(self):
        """Otherwise the recovery path re-runs the doomed allocation in a loop."""
        from vmlx_engine.scheduler import Scheduler

        probe = Scheduler.__new__(Scheduler)
        assert (
            Scheduler._is_cache_corruption_error(probe, RuntimeError(REAL_METAL_OVERMAX))
            is False
        )

    def test_transient_error_is_still_recoverable(self):
        from vmlx_engine.scheduler import Scheduler

        probe = Scheduler.__new__(Scheduler)
        assert (
            Scheduler._is_cache_corruption_error(
                probe, RuntimeError("MTLCommandBuffer failed: out of memory")
            )
            is True
        )
