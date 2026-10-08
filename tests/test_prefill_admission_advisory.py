"""ISSUES I-36 (Eric, 2026-10-08): prefill admission estimates ADVISE, never REFUSE."""
import logging

from vmlx_engine.utils.prefill_admission import (
    PrefillAdmissionError,
    advise_admission,
    hybrid_chunk_valve_check,
)


def test_advisory_logs_the_projection_and_does_not_raise(caplog):
    caplog.set_level(logging.WARNING)
    err = PrefillAdmissionError("hybrid prefill: prefill admission rejected chunk [8192:8256)")
    advise_admission(err, request_id="req-1")  # must return normally
    assert "Prefill admission ADVISORY" in caplog.text
    assert "never refuse" in caplog.text
    assert "request=req-1" in caplog.text
    assert "[8192:8256)" in caplog.text


def test_projection_functions_still_compute_the_decline_for_adaptation():
    # The check itself still raises so the per-chunk path can HALVE and retry
    # (adaptation); only the request-failing outcome became an advisory.
    gib = 1024**3
    try:
        hybrid_chunk_valve_check(
            int(96.56 * gib), int(107.5 * gib), 8 * gib, 4096, 8256, 0,
            chunk_start=8192, chunk_end=8256,
        )
    except PrefillAdmissionError:
        pass
    else:
        raise AssertionError("projection over the limit must still be detectable")
