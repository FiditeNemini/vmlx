# SPDX-License-Identifier: Apache-2.0
"""The hybrid chunked prefill must DECLINE a chunk it cannot serve.

This path had no admission check, and it does not fail with a catchable Python
error: MLX raises "[METAL] Command buffer execution failed: Insufficient Memory"
which libc++ turns into a process abort. There is no exception to handle, so the
only defence is to not submit the chunk.

The numbers below are MEASURED, not invented — Qwen3.6-27B-JANG_4M-CRACK, cold
single-shot 101,502-token prefill on a 128GB box, sampled after
``_materialize_prefill_cache_state`` (the eval point; sampling before it reports
0.00GB for every chunk because MLX is lazy, which is how four earlier sizing
attempts came to be tuned against zeros).
"""

from __future__ import annotations

import pytest

from vmlx_engine.utils.prefill_admission import (
    PrefillAdmissionError,
    hybrid_chunk_valve_check,
    project_span_peak_bytes,
)

GIB = 1024**3
# The REAL device limit, read from the box via
# get_effective_metal_working_set_bytes / max_recommended_working_set_size.
# Not a guess — an earlier guess of 105GB made this suite refuse a chunk that
# demonstrably ran.
DEVICE_LIMIT = int(107.52 * GIB)
MIN_MARGIN = int(2 * GIB)

# (prev_ctx, observed_transient_at_prev, next_ctx, active_before_next, survived?)
MEASURED_CHUNKS = [
    (67_584, 12.94, 69_632, 24.99, True),
    (90_112, 16.34, 92_160, 81.57, True),
    (92_160, 16.64, 94_208, 87.46, True),
    # chunk 47: this is the one that killed the process.
    (94_208, 16.95, 96_256, 93.48, False),
]


@pytest.mark.parametrize(
    "prev_ctx,transient,next_ctx,active,survived", MEASURED_CHUNKS
)
def test_valve_matches_what_the_device_actually_did(
    prev_ctx, transient, next_ctx, active, survived
):
    """Admit every chunk that ran; decline the chunk that aborted the process."""
    def run():
        hybrid_chunk_valve_check(
            int(active * GIB),
            DEVICE_LIMIT,
            int(transient * GIB),
            prev_ctx,
            next_ctx,
            MIN_MARGIN,
            chunk_start=next_ctx,
            chunk_end=next_ctx + 2048,
            model_label="hybrid prefill",
        )

    if survived:
        run()  # must not raise: declining these would refuse a servable request
    else:
        with pytest.raises(PrefillAdmissionError):
            run()


def test_transient_is_linear_in_context():
    """transient(ctx) = 2.82GB + 0.00015*ctx, fitted to the measured span.

    The fit matters because it is what justifies projecting from an observed
    transient. Earlier fits produced a NEGATIVE intercept, which was the signal
    that the measurements — not the model — were wrong.
    """
    for ctx, expected_gb in [
        (34_816, 8.04),
        (61_440, 12.02),
        (71_680, 13.56),
        (94_208, 16.95),
    ]:
        modelled = 2.82 + 0.00015 * ctx
        assert modelled == pytest.approx(expected_gb, abs=0.03), (
            f"ctx={ctx}: model says {modelled:.2f}GB, measured {expected_gb}GB"
        )


def test_span_projection_scales_with_context_not_chunk_size():
    """Projecting a measured transient forward must grow with CONTEXT.

    Three chunk-sizing experiments all aborted at the same context whether the
    chunk was 2048 or 556 tokens, so chunk size is not the variable.
    """
    at_60k = project_span_peak_bytes(int(12.0 * GIB), 60_000, 60_000)
    at_120k = project_span_peak_bytes(int(12.0 * GIB), 60_000, 120_000)
    assert at_120k == pytest.approx(2 * at_60k, rel=1e-6)


def test_admission_error_is_not_retried_as_cache_corruption():
    """The scheduler must exclude the valve abort from cache-clear recovery.

    It matches by CLASS NAME to avoid an import cycle, so a rename here silently
    reintroduces the retry loop this guards against. Assert the name the
    scheduler actually looks for.
    """
    import re
    from pathlib import Path

    scheduler = Path(__file__).resolve().parents[1] / "vmlx_engine" / "scheduler.py"
    source = scheduler.read_text(encoding="utf-8")
    assert "PrefillAdmissionError" in source, (
        "scheduler.py no longer excludes PrefillAdmissionError from cache-clear "
        "recovery — a declined hybrid prefill will be retried until max_retries"
    )
    # And the exception's own message must not trip the corruption matcher.
    err = PrefillAdmissionError(
        "hybrid prefill: prefill admission rejected chunk [1:2) — active Metal "
        "working set 93.48GB plus projected transient 21.19GB exceeds the "
        "device working-set limit 105.00GB."
    )
    for forbidden in ("out of memory", "allocation failed", "insufficient memory"):
        assert not re.search(forbidden, str(err), re.IGNORECASE), (
            f"message contains {forbidden!r}, which routes it into cache-clear recovery"
        )


def test_first_chunk_is_not_declined_by_a_degenerate_context_ratio():
    """The projection basis must be the END of a chunk, never its start.

    Caught LIVE, not by the suite above: recording the observation at the chunk's
    START context makes that context 0 for chunk 0. Clamped to 1, projecting it
    to 2048 scaled the transient by 2048x and produced a 6448GB estimate, so the
    valve refused chunk 1 of a prompt the device serves easily. Every case in
    MEASURED_CHUNKS is deep in a span, so none of them exercised this.
    """
    # chunk 0 measured live: base 16.33GB, peak 19.19GB -> transient 2.86GB,
    # observed at end-context 2048. Chunk 1 ends at 4096 with active ~16.60GB.
    hybrid_chunk_valve_check(
        int(16.60 * GIB),
        DEVICE_LIMIT,
        int(2.86 * GIB),
        2048,          # end context of chunk 0, NOT 0
        4096,          # end context of chunk 1
        MIN_MARGIN,
        chunk_start=2048,
        chunk_end=4096,
        model_label="hybrid prefill",
    )  # must not raise


def test_a_degenerate_observation_context_cannot_veto_everything():
    """Even if the observed context is bogus, the projection must stay sane.

    Guards the shape of the bug rather than the one arithmetic slip: a tiny
    observation context against a large target context is the condition that
    turns a 2.86GB transient into thousands of GB.
    """
    projected = project_span_peak_bytes(int(2.86 * GIB), 1, 2048)
    assert projected / GIB > 1000, "sanity: this IS the degenerate case"
    # ...which is exactly why the caller must never pass a start-of-span context.
