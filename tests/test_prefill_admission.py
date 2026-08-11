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


class TestChunkedSsmRederiveGate:
    """The chunk-safety rule for recurrent caches, and its opt-in override.

    Recurrent (SSM) slots force a ONE-SHOT clean re-derive, which the
    O(seq_len^2) memory guard then rejects past ~12k — so hybrid families reuse
    only the FIRST turn's blocks and re-prefill O(context) forever after
    (measured on Qwen3.6: cached frozen at 12,217 while TTFT grew 20.2s ->
    105.9s).

    The override exists to make that testable. It stays default OFF because the
    risk is asymmetric: a wrong recurrent state is STORED, not just used once,
    and these models collapse into token loops rather than failing loudly.
    """

    def test_recurrent_slots_force_one_shot_by_default(self):
        from vmlx_engine.mllm_batch_generator import _cache_requires_one_shot_rederive

        class _Recurrent:
            pass

        assert _cache_requires_one_shot_rederive([_Recurrent()]) is True

    def test_unclassifiable_slots_are_treated_as_recurrent(self):
        """Guessing chunk-safe would store silently wrong state."""
        from vmlx_engine.mllm_batch_generator import _cache_requires_one_shot_rederive

        assert _cache_requires_one_shot_rederive([object()]) is True

    def test_override_is_off_by_default(self):
        from vmlx_engine.mllm_batch_generator import _CHUNKED_SSM_REDERIVE

        assert _CHUNKED_SSM_REDERIVE is False


class TestChunkAttentionClamp:
    """A chunked prefill computes chunk x CONTEXT scores, not chunk^2.

    So the per-chunk buffer grows with the conversation even though the step
    size is fixed, and a step that is safe early becomes fatal later. MEASURED
    on Qwen3.6-27B: at a 67,292-token context the prefill chunked correctly at
    the configured step and the PROCESS STILL DIED with a Metal command-buffer
    OOM — 30 heads x 2048 x 67292 x 2 = 8.3 GB in one chunk. libc++ terminates
    on that, so there is nothing to catch; the chunk has to fit before it is
    submitted.
    """

    def test_clamp_shrinks_as_context_grows(self):
        from vmlx_engine.utils.prefill_admission import max_prefill_chunk_tokens

        caps = [max_prefill_chunk_tokens(30, ctx) for ctx in (10_000, 67_292, 300_000)]
        assert caps == sorted(caps, reverse=True), caps
        assert all(cap >= 1 for cap in caps)

    def test_the_measured_failure_would_now_be_clamped(self):
        from vmlx_engine.utils.prefill_admission import max_prefill_chunk_tokens

        # The exact configuration that aborted the process.
        assert max_prefill_chunk_tokens(30, 67_292) < 2048

    def test_a_short_context_is_not_penalised(self):
        from vmlx_engine.utils.prefill_admission import max_prefill_chunk_tokens

        assert max_prefill_chunk_tokens(30, 4_000) > 2048

    def test_degenerate_inputs_never_return_zero(self):
        """A zero chunk size would hang the prefill loop forever."""
        from vmlx_engine.utils.prefill_admission import max_prefill_chunk_tokens

        for heads, ctx in ((0, 0), (0, 10), (10, 0), (1, 10**9), (10**6, 10**6)):
            assert max_prefill_chunk_tokens(heads, ctx) >= 1

    def test_command_buffer_oom_is_flagged_process_fatal(self):
        from vmlx_engine.utils.prefill_admission import (
            is_process_fatal_allocation_signature,
        )

        assert is_process_fatal_allocation_signature(
            "[METAL] Command buffer execution failed: Insufficient Memory "
            "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
        )
        # metal::malloc is catchable and must NOT be lumped in with it
        assert not is_process_fatal_allocation_signature(
            "[metal::malloc] Attempting to allocate 4398046511104 bytes"
        )


class TestSpanAdmissionIsNotYetSufficient:
    """Whole-span admission: the idea is right, the linear model is NOT.

    Per-chunk admission cannot save the hybrid abort (three implementations,
    all failed at the same 73-75k context). Deciding once on the whole span
    before starting it is the right shape — it turns a process abort into a
    clean per-request error.

    But a transient that scales LINEARLY with context does not reproduce the
    observed failure, and this test pins that so nobody wires it up believing
    it works. Fed the real measurements it ADMITS the span that actually died.
    """

    def test_linear_projection_admits_the_known_fatal_span(self):
        from vmlx_engine.utils.prefill_admission import span_admission_check

        GIB = 1024**3
        # Real numbers: 18.32GB transient measured at ~65k context, span ran to
        # a final context of ~100,935, weights+cache baseline ~21GB, ~107GB limit.
        span_admission_check(
            active_bytes=21 * GIB,
            max_ws_bytes=107 * GIB,
            measured_transient_bytes=int(18.32 * GIB),
            measured_at_context=65_536,
            final_context=100_935,
            fresh_tokens=67_292,
            model_label="qwen3_5",
        )
        # No exception == admitted == would NOT have prevented the crash.
        # The observed active at that point was ~95GB, not the ~56GB this
        # projects, so the real growth is not linear in context.

    def test_it_does_decline_when_the_projection_is_large_enough(self):
        """The mechanism itself works; only the growth model is wrong."""
        from vmlx_engine.utils.prefill_admission import (
            PrefillAdmissionError,
            span_admission_check,
        )

        GIB = 1024**3
        with pytest.raises(PrefillAdmissionError) as excinfo:
            span_admission_check(
                active_bytes=80 * GIB,
                max_ws_bytes=107 * GIB,
                measured_transient_bytes=30 * GIB,
                measured_at_context=50_000,
                final_context=100_000,
                fresh_tokens=50_000,
                model_label="qwen3_5",
            )
        assert "cannot be served on this hardware" in str(excinfo.value)

    def test_unknown_readings_never_decline(self):
        from vmlx_engine.utils.prefill_admission import span_admission_check

        GIB = 1024**3
        for active, limit, transient in ((0, 107 * GIB, GIB), (21 * GIB, 0, GIB), (21 * GIB, 107 * GIB, 0)):
            span_admission_check(
                active_bytes=active,
                max_ws_bytes=limit,
                measured_transient_bytes=transient,
                measured_at_context=1000,
                final_context=100_000,
                fresh_tokens=99_000,
            )
