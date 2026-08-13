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


class TestSpanAdmission:
    """Whole-span admission: decide once, before burning the span.

    Per-chunk admission cannot save the hybrid abort cheaply (three
    implementations, all failed at the same 73-75k context). Deciding once on
    the whole span before starting it turns a process abort into a clean
    per-request error.

    HISTORY, because it is the whole reason this is shaped the way it is: the
    first version projected the per-chunk TRANSIENT and added the active
    reading taken at span START, and the test here pinned that it ADMITTED the
    span that actually died — so it was left deliberately unwired, with tests
    passing, for exactly that reason. Active is not constant across a span; it
    climbed from ~21GB to ~95GB as KV accumulated. Fitting the measured
    ABSOLUTE peak against context removes the term that was wrong, and the same
    measurements now decline. See fit_peak_model's docstring.
    """

    def test_peak_model_declines_the_known_fatal_span(self):
        """The regression this whole check exists for.

        The previous transient-plus-start-active formulation ADMITTED this span
        — it projected ~56GB against a 107GB limit for a prefill whose observed
        peak reached ~95GB and killed the process. Fitting the measured absolute
        peak instead declines it.

        Real numbers, Qwen3.6-27B: peak climbs roughly 74.4GB at 65,536 tokens
        to ~123GB at the final context of 100,935.
        """
        from vmlx_engine.utils.prefill_admission import (
            PrefillAdmissionError,
            fit_peak_model,
            span_admission_check,
        )

        GIB = 1024**3
        model = fit_peak_model(
            [(65_536, int(74.4 * GIB)), (80_000, int(94.3 * GIB))]
        )
        assert model is not None
        with pytest.raises(PrefillAdmissionError) as excinfo:
            span_admission_check(
                max_ws_bytes=107 * GIB,
                peak_model=model,
                final_context=100_935,
                fresh_tokens=67_292,
                model_label="qwen3_5",
            )
        assert "cannot be served on this hardware" in str(excinfo.value)

    def test_it_admits_a_span_that_fits(self):
        """The valve must not reject what the device actually serves."""
        from vmlx_engine.utils.prefill_admission import (
            fit_peak_model,
            span_admission_check,
        )

        GIB = 1024**3
        model = fit_peak_model(
            [(10_000, int(24.0 * GIB)), (20_000, int(26.0 * GIB))]
        )
        assert model is not None
        # Projects ~32GB at 50k, comfortably under a 107GB limit.
        span_admission_check(
            max_ws_bytes=107 * GIB,
            peak_model=model,
            final_context=50_000,
            fresh_tokens=30_000,
            model_label="qwen3_5",
        )

    def test_unknown_readings_never_decline(self):
        from vmlx_engine.utils.prefill_admission import span_admission_check

        GIB = 1024**3
        flat = (float(200 * GIB), 0.0)  # slope <= 0: nothing to extrapolate
        rising = (0.0, float(GIB))
        for limit, model in (
            (0, rising),           # no device limit exposed
            (107 * GIB, None),     # no fit learned yet (first span)
            (107 * GIB, flat),     # peak not growing with context
        ):
            span_admission_check(
                max_ws_bytes=limit,
                peak_model=model,
                final_context=100_000,
                fresh_tokens=99_000,
            )

    def test_fit_needs_two_distinct_contexts(self):
        from vmlx_engine.utils.prefill_admission import fit_peak_model

        assert fit_peak_model([]) is None
        assert fit_peak_model([(1000, 500)]) is None
        # Same context twice cannot determine a slope.
        assert fit_peak_model([(1000, 500), (1000, 900)]) is None
        assert fit_peak_model([(1000, 500), (2000, 900)]) is not None

    def test_fit_recovers_a_known_affine_model(self):
        from vmlx_engine.utils.prefill_admission import (
            fit_peak_model,
            project_peak_affine,
        )

        GIB = 1024**3
        # peak = 20GB + 0.0005GB/token, sampled at four contexts.
        samples = [
            (ctx, int((20.0 + 0.0005 * ctx) * GIB))
            for ctx in (10_000, 20_000, 40_000, 80_000)
        ]
        fitted = fit_peak_model(samples)
        assert fitted is not None
        projected = project_peak_affine(fitted[0], fitted[1], 100_000)
        assert abs(projected - int(70.0 * GIB)) < int(0.1 * GIB)

    def test_refuses_to_extrapolate_far_past_measured_range(self):
        """A false decline breaks a working request; a missed decline only costs
        wasted work, because the per-chunk valve still backstops it."""
        from vmlx_engine.utils.prefill_admission import (
            PrefillAdmissionError,
            fit_peak_model,
            span_admission_check,
        )

        GIB = 1024**3
        # Fit measured only up to 20k context, but with a steep slope, so a naive
        # projection to 500k would decline.
        model = fit_peak_model(
            [(10_000, int(24.0 * GIB)), (20_000, int(40.0 * GIB))]
        )
        assert model is not None
        # Sanity: without the range bound this WOULD decline.
        with pytest.raises(PrefillAdmissionError):
            span_admission_check(
                max_ws_bytes=107 * GIB,
                peak_model=model,
                final_context=500_000,
                fresh_tokens=480_000,
                fitted_max_context=0,  # unknown range = old behaviour
            )
        # With the measured range known, 500k is 25x past it — defer instead.
        span_admission_check(
            max_ws_bytes=107 * GIB,
            peak_model=model,
            final_context=500_000,
            fresh_tokens=480_000,
            fitted_max_context=20_000,
        )
        # Just inside the bound it still declines.
        with pytest.raises(PrefillAdmissionError):
            span_admission_check(
                max_ws_bytes=107 * GIB,
                peak_model=model,
                final_context=70_000,
                fresh_tokens=50_000,
                fitted_max_context=20_000,
            )

    def test_projection_never_falls_below_an_observed_peak(self):
        """A fit is an estimate; a measurement is not."""
        from vmlx_engine.utils.prefill_admission import project_peak_affine

        GIB = 1024**3
        assert project_peak_affine(0.0, 1.0, 10, floor_bytes=5 * GIB) == 5 * GIB


class TestMuseReasoningEffortParity:
    """The API path must reach the same depth control the panel already sets.

    Muse's template reads ONLY `reasoning_strength`; it ignores enable_thinking
    and reasoning_effort. The panel translates the UI level in its request
    builder (panel/src/main/ipc/chat.ts), so a UI user could steer reasoning
    depth while a plain API caller sending the standard reasoning_effort
    silently could not. MEASURED before the fix, temperature 0, same prompt:
    high and xhigh were BYTE-IDENTICAL (1073-char reasoning, 287-char answer),
    consistent with the template ignoring the field and defaulting to high.
    """

    def _merge(self, monkeypatch, effort, kwargs=None, model="Muse-Glimmer-30B-JANG_4M-CRACK"):
        from vmlx_engine import server

        monkeypatch.setattr(server, "_model_path", model, raising=False)
        monkeypatch.setattr(server, "_model_name", model, raising=False)
        monkeypatch.setattr(server, "_default_chat_template_kwargs", None, raising=False)
        return server._merge_ct_kwargs(kwargs, effort)

    def test_effort_becomes_reasoning_strength_for_muse(self, monkeypatch):
        assert self._merge(monkeypatch, "xhigh")["reasoning_strength"] == "xhigh"
        assert self._merge(monkeypatch, "low")["reasoning_strength"] == "low"
        # "max" is the OpenAI-ish top level; Muse's top is xhigh.
        assert self._merge(monkeypatch, "max")["reasoning_strength"] == "xhigh"

    def test_explicit_kwarg_wins_over_effort(self, monkeypatch):
        """A caller who names the kwarg directly meant it."""
        out = self._merge(monkeypatch, "low", {"reasoning_strength": "xhigh"})
        assert out["reasoning_strength"] == "xhigh"

    def test_unknown_effort_is_not_guessed(self, monkeypatch):
        assert "reasoning_strength" not in self._merge(monkeypatch, "banana")
        assert "reasoning_strength" not in self._merge(monkeypatch, None)

    def test_other_families_are_untouched(self, monkeypatch):
        """Only families that steer on reasoning_strength get the translation."""
        out = self._merge(monkeypatch, "xhigh", model="DSV4-Flash-JANG_2L")
        assert "reasoning_strength" not in out


class TestAtemStreamingEmitsEveryBlock:
    """A multi-step tool loop must emit EVERY completed call block.

    The streaming extractor used to return None for the whole rest of the turn
    once one `</atem:function_calls>` closed, so a turn with several blocks
    emitted the first and dropped the rest. Unemitted markup falls through to
    the renderer: MEASURED on Muse-Glimmer-30B, ~26 raw blocks (39 `<atem:`
    occurrences) were visible in the transcript.
    """

    BLOCK1 = (
        '<atem:function_calls>\n'
        '<atem:invoke name="create_directory">\n'
        '<atem:parameter name="path">/tmp/vmlx-tooltest</atem:parameter>\n'
        '</atem:invoke>\n'
        '</atem:function_calls>'
    )
    # Embedded double quotes inside the parameter — the real payload that leaked.
    BLOCK2 = (
        '<atem:function_calls>\n'
        '<atem:invoke name="run_applescript">\n'
        '<atem:parameter name="script">do shell script "ls -la /tmp/vmlx-tooltest"</atem:parameter>\n'
        '</atem:invoke>\n'
        '</atem:function_calls>'
    )

    def _parser(self):
        from vmlx_engine.tool_parsers.atem_tool_parser import AtemToolParser

        return AtemToolParser.__new__(AtemToolParser)

    def _stream(self, previous, current):
        return self._parser().extract_tool_calls_streaming(
            previous, current, "", [], [], [], None
        )

    def test_first_block_emits(self):
        out = self._stream("", self.BLOCK1)
        assert out is not None and out.tool_calls
        assert out.tool_calls[0]["name"] == "create_directory"

    def test_second_block_also_emits(self):
        """The regression: this returned None, so block 2 leaked as raw text."""
        out = self._stream(self.BLOCK1, self.BLOCK1 + "\n" + self.BLOCK2)
        assert out is not None, "second completed block was dropped"
        names = [c["name"] for c in out.tool_calls]
        assert names == ["run_applescript"], names

    def test_no_reemit_when_nothing_new_completed(self):
        """Emit-once still holds — a delta that closes no new block emits nothing."""
        assert self._stream(self.BLOCK1, self.BLOCK1 + "\nsome trailing prose") is None

    def test_partial_second_block_waits(self):
        partial = self.BLOCK1 + '\n<atem:function_calls>\n<atem:invoke name="list_directory">'
        assert self._stream(self.BLOCK1, partial) is None

    def test_embedded_quotes_survive(self):
        out = self._stream(self.BLOCK1, self.BLOCK1 + "\n" + self.BLOCK2)
        args = str(out.tool_calls[0]["arguments"])
        assert 'ls -la /tmp/vmlx-tooltest' in args


class TestAtemMarkupSuppressedFromVisibleText:
    """Muse was the ONE family missing from the visible-text suppression list.

    `_TOOL_CALL_MARKERS` is what keeps native tool-control payload out of what
    the user reads. It carried entries for tool_call, zyphra, minimax, DSML,
    gemma4, python_tag and more — but nothing for Muse's ATEM dialect. MEASURED
    live: a single three-call turn added 6 raw <atem:function_calls> blocks to
    the rendered transcript (39 -> 49 occurrences of "<atem:") while the calls
    themselves parsed and executed correctly.
    """

    def test_atem_is_in_the_marker_list(self):
        from vmlx_engine.server import _TOOL_CALL_MARKERS

        assert any("atem" in m.lower() for m in _TOOL_CALL_MARKERS)

    def test_atem_block_is_detected_as_tool_markup(self):
        from vmlx_engine.server import _has_tool_marker_or_partial_suffix

        payload = (
            'Sure.\n<atem:function_calls>\n<atem:invoke name="run_applescript">\n'
            '<atem:parameter name="script">do shell script "ls -la /tmp"</atem:parameter>\n'
            '</atem:invoke>\n</atem:function_calls>'
        )
        assert _has_tool_marker_or_partial_suffix(payload)

    def test_bare_invoke_without_wrapper_is_detected(self):
        """A block split across deltas can show the invoke tag alone."""
        from vmlx_engine.server import _has_tool_marker_or_partial_suffix

        assert _has_tool_marker_or_partial_suffix('<atem:invoke name="list_directory">')

    def test_ordinary_prose_is_not_flagged(self):
        """The suppression must not swallow normal answers."""
        from vmlx_engine.server import _has_tool_marker_or_partial_suffix

        assert not _has_tool_marker_or_partial_suffix(
            "The sky is blue because of Rayleigh scattering."
        )
        # A word merely containing the substring must not trip it.
        assert not _has_tool_marker_or_partial_suffix("we discussed the atem dialect")


class TestBlockDiskMissReasons:
    """A bare disk_misses count cannot separate "never stored" from "stored but
    unreadable/invalid" — and those need opposite fixes.

    MEASURED a live session with 0 disk_hits against 72 disk_misses where the
    cause was indistinguishable from the counters alone; every L2 lookup failed
    and nothing said why.
    """

    def _store(self):
        from vmlx_engine.block_disk_store import BlockDiskStore

        return BlockDiskStore.__new__(BlockDiskStore)

    def test_every_miss_path_has_a_reason_bucket(self):
        """The four buckets must cover the reasons the code actually records."""
        import inspect
        import re

        from vmlx_engine import block_disk_store

        src = inspect.getsource(block_disk_store)
        used = set(re.findall(r'_note_miss\("([a-z_]+)"\)', src))
        declared = {"absent", "load_failed", "budget_locked", "validation"}
        assert used, "no _note_miss call sites found — the breakdown is dead code"
        assert used <= declared, f"reason with no bucket: {used - declared}"

    def test_note_miss_counts_and_ignores_unknown(self):
        store = self._store()
        store.disk_miss_reasons = {
            "absent": 0, "load_failed": 0, "budget_locked": 0, "validation": 0,
        }
        store._note_miss("absent")
        store._note_miss("absent")
        store._note_miss("validation")
        # An unrecognised reason must not create a bucket or raise — telemetry
        # must never be able to break a cache read.
        store._note_miss("something_new")
        assert store.disk_miss_reasons["absent"] == 2
        assert store.disk_miss_reasons["validation"] == 1
        assert "something_new" not in store.disk_miss_reasons

    def test_every_disk_misses_increment_is_attributed(self):
        """A miss site without a _note_miss beside it is an unexplained miss."""
        import inspect

        from vmlx_engine import block_disk_store

        lines = inspect.getsource(block_disk_store).split("\n")
        unattributed = [
            i for i, l in enumerate(lines)
            if "self.disk_misses += 1" in l
            and "_note_miss" not in "\n".join(lines[i + 1:i + 3])
        ]
        assert not unattributed, f"disk_misses incremented with no reason at lines {unattributed}"


class TestBlockDiskWriteDropReasons:
    """Why a block was never WRITTEN — the other half of the L2 story.

    Live telemetry proved every L2 miss is "absent" (blocks never stored), so
    the write path is where the answer lives. `full` on a fast family is the
    known ~33%-drop defect: the admission timeout is 0 for non-path-dependent
    families, so a momentarily full FIFO drops the block outright.
    """

    def test_reasons_match_the_admission_outcomes(self):
        """Every non-success outcome the admission helper can return must have
        a bucket, or a drop would be counted nowhere."""
        import inspect
        import re

        from vmlx_engine import block_disk_store

        src = inspect.getsource(block_disk_store)
        # Outcomes returned by the queue-admission helpers.
        returned = set(re.findall(r'return "(full|quiescing|writer_stopped)"', src))
        declared = {"full", "quiescing", "writer_stopped"}
        assert returned, "admission helper returned no recognisable outcomes"
        assert returned <= declared, f"outcome with no bucket: {returned - declared}"

    def test_drop_site_records_the_reason(self):
        """The drop path must count, not just log — a warning is not telemetry."""
        import inspect

        from vmlx_engine import block_disk_store

        src = inspect.getsource(block_disk_store)
        idx = src.find("dropping block write")
        assert idx > 0, "drop site not found"
        window = src[max(0, idx - 500):idx]
        assert "write_drop_reasons" in window, "drop site does not record a reason"

    def test_reasons_are_exported_next_to_the_miss_breakdown(self):
        import inspect

        from vmlx_engine import block_disk_store

        src = inspect.getsource(block_disk_store)
        assert '"write_drop_reasons"' in src, "write_drop_reasons never exported to stats"
        assert '"disk_miss_reasons"' in src, "disk_miss_reasons never exported to stats"


    def test_byte_budget_drops_are_in_the_same_structure(self):
        """One structure must answer "why was this block not stored".

        MEASURED: a saturating run showed write_drop_reasons all zero while
        byte_budget_drops was 2 — the real drop path was outside the breakdown,
        which is exactly the blind spot the breakdown exists to remove.
        """
        import inspect

        from vmlx_engine import block_disk_store

        src = inspect.getsource(block_disk_store)
        idx = src.find("_pending_write_byte_drops += 1")
        assert idx > 0, "byte-budget drop site not found"
        # Every byte-budget increment must also record into write_drop_reasons.
        count_sites = src.count("_pending_write_byte_drops += 1")
        count_mirrored = src.count('write_drop_reasons["byte_budget"]')
        assert count_mirrored >= count_sites, (
            f"{count_sites} byte-budget drop sites but only {count_mirrored} recorded"
        )
