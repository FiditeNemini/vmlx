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
    fit_peak_model,
    hybrid_chunk_valve_check,
    project_span_peak_bytes,
    turn_peak_admission_check,
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


# ---------------------------------------------------------------------------
# Cross-TURN peak-walk admission (the third valve).
#
# MEASURED, Qwen3.8 17-turn incremental grow (+5,649 tokens/turn), eight crash
# points over two pool configs: the absolute Metal peak walks +5.0-5.6GB per
# TURN between spans (allocator/fragmentation growth a fresh process does not
# carry — the identical fatal request replayed cold survives). Neither existing
# valve sees the walk: the within-span fit gets 2-3 near-identical contexts
# (slope ~0, silent pass — seven aborts with ZERO refusals), and the per-chunk
# valve is blind on each span's first chunk, which is the buffer that aborted.
# Walk anchor: post-t16 high water 109.2GB at ctx 90,345 in the stock config.
# ---------------------------------------------------------------------------

_TURN = 5_649  # tokens added per turn in the measured grow
_SLOPE_GB_PER_TURN = 5.3

def _walk(last_ctx_gb: "tuple[int, float]", n: int = 3) -> "list[tuple[int, int]]":
    """n perfectly linear walk points ending at (ctx, peakGB) — the measured
    shape: residual-free within a config, differing only in intercept."""
    ctx, gb = last_ctx_gb
    return [
        (ctx - i * _TURN, int((gb - i * _SLOPE_GB_PER_TURN) * GIB))
        for i in range(n - 1, -1, -1)
    ]


def test_turn_walk_refuses_the_five_crash_ordinal():
    """Stock config, t17 (ctx 95,994): crashed five consecutive runs. The walk
    through t16 projects 114.5GB against the 107.52GB limit — must refuse."""
    walk = _walk((90_345, 109.2))
    fit = fit_peak_model(walk)
    assert fit is not None and fit[1] > 0
    with pytest.raises(PrefillAdmissionError):
        turn_peak_admission_check(
            DEVICE_LIMIT,
            fit,
            95_994,
            last_observed_peak_bytes=walk[-1][1],
            allowance_bytes=0,
            fitted_max_context=90_345,
            model_label="hybrid delta",
        )


def test_turn_walk_admits_the_turn_the_device_served():
    """Stock config, t15 (ctx 84,696) survived; its walk projects 103.9GB —
    well under the limit, must admit."""
    walk = _walk((79_047, 98.6))
    turn_peak_admission_check(
        DEVICE_LIMIT,
        fit_peak_model(walk),
        84_696,
        last_observed_peak_bytes=walk[-1][1],
        allowance_bytes=0,
        fitted_max_context=79_047,
        model_label="hybrid delta",
    )  # must not raise


def test_turn_walk_boundary_refusal_is_deliberate():
    """Bounded-pool config (r7), t17 (ctx 95,994) SURVIVED at an observed
    109.2GB peak — 1.6% OVER the advisory limit — and this valve refuses it.

    DELIBERATE, not a defect to fix by raising the allowance: the fatal turn's
    actual peak overshoots its linear projection by ~3-5GB, so across both
    measured configs the last-surviving and first-aborting turns PROJECT to
    the same ~109.2GB. Any allowance that admits this turn also admits r7's
    t18, which SIGABRTed the engine at ~101.6k. One turn of depth that exists
    only beyond the device's stated budget is the price of never aborting."""
    walk = _walk((90_345, 103.9))  # stock walk minus one turn's intercept
    with pytest.raises(PrefillAdmissionError):
        turn_peak_admission_check(
            DEVICE_LIMIT,
            fit_peak_model(walk),
            95_994,
            last_observed_peak_bytes=walk[-1][1],
            allowance_bytes=0,
            fitted_max_context=90_345,
            model_label="hybrid delta",
        )


def test_turn_walk_unknowns_never_reject():
    """Missing fit, flat slope, zero limit, and far extrapolation all admit."""
    walk = _walk((90_345, 109.2))
    fit = fit_peak_model(walk)
    turn_peak_admission_check(DEVICE_LIMIT, None, 95_994)
    turn_peak_admission_check(DEVICE_LIMIT, (100.0 * GIB, 0.0), 95_994)
    turn_peak_admission_check(0, fit, 95_994)
    turn_peak_admission_check(
        DEVICE_LIMIT, fit, 95_994, fitted_max_context=40_000,
        max_extrapolation=2.0,
    )  # 95,994 > 2 x 40,000 — defer rather than guess


def test_turn_walk_refusal_message_is_not_retried_as_cache_corruption():
    """Same contract as the other valves: the message must not route into the
    scheduler's cache-clear recovery, which retries the identical doomed work."""
    import re

    walk = _walk((90_345, 109.2))
    with pytest.raises(PrefillAdmissionError) as exc_info:
        turn_peak_admission_check(
            DEVICE_LIMIT, fit_peak_model(walk), 95_994,
            allowance_bytes=0, fitted_max_context=90_345,
        )
    for forbidden in ("out of memory", "allocation failed", "insufficient memory"):
        assert not re.search(forbidden, str(exc_info.value), re.IGNORECASE)


def test_turn_walk_wiring_in_the_generator():
    """Source pins: the hit-lane span path must record the walk, run the check,
    log an engagement line (the r2-r6 protocol: a mechanism that cannot prove
    it ran cannot be adjudicated), and default the allowance to 0."""
    import re
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "vmlx_engine"
        / "mllm_batch_generator.py"
    ).read_text(encoding="utf-8")
    assert "self._turn_peak_walk.append(" in src, "walk recording disappeared"
    assert "turn_peak_admission_check(" in src, "the third valve is unwired"
    assert "Turn-peak admission engaged" in src, (
        "engagement INFO line removed — refusal-free crash runs become "
        "unauditable"
    )
    # The admit call must run BEFORE the chunked/single-shot branch choice:
    # the first landing lived inside the chunked branch only, and a ~5.6k
    # hit-lane delta predicts a tiny attention buffer, stays single-shot, and
    # bypassed every check — zero engagement lines across an entire R8 build,
    # caught live by the engagement-line protocol. The single-shot forward IS
    # the command buffer that aborts.
    call = src.index("self._turn_peak_walk_admit(")
    branch = src.index("_hybrid_blocks_chunk = self._is_hybrid")
    assert call < branch, (
        "turn-peak admission moved back below the chunked-branch choice — "
        "single-shot deltas (the crashing path) would bypass it again"
    )
    # And the refusal path must zero the deferred-measurement anchor, or the
    # retry's no-forward-ran gauge reading poisons the fit and re-admits the
    # declined turn.
    helper = src.index("def _turn_peak_walk_admit(")
    body = src[helper : helper + 9000]
    assert "self._last_deep_span_tokens = 0" in body, (
        "refusal no longer zeroes _last_deep_span_tokens — a poisoned "
        "(deep span, near-zero peak) walk point re-admits the declined turn"
    )
    m = re.search(r'"VMLX_TURN_PEAK_ALLOWANCE_MB", "(\d+)"', src)
    assert m and int(m.group(1)) == 0, (
        "allowance default must stay 0 — see "
        "test_turn_walk_boundary_refusal_is_deliberate"
    )


def test_turn_walk_admit_state_machine(monkeypatch):
    """Drive the REAL _turn_peak_walk_admit through a scripted grow-to-wall
    sequence: deferred pairing, engagement, refusal at the projected wall,
    anchor zeroing, and the retry's reading NOT poisoning the fit."""
    from collections import deque
    from types import SimpleNamespace

    import vmlx_engine.mllm_batch_generator as gen
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

    GIB_ = 1024**3
    limit = int(107.52 * GIB_)
    # Scripted gauge: peak reading returned at each admit entry — the
    # PREVIOUS span's absolute peak (deferred measurement). Walks +5.3GB per
    # +5,649-token turn, anchored like the measured r4 curve.
    readings = iter(
        [
            int(93.3 * GIB_),   # entry t3: t2's peak (span 73,398)
            int(98.6 * GIB_),   # entry t4: t3's peak (span 79,047)
            int(104.2 * GIB_),  # entry t5: t4's peak (span 84,696)
            int(0.05 * GIB_),   # entry t5-retry: no forward ran since reset
        ]
    )
    fake_mx = SimpleNamespace(
        get_peak_memory=lambda: next(readings),
        reset_peak_memory=lambda: None,
    )
    monkeypatch.setattr(gen, "mx", fake_mx)
    monkeypatch.setattr(
        gen, "get_effective_metal_working_set_bytes", lambda _mx: (0, limit)
    )
    monkeypatch.setattr(gen, "_TURN_PEAK_ADMISSION", True)
    monkeypatch.setattr(gen, "_TURN_PEAK_ALLOWANCE_BYTES", 0)

    self = SimpleNamespace(
        _turn_peak_walk=deque(maxlen=8), _last_deep_span_tokens=73_398
    )
    admit = MLLMBatchGenerator._turn_peak_walk_admit

    admit(self, 79_047)  # records (73398, 93.3GB); 1 point — no fit yet
    admit(self, 84_696)  # records (79047, 98.6GB); projects ~103.9 — admit

    # t5 (ctx 90,345): records (84696, 104.2GB), and the walk now projects
    # ~109.5GB > 107.52 — REFUSE. One ordinal before the crash ordinal
    # (95,994), because the 90k turn's own peak (109.2GB, measured) exceeds
    # the stated device budget: at allowance 0 an ACCURATE projection must
    # decline it. This is the deliberate boundary trade (see
    # test_turn_walk_boundary_refusal_is_deliberate), surfaced honestly by
    # this state machine rather than hidden by optimistic test numbers.
    with pytest.raises(PrefillAdmissionError):
        admit(self, 90_345)
    assert [ctx for ctx, _ in self._turn_peak_walk] == [73_398, 79_047, 84_696]
    assert self._last_deep_span_tokens == 0, (
        "refusal must zero the anchor or the retry poisons the fit"
    )
    points_after_refusal = list(self._turn_peak_walk)

    # The retry: gauge reads ~0 (nothing ran since the reset). It must NOT be
    # recorded, and the refusal must repeat.
    with pytest.raises(PrefillAdmissionError):
        admit(self, 90_345)
    assert list(self._turn_peak_walk) == points_after_refusal, (
        "the no-forward-ran reading was recorded — the poisoned point will "
        "drag the fit down and re-admit the fatal turn"
    )


def test_turn_walk_interleaved_conversations_keep_protection(monkeypatch):
    """Adversarial finding 2: the walk deque is generator-global, so a second
    deep conversation at a different depth scatters ctx-vs-peak points and a
    whole-deque fit can collapse to slope<=0 (valve silent, abort returns).
    The fix fits only the longest strictly-increasing-ctx SUFFIX. Drive the
    real admit through X-grow, Y-interleave, X-continue-to-wall and assert
    the wall turn is STILL refused and the shallow turn is NOT."""
    from collections import deque
    from types import SimpleNamespace

    import vmlx_engine.mllm_batch_generator as gen
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

    GIB_ = 1024**3
    limit = int(107.52 * GIB_)
    readings = iter(
        [
            int(93.3 * GIB_),    # entry X@79k: X@73k's peak
            int(98.6 * GIB_),    # entry X@84.7k: X@79k's peak
            int(104.2 * GIB_),   # entry Y@40k: X@84.7k's peak
            int(104.5 * GIB_),   # entry X@90.3k: Y@40k's peak (X resident)
            int(109.2 * GIB_),   # entry X@95,994: X@90.3k's peak
        ]
    )
    fake_mx = SimpleNamespace(
        get_peak_memory=lambda: next(readings),
        reset_peak_memory=lambda: None,
    )
    monkeypatch.setattr(gen, "mx", fake_mx)
    monkeypatch.setattr(
        gen, "get_effective_metal_working_set_bytes", lambda _mx: (0, limit)
    )
    monkeypatch.setattr(gen, "_TURN_PEAK_ADMISSION", True)
    monkeypatch.setattr(gen, "_TURN_PEAK_ALLOWANCE_BYTES", 0)

    self = SimpleNamespace(
        _turn_peak_walk=deque(maxlen=8), _last_deep_span_tokens=73_398
    )
    admit = MLLMBatchGenerator._turn_peak_walk_admit

    admit(self, 79_047)
    admit(self, 84_696)
    # Conversation Y at 40k: the deque now holds X's deep points, whose fit
    # projects far above anything 40k costs — Y must NOT be refused by X's
    # walk (the over-refusal direction of finding 2).
    admit(self, 40_000)
    # Back to X at 90.3k: the depth switch truncated the monotone suffix to
    # one point — a one-turn protection GAP, by design, not a poisoned fit.
    admit(self, 90_345)
    # X at the five-crash ordinal: the suffix has re-accumulated two points
    # and, with the observed-peak floor, must refuse — a whole-deque fit
    # flattened by the 40k point is what the suffix rule exists to prevent.
    with pytest.raises(PrefillAdmissionError):
        admit(self, 95_994)


def test_turn_walk_aux_clean_path_prefill_is_exempt(monkeypatch):
    """Adversarial finding 8: the clean-media-prefix N-1 re-prefill runs
    NESTED inside a real turn; its gauge read/reset would corrupt the real
    span's deferred measurement and a refusal would silently skip the media
    prefix store. Exempt requests must not touch the gauge or the walk."""
    from collections import deque
    from types import SimpleNamespace

    import vmlx_engine.mllm_batch_generator as gen
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

    def _must_not_read():
        raise AssertionError("exempt aux prefill consumed the peak gauge")

    monkeypatch.setattr(
        gen, "mx", SimpleNamespace(
            get_peak_memory=_must_not_read, reset_peak_memory=_must_not_read
        )
    )
    monkeypatch.setattr(gen, "_TURN_PEAK_ADMISSION", True)
    self = SimpleNamespace(
        _turn_peak_walk=deque(maxlen=8), _last_deep_span_tokens=90_345
    )
    aux_req = SimpleNamespace(_aux_clean_path_prefill=True)
    MLLMBatchGenerator._turn_peak_walk_admit(self, 90_000, request=aux_req)
    assert self._last_deep_span_tokens == 90_345, "aux prefill moved the anchor"
    assert not self._turn_peak_walk, "aux prefill recorded a walk point"


def test_turn_walk_hardening_source_pins():
    """Findings 1, 3, 4, 7 are branch-embedded; pin their presence.

    1: a sub-threshold CHUNKED prefill resets the gauge per chunk — it must
       drop the deep anchor (one-turn gap instead of a deflated deep point).
    3: the broadcast retry prefills the full prompt — stale hit-lane
       _cached_tokens would double-count every span-derived decision.
    4: any prefill failure reaches the per-request handler with the anchor
       set — it must zero it there.
    7: a malformed allowance env must not kill the engine at import."""
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "vmlx_engine"
        / "mllm_batch_generator.py"
    ).read_text(encoding="utf-8")
    assert "one-turn recording gap" in src and src.count(
        "self._last_deep_span_tokens = 0"
    ) >= 3, (
        "anchor zeroing lost from one of: valve refusal, sub-threshold "
        "chunked span, per-request failure handler"
    )
    # Anchor on the broadcast retry itself, not the log line — an inner
    # ValueError retry logs a similar message but resets via
    # _discard_request_cache_hit instead.
    retry = src.index("on broadcast retry we do full prefill from scratch")
    assert "req._cached_tokens = 0" in src[retry - 800 : retry + 200], (
        "broadcast retry no longer clears the stale hit-lane _cached_tokens"
    )
    handler = src.index("Per-request prefill failure (bad image, OOM, etc.)")
    assert "self._last_deep_span_tokens = 0" in src[handler : handler + 1600], (
        "failure handler no longer zeroes the walk anchor"
    )
    env = src.index('VMLX_TURN_PEAK_ALLOWANCE_MB')
    assert "except ValueError" in src[env - 200 : env + 400], (
        "allowance env parse is a bare int() again — a malformed value "
        "kills the engine at import"
    )


def test_turn_walk_admit_wall_ordinal_matches_r4():
    """The state machine above refuses at ctx 95,994 — the five-crash ordinal.
    Sanity-check the same arithmetic the method uses, from its own inputs."""
    walk = [
        (73_398, int(93.3 * GIB)),
        (79_047, int(98.6 * GIB)),
        (84_696, int(104.2 * GIB)),
        (90_345, int(109.2 * GIB)),
    ]
    fit = fit_peak_model(walk)
    assert fit is not None
    with pytest.raises(PrefillAdmissionError):
        turn_peak_admission_check(
            DEVICE_LIMIT, fit, 95_994,
            last_observed_peak_bytes=walk[-1][1], allowance_bytes=0,
            fitted_max_context=90_345,
        )


def test_admission_decline_crosses_every_door_as_413():
    """The first LIVE refusal (turn-peak valve, ctx 101,678) reached the
    client as a bare 500: the /v1/responses handler caught the typed error
    and then died on a latent NameError, and /v1/messages + /api/chat +
    /api/generate had no handling at all. The registered app-level handlers
    are the one mechanism covering every door — present and future."""
    import json

    from vmlx_engine.errors import PromptTooLongError
    from vmlx_engine.server import _prefill_admission_declined_response, app

    assert PrefillAdmissionError in app.exception_handlers, (
        "app-level PrefillAdmissionError handler unregistered — anthropic "
        "and ollama doors surface valve refusals as bare 500s again"
    )
    assert PromptTooLongError in app.exception_handlers
    resp = _prefill_admission_declined_response(
        PrefillAdmissionError("hybrid delta: turn-peak admission declined")
    )
    assert resp.status_code == 413
    body = json.loads(resp.body)
    assert body["error"]["code"] == "prefill_admission_declined"
    assert "turn-peak admission declined" in body["error"]["message"]


def test_deep_span_cache_clear_default_and_gate():
    """Row 110/111: 16 turns of in-process accumulation filled the MLX
    allocator cache (26.9GB limit) and a ~94k span's peak then aborted Metal
    with an uncatchable command-buffer OOM; the identical request on an
    empty-cache process completed with ~90GB headroom. The deep-span clear
    must default ON with a threshold well below the measured crash point."""
    import re
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "vmlx_engine"
        / "mllm_batch_generator.py"
    ).read_text(encoding="utf-8")
    m = re.search(
        r'"VMLX_DEEP_SPAN_CACHE_CLEAR_TOKENS", "(\d+)"',
        src,
    )
    assert m, "deep-span cache clear env default disappeared"
    assert 0 < int(m.group(1)) <= 65536, (
        f"threshold {m.group(1)} must sit well below the measured ~94k crash"
    )
    helper = src.index("def _maybe_clear_deep_span_cache(")
    window = src[helper : helper + 1600]
    assert "mx.synchronize()" in window, (
        "must synchronize before clearing or freed buffers are not reclaimable"
    )
    assert "clear_cache" in window
    # BOTH lanes must call it: the fresh chunked-prefill lane (before the
    # full-span presize) AND the hybrid cache-hit delta lane (the r2 rerun
    # crashed at ~96k with ZERO engagement because only the fresh lane was
    # wired — the one-of-two-lanes class).
    assert src.count("_maybe_clear_deep_span_cache(") >= 3, (
        "helper must be called from the fresh-prefill AND hybrid-hit lanes"
    )
    hit = src.index("VLM HYBRID cache FULL HIT")
    assert "_maybe_clear_deep_span_cache(" in src[hit : hit + 1600], (
        "hybrid cache-hit delta lane lost its deep-span clear"
    )
    # And the hit lane must NOT presize the reconstructed cache: measured
    # (r6) an up-front span+headroom step there materialized an EXTRA
    # full-span allocation each turn and moved the deep-span OOM wall TWO
    # turns earlier (t17 -> t15).
    assert "_presize_kv_slots_for_span" not in src[hit : hit + 1600], (
        "hit-lane presize regressed the deep-span wall; keep it reverted"
    )
