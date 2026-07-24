"""F6: the bounded thinking-off answer pass streams token-by-token.

When a caller explicitly supplies ``max_thinking_tokens`` and the bounded
thinking-on pass produces no visible content, the server may use the reserved
remainder for a thinking-off answer pass. That pass streams incrementally via
``_answer_pass_visible_delta``.

These tests cover the pure delta helper directly (the two streaming call sites —
Chat Completions and the Responses API — both delegate to it) plus a source-level
assertion that neither site regressed to the single-chunk form.
"""

import inspect
from types import SimpleNamespace

import vmlx_engine.server as server_mod
from vmlx_engine.api.utils import clean_output_text
from vmlx_engine.server import (
    _ANS_MARKER_HOLDBACK,
    _ANSWER_PASS_LEAK_GUARD_FAMILIES,
    _answer_pass_stream_holdback,
    _answer_pass_visible_delta,
    _main_pass_finish_reason,
)


def _req(enable_thinking=True):
    return SimpleNamespace(enable_thinking=enable_thinking)


def _drive(raw_chunks, request, *, holdback=0):
    """Replicate the streaming loop: accumulate new_text, collect emitted deltas.

    The last chunk is delivered with finished=True (the engine's terminal chunk),
    which flushes the held-back tail.
    """
    raw = ""
    sent = ""
    deltas = []
    for i, piece in enumerate(raw_chunks):
        raw += piece
        finished = i == len(raw_chunks) - 1
        delta, sent = _answer_pass_visible_delta(
            raw,
            sent,
            request,
            finished,
            holdback=holdback,
        )
        if delta:
            deltas.append(delta)
    return deltas, raw


def test_streams_multiple_deltas_for_long_answer():
    # A long plain answer streams incrementally, not as one blob.
    words = ("The ocean covers most of the planet and regulates its climate "
             "while sustaining an immense diversity of life across every depth. ") * 3
    chunks = [w + " " for w in words.split(" ") if w]
    deltas, raw = _drive(chunks, _req())
    assert len(deltas) > 1
    # Concatenated deltas reproduce the finalized text exactly (no loss, no dup).
    assert "".join(deltas) == clean_output_text(raw).strip()


def test_no_partial_marker_leak_gemma_channel():
    # gemma degraded form: clean_output_text turns "<|channel>thought" into a
    # bare "thought" until "<channel|>" arrives, then re-strips it. The dynamic
    # control state must ensure that transient "thought" never escapes.
    answer = ("The ocean is a vast interconnected system that stores heat, "
              "drives weather, and shelters countless species in its depths.")
    raw_chunks = ["<|channel>", "thought", "\n", "<channel|>"]
    raw_chunks += [answer[i:i + 4] for i in range(0, len(answer), 4)]
    deltas, raw = _drive(raw_chunks, _req())
    joined = "".join(deltas)
    assert "thought" not in joined
    assert not joined.lstrip().startswith(("channel", "<"))
    assert joined == clean_output_text(raw).strip()
    assert joined.startswith("The ocean is a vast")


def test_short_plain_answer_streams_without_family_allowlist():
    # Ordinary text is not a control marker and must stream even when shorter than
    # the old global 48-character tail.
    answer = "The capital of France is Paris."
    assert len(answer) < _ANS_MARKER_HOLDBACK
    raw_chunks = [answer[i:i + 3] for i in range(0, len(answer), 3)]
    deltas, raw = _drive(raw_chunks, _req())
    assert len(deltas) == len(raw_chunks)
    assert "".join(deltas) == answer


def test_hy3_direct_rail_short_answer_streams_incrementally():
    """Hy3 no_think has no late reasoning marker, so it needs no holdback."""
    chunks = ["TURN", "1: ", "ORBIT", "-731"]
    deltas, raw = _drive(chunks, _req(enable_thinking=False), holdback=0)
    assert len(deltas) == len(chunks)
    assert "".join(deltas) == raw


def test_direct_rail_short_answer_is_not_model_allowlisted():
    """All non-buffered retries stream; behavior is based on text state, not family."""
    chunks = ["B1", "-", "OK"]
    for family in (
        "hy_v3",
        "minimax_m3",
        "qwen3_5",
        "qwen3_5_moe",
        "step3p7",
        "minimax",
        "gemma4",
        "laguna",
        "nemotron_h",
    ):
        holdback = _answer_pass_stream_holdback(family, buffer_answer_pass=False)
        assert holdback == 0
        deltas, raw = _drive(chunks, _req(enable_thinking=False), holdback=holdback)
        assert deltas == chunks
        assert "".join(deltas) == raw


def test_no_family_batches_ordinary_answer_text_by_name():
    assert _ANSWER_PASS_LEAK_GUARD_FAMILIES == frozenset()
    assert _answer_pass_stream_holdback("step3p7", buffer_answer_pass=False) == 0
    assert _answer_pass_stream_holdback("minimax", buffer_answer_pass=False) == 0
    assert _answer_pass_stream_holdback("gemma4", buffer_answer_pass=False) == 0
    assert _answer_pass_stream_holdback("deepseek_v4", buffer_answer_pass=False) == 0


def test_laguna_does_not_run_a_synthetic_thinking_off_answer_pass():
    """Preserve Laguna's real first-pass terminal instead of inventing a final.

    Live XS-2.1 evidence showed a reasoning pass exhausting its requested
    output budget, followed by the old thinking-off retry returning a
    confident but wrong answer. Laguna's native variable-reasoning contract
    does not make a second independently sampled pass semantically equivalent
    to continuing the first turn.
    """
    for handler in (
        server_mod.create_chat_completion,
        server_mod.create_response,
        server_mod.stream_chat_completion,
        server_mod.stream_responses_api,
    ):
        source = inspect.getsource(handler)
        assert "max_thinking_tokens" in source
        assert "_auto_thinking_partition_allowed(" not in source


def test_partial_close_think_marker_never_leaks_then_answer_streams():
    chunks = ["<", "/t", "hink", ">", "B1", "-", "OK"]
    deltas, raw = _drive(chunks, _req(enable_thinking=False))
    assert deltas == ["B1", "-", "OK"]
    assert "".join(deltas) == "B1-OK"


def test_reopened_reasoning_is_hidden_until_close_then_answer_streams():
    chunks = ["<think>", "private plan", "</think>", "STEP", "-", "OK"]
    deltas, raw = _drive(chunks, _req(enable_thinking=False))
    assert deltas == ["STEP", "-", "OK"]
    assert "private plan" not in "".join(deltas)


def test_dsv4_thinking_variant_is_hidden_then_answer_streams():
    chunks = ["<thi", "nking>", "private plan", "</thinking>", "D4", "-", "OK"]
    deltas, raw = _drive(chunks, _req(enable_thinking=False))
    assert deltas == ["D4", "-", "OK"]
    assert "private plan" not in "".join(deltas)


def test_terminal_unclosed_reasoning_variant_never_leaks():
    deltas, raw = _drive(
        ["<thinking>", "private plan without a close"],
        _req(enable_thinking=False),
    )
    assert deltas == []


def test_deltas_are_monotonic_prefix_extensions():
    # Each emitted delta only appends to what was already sent (no rewrite of
    # already-streamed text), which is what a streaming client relies on.
    answer = "Sea water is salty because rivers carry dissolved minerals into it " * 4
    raw_chunks = [answer[i:i + 5] for i in range(0, len(answer), 5)]
    sent = ""
    request = _req()
    raw = ""
    for i, piece in enumerate(raw_chunks):
        raw += piece
        finished = i == len(raw_chunks) - 1
        delta, new_sent = _answer_pass_visible_delta(raw, sent, request, finished)
        # new cursor is always the old cursor + the delta
        assert new_sent == sent + delta or delta == ""
        sent = new_sent
    assert sent == clean_output_text(raw).strip()


def test_finished_empty_answer_yields_nothing():
    # A pass that produces no visible text must not emit an empty content delta.
    deltas, _ = _drive(["", ""], _req())
    assert deltas == []


def test_enable_thinking_false_splits_on_close_think():
    # Defensive: with the request's enable_thinking False and a </think> present,
    # only the post-</think> text is visible (mirrors the non-streaming path).
    raw = "<think>internal deliberation here</think>The final answer is 42 and it is complete."
    chunks = [raw[i:i + 6] for i in range(0, len(raw), 6)]
    deltas, _ = _drive(chunks, _req(enable_thinking=False))
    joined = "".join(deltas)
    assert "internal deliberation" not in joined
    assert joined.endswith("complete.")


def test_both_stream_sites_use_the_delta_helper():
    # Neither streaming answer pass may regress to the old single-chunk emit.
    chat_src = inspect.getsource(server_mod.stream_chat_completion)
    assert "_answer_pass_visible_delta" in chat_src
    assert "_answer_pass_reconcile_delta" not in chat_src
    assert "engine.stream_chat(messages=answer_messages" in chat_src
    # Responses API streaming answer pass lives in stream_responses_api.
    resp_src = inspect.getsource(server_mod.stream_responses_api)
    assert "_answer_pass_visible_delta" in resp_src
    assert "_answer_pass_reconcile_delta" not in resp_src
    assert "engine.stream_chat(messages=answer_messages" in resp_src


def test_partial_prefix_regeneration_helper_is_removed():
    assert not hasattr(server_mod, "_answer_pass_reconcile_delta")


def test_internal_reasoning_pass_terminal_is_held_until_visible_answer_finishes():
    """A coding client must not see terminal length before the answer pass."""
    assert _main_pass_finish_reason(
        "length",
        finished=True,
        content_was_emitted=False,
        accumulated_reasoning="private planning",
        answer_pass_pending=True,
    ) is None

    # Once native visible content exists, the first generation owns the
    # terminal. A fresh sample cannot retract or "continue" that prefix.
    assert _main_pass_finish_reason(
        "length",
        finished=True,
        content_was_emitted=True,
        accumulated_reasoning="private planning",
        answer_pass_pending=True,
    ) == "length"
    assert _main_pass_finish_reason(
        "stop",
        finished=True,
        content_was_emitted=False,
        accumulated_reasoning="private planning",
        answer_pass_pending=True,
    ) is None


def test_genuine_main_pass_terminal_reasons_are_preserved():
    assert _main_pass_finish_reason(
        "stop",
        finished=True,
        content_was_emitted=True,
        accumulated_reasoning="private planning",
        answer_pass_pending=True,
    ) == "stop"
    assert _main_pass_finish_reason(
        "length",
        finished=True,
        content_was_emitted=False,
        accumulated_reasoning="private planning",
        answer_pass_pending=False,
    ) == "length"
    assert _main_pass_finish_reason(
        "stop",
        finished=False,
        content_was_emitted=True,
        accumulated_reasoning="",
        answer_pass_pending=False,
    ) is None


def test_chat_legacy_reasoning_fallback_cannot_precede_answer_pass():
    """An armed answer pass must be the sole visible fallback.

    Otherwise the legacy block emits accumulated reasoning as content without
    updating ``content_was_emitted``, and the answer pass emits a second answer.
    """
    chat_src = inspect.getsource(server_mod.stream_chat_completion)
    legacy_start = chat_src.index(
        "# A parser may surface its full reasoning block only at stream finalization."
    )
    answer_start = chat_src.index(
        "and (m3_reasoning_only_answer_enabled or reasoning_only_answer_enabled)",
        legacy_start,
    )
    legacy_block = chat_src[legacy_start:answer_start]
    assert (
        "not (m3_reasoning_only_answer_enabled or reasoning_only_answer_enabled)"
        in legacy_block
    )


def test_main_chat_reasoning_content_is_never_terminally_buffered():
    """Parser-exposed content streams now; only content-empty runs use retry."""
    source = inspect.getsource(server_mod.stream_chat_completion)
    assert "not content_was_emitted" in source
    assert "and (m3_reasoning_only_answer_enabled or reasoning_only_answer_enabled)" in source
    assert "deferred_reasoning_visible_content" not in source
    assert 'finish_reason="length"' in source
    assert "_answer_pass_stream_holdback(" in source
    assert "_answer_pass_reconcile_delta" not in source


def test_nonstream_answer_pass_only_arms_when_visible_content_is_empty():
    chat_source = inspect.getsource(server_mod.create_chat_completion)
    responses_source = inspect.getsource(server_mod.create_response)
    for source in (chat_source, responses_source):
        assert "_ns_visible_content_for_answer_gate = content_for_parsing" in source
        assert "_clean_suppressed_tool_markup_for_display(" in source
        assert "_ns_reasoning_truncated" not in source
        assert "not _ns_visible_content_for_answer_gate" in source


def test_streaming_answer_pass_announces_phase_without_nonstandard_json():
    """Strict API clients ignore the phase while the local panel stays live."""
    chat_source = inspect.getsource(server_mod.stream_chat_completion)
    responses_source = inspect.getsource(server_mod.stream_responses_api)
    for source in (chat_source, responses_source):
        assert 'yield ": vmlx-answer-pass-start\\n\\n"' in source
        assert '"type": "vmlx-answer-pass-start"' not in source


def test_responses_answer_pass_continues_incremental_usage_counter():
    source = inspect.getsource(server_mod.stream_responses_api)
    marker = source.index('yield ": vmlx-answer-pass-start\\n\\n"')
    answer_pass = source[marker:]
    assert "if incremental_usage_extension:" in answer_pass
    assert '"output_tokens": completion_tokens + _ans_ct' in answer_pass
    assert '"response.usage"' in answer_pass
