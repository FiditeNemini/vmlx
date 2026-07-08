"""F6: the bounded thinking-off answer pass streams token-by-token.

When a reasoning model exhausts its token budget in the first (thinking-on) pass
it produces no visible content, so the server runs a bounded thinking-off answer
pass. That pass used to be generated in full and emitted as ONE content delta,
so the whole answer landed at once after a long silent wait (a progressive-UI /
coding-harness hazard). It now streams incrementally via
``_answer_pass_visible_delta``.

These tests cover the pure delta helper directly (the two streaming call sites —
Chat Completions and the Responses API — both delegate to it) plus a source-level
assertion that neither site regressed to the single-chunk form.
"""

import inspect
from types import SimpleNamespace

import vmlx_engine.server as server_mod
from vmlx_engine.server import _ANS_MARKER_HOLDBACK, _answer_pass_visible_delta
from vmlx_engine.api.utils import clean_output_text


def _req(enable_thinking=True):
    return SimpleNamespace(enable_thinking=enable_thinking)


def _drive(raw_chunks, request):
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
        delta, sent = _answer_pass_visible_delta(raw, sent, request, finished)
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
    # bare "thought" until "<channel|>" arrives, then re-strips it. The holdback
    # must ensure that transient "thought" never escapes to the client.
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


def test_short_answer_flushes_once_on_finish():
    # An answer shorter than the holdback stays buffered until the finished chunk,
    # then flushes intact — the openPangu #66 floor case ("capital of France").
    answer = "The capital of France is Paris."
    assert len(answer) < _ANS_MARKER_HOLDBACK
    raw_chunks = [answer[i:i + 3] for i in range(0, len(answer), 3)]
    deltas, raw = _drive(raw_chunks, _req())
    assert deltas == [answer]


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
    assert "engine.stream_chat(messages=answer_messages" in chat_src
    # Responses API streaming answer pass lives in stream_responses_api.
    resp_src = inspect.getsource(server_mod.stream_responses_api)
    assert "_answer_pass_visible_delta" in resp_src
    assert "engine.stream_chat(messages=answer_messages" in resp_src
