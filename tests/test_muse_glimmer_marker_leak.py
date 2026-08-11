# SPDX-License-Identifier: Apache-2.0
"""Muse Glimmer: a message terminator must never render as chat text.

Emitted counters in ``extract_reasoning_streaming`` are monotonic, so anything
classified from a half-arrived header is unrecoverable — it renders in chat and
no later re-split can retract it. These cases are the three shapes that leaked
``<|eom|>`` / ``<|eot|>`` live:

1. terminator at the very end of the stable prefix, header not yet arrived
2. terminator followed by a BARE header (``<|eom|>assistant to=user``) — the
   generation prompt already spent the ``<|start|>``
3. terminator followed immediately by ``<|start|>`` — both resolve to the same
   boundary, so a ``>=`` tie-break released the terminator behind the start tag
"""

import pytest

from vmlx_engine.reasoning.muse_glimmer_parser import MuseGlimmerReasoningParser

MARKERS = ("<|start|>", "<|message|>", "<|eom|>", "<|eot|>")

LEAK_SHAPES = {
    "terminator_then_start_tag_header": (
        "Oslo.<|eom|><|start|>assistant to=user<|message|>And it is cold.<|eot|>",
        "Oslo.And it is cold.",
    ),
    "terminator_then_bare_header": (
        "Oslo.<|eom|>assistant to=user<|message|>And it is cold.<|eot|>",
        "Oslo.And it is cold.",
    ),
    "eot_then_bare_header": (
        "Partial.<|eot|>assistant to=user<|message|>Rest of it.<|eot|>",
        "Partial.Rest of it.",
    ),
}


def _stream(raw, chunks):
    parser = MuseGlimmerReasoningParser()
    reasoning, content, prev = [], [], ""
    for cur in chunks:
        delta = parser.extract_reasoning_streaming(prev, cur, cur[len(prev):])
        if delta:
            if delta.reasoning:
                reasoning.append(delta.reasoning)
            if delta.content:
                content.append(delta.content)
        prev = cur
    return "".join(reasoning), "".join(content)


def _char_chunks(raw):
    return [raw[: i + 1] for i in range(len(raw))]


@pytest.mark.parametrize("name", sorted(LEAK_SHAPES))
def test_terminator_never_reaches_streamed_content(name):
    raw, expected = LEAK_SHAPES[name]
    _, content = _stream(raw, _char_chunks(raw))
    assert content == expected
    for marker in MARKERS:
        assert marker not in content


@pytest.mark.parametrize("name", sorted(LEAK_SHAPES))
def test_terminator_never_reaches_oneshot_content(name):
    raw, expected = LEAK_SHAPES[name]
    _, content = MuseGlimmerReasoningParser().extract_reasoning(raw)
    assert content == expected


@pytest.mark.parametrize("name", sorted(LEAK_SHAPES))
def test_streaming_agrees_with_oneshot(name):
    """Whatever the chunk boundaries, the user sees the same text."""
    raw, _ = LEAK_SHAPES[name]
    _, oneshot = MuseGlimmerReasoningParser().extract_reasoning(raw)
    # split inside every marker occurrence — the worst case for a chunk boundary
    cuts = set()
    for marker in MARKERS:
        at = raw.find(marker)
        while at >= 0:
            cuts.update(at + k for k in range(1, len(marker)))
            at = raw.find(marker, at + 1)
    cuts.add(len(raw))
    _, streamed = _stream(raw, [raw[:p] for p in sorted(cuts)])
    assert streamed.strip() == (oneshot or "").strip()


def test_stable_prefix_never_shrinks():
    """Content may only grow; a shrink means an emitted char cannot be retracted."""
    from vmlx_engine.reasoning.muse_glimmer_parser import _stable_length

    for raw, _ in LEAK_SHAPES.values():
        parser = MuseGlimmerReasoningParser()
        seen = 0
        for i in range(len(raw)):
            cur = raw[: i + 1]
            _, content = parser._split(cur[: _stable_length(cur)])
            assert len(content) >= seen, f"{raw!r} shrank at {i}: {content!r}"
            seen = len(content)


def test_prose_that_merely_mentions_a_marker_is_preserved():
    """A lone control token in prose is NOT a leak and must survive."""
    raw = " to=user<|message|>Muse opens a turn with the <|start|> token."
    _, content = MuseGlimmerReasoningParser().extract_reasoning(raw)
    assert content == "Muse opens a turn with the <|start|> token."


PROSE_KEPT = {
    # This exact sentence rendered live as "It opens with the control token" —
    # the scrub scanned the whole body and ate a correct answer.
    "header_named_mid_sentence": (
        "It opens with the <|start|>assistant control token.",
        "It opens with the <|start|>assistant control token.",
    ),
    "bare_token_mid_sentence": (
        "Use <|start|> then the role name.",
        "Use <|start|> then the role name.",
    ),
    "recipient_named_mid_sentence": (
        "The reasoning rail is to=self and the answer rail is to=user here.",
        "The reasoning rail is to=self and the answer rail is to=user here.",
    ),
}


@pytest.mark.parametrize("name", sorted(PROSE_KEPT))
def test_prose_about_the_template_survives_the_scrub(name):
    """Explaining the chat template is exactly when a user spells a header."""
    body, expected = PROSE_KEPT[name]
    _, content = MuseGlimmerReasoningParser().extract_reasoning(
        f" to=user<|message|>{body}<|eot|>"
    )
    assert content == expected


def test_residue_is_still_scrubbed_at_a_message_boundary():
    """The backstop must keep working where a real leak actually lands."""
    parser = MuseGlimmerReasoningParser
    # generation cut mid-header -> fragment at the TAIL of the body
    _, tail = parser().extract_reasoning(" to=user<|message|>Answer here<|start|>assistant")
    assert tail == "Answer here"
    # structure that outran the segmenter -> fragment at the HEAD of the body
    _, head = parser().extract_reasoning("<|eom|>assistant to=user<|message|>Real answer.<|eot|>")
    assert head == "Real answer."


def test_reply_ending_in_letters_keeps_its_last_characters():
    """Holding `assistant`/`to=` globally used to eat the tail of a real reply."""
    for tail in ("The capital is Oslo", "I do not know what this refers to",
                 "You are speaking with an assistant"):
        raw = f" to=user<|message|>{tail}"
        _, content = _stream(raw, _char_chunks(raw))
        assert content == tail
