# SPDX-License-Identifier: Apache-2.0
"""Whitespace must survive every reasoning parser, on both output paths.

Marker-aware parsers hold and re-emit text around control tokens, so whitespace
is exactly where they corrupt output silently: a newline right after a header, a
blank line between markdown paragraphs, indentation inside a code fence, a hard
line break's trailing spaces.

Two invariants per parser, per payload:

  I1  INTERIOR whitespace is preserved byte-for-byte (a lost \\n\\n runs two
      paragraphs together; a lost fence newline breaks code rendering)
  I2  concat(streamed deltas) == the one-shot result, ignoring only
      leading/trailing whitespace — the UI streams and the API does not, and the
      two must not disagree about the body

Every concrete parser is discovered by reflection, so a new family is covered the
day it lands rather than the day someone remembers to add it here.
"""

import importlib
import inspect
import pkgutil

import pytest

import vmlx_engine.reasoning as reasoning_pkg
from vmlx_engine.reasoning import base as reasoning_base

# Answer bodies whose interior whitespace must survive verbatim.
BODIES = {
    "paragraph_break": "First paragraph.\n\nSecond paragraph.",
    "triple_newline": "A\n\n\nB",
    "code_fence": "Here:\n```python\ndef f():\n    return 1\n```\nDone.",
    "fence_blank_line": "```py\na = 1\n\nb = 2\n```",
    "crlf": "Line one.\r\nLine two.",
    "tabs": "col1\tcol2\tcol3",
    "hard_break_spaces": "hard break  \nnext line",
    "nbsp": "a b",
    "latex_block": "$$\n\\frac{a}{b}\n$$",
    "markdown_list": "- one\n- two\n  - nested",
    "indented_html": "<div>\n  <p>x</p>\n</div>",
}


def _concrete_parsers():
    found = {}
    for mod in pkgutil.iter_modules(reasoning_pkg.__path__):
        if mod.name in ("base", "__init__"):
            continue
        module = importlib.import_module(f"vmlx_engine.reasoning.{mod.name}")
        for _name, obj in vars(module).items():
            if (
                inspect.isclass(obj)
                and issubclass(obj, reasoning_base.ReasoningParser)
                and obj is not reasoning_base.ReasoningParser
                and obj.__module__ == module.__name__
                and not getattr(obj, "__abstractmethods__", None)
            ):
                found[mod.name] = obj
    return found


PARSERS = _concrete_parsers()

# How to present an answer body so each family routes it to CONTENT. Families
# whose markers are plain start/end tokens are derived from the instance.
SPECIAL_WRAPPERS = {
    "muse_glimmer_parser": lambda b: f" to=user<|message|>{b}<|eot|>",
    "gemma4_parser": lambda b: f"<|channel>thought\nreasoning<channel|>{b}<turn|>",
    "gptoss_parser": (
        lambda b: "<|channel|>analysis<|message|>reasoning"
        f"<|start|>assistant<|channel|>final<|message|>{b}"
    ),
}


def _wrap(name, parser_cls, body):
    special = SPECIAL_WRAPPERS.get(name)
    if special:
        return special(body)
    inst = parser_cls()
    start = getattr(inst, "start_token", None)
    end = getattr(inst, "end_token", None)
    if not start or not end:
        return None
    return f"{start}reasoning{end}{body}"


def _stream(parser_cls, raw, chunks):
    parser = parser_cls()
    content, prev = [], ""
    for cur in chunks:
        delta = parser.extract_reasoning_streaming(prev, cur, cur[len(prev):])
        if delta and getattr(delta, "content", None):
            content.append(delta.content)
        prev = cur
    return "".join(content)


CASES = [
    (pname, bname)
    for pname in sorted(PARSERS)
    for bname in sorted(BODIES)
    if _wrap(pname, PARSERS[pname], BODIES[bname]) is not None
]


def test_every_concrete_parser_is_exercised():
    """Guard against the sweep silently covering nothing."""
    covered = {pname for pname, _ in CASES}
    uncovered = set(PARSERS) - covered
    assert not uncovered, f"no wrapper for {sorted(uncovered)} — extend SPECIAL_WRAPPERS"
    assert len(PARSERS) >= 8, f"only found {sorted(PARSERS)}"


@pytest.mark.parametrize("pname,bname", CASES, ids=[f"{p}-{b}" for p, b in CASES])
def test_interior_whitespace_survives_the_oneshot_path(pname, bname):
    body = BODIES[bname]
    raw = _wrap(pname, PARSERS[pname], body)
    _reasoning, content = PARSERS[pname]().extract_reasoning(raw)
    assert content is not None, f"{pname} dropped the whole answer for {bname}"
    # Leading/trailing trim is tolerated; the BODY must be intact.
    assert content.strip() == body.strip(), (
        f"{pname} altered interior whitespace for {bname}:\n"
        f"  want {body.strip()!r}\n  got  {content.strip()!r}"
    )


@pytest.mark.parametrize("pname,bname", CASES, ids=[f"{p}-{b}" for p, b in CASES])
def test_streaming_agrees_with_oneshot(pname, bname):
    """The UI streams and the API does not; they must not disagree on the body."""
    body = BODIES[bname]
    raw = _wrap(pname, PARSERS[pname], body)
    _reasoning, oneshot = PARSERS[pname]().extract_reasoning(raw)
    for label, chunks in (
        ("char", [raw[: i + 1] for i in range(len(raw))]),
        ("whole", [raw]),
        ("half", [raw[: max(1, len(raw) // 2)], raw]),
    ):
        streamed = _stream(PARSERS[pname], raw, chunks)
        assert streamed.strip() == (oneshot or "").strip(), (
            f"{pname}/{bname}/{label}: streaming and one-shot disagree\n"
            f"  stream  {streamed!r}\n  oneshot {oneshot!r}"
        )


def test_gemma4_literal_turn_marker_in_prose_survives_streaming():
    """A literal <turn|> in the answer must not desync the emission counter.

    _plain_content_before_possible_thought did text.replace(_EOT, "") over the
    WHOLE accumulated buffer. That shrinks the text under a monotonic counter,
    so "Use <turn|> to end turns." streamed as "Use <turn|d turns." while the
    one-shot path returned it intact — the signature defect class of applying a
    whole-string operation to a growing buffer. Only a TRAILING marker may be
    trimmed.
    """
    from vmlx_engine.reasoning.gemma4_parser import Gemma4ReasoningParser

    for raw, expected in [
        ("Use <turn|> to end turns.", "Use <turn|> to end turns."),
        ("Ends with marker<turn|>", "Ends with marker"),
    ]:
        parser = Gemma4ReasoningParser()
        parts = []
        for i in range(1, len(raw) + 1):
            delta = parser.extract_reasoning_streaming(raw[: i - 1], raw[:i], raw[i - 1])
            if delta is not None and getattr(delta, "content", None):
                parts.append(delta.content)
        streamed = "".join(parts)
        one = Gemma4ReasoningParser().extract_reasoning(raw)
        one_text = one[1] if isinstance(one, tuple) else one
        assert streamed == expected, f"streamed {streamed!r} for {raw!r}"
        assert str(one_text) == expected


def test_gemma4_visible_prefix_before_thought_channel_streams_monotonically():
    """Content before a thought channel must not desync the emission counter.

    _join_visible_content deduped whenever the suffix began with the prefix and
    returned the suffix ALONE. Mid-stream that is a coincidence, not a
    duplicate, and it makes the joined string SHRINK while the counter only
    grows: "Hi" + a suffix rail opening "Hi\\nMore" streamed as "Hi\\nHore" — one
    character duplicated, another lost — while one-shot returned "Hi\\nMore".

    The invariant under test is streamed == one-shot, not a particular dedup
    policy. NOTE the marker is <|channel> (no pipe before '>'); an audit
    write-up used <|channel|> and the case silently did not reproduce.
    """
    from vmlx_engine.reasoning.gemma4_parser import Gemma4ReasoningParser

    for raw in [
        "Hi<|channel>thought\nplan\n<channel|>Hi\nMore",
        "Hi<|channel>thought\nr\n<channel|>Answer",
        "A<|channel>thought\nx\n<channel|>A",
    ]:
        parser = Gemma4ReasoningParser()
        parts = []
        for i in range(1, len(raw) + 1):
            delta = parser.extract_reasoning_streaming(raw[: i - 1], raw[:i], raw[i - 1])
            if delta is not None and getattr(delta, "content", None):
                parts.append(delta.content)
        streamed = "".join(parts)
        one = Gemma4ReasoningParser().extract_reasoning(raw)
        one_text = one[1] if isinstance(one, tuple) else one
        assert streamed == str(one_text), (
            f"streaming/one-shot disagree for {raw!r}: "
            f"{streamed!r} vs {str(one_text)!r}"
        )


def test_marker_only_delta_keeps_its_leading_whitespace():
    """A delta that is ONLY a marker still carries real answer whitespace.

    The parser returned None for any delta whose strip() equalled a marker, so a
    chunking where one delta is exactly "\\n<think>" lost the newline: "Intro"
    streamed straight into "Answer". Character-wise streaming never produces
    such a delta, which is why the 199-case suite did not catch it — the gap was
    the CHUNKING, not the body.

    Only the leading whitespace of an OPENING marker is asserted here; the
    remaining "\\n" before the answer is a separate defect (the .lstrip() on the
    post-close transition) and is still open.
    """
    from vmlx_engine.reasoning import get_parser

    parser_cls = get_parser("qwen3")
    parser = parser_cls()
    previous = ""
    parts = []
    for chunk in ["Intro", "\n<think>", "r", "</think>", "\nAnswer"]:
        current = previous + chunk
        delta = parser.extract_reasoning_streaming(previous, current, chunk)
        if delta is not None and getattr(delta, "content", None):
            parts.append(delta.content)
        previous = current
    streamed = "".join(parts)
    assert streamed.startswith("Intro\n"), (
        f"newline before the think block was swallowed: {streamed!r}"
    )


def test_quoted_close_tag_is_prose_not_a_reasoning_boundary():
    """`</think>` inside backticks must not swallow the answer's opening text.

    The bare-close branch (implicit mode: the template injects <think> as a
    special token that gets eaten, so the model emits only the close) treated
    ANY occurrence as the boundary. So an answer that merely MENTIONED the tag
    lost everything before it to the reasoning rail:

        "Code: `</think>` here"   ->   "` here"

    Gating the branch on _think_in_prompt was tried first and BROKE implicit
    mode — three existing tests caught it. The working discriminator is the same
    one used for tool markers in the suppressed-tool display path: skip
    occurrences preceded by a backtick, since a model never wraps a real emitted
    close in backticks.
    """
    from vmlx_engine.reasoning import get_parser

    parser_cls = get_parser("qwen3")

    for raw in (
        "Code: `</think>` here",
        "Use `</think>` to close the block.",
        "Plain answer with no tags.",
    ):
        result = parser_cls().extract_reasoning(raw)
        content = result[1] if isinstance(result, tuple) else result
        assert str(content) == raw, (
            f"a quoted mention was treated as a reasoning boundary: {content!r}"
        )

    # ...and a genuine bare close must still split (implicit mode).
    result = parser_cls().extract_reasoning("reasoning here</think>the answer")
    assert result == ("reasoning here", "the answer")
