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
