"""Reasoning control markers must never leak at arbitrary stream boundaries."""

import pytest

from vmlx_engine.reasoning.deepseek_r1_parser import DeepSeekR1ReasoningParser
from vmlx_engine.reasoning.minimax_m3_parser import MiniMaxM3ReasoningParser
from vmlx_engine.reasoning.qwen3_parser import Qwen3ReasoningParser
from vmlx_engine.reasoning.think_xml_parser import ThinkXmlReasoningParser


def _stream_one_character_at_a_time(parser, text: str) -> tuple[str, str]:
    previous = ""
    reasoning: list[str] = []
    content: list[str] = []
    for char in text:
        current = previous + char
        delta = parser.extract_reasoning_streaming(previous, current, char)
        if delta is not None:
            if delta.reasoning:
                reasoning.append(delta.reasoning)
            if delta.content:
                content.append(delta.content)
        previous = current
    return "".join(reasoning), "".join(content)


@pytest.mark.parametrize(
    ("parser", "text", "expected_reasoning", "expected_content"),
    [
        (
            Qwen3ReasoningParser(),
            "prefix<think>private plan</think>VISIBLE",
            "private plan",
            "prefixVISIBLE",
        ),
        (
            ThinkXmlReasoningParser(),
            "prefix<think>private plan</think>VISIBLE",
            "private plan",
            "prefixVISIBLE",
        ),
        (
            MiniMaxM3ReasoningParser(),
            "prefix<mm:think>private plan</mm:think>VISIBLE",
            "private plan",
            "prefixVISIBLE",
        ),
        (
            MiniMaxM3ReasoningParser(),
            "prefix<think>private plan</think>VISIBLE",
            "private plan",
            "prefixVISIBLE",
        ),
        (
            Qwen3ReasoningParser(),
            (
                "<think>first rail</think>"
                "<thinking>secondary private plan</thinking>"
                "VISIBLE"
            ),
            "first railsecondary private plan",
            "VISIBLE",
        ),
        (
            ThinkXmlReasoningParser(),
            (
                "<think>first rail</think>"
                "<thinking>secondary private plan</thinking>"
                "VISIBLE"
            ),
            "first railsecondary private plan",
            "VISIBLE",
        ),
        # deepseek_r1 is the parser nemotron/nemotron_h, deepseek_v4, laguna
        # and the qwen3 families resolve to — by request volume the most-used
        # rail in the product, and it had no every-character-boundary row.
        (
            DeepSeekR1ReasoningParser(),
            "prefix<think>private plan</think>VISIBLE",
            "private plan",
            "prefixVISIBLE",
        ),
        # DSV4 live-emits the "<thinking>" spelling; a literal "<think>"
        # needle misses it (its closing ">" never aligns with "ing>").
        (
            DeepSeekR1ReasoningParser(),
            "<thinking>private plan</thinking>VISIBLE",
            "private plan",
            "VISIBLE",
        ),
    ],
)
def test_explicit_reasoning_markers_are_safe_at_every_character_boundary(
    parser,
    text,
    expected_reasoning,
    expected_content,
):
    parser.reset_state(think_in_prompt=False)

    reasoning, content = _stream_one_character_at_a_time(parser, text)

    assert reasoning == expected_reasoning
    assert content == expected_content
    assert "<think" not in reasoning + content
    assert "</think" not in reasoning + content
    assert "<thinking" not in reasoning + content
    assert "</thinking" not in reasoning + content
    assert "<mm:think" not in reasoning + content
    assert "</mm:think" not in reasoning + content


@pytest.mark.parametrize(
    ("parser", "text"),
    [
        (Qwen3ReasoningParser(), "private plan</think>VISIBLE"),
        (MiniMaxM3ReasoningParser(), "private plan</mm:think>VISIBLE"),
        # The implicit rail is deepseek_r1's LIVE mode: the template opens
        # <think> in the prompt, so the model emits only the close marker.
        (DeepSeekR1ReasoningParser(), "private plan</think>VISIBLE"),
    ],
)
def test_prompt_opened_close_marker_is_safe_at_every_character_boundary(parser, text):
    parser.reset_state(think_in_prompt=True)

    reasoning, content = _stream_one_character_at_a_time(parser, text)

    assert reasoning == "private plan"
    assert content == "VISIBLE"
    assert "<" not in reasoning + content


def test_marker_like_literal_is_released_once_it_cannot_become_a_marker():
    parser = Qwen3ReasoningParser()
    parser.reset_state(think_in_prompt=False)

    reasoning, content = _stream_one_character_at_a_time(parser, "math: 2 < three")

    assert reasoning == ""
    assert content == "math: 2 < three"


def test_qwen_complete_extraction_routes_secondary_thinking_alias_to_reasoning():
    parser = Qwen3ReasoningParser()
    parser.reset_state(think_in_prompt=False)

    reasoning, content = parser.extract_reasoning(
        "<think>canonical private</think>"
        "<thinking>secondary private</thinking>"
        "R18-Q27-UI-1-DONE"
    )

    assert reasoning == "canonical private\nsecondary private"
    assert content == "R18-Q27-UI-1-DONE"
    assert "<thinking" not in content


def _stream_chunks(parser, chunks, think_in_prompt=False):
    parser.reset_state(think_in_prompt=think_in_prompt)
    previous = ""
    reasoning: list[str] = []
    content: list[str] = []
    for chunk in chunks:
        current = previous + chunk
        delta = parser.extract_reasoning_streaming(previous, current, chunk)
        if delta is not None:
            if delta.reasoning:
                reasoning.append(delta.reasoning)
            if delta.content:
                content.append(delta.content)
        previous = current
    return "".join(reasoning), "".join(content)


@pytest.mark.parametrize(
    "chunks",
    [
        ["<think>", "plan", "</think>", "\n\n", "Answer here"],
        ["<think>plan</think>", "\n\nAnswer here"],
        ["<think>plan</think>\n\nAnswer here"],
        ["<thinking>plan</thinking>", "\n\nAnswer here"],
    ],
)
def test_deepseek_r1_strips_the_separator_after_a_genuine_rail(chunks):
    """The newline(s) after a close that ended the model's OWN <think> rail
    are a structural boundary, not answer text.

    On the explicit direct rail (reset_state(think_in_prompt=False) — what the
    server does for every request whose template did not open the rail) these
    used to stream through verbatim, so every nemotron / deepseek_v4 / laguna
    answer began with a blank line and streaming disagreed with the parser's
    own non-stream output. qwen3 on identical input has always stripped them.
    """
    _, content = _stream_chunks(DeepSeekR1ReasoningParser(), chunks)
    assert content == "Answer here"


def test_deepseek_r1_keeps_whitespace_after_a_stray_close():
    """The case the preserve-whitespace branch exists for: a close with NO
    opener, after prose that was already streamed as visible content. That is
    redundant markup between two visible runs, so the spacing is the user's."""
    _, content = _stream_chunks(
        DeepSeekR1ReasoningParser(), ["Visible prose", "</think>", "\n\nmore text"]
    )
    assert content == "Visible prose\n\nmore text"


def test_deepseek_r1_preserves_blank_lines_inside_the_answer():
    """Only the leading separator is structural — paragraph breaks within the
    answer are content and must survive."""
    _, content = _stream_chunks(
        DeepSeekR1ReasoningParser(), ["<think>p</think>", "\n\nLine1\n\nLine2"]
    )
    assert content == "Line1\n\nLine2"


@pytest.mark.parametrize(
    "chunks",
    [
        ["<think>", "plan", "</think>", "\n\nAnswer."],
        ["<think>", "plan", "</think>\n\nAnswer."],
        ["<think>plan</think>\n\nAnswer."],
        list("<think>plan</think>\n\nAnswer."),
    ],
)
def test_minimax_m3_think_alias_matches_the_canonical_dialect(chunks):
    """MiniMax-M3 accepts BOTH <mm:think> and a plain <think> fallback.

    The canonical dialect delegates to the base parser (which strips the
    structural separator after the close); the alias went through a standalone
    streaming helper that emitted post-close text verbatim. Same model, same
    prompt, two dialects — one answered with a leading blank line and the other
    did not. M3-VL is a media family, so this sat in the multimodal campaign.
    """
    reasoning, content = _stream_chunks(MiniMaxM3ReasoningParser(), chunks)
    assert reasoning == "plan"
    assert content == "Answer."


def test_minimax_m3_canonical_dialect_is_the_parity_reference():
    reasoning, content = _stream_chunks(
        MiniMaxM3ReasoningParser(), ["<mm:think>", "plan", "</mm:think>", "\n\nAnswer."]
    )
    assert reasoning == "plan"
    assert content == "Answer."


def test_minimax_m3_alias_preserves_blank_lines_inside_the_answer():
    _, content = _stream_chunks(
        MiniMaxM3ReasoningParser(), ["<think>", "p", "</think>", "\n\nL1\n\nL2"]
    )
    assert content == "L1\n\nL2"
