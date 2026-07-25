"""Reasoning control markers must never leak at arbitrary stream boundaries."""

import pytest

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
