# SPDX-License-Identifier: Apache-2.0
"""Muse Glimmer reasoning split.

Muse routes by RECIPIENT rather than by an inline <think> pair or a named
channel: "to=self" is the reasoning rail, "to=user" (or no recipient) is the
visible answer, and a tool recipient carries an ATEM call that belongs to the
tool parser. think_in_template is false, so output never begins mid-reasoning.
"""

import pytest

from vmlx_engine.reasoning import get_parser


@pytest.fixture
def parser():
    p = get_parser("muse_glimmer")()
    p.reset_state()
    return p


# The prompt ends at "<|start|>assistant", so the model's first emission is a
# bare " to=..." with no leading <|start|>.
REASON_THEN_ANSWER = (
    " to=self<|message|>Weighing the options.<|eom|>"
    "<|start|>assistant to=user<|message|>Pick the second one.<|eot|>"
)


def test_registered_under_both_aliases():
    for alias in ("muse_glimmer", "muse"):
        assert get_parser(alias).__name__ == "MuseGlimmerReasoningParser"


def test_splits_reasoning_from_answer(parser):
    reasoning, content = parser.extract_reasoning(REASON_THEN_ANSWER)
    assert reasoning == "Weighing the options."
    assert content == "Pick the second one."


def test_answer_with_no_recipient_is_content(parser):
    """The template treats a missing recipient as 'user'."""
    reasoning, content = parser.extract_reasoning("<|message|>Just an answer.<|eot|>")
    assert reasoning is None
    assert content == "Just an answer."


def test_answer_only_turn(parser):
    reasoning, content = parser.extract_reasoning(
        " to=user<|message|>No thinking needed.<|eot|>"
    )
    assert reasoning is None
    assert content == "No thinking needed."


def test_reasoning_only_so_far(parser):
    """A turn cut off before the answer still yields its reasoning."""
    reasoning, content = parser.extract_reasoning(" to=self<|message|>Still working")
    assert reasoning == "Still working"
    assert content is None


def test_multiple_reasoning_messages_concatenate(parser):
    out = (
        " to=self<|message|>First thought.<|eom|>"
        "<|start|>assistant to=self<|message|> Second thought.<|eom|>"
        "<|start|>assistant to=user<|message|>Done.<|eot|>"
    )
    reasoning, content = parser.extract_reasoning(out)
    assert reasoning == "First thought. Second thought."
    assert content == "Done."


def test_tool_recipient_body_stays_in_content(parser):
    """The ATEM tool parser owns that markup and must see it verbatim."""
    out = (
        " to=self<|message|>Need the weather.<|eom|>"
        '<|start|>assistant to=get_weather<|message|><atem:function_calls>\n'
        '<atem:invoke name="get_weather">\n'
        '<atem:parameter name="location">Oslo</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls><|eot|>"
    )
    reasoning, content = parser.extract_reasoning(out)

    assert reasoning == "Need the weather."
    assert "<atem:invoke" in content
    assert "Oslo" in content


def test_empty_output(parser):
    assert parser.extract_reasoning("") == (None, None)


def test_unmarked_text_is_content_not_reasoning(parser):
    """think_in_template is false, so there is no injected reasoning prefix."""
    reasoning, content = parser.extract_reasoning("Plain answer with no markers.")
    assert reasoning is None
    assert content == "Plain answer with no markers."


class TestStreaming:
    def test_incremental_reasoning_then_content(self, parser):
        chunks = [
            " to=self<|message|>Think",
            "ing hard.<|eom|>",
            "<|start|>assistant to=user<|message|>Answer",
            " here.<|eot|>",
        ]
        reasoning_seen, content_seen = "", ""
        prev = ""
        for chunk in chunks:
            curr = prev + chunk
            msg = parser.extract_reasoning_streaming(prev, curr, chunk)
            if msg:
                reasoning_seen += msg.reasoning or ""
                content_seen += msg.content or ""
            prev = curr

        assert reasoning_seen == "Thinking hard."
        assert content_seen == "Answer here."

    def test_header_split_across_chunks_is_not_leaked(self, parser):
        """A header straddling a boundary must not be emitted as body text."""
        chunks = [" to=self<|message|>Idea.<|eom|><|start|>assis", "tant to=user<|mess", "age|>Final.<|eot|>"]
        reasoning_seen, content_seen = "", ""
        prev = ""
        for chunk in chunks:
            curr = prev + chunk
            msg = parser.extract_reasoning_streaming(prev, curr, chunk)
            if msg:
                reasoning_seen += msg.reasoning or ""
                content_seen += msg.content or ""
            prev = curr

        assert reasoning_seen == "Idea."
        assert content_seen == "Final."
        assert "assistant" not in content_seen
        assert "message" not in content_seen

    def test_no_duplicate_emission(self, parser):
        text = " to=self<|message|>Once.<|eom|>"
        first = parser.extract_reasoning_streaming("", text, text)
        assert first is not None and first.reasoning == "Once."

        again = parser.extract_reasoning_streaming(text, text, "")
        assert again is None, "the same reasoning was emitted twice"

    def test_reset_state_clears_counters(self, parser):
        text = " to=self<|message|>A.<|eom|>"
        parser.extract_reasoning_streaming("", text, text)
        parser.reset_state()
        again = parser.extract_reasoning_streaming("", text, text)
        assert again is not None and again.reasoning == "A."
