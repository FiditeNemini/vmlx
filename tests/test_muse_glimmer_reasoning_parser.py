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


class TestServerContract:
    """The server calls reset_state with keywords every parser must tolerate.

    Missing **kwargs turned EVERY Muse chat request into a 500:
    "MuseGlimmerReasoningParser.reset_state() got an unexpected keyword
    argument 'think_in_prompt'". Load-testing caught it; unit tests had not,
    because they called reset_state() bare.
    """

    def test_reset_state_accepts_the_server_keywords(self, parser):
        parser.reset_state(think_in_prompt=False, harmony_active=False)
        parser.reset_state(think_in_prompt=True, harmony_active=True)
        parser.reset_state()

    def test_reset_state_still_clears_counters_with_keywords(self, parser):
        text = " to=self<|message|>A.<|eom|>"
        parser.extract_reasoning_streaming("", text, text)
        parser.reset_state(think_in_prompt=False, harmony_active=False)
        again = parser.extract_reasoning_streaming("", text, text)
        assert again is not None and again.reasoning == "A."

    def test_matches_the_base_class_signature(self):
        """Any future parser method the server calls must stay keyword-tolerant."""
        import inspect

        from vmlx_engine.reasoning.base import ReasoningParser
        from vmlx_engine.reasoning.muse_glimmer_parser import (
            MuseGlimmerReasoningParser,
        )

        base = inspect.signature(ReasoningParser.reset_state)
        mine = inspect.signature(MuseGlimmerReasoningParser.reset_state)
        base_var_kw = any(p.kind is p.VAR_KEYWORD for p in base.parameters.values())
        mine_var_kw = any(p.kind is p.VAR_KEYWORD for p in mine.parameters.values())
        assert base_var_kw and mine_var_kw, "reset_state must accept **kwargs"


class TestStreamingNeverLeaksMarkers:
    """The streaming path must never print a header or marker as prose.

    Emitted counters only move forward, so a character classified from a
    half-arrived header can never be taken back. Live in the Electron app this
    surfaced as `to=self` and a bare `<|message|` rendered as the answer, and
    as `<|eom|>assistant to=user` sitting in the middle of the reply.

    Every case is driven ONE CHARACTER AT A TIME — the worst case for a
    streaming parser, and the one that exposed the bug.
    """

    LEAKS = ("to=self", "to=user", "<|message|", "<|eom|", "<|eot|", "<|start|")

    CASES = {
        "two_rail": (
            "to=self<|message|>Let me think.<|eom|>"
            "assistant to=user<|message|>Paris is the capital.<|eot|>"
        ),
        "explicit_start_tags": (
            "<|start|>assistant to=self<|message|>Thinking.<|eom|>"
            "<|start|>assistant to=user<|message|>Answer here.<|eot|>"
        ),
        "answer_only": "to=user<|message|>Direct answer.<|eot|>",
        "no_header_at_all": "Just plain prose with no markers whatsoever.",
        "prose_containing_header_words": (
            "to=user<|message|>Go to=the store and ask an assistant.<|eot|>"
        ),
        "three_messages": (
            "to=self<|message|>A<|eom|>assistant to=self<|message|>B<|eom|>"
            "assistant to=user<|message|>Final.<|eot|>"
        ),
        "tool_recipient": (
            "to=self<|message|>Need a tool.<|eom|>"
            'assistant to=atem.get_weather<|message|><atem:invoke name="x"/><|eot|>'
        ),
    }

    @staticmethod
    def _stream(raw):
        parser = get_parser("muse_glimmer")()
        parser.reset_state()
        previous, reasoning, content = "", "", ""
        for index in range(1, len(raw) + 1):
            current = raw[:index]
            delta = parser.extract_reasoning_streaming(previous, current, raw[index - 1])
            previous = current
            if delta:
                reasoning += delta.reasoning or ""
                content += delta.content or ""
        return reasoning, content

    @pytest.mark.parametrize("name", sorted(CASES))
    def test_no_marker_reaches_the_user(self, name):
        reasoning, content = self._stream(self.CASES[name])
        for marker in self.LEAKS:
            assert marker not in content, f"{marker!r} leaked into content"
            assert marker not in reasoning, f"{marker!r} leaked into reasoning"

    @pytest.mark.parametrize("name", sorted(CASES))
    def test_streamed_text_matches_the_final_split(self, name):
        raw = self.CASES[name]
        reasoning, content = self._stream(raw)
        final = get_parser("muse_glimmer")()
        final.reset_state()
        final_reasoning, final_content = final.extract_reasoning(raw)
        # Streaming is incremental, so it must be a prefix of the final answer —
        # and for these complete messages it should be the WHOLE answer.
        assert (final_content or "").strip() == content.strip()
        assert (final_reasoning or "").strip() == reasoning.strip()

    def test_plain_text_still_streams_rather_than_being_held(self):
        """Holding an ambiguous tail must not stall an ordinary reply.

        An over-eager hold made plain replies emit NOTHING until generation
        finished — correct output, unusable streaming.
        """
        _, content = self._stream("Just plain prose with no markers whatsoever.")
        assert content.strip() == "Just plain prose with no markers whatsoever."
