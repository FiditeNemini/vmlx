

class TestProtocolResidueDoesNotEatProse:
    """"assistant", "analysis" and "final" are ordinary English words.

    The leading/trailing bare-word strips ran unconditionally, so any reply
    ENDING in one lost its last word and any reply STARTING with one lost its
    first. MEASURED before the guard:
        "I am your assistant"         -> "I am your"
        "Give this to your assistant" -> "Give this to your"
        "Run the analysis"            -> "Run the"
        "This is final"               -> "This is"
        "assistant helped me"         -> "helped me"
    The strips now require evidence of actual protocol debris: either a marker
    fragment was removed from this text, or the bare word IS the whole text.
    """

    def _clean(self, text):
        from vmlx_engine.reasoning.gptoss_parser import GptOssReasoningParser

        return GptOssReasoningParser()._clean_protocol_residue(text)

    def test_prose_ending_in_a_protocol_word_is_preserved(self):
        for text in (
            "I am your assistant",
            "Give this to your assistant",
            "Run the analysis",
            "This is final",
        ):
            assert self._clean(text) == text

    def test_prose_starting_with_a_protocol_word_is_preserved(self):
        assert self._clean("assistant helped me") == "assistant helped me"

    def test_ordinary_text_is_untouched(self):
        assert self._clean("The final answer is 42") == "The final answer is 42"
        assert self._clean("The report is complete.") == "The report is complete."

    def test_marker_fragments_are_still_cleaned(self):
        assert self._clean("<|channel|>final<|message|>Hello there") == "Hello there"
        assert self._clean("<|assistant Hello") == "Hello"
        assert self._clean("<|start|>assistant The answer is 7") == "The answer is 7"

    def test_a_bare_protocol_word_alone_is_still_dropped(self):
        for text in ("final", "assistant", "  analysis  "):
            assert self._clean(text) == ""
