"""Stop boundaries must not leak, including across token/verify blocks."""
import pytest
from vmlx_engine.dflash2_runtime import _StopDetokenizer


class Pieces:
    def __init__(self):
        self.segment = ""
    def add_token(self, token):
        self.segment += token
    def finalize(self):
        pass
    @property
    def last_segment(self):
        text, self.segment = self.segment, ""
        return text


@pytest.mark.parametrize("pieces,stops,expected,matched", [
    (["alpha ", "ST", "OP", " omega"], ["STOP"], "alpha ", True),
    (["alpha STOP omega"], ["STOP"], "alpha ", True),
    (["S", "T"], ["STOP"], "ST", False),
    (["ST", "uff"], ["STOP"], "STuff", False),
    (["abcENDtail"], ["END", "bc"], "a", True),
    (["ab", "ab", "x"], ["abab"], "", True),
    (["hello", " world"], ["STOP"], "hello world", False),
])
def test_stop_stream_boundary(pieces, stops, expected, matched):
    d = _StopDetokenizer(Pieces(), stops)
    output = ""
    for piece in pieces:
        d.add_token(piece)
        output += d.last_segment
        if d.matched:
            break
    d.finalize()
    output += d.last_segment
    assert output == expected
    assert d.matched == matched


def test_incomplete_stop_is_withheld_until_disambiguated():
    d = _StopDetokenizer(Pieces(), ["STOP"])
    d.add_token("text ST")
    assert d.last_segment == "text "
    d.add_token("uff")
    assert d.last_segment == "STuff"
