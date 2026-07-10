# SPDX-License-Identifier: Apache-2.0
"""Variant-suffixed special tags are canonicalized at the token→text boundary.

Regression origin: Hy3-JANG_2L returned an EMPTY `content` with the whole
answer misfiled as `reasoning_content`, because the shipped tencent/Hy3
tokenizer closes reasoning with `</think:opensource>` and every vMLX parser
matches the bare `</think>`.
"""

from __future__ import annotations

import pytest

from vmlx_engine.special_tag_dialect import (
    build_canonical_map,
    compile_canonicalizer,
    detect_tag_variant,
)


class _Tok:
    """Minimal stand-in for an HF tokenizer's added-token table."""

    def __init__(self, contents):
        self.added_tokens_decoder = {
            120000 + i: type("AddedToken", (), {"content": c})()
            for i, c in enumerate(contents)
        }


HY3_TOKENS = [
    "<｜hy_begin_of_sentence:opensource｜>",
    "<｜hy_end▁of▁sentence｜>",
    "<｜hy_User:opensource｜>",
    "<｜hy_Assistant:opensource｜>",
    "<｜hy_eos:opensource｜>",
    "<think:opensource>",
    "</think:opensource>",
    "<answer>",
    "</answer>",
    "<tool_call:opensource>",
    "</tool_call:opensource>",
    "<tool_calls:opensource>",
    "</tool_calls:opensource>",
    "<tool_sep:opensource>",
    "<arg_key:opensource>",
    "</arg_key:opensource>",
]


@pytest.fixture
def canon():
    return compile_canonicalizer(build_canonical_map(_Tok(HY3_TOKENS)))


class TestDetect:
    def test_variant_detected_from_think_close_tag(self):
        assert detect_tag_variant(_Tok(HY3_TOKENS)) == "opensource"

    def test_no_variant_when_think_close_is_bare(self):
        assert detect_tag_variant(_Tok(["<think>", "</think>", "<｜eos｜>"])) is None

    def test_plain_tokenizer_yields_identity(self):
        assert build_canonical_map(_Tok(["<think>", "</think>"])) == {}
        assert compile_canonicalizer({}) is None


class TestCanonicalize:
    def test_reasoning_close_tag_rewritten(self, canon):
        raw = "Answer: 9.</think:opensource>9 sheep are left."
        assert canon(raw) == "Answer: 9.</think>9 sheep are left."

    def test_role_and_eos_markers_rewritten(self, canon):
        assert canon("x<｜hy_User:opensource｜>y") == "x<｜hy_User｜>y"
        assert canon("<｜hy_eos:opensource｜>") == "<｜hy_eos｜>"

    def test_tool_tags_rewritten(self, canon):
        raw = ("<tool_calls:opensource><tool_call:opensource>fn"
               "<tool_sep:opensource></tool_call:opensource></tool_calls:opensource>")
        assert canon(raw) == "<tool_calls><tool_call>fn<tool_sep></tool_call></tool_calls>"

    def test_tool_call_key_does_not_eat_tool_calls(self, canon):
        """Longest-first ordering: `<tool_call:x>` must not consume the prefix
        of `<tool_calls:x>` and leave a stray `s`."""
        assert canon("<tool_calls:opensource>") == "<tool_calls>"
        assert "<tool_call>s>" not in canon("<tool_calls:opensource>")
        assert canon("</tool_calls:opensource>") == "</tool_calls>"

    def test_unsuffixed_tokens_untouched(self, canon):
        assert canon("<answer>hi</answer>") == "<answer>hi</answer>"
        assert canon("<｜hy_end▁of▁sentence｜>") == "<｜hy_end▁of▁sentence｜>"

    def test_idempotent(self, canon):
        once = canon("a</think:opensource>b")
        assert canon(once) == once

    def test_empty_text(self, canon):
        assert canon("") == ""

    def test_reasoning_split_now_works(self, canon):
        """The actual defect: partition on `</think>` recovers the answer."""
        raw = "reasoning here</think:opensource>The answer is 9."
        _, sep, visible = canon(raw).partition("</think>")
        assert sep == "</think>"
        assert visible == "The answer is 9."


class TestStreamingDetokenizer:
    def test_offsets_stay_in_canonical_space(self):
        """`last_segment` slices `self.text` by `self.offset`; the override must
        keep both in canonical-text space so no bytes are dropped or repeated."""
        from vmlx_engine.special_tag_dialect import compile_canonicalizer

        canon = compile_canonicalizer(build_canonical_map(_Tok(HY3_TOKENS)))

        class FakeNaive:
            def __init__(self):
                self.offset = 0
                self._raw = ""

            @property
            def text(self):
                return self._raw

            @property
            def last_segment(self):
                t = self.text
                seg = t[self.offset:]
                self.offset = len(t)
                return seg

        class Canonical(FakeNaive):
            @property
            def text(self):
                return canon(FakeNaive.text.fget(self))

        d = Canonical()
        d._raw = "Answer: 9."
        assert d.last_segment == "Answer: 9."
        d._raw += "</think:opensource>"
        assert d.last_segment == "</think>"
        d._raw += "9 sheep."
        assert d.last_segment == "9 sheep."
        assert d.text == "Answer: 9.</think>9 sheep."
