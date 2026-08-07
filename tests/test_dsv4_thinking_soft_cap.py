"""DSV4 answer-reserve thinking SOFT cap.

The v1.6.23 reserve reduced the first pass's max_tokens outright. When the
model naturally closed reasoning and began the visible answer just before that
reduced cap, the pass ended finish=length mid-sentence and the reserved tokens
were stranded — the answer pass only arms when NO content was emitted.

The soft cap keeps the full output budget on the request and enforces the
thinking budget only while generation is still inside the thinking rail. When
the model emits </think>, the cap lifts and the SAME pass continues into the
reserved budget. Reasoning-only runs still finish=length at the cap, so the
fresh-context answer pass arms exactly as before. No tokens are ever injected.
"""
import pytest

from vmlx_engine.utils.dsv4_batch_generator import (
    DSV4BatchGenerator,
    DSV4_EOS_ID,
    THINK_CLOSE_ID,
    THINK_OPEN_ID,
    _Request,
)


def _gen(stop_tokens=frozenset()):
    gen = object.__new__(DSV4BatchGenerator)
    gen.stop_tokens = set(stop_tokens)
    return gen


def _req(max_tokens, soft_cap=None, visible_started=False, out=0):
    return _Request(
        uid=1,
        prompt_tokens=[1, 2, 3],
        context_tokens=[1, 2, 3],
        cache=None,
        out_tokens=[7] * out,
        max_tokens=max_tokens,
        thinking_soft_cap=soft_cap,
        visible_started=visible_started,
    )


class TestEffectiveMaxTokens:
    def test_no_soft_cap_uses_full_budget(self):
        assert _gen()._effective_max_tokens(_req(220)) == 220

    def test_inside_thinking_rail_enforces_the_cap(self):
        assert _gen()._effective_max_tokens(_req(220, soft_cap=110)) == 110

    def test_after_think_close_the_full_budget_returns(self):
        req = _req(220, soft_cap=110, visible_started=True)
        assert _gen()._effective_max_tokens(req) == 220

    def test_cap_never_raises_the_budget(self):
        assert _gen()._effective_max_tokens(_req(64, soft_cap=110)) == 64


class TestFinishReason:
    def test_think_close_lifts_the_cap_and_continues_the_same_pass(self):
        gen, req = _gen(), _req(220, soft_cap=110, out=107)
        gen._update_finish_reason_after_token(req, THINK_CLOSE_ID)
        assert req.visible_started is True
        assert req.finish_reason is None  # pass continues into the reserve

    def test_reasoning_only_still_lengths_at_the_cap(self):
        # The answer pass arms off this exact outcome: finish=length with no
        # visible content at completion == thinking cap.
        gen, req = _gen(), _req(220, soft_cap=110, out=110)
        gen._update_finish_reason_after_token(req, 42)
        assert req.finish_reason == "length"
        assert req.visible_started is False

    def test_content_just_before_the_cap_is_not_truncated(self):
        # The v1.6.23 defect: </think> at token ~107 of a 110 hard cap ended
        # the stream mid-sentence. With the soft cap the pass runs on.
        gen, req = _gen(), _req(220, soft_cap=110, out=108)
        gen._update_finish_reason_after_token(req, THINK_CLOSE_ID)
        assert req.finish_reason is None
        req.out_tokens.extend([7] * 30)  # tokens 109..138 of 220
        gen._update_finish_reason_after_token(req, 42)
        assert req.finish_reason is None

    def test_lifted_pass_still_lengths_at_the_full_budget(self):
        gen, req = _gen(), _req(220, soft_cap=110, visible_started=True, out=220)
        gen._update_finish_reason_after_token(req, 42)
        assert req.finish_reason == "length"

    def test_eos_still_stops(self):
        gen, req = _gen(), _req(220, soft_cap=110, out=50)
        gen._update_finish_reason_after_token(req, DSV4_EOS_ID)
        assert req.finish_reason == "stop"

    def test_stop_token_wins_over_the_cap(self):
        gen, req = _gen(stop_tokens={99}), _req(220, soft_cap=110, out=110)
        gen._update_finish_reason_after_token(req, 99)
        assert req.finish_reason == "stop"

    def test_think_open_does_not_lift(self):
        gen, req = _gen(), _req(220, soft_cap=110, out=50)
        gen._update_finish_reason_after_token(req, THINK_OPEN_ID)
        assert req.visible_started is False


class TestSchedulerRemainingSoftCap:
    def _scheduler(self):
        from vmlx_engine.scheduler import Scheduler
        return object.__new__(Scheduler)

    class _FakeRequest:
        def __init__(self, cap=None, total_output_tokens=0):
            if cap is not None:
                self._dsv4_thinking_soft_cap = cap
            self.total_output_tokens = total_output_tokens

    def test_absent_cap_is_none(self):
        s = self._scheduler()
        assert s._dsv4_remaining_thinking_soft_cap(self._FakeRequest()) is None

    def test_fresh_request_gets_the_full_cap(self):
        s = self._scheduler()
        assert s._dsv4_remaining_thinking_soft_cap(self._FakeRequest(110)) == 110

    def test_reinserted_request_resumes_with_the_remainder(self):
        # Mirrors Request.remaining_output_budget: total_output_tokens is a
        # lifetime counter, so a rescheduled request must not get a fresh cap.
        s = self._scheduler()
        req = self._FakeRequest(110, total_output_tokens=40)
        assert s._dsv4_remaining_thinking_soft_cap(req) == 70

    @pytest.mark.parametrize("cap", [0, -5, "junk"])
    def test_invalid_caps_are_ignored(self, cap):
        s = self._scheduler()
        assert s._dsv4_remaining_thinking_soft_cap(self._FakeRequest(cap)) is None
