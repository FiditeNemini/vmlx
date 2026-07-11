"""C1: every visible-answer retry consumes the same total output cap."""

import inspect

import vmlx_engine.server as server


def test_remaining_answer_pass_budget_bounded_floor():
    """The answer pass draws the unspent cap, but is floored so a runaway
    reasoning pass that consumed the whole cap still gets a bounded budget to
    emit a visible answer (empty turn is worse than a small bounded overage).
    The overage is bounded by ANSWER_PASS_FLOOR, NOT the unbounded
    fresh-full-budget the audit removed."""
    budget = server._remaining_answer_pass_budget
    floor = server.ANSWER_PASS_FLOOR
    # ample remaining -> exact draw-down (no overage)
    assert budget(1024, 0) == 1024
    assert budget(1024, 600) == 424
    # exhausted cap -> floored (bounded overage, never zero/empty)
    assert budget(384, 384) == floor
    assert budget(384, 500) == floor
    # remaining above the floor is used as-is
    assert budget(1024, 800) == max(floor, 224)
    # explicit floor=0 -> strict draw-down (for callers that opt out)
    assert budget(384, 384, floor=0) == 0


def test_all_chat_and_responses_answer_passes_use_remaining_budget():
    source = inspect.getsource(server)
    assert source.count("_remaining_answer_pass_budget(") >= 9
    assert 'answer_kwargs["max_tokens"] = max(\n            256' not in source
    assert "_ns_budget = max(32" not in source

