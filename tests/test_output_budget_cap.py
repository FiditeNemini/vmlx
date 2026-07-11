"""C1: every visible-answer retry consumes the same total output cap."""

import inspect

import vmlx_engine.server as server


def test_remaining_answer_pass_budget_never_overdraws():
    budget = server._remaining_answer_pass_budget
    assert budget(64, 0) == 64
    assert budget(64, 63) == 1
    assert budget(64, 64) == 0
    assert budget(64, 80) == 0


def test_all_chat_and_responses_answer_passes_use_remaining_budget():
    source = inspect.getsource(server)
    assert source.count("_remaining_answer_pass_budget(") >= 9
    assert 'answer_kwargs["max_tokens"] = max(\n            256' not in source
    assert "_ns_budget = max(32" not in source

