"""C1: explicit visible-answer retries stay inside the request output cap."""

import inspect

import vmlx_engine.server as server


def test_remaining_answer_pass_budget_is_strict_draw_down():
    """No answer pass may overrun the caller's total output-token cap."""
    budget = server._remaining_answer_pass_budget
    assert budget(1024, 0) == 1024
    assert budget(1024, 600) == 424
    assert budget(1024, 800) == 224
    assert budget(384, 384) == 0
    assert budget(384, 500) == 0
    assert budget(0, 0) == 0


def test_auto_reasoning_has_no_implicit_partition_or_retry_policy():
    """Absent max_thinking_tokens must remain one native full-cap sample."""
    source = inspect.getsource(server)
    assert "AUTO_THINKING_PASS_LIMIT" not in source
    assert "AUTO_VISIBLE_ANSWER_RESERVE" not in source
    assert "_AUTO_THINKING_PARTITION_FAMILIES" not in source
    assert "_auto_thinking_pass_budget" not in source
    assert "_auto_thinking_partition_allowed" not in source


def test_explicit_chat_and_responses_answer_passes_use_remaining_budget():
    source = inspect.getsource(server)
    assert source.count("_remaining_answer_pass_budget(") >= 5
    assert 'answer_kwargs["max_tokens"] = max(\n            256' not in source
    assert "_ns_budget = max(32" not in source
    assert "ANSWER_PASS_FLOOR" not in source
