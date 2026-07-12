"""deepseek_v4 / step3p7 join the never-empty answer-pass families.

Regression guard for the DSV4/Step-3.7 empty-content-after-reasoning fix:
- Both must be in _REASONING_ANSWER_PASS_FAMILIES so the bounded thinking-off
  answer pass fires when the thinking block consumes the whole budget.
- Both must be EXCLUDED from _THINKING_BUDGET_CAP_FAMILIES: they key on
  reasoning_effort, not a thinking-token budget, so max_thinking_tokens is an
  inert field and must not cap max_tokens for them (#89).
"""
import vmlx_engine.server as server_mod


def test_dsv4_and_step37_in_answer_pass_families():
    fams = server_mod._REASONING_ANSWER_PASS_FAMILIES
    assert "deepseek_v4" in fams
    assert "step3p7" in fams
    # existing families still covered
    for f in ("qwen3_5", "qwen3_5_moe", "gemma4", "hy_v3", "laguna",
              "minimax_m2", "openpangu_v2"):
        assert f in fams


def test_dsv4_and_step37_excluded_from_thinking_budget_cap():
    cap = server_mod._THINKING_BUDGET_CAP_FAMILIES
    assert "deepseek_v4" not in cap
    assert "step3p7" not in cap
    # families that DO honor a thinking-token budget remain capped
    for f in ("qwen3_5", "qwen3_5_moe", "gemma4", "hy_v3", "laguna",
              "minimax_m2", "openpangu_v2"):
        assert f in cap


def test_answer_pass_labels_for_new_families():
    label = server_mod._reasoning_answer_pass_family_label
    assert label("deepseek_v4") == "DeepSeek-V4"
    assert label("step3p7") == "Step-3.7"
    # unchanged defaults
    assert label("hy_v3") == "Hy3"
    assert label("qwen3_5") == "Qwen3.5"
