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


_MSGS = [{"role": "user", "content": "Remember codeword BLUE-FALCON."}]
_TRUNC = "We are given the task. Interpretation: We need BLUE-F"


def test_answer_pass_fresh_context_families():
    """Malformed/double assistant templates and Qwen's live-proven planning
    continuation must re-run the ORIGINAL messages with nothing appended."""
    for fam in (
        "deepseek_v4", "step3p7", "minimax", "minimax_m2",
        "qwen3_5", "qwen3_5_moe",
    ):
        out = server_mod._answer_pass_messages(_MSGS, fam, _TRUNC)
        assert out == _MSGS
        assert out is not _MSGS  # fresh copy, caller list not aliased


def test_answer_pass_appends_reasoning_turn_for_legacy_families():
    """Other legacy families keep the truncated
    reasoning rides along as an assistant turn."""
    for fam in ("gemma4", "hy_v3", "laguna", "openpangu_v2", None,
                "reasoning model"):
        out = server_mod._answer_pass_messages(_MSGS, fam, _TRUNC)
        assert out[:-1] == _MSGS
        assert out[-1] == {
            "role": "assistant",
            "content": "",
            "reasoning_content": _TRUNC,
        }


def test_leak_guard_families_keep_qwen_progressive():
    """Qwen's fresh direct rail streams progressively; families with a
    live-proven thinking re-entry remain buffered."""
    guard = server_mod._ANSWER_PASS_LEAK_GUARD_FAMILIES
    for fam in ("deepseek_v4", "step3p7", "minimax", "minimax_m2"):
        assert fam in guard
    for fam in ("qwen3_5", "qwen3_5_moe"):
        assert fam not in guard
    # families with live-proven coherent partial salvages stay unbuffered
    for fam in ("gemma4", "hy_v3", "laguna", "openpangu_v2"):
        assert fam not in guard


def test_thinking_reentry_matches_tag_variants():
    """DSV4 live-emitted "<thinking>..." (deterministic 3/3). A literal
    "<think>" needle does NOT match it — the closing ">" never aligns with
    "ing>" — which is exactly the miss the open-prefix helper fixes."""
    reentry = server_mod._answer_pass_thinking_reentry
    assert "<think>" not in "<thinking>Let's parse the user's request."
    assert reentry("<thinking>Let's parse the user's request.")
    assert reentry("<think>\nplanning\n</think>")
    assert reentry("prefix text <think>")
    assert not reentry("A clean answer: BLUE-FALCON 37 Paris.")
    assert not reentry("")
    assert not reentry(None)


def test_minimax_family_armed_for_answer_pass():
    """MiniMax-M2.x bundles report family_name "minimax" — the parser name
    "minimax_m2" alone left M2.7 reasoning-only turns EMPTY (live-proven
    2026-07-12). Both spellings must arm the rail and label as MiniMax-M2."""
    assert "minimax" in server_mod._REASONING_ANSWER_PASS_FAMILIES
    assert "minimax" in server_mod._THINKING_BUDGET_CAP_FAMILIES
    label = server_mod._reasoning_answer_pass_family_label
    assert label("minimax") == "MiniMax-M2"
    assert label("minimax_m2") == "MiniMax-M2"
