"""DSV4 answer-reserve split table.

DeepSeek V4 Flash does not reliably emit `</think>`, so without reserving part
of the output budget for the answer the reasoning phase consumes all of it and
the visible answer is empty. The reserve is a bounded slice rather than a
percentage so that deep reasoning is not taxed at large budgets.

Three properties are pinned here:

  * small budgets still split. A flat 256-token floor previously made every
    budget under 512 unsplittable, and 160/256/400 all measured content=0 while
    512 and above answered.
  * nothing at or above 512 changed when that floor was fixed.
  * every splittable budget (>= 2 tokens) splits. Budgets under 128 were left
    unsplit, which disarmed the never-empty answer pass entirely: reasoning-only
    runs returned empty content at finish=length, and a hard 502
    reasoning_only_no_content when the model EOSed inside the thinking block
    (live-hit 2026-08-07: max_tokens=48 at 243K ctx).
"""
import pytest

from vmlx_engine.server import (
    _dsv4_answer_pass_thinking_cap,
    _dsv4_default_thinking_cap,
)


class TestSplitTable:
    @pytest.mark.parametrize("budget,cap", [
        (512, 256),
        (700, 444),
        (1000, 744),
        (2000, 1500),
        (8000, 6000),
        (16384, 14336),
        (32000, 29952),
        (393216, 391168),
    ])
    def test_budgets_at_or_above_512_are_unchanged(self, budget, cap):
        assert _dsv4_default_thinking_cap(budget) == cap

    @pytest.mark.parametrize("budget", [128, 160, 256, 400])
    def test_small_budgets_now_split_instead_of_returning_empty(self, budget):
        cap = _dsv4_default_thinking_cap(budget)
        assert cap is not None, budget
        assert 0 < cap < budget
        # The answer must get a real share of a small budget, not a token or two.
        assert budget - cap >= budget // 4

    @pytest.mark.parametrize("budget", [0, 1])
    def test_budgets_too_small_to_split_are_left_alone(self, budget):
        assert _dsv4_default_thinking_cap(budget) is None

    @pytest.mark.parametrize("budget", [2, 8, 48, 64, 127])
    def test_sub_128_budgets_split_so_the_answer_pass_can_arm(self, budget):
        # Unsplit budgets disarm the never-empty answer pass: reasoning-only
        # runs return empty content (finish=length) or a hard 502 (EOS inside
        # the thinking block). The reserve is exactly half the budget here.
        cap = _dsv4_default_thinking_cap(budget)
        assert cap is not None, budget
        assert 0 < cap < budget
        assert budget - cap == budget // 2

    def test_reserve_never_exceeds_the_ceiling(self):
        for budget in (2000, 8000, 32000, 131072, 393216):
            cap = _dsv4_default_thinking_cap(budget)
            assert budget - cap <= 2048, budget

    def test_thinking_share_grows_with_the_budget(self):
        budgets = [512, 2000, 8000, 32000, 393216]
        shares = [_dsv4_default_thinking_cap(b) / b for b in budgets]
        assert shares == sorted(shares)
        assert shares[-1] > 0.99

    def test_deep_reasoning_is_not_taxed_at_the_card_maximum(self):
        # The model card allows up to 384K output tokens at high/max effort.
        cap = _dsv4_default_thinking_cap(393216)
        assert cap / 393216 > 0.99


class TestFamilyGate:
    def test_only_server_managed_budget_families_are_affected(self):
        for family in ("qwen3", "laguna", "minimax_m3", "gemma4", "llama", None):
            assert _dsv4_answer_pass_thinking_cap(family, True, 8000) is None, family

    def test_thinking_off_is_a_no_op(self):
        assert _dsv4_answer_pass_thinking_cap("deepseek_v4", False, 8000) is None
        assert _dsv4_answer_pass_thinking_cap("nemotron_h", False, 8000) is None

    def test_deepseek_v4_with_thinking_on_splits(self):
        assert _dsv4_answer_pass_thinking_cap("deepseek_v4", True, 8000) == 6000

    def test_thinking_unspecified_still_splits(self):
        # Auto/None must behave like on: the rail is open by default.
        assert _dsv4_answer_pass_thinking_cap("deepseek_v4", None, 8000) == 6000

    def test_nemotron_gets_the_default_reserve_too(self):
        """The bundle declares budget: server_side and callers never send
        max_thinking_tokens, so without a default reserve the never-empty
        answer pass can never arm — live-hit through the real UI 2026-08-11
        (Lightning MXFP8, max_tokens=120: 393 chars reasoning, EMPTY content)
        even with the family already in the answer-pass sets."""
        for family in ("nemotron", "nemotron_h"):
            assert _dsv4_answer_pass_thinking_cap(family, True, 8000) == 6000, family
            assert _dsv4_answer_pass_thinking_cap(family, None, 8000) == 6000, family

    def test_qwen4_exp_is_not_in_the_answer_pass_sets(self):
        """qwen4_exp's enrollment (f58c9c196) was corrected 2026-08-28: it
        hard-split the first pass, then silently reran a second thinking-off
        generation with tools popped and a FABRICATED reasoning_content
        history item spliced in ahead of it — forced behavior forbidden by
        AGENTS.md:54, not a legitimate cap. Removed, not relocated to
        _ANSWER_PASS_FRESH_CONTEXT_FAMILIES or any other exemption set —
        that would still be a hidden second generation."""
        from vmlx_engine.server import (
            _DEFAULT_ANSWER_RESERVE_FAMILIES,
            _REASONING_ANSWER_PASS_FAMILIES,
            _THINKING_BUDGET_CAP_FAMILIES,
            _ANSWER_PASS_FRESH_CONTEXT_FAMILIES,
        )

        assert "qwen4_exp" not in _DEFAULT_ANSWER_RESERVE_FAMILIES
        assert "qwen4_exp" not in _REASONING_ANSWER_PASS_FAMILIES
        assert "qwen4_exp" not in _THINKING_BUDGET_CAP_FAMILIES
        assert "qwen4_exp" not in _ANSWER_PASS_FRESH_CONTEXT_FAMILIES

    def test_qwen4_exp_gets_no_hard_cap_at_any_thinking_state(self):
        # _dsv4_answer_pass_thinking_cap gates on _DEFAULT_ANSWER_RESERVE_FAMILIES;
        # qwen4_exp is not a member, so it must return None unconditionally,
        # not a split budget.
        for effective_thinking in (True, False, None):
            assert _dsv4_answer_pass_thinking_cap("qwen4_exp", effective_thinking, 8000) is None

    def test_nemotron_reserve_env_override_is_family_neutral(self, monkeypatch):
        """Nemotron honors VMLX_ANSWER_RESERVE, and the DSV4-named env does
        not leak across families."""
        monkeypatch.setenv("VMLX_ANSWER_RESERVE", "0")
        assert _dsv4_answer_pass_thinking_cap("nemotron_h", True, 8000) is None
        assert _dsv4_answer_pass_thinking_cap("deepseek_v4", True, 8000) == 6000
        monkeypatch.delenv("VMLX_ANSWER_RESERVE")
        monkeypatch.setenv("DSV4_ANSWER_RESERVE", "0")
        assert _dsv4_answer_pass_thinking_cap("nemotron_h", True, 8000) == 6000
        assert _dsv4_answer_pass_thinking_cap("deepseek_v4", True, 8000) is None
