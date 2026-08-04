"""DSV4 answer-reserve split table.

DeepSeek V4 Flash does not reliably emit `</think>`, so without reserving part
of the output budget for the answer the reasoning phase consumes all of it and
the visible answer is empty. The reserve is a bounded slice rather than a
percentage so that deep reasoning is not taxed at large budgets.

Two properties are pinned here:

  * small budgets still split. A flat 256-token floor previously made every
    budget under 512 unsplittable, and 160/256/400 all measured content=0 while
    512 and above answered.
  * nothing at or above 512 changed when that floor was fixed.
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

    @pytest.mark.parametrize("budget", [0, 1, 64, 127])
    def test_budgets_too_small_to_split_are_left_alone(self, budget):
        assert _dsv4_default_thinking_cap(budget) is None

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
    def test_only_deepseek_v4_is_affected(self):
        for family in ("qwen3", "laguna", "minimax_m3", "gemma4", "llama", None):
            assert _dsv4_answer_pass_thinking_cap(family, True, 8000) is None, family

    def test_thinking_off_is_a_no_op(self):
        assert _dsv4_answer_pass_thinking_cap("deepseek_v4", False, 8000) is None

    def test_deepseek_v4_with_thinking_on_splits(self):
        assert _dsv4_answer_pass_thinking_cap("deepseek_v4", True, 8000) == 6000

    def test_thinking_unspecified_still_splits(self):
        # Auto/None must behave like on: the rail is open by default.
        assert _dsv4_answer_pass_thinking_cap("deepseek_v4", None, 8000) == 6000
