"""DFlash2 verify width chooser: expected tokens (from per-position acceptance) per measured second."""
import random

from vmlx_engine.dflash2_runtime import _BlockChooser, _dflash2_block_plan


def _run(p, cost5, cost8, cycles=200, seed=0):
    rng, c, seq = random.Random(seed), _BlockChooser(5, 8), []
    for _ in range(cycles):
        w = c.width
        seq.append(w)
        accepted = 0
        while accepted < w - 1 and rng.random() < p:
            accepted += 1
        c.observe(w, accepted + 1, cost5 if w == 5 else cost8)
    return seq


def test_expensive_wide_verify_on_prose_stays_at_five():   # JANG_4D: 8 rows cost 1.49x
    seq = _run(0.55, 0.053, 0.079)
    assert seq.count(5) > 180 and 8 in seq                 # still re-times 8


def test_cheap_wide_verify_on_easy_text_goes_to_eight():   # JANGH2: 8 rows cost 1.13x
    seq = _run(0.97, 0.078, 0.088)
    assert seq.count(8) > 180 and 5 in seq


def test_width_five_cycles_inform_width_eight():
    c = _BlockChooser(5, 8)
    for _ in range(20):
        c.observe(5, 5, 0.05)                              # every block fully accepted
    assert c.expected_tokens(8) > 6.5


def test_clipped_final_block_is_ignored():
    c = _BlockChooser(5, 8)
    c.observe(3, 3, 0.04)
    assert c.cycles == 0 and c.cost == {}


def test_fixed_width_env(monkeypatch):
    monkeypatch.setenv("VMLX_DFLASH2_BLOCK", "8")
    assert _dflash2_block_plan(8) == (8, 8)
    monkeypatch.setenv("VMLX_DFLASH2_BLOCK", "auto")
    assert _dflash2_block_plan(8) == (5, 8)
    assert _dflash2_block_plan(4) == (4, 4)


def test_first_cycle_at_a_width_is_warmup_and_costs_persist_across_requests():
    shared: dict = {}
    c = _BlockChooser(5, 8, shared)
    widths = []
    for _ in range(4):                                     # warm 5, time 5, warm 8, time 8
        w = c.width
        widths.append(w)
        c.observe(w, w, 2.0 if w not in shared.get("warm", set()) else (0.05 if w == 5 else 0.06))
    assert widths == [5, 5, 8, 8]
    assert shared[5] == 0.05 and shared[8] == 0.06        # the 2.0 s warm-up cycles were discarded
    c2 = _BlockChooser(5, 8, shared)                       # the next request starts with measured costs
    c2.observe(5, 5, 0.05)
    assert c2.width == 8                                   # 5/5 accepted -> 8 predicted to pay at 1.2x cost
