"""Terminal diagnostics must not modify the measured speculation policy."""
from copy import deepcopy

from vmlx_engine.dspark.generate import ARFallback


def test_cost_snapshot_is_read_only_and_detached():
    policy = ARFallback()
    policy.record_draft(2, 0.03, rows=4)
    policy.record_ar(0.01)
    before = deepcopy(policy.__dict__)
    snapshot = policy.cost_snapshot()
    assert snapshot == {
        "ar_step_seconds": 0.01,
        "draft_tokens_per_second": 2 / 0.03,
        "round_seconds_by_rows": {4: 0.03},
    }
    snapshot["round_seconds_by_rows"][4] = 100
    assert policy.__dict__ == before
    control = ARFallback()
    control.__dict__.update(deepcopy(before))
    assert policy.next_is_ar() == control.next_is_ar()
    assert policy.__dict__ == control.__dict__


def test_unmeasured_costs_remain_missing():
    assert ARFallback().cost_snapshot() == {
        "ar_step_seconds": None,
        "draft_tokens_per_second": None,
        "round_seconds_by_rows": {},
    }
