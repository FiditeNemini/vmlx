"""CPU checks for submitted-only candidate observations, without MLX imports."""
import copy
from types import SimpleNamespace

import pytest

from .test_cache_cleanup_phase_timing import ROOT, load_functions
from .test_qwen4_prefill_reduce_route import fixture


def setup_status(monkeypatch):
    ns, switch, x, ids, scores, events = fixture(monkeypatch)
    ns.update(copy=copy, _STATUS={"observation": "submitted_only",
              "dispatch_calls": 0, "dispatched_rows": 0, "maximum_rows": 0})
    load_functions(ROOT / "metal/qwen4_prefill_reduce.py",
                   {"status", "_record_submission"}, ns)
    ns["weighted_unsort"] = lambda *args: SimpleNamespace(dtype="f16")
    return ns, switch, x, ids, scores


def test_success_records_submission_metadata_without_retaining_tensors(monkeypatch):
    ns, switch, x, ids, scores = setup_status(monkeypatch)
    output = ns["prefill_reduce"](switch, x, ids, scores)
    assert output.dtype == "f16"
    state = ns["status"]()
    assert state == {
        "observation": "submitted_only", "dispatch_calls": 1,
        "dispatched_rows": 33, "maximum_rows": 33, "last_rows": 33,
        "last_input_dtype": "f16", "last_score_dtype": "f16",
        "last_output_dtype": "f16", "requested": True,
        "projections": {name: {"bits": 4, "group_size": 64}
                        for name in ("up_proj", "gate_proj", "down_proj")},
    }
    # Metadata consists only of scalars/dicts and is detached at every depth.
    state["projections"]["up_proj"]["bits"] = 999
    state["dispatch_calls"] = 999
    assert ns["status"]()["projections"]["up_proj"]["bits"] == 4
    assert ns["status"]()["dispatch_calls"] == 1
    ns["prefill_reduce"](switch, x, ids, scores)
    assert ns["status"]()["dispatched_rows"] == 66


def test_declined_or_failed_submission_does_not_increment(monkeypatch):
    ns, switch, x, ids, scores = setup_status(monkeypatch)
    monkeypatch.setenv("VMLX_QWEN4_PREFILL_REDUCE", "0")
    assert ns["prefill_reduce"](switch, x, ids, scores) is None
    assert ns["status"]()["dispatch_calls"] == 0
    monkeypatch.setenv("VMLX_QWEN4_PREFILL_REDUCE", "1")

    def fail(*args):
        raise RuntimeError("submission failed")

    ns["weighted_unsort"] = fail
    with pytest.raises(RuntimeError, match="submission failed"):
        ns["prefill_reduce"](switch, x, ids, scores)
    assert ns["status"]()["dispatch_calls"] == 0


def test_existing_health_hook_includes_detached_candidate_status(monkeypatch):
    ns, switch, x, ids, scores = setup_status(monkeypatch)
    ns["prefill_reduce"](switch, x, ids, scores)
    health = {"_STATUS": {"installed": 0, "observed_calls": 1},
              "prefill_reduce_status": ns["status"]}
    load_functions(ROOT / "metal/qwen4_affine_moe_decode.py",
                   {"qwen4_affine_moe_status"}, health)
    observed = health["qwen4_affine_moe_status"]()
    assert observed["observed_calls"] == 1  # Existing owner's semantics unchanged.
    assert observed["prefill_reduce"]["dispatch_calls"] == 1
    observed["prefill_reduce"]["projections"]["up_proj"]["bits"] = 99
    assert health["qwen4_affine_moe_status"]()["prefill_reduce"]["projections"]["up_proj"]["bits"] == 4
