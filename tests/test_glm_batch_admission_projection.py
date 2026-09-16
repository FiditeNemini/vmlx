# SPDX-License-Identifier: Apache-2.0
"""Execute the real health/cache-stats projections without importing MLX.

This isolates telemetry, not HTTP serving, model execution or queue behavior.
"""

import ast
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(params=[("health", "scheduler"), ("cache_stats", "scheduler_stats")])
def project(request):
    path = Path(__file__).resolve().parents[1] / "vmlx_engine" / "server.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    namespace = {"Any": Any}
    # Baseline has no helper; its actual projection still executes and fails
    # on the missing policy/false single-active field rather than import setup.
    helpers = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_scheduler_single_active_admission"
    ]
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(path), "exec"), namespace)
    handler, field = request.param
    function = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == handler)
    assignments = [
        node for node in ast.walk(function)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name) and target.value.id == "result"
            and isinstance(target.slice, ast.Constant) and target.slice.value == field
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    code = compile(ast.Expression(assignments[0].value), str(path), "eval")
    return lambda stats: eval(code, namespace, {"stats": stats, "scheduler_stats": stats})


def _policy(requested=1):
    fields = ("max_num_seqs", "prefill_batch_size", "completion_batch_size")
    return {
        "policy": "glm5_native_single_active",
        "requested": dict.fromkeys(fields, requested),
        "effective": dict.fromkeys(fields, 1),
        "concurrent_clients": "queued",
    }


@pytest.mark.parametrize("requested", [1, 512])
def test_glm_policy_is_projected_without_inventing_saved_preferences(project, requested):
    stats = {"batch_admission": _policy(requested), "batch_generator": {"single_active_decode": False}}
    before = deepcopy(stats)
    result = project(stats)
    assert result["batch_admission"] == stats["batch_admission"]
    assert result["batch_admission"]["requested"]["max_num_seqs"] == requested
    assert result["single_active_decode"] is True
    assert result["batch_generator"]["single_active_decode"] is False
    assert stats == before


@pytest.mark.parametrize("generator", [None, {}, {"single_active_decode": False}, {"single_active_decode": True}])
def test_other_families_keep_generator_semantics(project, generator):
    result = project({"batch_generator": generator})
    assert result["single_active_decode"] is bool((generator or {}).get("single_active_decode", False))
    assert result.get("batch_admission") is None


@pytest.mark.parametrize("change", ["unknown_policy", "missing_effective", "larger_batch", "boolean_batch"])
def test_unproven_policy_does_not_claim_single_active(project, change):
    policy = _policy()
    if change == "unknown_policy":
        policy["policy"] = "another_family"
    elif change == "missing_effective":
        policy.pop("effective")
    elif change == "larger_batch":
        policy["effective"]["max_num_seqs"] = 2
    else:
        policy["effective"]["max_num_seqs"] = True
    assert project({"batch_admission": policy})["single_active_decode"] is False
