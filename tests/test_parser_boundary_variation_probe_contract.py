# SPDX-License-Identifier: Apache-2.0
"""Contract pins for bench/parser_boundary_variation_probe.py (campaign #181
arms A2/A3/A7). Mirrors the DSV4 equivalence-probe contract style: the probe
module must load standalone, demand a nonce, exercise the variation matrix,
and never pass without proven reuse."""

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "bench"


@pytest.fixture(scope="module")
def probe() -> Any:
    path = BENCH_DIR / "parser_boundary_variation_probe.py"
    spec = importlib.util.spec_from_file_location(
        "parser_boundary_variation_probe", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_nonce_required_with_no_default(probe: Any) -> None:
    parser = probe.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "/tmp/x", "--out", "/tmp/y.json"])


def test_script_covers_the_variation_matrix(probe: Any) -> None:
    script = probe.conversation_script("NONCE-ABC")
    keys = [turn["key"] for turn in script]
    assert keys == [
        "t1_plain",
        "t2_preserved_thinking",
        "t3_tools_appear",
        "t4_toolset_changes",
        "t5_tools_off_effort_change",
        "t6_reestablish",
    ]
    assert all("NONCE-ABC" in turn["user"] for turn in script)
    # Preserved thinking rides into t2/t3 history; tools appear at t3 and the
    # SET changes at t4; t5 drops tools and changes effort; t6 repeats t5's
    # settings (the re-establishment turn).
    assert script[1]["preserve_prior_reasoning"] is True
    assert script[2]["preserve_prior_reasoning"] is True
    assert script[2]["tools"] and len(script[2]["tools"]) == 1
    assert script[3]["tools"] and len(script[3]["tools"]) == 2
    assert script[4]["tools"] is None
    assert script[4]["reasoning_effort"] == "medium"
    assert script[5]["tools"] is None
    assert script[5]["reasoning_effort"] == "medium"


def test_verdict_never_passes_without_reuse(probe: Any) -> None:
    cold = [
        {"key": "t1_plain", "status_code": 200, "content": "A", "cached_tokens": 0}
    ]
    warm = [
        {"key": "t1_plain", "status_code": 200, "content": "A", "cached_tokens": 0}
    ]
    status, _ = probe.compute_verdict(cold, warm, table_hit_evidence=False)
    assert status == "inconclusive_no_reuse"
    status, _ = probe.compute_verdict(cold, warm, table_hit_evidence=True)
    assert status == "pass"


def test_verdict_fails_on_any_byte_divergence(probe: Any) -> None:
    cold = [
        {"key": "t1_plain", "status_code": 200, "content": "A", "cached_tokens": 0}
    ]
    warm = [
        {"key": "t1_plain", "status_code": 200, "content": "B", "cached_tokens": 64}
    ]
    status, turns = probe.compute_verdict(cold, warm, table_hit_evidence=False)
    assert status == "fail"
    assert turns[0]["byte_equal"] is False


def test_effort_stays_in_qwen38_stamped_set(probe: Any) -> None:
    # qwen3.8 stamps low/medium/xhigh; the probe's effort-change turn must use
    # an IN-SET tier so the substitution policy never rewrites the request
    # (that would silently change what the A/B exercises).
    script = probe.conversation_script("N")
    efforts = {t["reasoning_effort"] for t in script if t["reasoning_effort"]}
    assert efforts == {"medium"}
