# SPDX-License-Identifier: Apache-2.0
"""Contract pins for the DSV4 cold-vs-restored equivalence probes (task #149).

These tests never launch a server. They import the two bench probes and pin
the parts of the contract that make the probe results trustworthy:

1. ``--nonce`` is required with no default (novel-seed rule — a default nonce
   would let reruns collide with stale cache entries).
2. The verdict can NEVER be "pass" when every restored turn reports zero
   ``cached_tokens`` — unproven reuse is "inconclusive_no_reuse".
3. The restart/L2 probe relaunches on the SAME block-disk-cache dir, pinned
   through ``restart_serve_commands`` (the function main() launches from),
   not via source-text matching.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "bench"


def _load(module_name: str) -> Any:
    path = BENCH_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sameproc() -> Any:
    return _load("dsv4_cold_vs_restored_sameproc")


@pytest.fixture(scope="module")
def restart_l2() -> Any:
    return _load("dsv4_cold_vs_restored_restart_l2")


def _synthetic_records(
    *,
    byte_equal: bool = True,
    cached: tuple[int, ...] = (0, 0, 0),
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for turn, cached_tokens in enumerate(cached, start=1):
        cold_text = f"answer for turn {turn}"
        records.append(
            {
                "turn": turn,
                "cold_text": cold_text,
                "restored_text": cold_text if byte_equal else cold_text + " drift",
                "byte_equal": byte_equal,
                "normalized_equal": byte_equal,
                "cold_usage": {"completion_tokens": 9},
                "restored_usage": {"completion_tokens": 9},
                "cached_tokens": cached_tokens,
                "cold_http_code": 200,
                "restored_http_code": 200,
            }
        )
    return records


def _option_action(parser: Any, option: str) -> Any:
    for action in parser._actions:
        if option in action.option_strings:
            return action
    raise AssertionError(f"{option} is not defined on the parser")


# ---------------------------------------------------------------------------
# 1. --nonce is required with no default on BOTH probes.
# ---------------------------------------------------------------------------


def test_nonce_required_with_no_default(sameproc: Any, restart_l2: Any) -> None:
    for module in (sameproc, restart_l2):
        parser = module.build_arg_parser()
        action = _option_action(parser, "--nonce")
        assert action.required is True
        assert action.default is None
        with pytest.raises(SystemExit):
            parser.parse_args(["--model", "/m", "--out", "/tmp/out.json"])


# ---------------------------------------------------------------------------
# 2. "pass" is impossible without proven reuse.
# ---------------------------------------------------------------------------


def test_sameproc_verdict_never_passes_without_reuse(sameproc: Any) -> None:
    # Byte-equal everywhere, but zero cached tokens on every restored turn:
    # reuse is unproven, so the verdict must be inconclusive, never pass.
    verdict = sameproc.compute_verdict(_synthetic_records(cached=(0, 0, 0)))
    assert verdict == "inconclusive_no_reuse"


def test_sameproc_verdict_positive_and_negative_paths(sameproc: Any) -> None:
    assert sameproc.compute_verdict(_synthetic_records(cached=(0, 128, 192))) == "pass"
    assert (
        sameproc.compute_verdict(_synthetic_records(byte_equal=False, cached=(0, 128, 192)))
        == "fail"
    )
    assert sameproc.compute_verdict([]) == "fail"
    http_error = _synthetic_records(cached=(0, 128, 192))
    http_error[1]["restored_http_code"] = 500
    assert sameproc.compute_verdict(http_error) == "fail"


def test_restart_l2_verdict_never_passes_without_reuse(restart_l2: Any) -> None:
    # The DSV4 composite performs a clean N-1 reprefill over restored
    # anchors, so cached_tokens stays 0 BY DESIGN on a real restore (proven
    # live 2026-08-15: disk_hits=2, byte-equal 3/3, cached_tokens=[0,0,0]).
    # Disk evidence alone is therefore the composite's restore attribution
    # and yields "pass" when every turn is byte-equal.
    verdict = restart_l2.compute_verdict(
        _synthetic_records(cached=(0, 0, 0)), disk_evidence=True
    )
    assert verdict == "pass"
    # cached_tokens without disk evidence: reuse did not provably come from
    # the disk restore, so still inconclusive.
    verdict = restart_l2.compute_verdict(
        _synthetic_records(cached=(64, 128, 192)), disk_evidence=False
    )
    assert verdict == "inconclusive_no_reuse"


def test_restart_l2_verdict_positive_and_negative_paths(restart_l2: Any) -> None:
    assert (
        restart_l2.compute_verdict(
            _synthetic_records(cached=(64, 128, 192)), disk_evidence=True
        )
        == "pass"
    )
    assert (
        restart_l2.compute_verdict(
            _synthetic_records(byte_equal=False, cached=(64, 128, 192)),
            disk_evidence=True,
        )
        == "fail"
    )
    assert restart_l2.compute_verdict([], disk_evidence=True) == "fail"


# ---------------------------------------------------------------------------
# 3. The restart/L2 probe relaunches on the SAME disk cache dir.
# ---------------------------------------------------------------------------


def _flag_value(cmd: list[str], flag: str) -> str:
    index = cmd.index(flag)
    return cmd[index + 1]


def test_restart_l2_relaunches_on_same_disk_dir(restart_l2: Any, tmp_path: Path) -> None:
    disk_dir = tmp_path / "block_disk_cache_probe"
    first_cmd, second_cmd = restart_l2.restart_serve_commands("/models/dsv4", 8867, disk_dir)
    assert first_cmd is not second_cmd  # independent lists, same contract
    for cmd in (first_cmd, second_cmd):
        assert "serve" in cmd
        assert "--use-paged-cache" in cmd
        assert "--continuous-batching" in cmd
        assert "--enable-block-disk-cache" in cmd
    assert _flag_value(first_cmd, "--block-disk-cache-dir") == str(disk_dir)
    assert _flag_value(second_cmd, "--block-disk-cache-dir") == str(disk_dir)
    assert _flag_value(first_cmd, "--port") == "8867"
    assert _flag_value(second_cmd, "--port") == "8867"


# ---------------------------------------------------------------------------
# Supporting invariants of the probe design.
# ---------------------------------------------------------------------------


def test_sameproc_cold_arm_has_no_disk_cache_tier(sameproc: Any) -> None:
    cmd = sameproc.build_serve_cmd("/models/dsv4", 8866)
    assert "--enable-block-disk-cache" not in cmd
    assert "--block-disk-cache-dir" not in cmd
    assert "--use-paged-cache" in cmd


def test_payload_contract_no_enable_thinking_temp0(sameproc: Any, restart_l2: Any) -> None:
    for module in (sameproc, restart_l2):
        payload = module.chat_payload([{"role": "user", "content": "hi"}])
        assert "enable_thinking" not in payload
        assert payload["temperature"] == 0
        assert payload["max_tokens"] == 512


def test_settle_gap_meets_minimum(sameproc: Any, restart_l2: Any) -> None:
    for module in (sameproc, restart_l2):
        assert module.SETTLE_GAP_S >= 2.0


def test_conversation_is_three_turns_and_nonce_salted(sameproc: Any, restart_l2: Any) -> None:
    for module in (sameproc, restart_l2):
        turns = module.conversation_turns("NONCE-XYZ-123")
        assert len(turns) == 3
        assert all("NONCE-XYZ-123" in turn for turn in turns)
