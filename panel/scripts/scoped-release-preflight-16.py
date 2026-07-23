#!/usr/bin/env python3
"""Fail-closed preflight for the vMLX 1.6.16 emergency parser/cache release.

The broad historical release manifest still tracks older full-matrix artifacts
whose filenames and required families predate the July 22 emergency scope. This
preflight is intentionally narrower and explicit: it only clears packaging for
the named 1.6.16 checkpoint scope, and it records every retained proof artifact
that justified doing so. It does not mark the broader family/media/stress
matrix as complete.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCOPE = "r16_parser_cache"
VERSION = "1.6.16"

CACHE_ROOT = ROOT / "docs/internal/release-gates/20260722_cache_partial_bonsai_gemma_laguna"
LAGUNA_ROOT = ROOT / "docs/internal/release-gates/20260722_laguna_r16_parser_ui_api"
GEMMA_ROOT = ROOT / "docs/internal/release-gates/20260722_gemma_r16_ui_api"
BONSAI_ROOT = ROOT / "docs/internal/release-gates/20260722_bonsai_r16_ui_api"
QWEN_ROOT = ROOT / "docs/internal/release-gates/20260722_qwen35_release_checkpoint"
CACHE_LABEL_ROOT = ROOT / "docs/internal/release-gates/20260722_cache_names_ram_ssd"
CAMPAIGN_ROOT = ROOT / "docs/internal/release-gates/20260722_v1_6_16_campaign"


def load_json(path: Path, failures: list[str]) -> dict[str, Any]:
    if not path.exists():
        failures.append(f"missing JSON proof artifact: {path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - preflight reports exact load failure
        failures.append(f"invalid JSON proof artifact {path}: {exc}")
        return {}
    if not isinstance(payload, dict):
        failures.append(f"JSON proof artifact is not an object: {path}")
        return {}
    return payload


def require(condition: bool, failures: list[str], message: str) -> None:
    if not condition:
        failures.append(message)


def _get(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def _read(path: Path, failures: list[str]) -> str:
    if not path.exists():
        failures.append(f"missing text artifact: {path}")
        return ""
    return path.read_text(encoding="utf-8")


def _version_from_pyproject() -> str | None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else None


def validate_versions(failures: list[str]) -> dict[str, Any]:
    package = load_json(ROOT / "panel/package.json", failures)
    lock = load_json(ROOT / "panel/package-lock.json", failures)
    latest = load_json(ROOT / "latest.json", failures)
    init_text = _read(ROOT / "vmlx_engine/__init__.py", failures)
    versions = {
        "pyproject": _version_from_pyproject(),
        "panel_package": package.get("version"),
        "panel_package_lock": lock.get("version"),
        "panel_package_lock_root": _get(lock, "packages", "", "version"),
        "engine_init": VERSION if f'__version__ = "{VERSION}"' in init_text else None,
        "latest_json": latest.get("version"),
    }
    require(versions["pyproject"] == VERSION, failures, f"pyproject version={versions['pyproject']!r}")
    require(versions["panel_package"] == VERSION, failures, f"panel/package.json version={versions['panel_package']!r}")
    require(versions["panel_package_lock"] == VERSION, failures, f"panel/package-lock.json version={versions['panel_package_lock']!r}")
    require(versions["panel_package_lock_root"] == VERSION, failures, f"panel/package-lock root version={versions['panel_package_lock_root']!r}")
    require(versions["engine_init"] == VERSION, failures, "vmlx_engine/__init__.py version stamp missing")
    require(versions["latest_json"] == "1.6.15", failures, f"latest.json should stay 1.6.15 before publish, got {versions['latest_json']!r}")
    return versions


def validate_cache_label_artifacts(failures: list[str]) -> dict[str, Any]:
    local = _read(CACHE_LABEL_ROOT / "local-focused-tests.txt", failures)
    remote = _read(CACHE_LABEL_ROOT / "remote-focused-tests.txt", failures)
    cli = _read(CACHE_LABEL_ROOT / "r16-cache-cli-help.txt", failures)
    readme = _read(CACHE_LABEL_ROOT / "README.md", failures)
    screenshots = [
        CACHE_LABEL_ROOT / "r16-cache-names-block-tooltip-fixed2.png",
        CACHE_LABEL_ROOT / "r16-cache-panel-names.png",
        CACHE_LABEL_ROOT / "r16-perf-panel-cache-names.png",
    ]
    for shot in screenshots:
        require(shot.exists() and shot.stat().st_size > 0, failures, f"missing/nonempty screenshot: {shot}")
    for label in ["In-Memory Paged Cache (RAM)", "Block Disk Cache (SSD / L2)"]:
        require(label in readme, failures, f"cache label missing from README: {label}")
    for text, name in [(local, "local-focused-tests"), (remote, "remote-focused-tests")]:
        require("passed" in text.lower(), failures, f"{name} does not record a passed focused test run")
    require("Block Disk Cache (SSD / L2)" in cli, failures, "CLI/help artifact does not expose SSD/L2 label")
    return {
        "readme": str(CACHE_LABEL_ROOT / "README.md"),
        "local_focused_tests": str(CACHE_LABEL_ROOT / "local-focused-tests.txt"),
        "remote_focused_tests": str(CACHE_LABEL_ROOT / "remote-focused-tests.txt"),
        "screenshots": [str(path) for path in screenshots],
    }


def _iter_request_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(summary.get("requests"), list):
        return [row for row in summary["requests"] if isinstance(row, dict)]
    return [summary]


def validate_cache_summary(
    path: Path,
    failures: list[str],
    *,
    required_detail: str,
    min_cached_tokens: int,
    require_disk_hit: bool,
    require_ram_zero: bool = False,
    required_tags: set[str] | None = None,
) -> dict[str, Any]:
    data = load_json(path, failures)
    rows = _iter_request_rows(data)
    require(bool(rows), failures, f"{path} has no request rows")
    seen_tags: set[str] = set()
    passing_rows = 0
    for row in rows:
        tag = str(row.get("tag") or "")
        if required_tags and tag not in required_tags:
            continue
        seen_tags.add(tag)
        require(row.get("status_code") == 200, failures, f"{path}:{tag} status_code={row.get('status_code')!r}")
        require(row.get("terminal_ok") is True, failures, f"{path}:{tag} terminal_ok={row.get('terminal_ok')!r}")
        require(row.get("marker_ok") is True, failures, f"{path}:{tag} marker_ok={row.get('marker_ok')!r}")
        cached = int(row.get("cached_tokens") or 0)
        require(cached >= min_cached_tokens, failures, f"{path}:{tag} cached_tokens={cached}, expected >= {min_cached_tokens}")
        detail = str(row.get("cache_detail") or "")
        require(required_detail in detail, failures, f"{path}:{tag} cache_detail={detail!r}, expected contains {required_detail!r}")
        last = row.get("last_cache_execution") if isinstance(row.get("last_cache_execution"), dict) else {}
        if require_disk_hit:
            require(last.get("disk_hit") is True or int(last.get("disk_blocks") or 0) > 0, failures, f"{path}:{tag} missing disk hit evidence")
        require(last.get("reconstruction_ok") is not False, failures, f"{path}:{tag} reconstruction_ok is false")
        if require_ram_zero:
            totals = row.get("totals") if isinstance(row.get("totals"), dict) else {}
            require(int(totals.get("ram_tokens_cached") or 0) == 0, failures, f"{path}:{tag} ram_tokens_cached={totals.get('ram_tokens_cached')!r}, expected 0")
        passing_rows += 1
    if required_tags:
        missing = sorted(required_tags - seen_tags)
        require(not missing, failures, f"{path} missing required request tags: {missing}")
    require(passing_rows > 0, failures, f"{path} had no checked passing rows")
    return {"path": str(path), "checked_tags": sorted(seen_tags), "required_detail": required_detail}


def validate_cache_proofs(failures: list[str]) -> list[dict[str, Any]]:
    checks = [
        validate_cache_summary(
            CACHE_ROOT / "bonsai_paged_on_store/summary.json",
            failures,
            required_detail="paged+ssm+tq-native",
            min_cached_tokens=9000,
            require_disk_hit=False,
            required_tags={"warm_a", "partial_b"},
        ),
        validate_cache_summary(
            CACHE_ROOT / "bonsai_paged_on_suffix_c_partial_after_restart/summary.json",
            failures,
            required_detail="paged+ssm+disk+tq-native",
            min_cached_tokens=9000,
            require_disk_hit=True,
            required_tags={"suffix_c"},
        ),
        validate_cache_summary(
            CACHE_ROOT / "bonsai_paged_off_disk_only_store/summary.json",
            failures,
            required_detail="block-disk+ssm+tq-native",
            min_cached_tokens=9000,
            require_disk_hit=True,
            required_tags={"warm_a", "partial_b"},
        ),
        validate_cache_summary(
            CACHE_ROOT / "bonsai_paged_off_suffix_d_partial_after_restart/summary.json",
            failures,
            required_detail="block-disk+ssm+tq-native",
            min_cached_tokens=9000,
            require_disk_hit=True,
            require_ram_zero=True,
            required_tags={"suffix_d"},
        ),
        validate_cache_summary(
            CACHE_ROOT / "laguna_jang4m_paged_off_disk_only_store/summary.json",
            failures,
            required_detail="block-disk+tq-native",
            min_cached_tokens=6000,
            require_disk_hit=True,
            required_tags={"warm_a", "partial_b"},
        ),
        validate_cache_summary(
            CACHE_ROOT / "laguna_jang4m_paged_off_suffix_c_partial/summary.json",
            failures,
            required_detail="block-disk+tq-native",
            min_cached_tokens=6000,
            require_disk_hit=True,
            require_ram_zero=True,
            required_tags={"suffix_c"},
        ),
        validate_cache_summary(
            CACHE_ROOT / "gemma_e2b_paged_on_store/summary.json",
            failures,
            required_detail="paged+mixed_swa",
            min_cached_tokens=4500,
            require_disk_hit=False,
            required_tags={"warm_a", "partial_b"},
        ),
        validate_cache_summary(
            CACHE_ROOT / "gemma_e2b_paged_on_suffix_d_partial_after_clean_restart/summary.json",
            failures,
            required_detail="paged+mixed_swa+disk",
            min_cached_tokens=4500,
            require_disk_hit=True,
            required_tags={"suffix_d"},
        ),
        validate_cache_summary(
            CACHE_ROOT / "gemma_e2b_paged_off_disk_only_store/summary.json",
            failures,
            required_detail="block-disk+mixed_swa",
            min_cached_tokens=4500,
            require_disk_hit=True,
            required_tags={"warm_a", "partial_b"},
        ),
        validate_cache_summary(
            CACHE_ROOT / "gemma_e2b_paged_off_suffix_c_partial_after_restart/summary.json",
            failures,
            required_detail="block-disk+mixed_swa",
            min_cached_tokens=4500,
            require_disk_hit=True,
            required_tags={"suffix_c"},
        ),
    ]
    return checks


def validate_case(
    case: dict[str, Any],
    failures: list[str],
    *,
    label: str,
    content: str | None = None,
    min_reasoning: int = 0,
    finish: str | None = None,
    terminal: str | None = None,
    tool_name: str | None = None,
    tool_path: str | None = None,
) -> None:
    require(case.get("status") == 200, failures, f"{label} status={case.get('status')!r}")
    require(not case.get("raw_markers_in_content"), failures, f"{label} raw markers leaked into visible content")
    require(not case.get("raw_marker_in_reasoning"), failures, f"{label} raw marker leaked into reasoning")
    require(int(case.get("reasoning_chars") or 0) >= min_reasoning, failures, f"{label} reasoning_chars={case.get('reasoning_chars')!r}, expected >= {min_reasoning}")
    if content is not None:
        require(str(case.get("content") or "") == content, failures, f"{label} content={case.get('content')!r}")
    if finish is not None:
        require(finish in (case.get("finishes") or []), failures, f"{label} missing finish {finish!r}: {case.get('finishes')!r}")
    if terminal is not None:
        require(terminal in (case.get("terminals") or []), failures, f"{label} missing terminal {terminal!r}: {case.get('terminals')!r}")
    if tool_name is not None:
        require(case.get("tool_call_name") == tool_name, failures, f"{label} tool_call_name={case.get('tool_call_name')!r}")
    if tool_path is not None:
        args = case.get("tool_call_args") if isinstance(case.get("tool_call_args"), dict) else {}
        require(args.get("path") == tool_path, failures, f"{label} tool path={args.get('path')!r}")


def validate_api_proofs(failures: list[str]) -> dict[str, Any]:
    gemma = load_json(GEMMA_ROOT / "gemma-api-gateway-proof.json", failures)
    gemma_cases = {case.get("tag"): case for case in gemma.get("cases", []) if isinstance(case, dict)}
    validate_case(gemma_cases.get("chat_reasoning", {}), failures, label="Gemma Chat reasoning", content="GEMMA-R16-API-CHAT-DONE", min_reasoning=500, finish="stop")
    validate_case(gemma_cases.get("responses_reasoning", {}), failures, label="Gemma Responses reasoning", content="GEMMA-R16-API-RESP-DONE", min_reasoning=500, terminal="response.completed")
    validate_case(gemma_cases.get("chat_tool_call", {}), failures, label="Gemma Chat tool call", min_reasoning=200, finish="tool_calls")
    validate_case(gemma_cases.get("chat_tool_continuation", {}), failures, label="Gemma Chat tool continuation", content="GEMMA-R16-API-TOOL-DONE SIZE=5.2 KB", min_reasoning=500, finish="stop", tool_name="file_info", tool_path="panel/package.json")

    laguna = load_json(LAGUNA_ROOT / "laguna-api-gateway-proof.json", failures)
    laguna_cases = {case.get("tag"): case for case in laguna.get("cases", []) if isinstance(case, dict)}
    validate_case(laguna_cases.get("chat_tool_call", {}), failures, label="Laguna Chat tool call", finish="tool_calls")
    validate_case(laguna_cases.get("chat_tool_continuation", {}), failures, label="Laguna Chat tool continuation", content="The package file is 5.2 KB.", finish="stop", tool_name="file_info", tool_path="panel/package.json")
    validate_case(laguna_cases.get("chat_auto_reasoning", {}), failures, label="Laguna Chat hard transport", min_reasoning=1500)
    validate_case(laguna_cases.get("responses_on_reasoning", {}), failures, label="Laguna Responses hard transport", min_reasoning=1500, terminal="response.incomplete")

    terminal = load_json(LAGUNA_ROOT / "laguna-api-terminal-addendum.json", failures)
    terminal_cases = {case.get("tag"): case for case in terminal.get("cases", []) if isinstance(case, dict)}
    validate_case(terminal_cases.get("chat_auto_terminal", {}), failures, label="Laguna Chat terminal addendum", content="30 times 25 is larger.", min_reasoning=1200, finish="stop")
    validate_case(terminal_cases.get("responses_on_terminal", {}), failures, label="Laguna Responses terminal addendum", content="43\u202f×\u202f18 is larger.", terminal="response.completed")

    laguna_four = load_json(LAGUNA_ROOT / "laguna-anthropic-ollama-gateway-proof.json", failures)
    checks = laguna_four.get("checks") if isinstance(laguna_four.get("checks"), dict) else {}
    for key in [
        "anthropic_hard_reasoning_content_stop",
        "anthropic_tool_initial_exact",
        "anthropic_tool_follow_final",
        "ollama_hard_reasoning_content_stop_overgenerated",
        "ollama_tool_initial_exact",
        "ollama_tool_follow_final",
    ]:
        require(checks.get(key) is True, failures, f"Laguna Anthropic/Ollama check {key}={checks.get(key)!r}")
    for label in ["anthropic_hard", "ollama_hard"]:
        row = laguna_four.get(label) if isinstance(laguna_four.get(label), dict) else {}
        require(int(row.get("reasoning_deltas") or 0) > 0, failures, f"{label} missing reasoning/thinking deltas")
        require(int(row.get("content_deltas") or 0) > 0, failures, f"{label} missing content deltas")
        require(row.get("marker_leak") is False, failures, f"{label} marker leak")

    bonsai = load_json(BONSAI_ROOT / "bonsai-api-gateway-proof.json", failures)
    bonsai_cases = [case for case in bonsai.get("cases", []) if isinstance(case, dict)]
    require(len(bonsai_cases) >= 3, failures, "Bonsai API proof missing expected cases")
    if len(bonsai_cases) >= 2:
        validate_case(bonsai_cases[0], failures, label="Bonsai Chat reasoning", content="BONSAI-R16-API-CHAT-B-DONE", min_reasoning=1000, finish="stop")
        validate_case(bonsai_cases[1], failures, label="Bonsai Responses reasoning", content="BONSAI-R16-API-RESP-B-DONE", min_reasoning=900)
    if len(bonsai_cases) >= 3:
        validate_case(bonsai_cases[2], failures, label="Bonsai Chat tool call", min_reasoning=100, finish="tool_calls")

    qwen = load_json(QWEN_ROOT / "all-protocols-thinking-off.json", failures)
    qwen_checks = qwen.get("checks") if isinstance(qwen.get("checks"), dict) else {}
    require(qwen.get("pass") is True, failures, "Qwen all-protocols proof pass flag is not true")
    require(qwen_checks.get("all_flows_pass") is True, failures, "Qwen all flows did not pass")
    require(qwen_checks.get("all_requested_flows_present") is True, failures, "Qwen requested flows missing")

    return {
        "gemma": str(GEMMA_ROOT / "gemma-api-gateway-proof.json"),
        "laguna_chat_responses": str(LAGUNA_ROOT / "laguna-api-gateway-proof.json"),
        "laguna_terminal": str(LAGUNA_ROOT / "laguna-api-terminal-addendum.json"),
        "laguna_anthropic_ollama": str(LAGUNA_ROOT / "laguna-anthropic-ollama-gateway-proof.json"),
        "bonsai": str(BONSAI_ROOT / "bonsai-api-gateway-proof.json"),
        "qwen": str(QWEN_ROOT / "all-protocols-thinking-off.json"),
    }


def validate_laguna_low_limit(failures: list[str]) -> dict[str, Any]:
    four = load_json(LAGUNA_ROOT / "laguna-four-block-eviction-refault.json", failures)
    rows = {row.get("label"): row for row in four.get("rows", []) if isinstance(row, dict)}
    require(set(rows) >= {"store_a", "partial_b", "pressure_1", "pressure_2", "pressure_3", "refault_c"}, failures, "Laguna four-block proof missing expected rows")
    for label, row in rows.items():
        result = row.get("result") if isinstance(row.get("result"), dict) else {}
        require(result.get("status") == 200, failures, f"four-block {label} status={result.get('status')!r}")
        require(result.get("content") == row.get("expected"), failures, f"four-block {label} content={result.get('content')!r} expected={row.get('expected')!r}")
        require(result.get("marker_leak") is False, failures, f"four-block {label} marker leak")
        require("[DONE]" in (result.get("terminals") or []), failures, f"four-block {label} missing DONE terminal")
    refault = rows.get("refault_c", {})
    refault_after = refault.get("after_summary") if isinstance(refault.get("after_summary"), dict) else {}
    require(int(refault_after.get("disk_hit_delta") or 0) >= 3, failures, f"four-block refault disk_hit_delta={refault_after.get('disk_hit_delta')!r}")
    require(int(refault_after.get("evictions") or 0) >= 6, failures, f"four-block evictions={refault_after.get('evictions')!r}")

    disk = load_json(LAGUNA_ROOT / "laguna-block-disk-gb-cap-eviction-025gb.json", failures)
    disk_rows = {row.get("label"): row for row in disk.get("rows", []) if isinstance(row, dict)}
    require(set(disk_rows) >= {"alpha_store", "alpha_partial", "bravo_store", "charlie_store", "charlie_refault", "alpha_after_pressure"}, failures, "Laguna disk GB proof missing expected rows")
    for label, row in disk_rows.items():
        result = row.get("result") if isinstance(row.get("result"), dict) else {}
        require(result.get("status") == 200, failures, f"disk-budget {label} status={result.get('status')!r}")
        require(result.get("content") == row.get("expected"), failures, f"disk-budget {label} content={result.get('content')!r} expected={row.get('expected')!r}")
        require(result.get("marker_leak") is False, failures, f"disk-budget {label} marker leak")
        require("[DONE]" in (result.get("terminals") or []), failures, f"disk-budget {label} missing DONE terminal")
    disk_after = disk.get("health_after", {}).get("bd") if isinstance(disk.get("health_after"), dict) else {}
    require(int(disk_after.get("disk_writes") or 0) >= 60, failures, f"disk-budget writes={disk_after.get('disk_writes')!r}")
    require(int(disk_after.get("disk_hits") or 0) >= 50, failures, f"disk-budget hits={disk_after.get('disk_hits')!r}")
    require(int(disk_after.get("disk_evictions") or 0) >= 50, failures, f"disk-budget evictions={disk_after.get('disk_evictions')!r}")
    require(float(disk_after.get("disk_size_gb") or 0) <= 0.25, failures, f"disk-budget size_gb={disk_after.get('disk_size_gb')!r}")
    charlie_after = disk_rows.get("charlie_refault", {}).get("after") or {}
    require(int(charlie_after.get("hit_delta") or 0) >= 2, failures, f"charlie_refault hit_delta={charlie_after.get('hit_delta')!r}")
    return {
        "four_block": str(LAGUNA_ROOT / "laguna-four-block-eviction-refault.json"),
        "disk_gb_cap": str(LAGUNA_ROOT / "laguna-block-disk-gb-cap-eviction-025gb.json"),
    }


def validate_ui_readmes(failures: list[str]) -> dict[str, Any]:
    expectations = {
        LAGUNA_ROOT / "README.md": [
            "separate Reasoning rail `2305 chars`",
            "exactly one visible",
            "`Info panel/package.json` tool card",
            "`4861 block-disk+tq-native cached`",
        ],
        GEMMA_ROOT / "README.md": [
            "Separate Reasoning rail `1106 chars`",
            "GEMMA-R16-UI-T2-DONE PREV=GEMMA-R16-UI-T1-DONE",
            "exactly one visible `Info panel/package.json` tool card",
            "GEMMA-R16-UI-T3-DONE SIZE=5.2 KB",
        ],
        CACHE_ROOT / "README.md": [
            "Paged-Off + SSD/L2 disk-only",
            "Paged-On + SSD/L2",
            "Block Disk Cache (SSD / L2)",
        ],
        CAMPAIGN_ROOT / "README.md": [
            "`R16-CACHE-HIERARCHY`",
            "Laguna JANG_4M",
            "disk_writes=62",
            "disk_hits=53",
        ],
    }
    for path, markers in expectations.items():
        text = _read(path, failures)
        for marker in markers:
            require(marker in text, failures, f"{path} missing UI/doc marker: {marker}")
    return {path.name: str(path) for path in expectations}


def validate_source_trace(failures: list[str]) -> dict[str, Any]:
    server = _read(ROOT / "vmlx_engine/server.py", failures)
    prefix_cache = _read(ROOT / "vmlx_engine/prefix_cache.py", failures)
    math = _read(ROOT / "panel/src/renderer/src/components/chat/mathMarkdown.ts", failures)
    policy = _read(ROOT / "panel/src/shared/cacheControlPolicy.ts", failures)
    require("_resolve_enable_thinking" in server, failures, "server missing reasoning resolver")
    require("block-disk+tq-native" in prefix_cache or "tq_native" in prefix_cache, failures, "prefix cache missing TQ-native source markers")
    require("prepareMarkdownWithMath" in math, failures, "renderer math/markdown shim missing")
    require("blockDiskCacheDisabled" in policy, failures, "cache control policy source missing block-disk disabled rule")
    return {
        "server": "vmlx_engine/server.py",
        "prefix_cache": "vmlx_engine/prefix_cache.py",
        "math_markdown": "panel/src/renderer/src/components/chat/mathMarkdown.ts",
        "cache_control_policy": "panel/src/shared/cacheControlPolicy.ts",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", default=VERSION)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "build/current-scoped-release-preflight-16-parser-cache.json",
    )
    args = parser.parse_args()
    failures: list[str] = []
    require(args.expected_version == VERSION, failures, f"unsupported expected version {args.expected_version!r}")

    manifest = {
        "schema_version": 1,
        "scope": SCOPE,
        "version": VERSION,
        "status": "pass",
        "failures": failures,
        "versions": validate_versions(failures),
        "source_trace": validate_source_trace(failures),
        "ui_evidence": validate_ui_readmes(failures),
        "cache_label_evidence": validate_cache_label_artifacts(failures),
        "cache_partial_ssd_evidence": validate_cache_proofs(failures),
        "laguna_low_limit_evidence": validate_laguna_low_limit(failures),
        "api_reasoning_tool_evidence": validate_api_proofs(failures),
        "deferred_full_matrix_boundaries": [
            "full cross-family media breadth",
            "MiniMax M3 sparse/MSA restart/eviction",
            "DSV4 composite long-output quality",
            "openPangu native prompt disk",
            "Nemotron/Step/LFM remaining protocol/media rows",
            "broad old June proof-sweep artifacts",
        ],
        "scope_note": (
            "Clears only the 1.6.16 emergency parser/cache checkpoint for packaging. "
            "It does not declare the broad release matrix complete."
        ),
    }
    manifest["status"] = "pass" if not failures else "fail"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    print(f"scope={SCOPE}")
    print(f"version={VERSION}")
    print(f"status={manifest['status']}")
    if failures:
        print("failures:")
        for failure in failures:
            print(f"- {failure}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
