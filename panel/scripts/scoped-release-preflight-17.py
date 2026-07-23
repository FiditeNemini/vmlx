#!/usr/bin/env python3
"""Fail-closed preflight for the scoped vMLX 1.6.17 consolidation release.

This gate clears only the named usable checkpoint for packaging. It validates
the current source-suite, panel, Electron, raw protocol, and agentic evidence
without declaring the retained family/media/cache/stress matrix complete.
Signing, notarization, installed-app smoke, tagging, and publication remain
separate downstream gates.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCOPE = "r17_consolidation"
VERSION = "1.6.17"

CAMPAIGN_ROOT = ROOT / "docs/internal/release-gates/20260722_v1_6_17_consolidation"
SMOKE_ROOT = CAMPAIGN_ROOT / "release-candidate-smoke"
SUITE_ROOT = CAMPAIGN_ROOT / "release-head-suite-audit"


def require(condition: bool, failures: list[str], message: str) -> None:
    if not condition:
        failures.append(message)


def read_text(path: Path, failures: list[str]) -> str:
    if not path.exists():
        failures.append(f"missing text proof artifact: {path}")
        return ""
    return path.read_text(encoding="utf-8")


def load_json(path: Path, failures: list[str]) -> dict[str, Any]:
    if not path.exists():
        failures.append(f"missing JSON proof artifact: {path}")
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report exact preflight failure
        failures.append(f"invalid JSON proof artifact {path}: {exc}")
        return {}
    if not isinstance(data, dict):
        failures.append(f"JSON proof artifact is not an object: {path}")
        return {}
    return data


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def project_version() -> str | None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else None


def validate_versions(failures: list[str]) -> dict[str, Any]:
    package = load_json(ROOT / "panel/package.json", failures)
    lock = load_json(ROOT / "panel/package-lock.json", failures)
    latest = load_json(ROOT / "latest.json", failures)
    engine_init = read_text(ROOT / "vmlx_engine/__init__.py", failures)
    values = {
        "pyproject": project_version(),
        "panel_package": package.get("version"),
        "panel_package_lock": lock.get("version"),
        "panel_package_lock_root": nested(lock, "packages", "", "version"),
        "engine_init": VERSION if f'__version__ = "{VERSION}"' in engine_init else None,
        "source_latest_json": latest.get("version"),
    }
    for key in [
        "pyproject",
        "panel_package",
        "panel_package_lock",
        "panel_package_lock_root",
        "engine_init",
    ]:
        require(values[key] == VERSION, failures, f"{key}={values[key]!r}, expected {VERSION}")
    require(
        values["source_latest_json"] in {"1.6.15", "1.6.16"},
        failures,
        "source latest.json must remain a prior public version before publication, "
        f"got {values['source_latest_json']!r}",
    )
    require(
        values["source_latest_json"] != VERSION,
        failures,
        "source latest.json was advanced before signed artifact publication",
    )
    return values


def validate_source_suite(failures: list[str]) -> dict[str, Any]:
    readme = read_text(SUITE_ROOT / "README.md", failures)
    full = read_text(SUITE_ROOT / "final-head-source-suite.txt", failures)
    require(
        "6407 passed, 96 skipped, 93 deselected" in readme,
        failures,
        "source-suite README is missing the corrected full-suite result",
    )
    require(
        "6407 passed, 96 skipped, 93 deselected" in full,
        failures,
        "source-suite log is missing the corrected full-suite result",
    )
    require(
        "bundled-Python integrity row manually deselected" in readme,
        failures,
        "source-suite README must retain the bundled-Python integrity boundary",
    )
    return {
        "readme": str(SUITE_ROOT / "README.md"),
        "full_source_suite": str(SUITE_ROOT / "final-head-source-suite.txt"),
    }


def validate_panel_suite(failures: list[str]) -> dict[str, Any]:
    panel = read_text(SMOKE_ROOT / "r17-panel-full-8eb8468bd.log", failures)
    typecheck = read_text(SMOKE_ROOT / "r17-panel-typecheck-8eb8468bd.log", failures)
    build = read_text(SMOKE_ROOT / "r17-panel-build-8eb8468bd.log", failures)
    require("Test Files  86 passed (86)" in panel, failures, "panel log missing 86/86 file pass")
    require(
        "Tests  2491 passed | 3 skipped (2494)" in panel,
        failures,
        "panel log missing 2491 passed / 3 skipped result",
    )
    require("tsc --noEmit" in typecheck, failures, "typecheck log missing tsc invocation")
    require(
        "built in" in build and "KaTeX_Main-Regular" in build,
        failures,
        "production build log missing completed renderer/KaTeX assets",
    )
    return {
        "panel": str(SMOKE_ROOT / "r17-panel-full-8eb8468bd.log"),
        "typecheck": str(SMOKE_ROOT / "r17-panel-typecheck-8eb8468bd.log"),
        "production_build": str(SMOKE_ROOT / "r17-panel-build-8eb8468bd.log"),
    }


def validate_raw_streams(failures: list[str]) -> dict[str, Any]:
    data = load_json(SMOKE_ROOT / "r17-raw-wire-aa49b699d.json", failures)
    rows = [row for row in data.get("rows", []) if isinstance(row, dict)]
    require(len(rows) == 8, failures, f"raw stream row count={len(rows)}, expected 8")
    seen = {(str(row.get("base")), str(row.get("protocol"))) for row in rows}
    expected = {
        (base, protocol)
        for base in ("direct", "gateway")
        for protocol in ("chat", "responses", "anthropic", "ollama")
    }
    require(seen == expected, failures, f"raw stream coverage mismatch: {sorted(seen)}")
    exact_misses: list[str] = []
    for row in rows:
        label = f"{row.get('base')}:{row.get('protocol')}"
        checks = row.get("checks") if isinstance(row.get("checks"), dict) else {}
        for check in [
            "http_200",
            "stream_parsed",
            "reasoning_present",
            "content_progressive",
            "content_has_no_control_markers",
            "currency_preserved",
            "math_preserved",
            "terminal_present",
        ]:
            require(checks.get(check) is True, failures, f"{label} failed {check}")
        require(int(row.get("reasoning_delta_count") or 0) > 0, failures, f"{label} no reasoning deltas")
        require(int(row.get("content_delta_count") or 0) > 1, failures, f"{label} content not progressive")
        require(bool(row.get("raw_sha256")), failures, f"{label} missing raw stream hash")
        require(bool(row.get("reasoning_sha256")), failures, f"{label} missing reasoning hash")
        if checks.get("content_exact") is not True:
            exact_misses.append(label)
    require(
        exact_misses == ["direct:chat"],
        failures,
        f"unexpected strict exact-string misses: {exact_misses}",
    )
    direct_chat = next(
        (row for row in rows if row.get("base") == "direct" and row.get("protocol") == "chat"),
        {},
    )
    require(
        direct_chat.get("content") == "R17-WIRE-CHAT-DONE CURRENCY=$43 MATH=9×6=54",
        failures,
        "direct Chat variation was not the documented final-period-only omission",
    )
    return {
        "artifact": str(SMOKE_ROOT / "r17-raw-wire-aa49b699d.json"),
        "rows": len(rows),
        "strict_exact_misses": exact_misses,
        "structural_stream_rows_passed": len(rows),
    }


def validate_agentic_streams(failures: list[str]) -> dict[str, Any]:
    data = load_json(SMOKE_ROOT / "r17-agentic-stream-aa49b699d.json", failures)
    checks = data.get("checks") if isinstance(data.get("checks"), dict) else {}
    require(data.get("pass") is True, failures, "agentic protocol artifact pass is not true")
    require(checks.get("all_flows_pass") is True, failures, "agentic protocol flows did not all pass")
    require(
        checks.get("all_requested_flows_present") is True,
        failures,
        "agentic protocol artifact is missing requested flows",
    )
    flows = data.get("flows") if isinstance(data.get("flows"), dict) else {}
    checked = 0
    for base in ("direct", "gateway"):
        base_flows = flows.get(base) if isinstance(flows.get(base), dict) else {}
        for protocol in ("chat", "responses", "anthropic", "ollama"):
            protocol_flows = (
                base_flows.get(protocol)
                if isinstance(base_flows.get(protocol), dict)
                else {}
            )
            flow = (
                protocol_flows.get("stream")
                if isinstance(protocol_flows.get("stream"), dict)
                else {}
            )
            label = f"{base}:{protocol}:stream"
            require(flow.get("pass") is True, failures, f"{label} did not pass")
            flow_checks = flow.get("checks") if isinstance(flow.get("checks"), dict) else {}
            for check in [
                "status_200",
                "round1_exact_tool",
                "round2_exact_tool",
                "final_exact",
                "reasoning_present",
                "reasoning_not_duplicated_as_content",
                "reasoning_not_stale_when_present",
                "stream_final_progressive",
                "tool_rounds_have_no_visible_prose",
                "no_visible_control_markup",
                "terminals_truthful",
            ]:
                require(flow_checks.get(check) is True, failures, f"{label} failed {check}")
            executions = flow.get("executions") if isinstance(flow.get("executions"), list) else []
            require(
                [item.get("name") for item in executions if isinstance(item, dict)]
                == ["file_info", "run_command"],
                failures,
                f"{label} execution sequence is not exact",
            )
            checked += 1
    require(checked == 8, failures, f"agentic flow count={checked}, expected 8")
    return {
        "artifact": str(SMOKE_ROOT / "r17-agentic-stream-aa49b699d.json"),
        "flows": checked,
    }


def validate_electron(failures: list[str]) -> dict[str, Any]:
    data = load_json(SMOKE_ROOT / "r17-electron-postfix-8eb8468bd.json", failures)
    require(data.get("source_head") == "8eb8468bd", failures, "Electron proof head mismatch")
    require(data.get("fresh_chat") is True, failures, "Electron proof did not use a fresh chat")
    require(data.get("turn2_succeeded") is True, failures, "Electron real tool turn did not succeed")
    rows = [row for row in data.get("rows", []) if isinstance(row, dict)]
    require(len(rows) == 3, failures, f"Electron turn count={len(rows)}, expected 3")
    for index, row in enumerate(rows, start=1):
        require(row.get("turn") == index, failures, f"Electron turn ordering mismatch at {index}")
        require(row.get("saw_generation_control") is True, failures, f"Electron turn {index} did not show generation control")
        require(int(row.get("reasoning_rails") or 0) >= 1, failures, f"Electron turn {index} missing reasoning rail")
        require(row.get("raw_control_markup_visible") is False, failures, f"Electron turn {index} leaked control markup")
        blocks = row.get("assistant_blocks") if isinstance(row.get("assistant_blocks"), list) else []
        require(bool(blocks) and all(str(block).strip() for block in blocks), failures, f"Electron turn {index} empty output")
    if len(rows) == 3:
        require(int(rows[0].get("katex_nodes") or 0) >= 1, failures, "Electron math turn missing KaTeX node")
        require("Currency $43. Math 9×6=54." in "\n".join(rows[0].get("assistant_blocks") or []), failures, "Electron math/currency output mismatch")
        require(int(rows[1].get("new_exact_path_cards") or 0) == 1, failures, "Electron tool card count is not exactly one")
        require("R17-HY3-POSTFIX-UI-2-DONE SIZE=5.2 KB" in "\n".join(rows[1].get("assistant_blocks") or []), failures, "Electron tool continuation mismatch")
        require("PATH=panel/package.json SIZE=5.2 KB CURRENCY=$43 MATH=7×8=56" in "\n".join(rows[2].get("assistant_blocks") or []), failures, "Electron history recall mismatch")
    for name in [
        "r17-electron-postfix-turn-1.png",
        "r17-electron-postfix-turn-2.png",
        "r17-electron-postfix-turn-3.png",
    ]:
        path = SMOKE_ROOT / name
        require(path.exists() and path.stat().st_size > 0, failures, f"missing/nonempty screenshot: {path}")
    return {
        "artifact": str(SMOKE_ROOT / "r17-electron-postfix-8eb8468bd.json"),
        "screenshots": [
            str(SMOKE_ROOT / "r17-electron-postfix-turn-1.png"),
            str(SMOKE_ROOT / "r17-electron-postfix-turn-2.png"),
            str(SMOKE_ROOT / "r17-electron-postfix-turn-3.png"),
        ],
    }


def validate_campaign_boundary(failures: list[str]) -> dict[str, Any]:
    campaign = read_text(CAMPAIGN_ROOT / "README.md", failures)
    smoke = read_text(SMOKE_ROOT / "README.md", failures)
    for marker in [
        "Status: `ACTIVE / PARTIAL / NOT RELEASE-READY`",
        "### R17-026 Release-candidate UI/API smoke",
        "Broader retained",
    ]:
        require(marker in campaign, failures, f"campaign boundary missing marker: {marker}")
    for marker in [
        "DEV UI+API+PANEL PASS / BUNDLE+SIGNED APP OPEN",
        "Remaining release gates",
        "Retained broader family/media/cache/stress rows remain follow-up work",
    ]:
        require(marker in smoke, failures, f"smoke boundary missing marker: {marker}")
    return {
        "campaign": str(CAMPAIGN_ROOT / "README.md"),
        "smoke": str(SMOKE_ROOT / "README.md"),
    }


def validate_source_trace(failures: list[str]) -> dict[str, Any]:
    chat = read_text(ROOT / "panel/src/main/ipc/chat.ts", failures)
    recovery = read_text(ROOT / "panel/src/shared/responsesStreamRecovery.ts", failures)
    tests = read_text(ROOT / "panel/tests/responses-stream-recovery.test.ts", failures)
    require(
        "rejectedControlMarkup" in chat and "rejectedControlMarkup" in recovery,
        failures,
        "panel rejected-control-markup source correction is missing",
    )
    require(
        "rejects parser-missed tool control markup instead of restoring it as prose" in tests,
        failures,
        "panel rejected-control-markup regression test is missing",
    )
    return {
        "panel_chat": "panel/src/main/ipc/chat.ts",
        "stream_recovery": "panel/src/shared/responsesStreamRecovery.ts",
        "stream_recovery_tests": "panel/tests/responses-stream-recovery.test.ts",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", default=VERSION)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "build/current-scoped-release-preflight-17.json",
    )
    args = parser.parse_args()
    failures: list[str] = []
    require(
        args.expected_version == VERSION,
        failures,
        f"unsupported expected version {args.expected_version!r}",
    )
    manifest = {
        "schema_version": 1,
        "scope": SCOPE,
        "version": VERSION,
        "status": "fail",
        "failures": failures,
        "versions": validate_versions(failures),
        "source_trace": validate_source_trace(failures),
        "source_suite": validate_source_suite(failures),
        "panel_suite": validate_panel_suite(failures),
        "electron": validate_electron(failures),
        "raw_protocol_streams": validate_raw_streams(failures),
        "agentic_protocol_streams": validate_agentic_streams(failures),
        "campaign_boundary": validate_campaign_boundary(failures),
        "downstream_release_gates": [
            "rebuild bundled Python from frozen versioned source",
            "verify bundled Python provenance and rerun the full suite",
            "build Developer-ID-signed Sequoia and Tahoe DMGs",
            "Apple notarization, stapling, Gatekeeper, and codesign validation",
            "signed installed-app Electron and raw API smoke",
            "tag, GitHub release, PyPI, mlxstudio latest.json, and feed publication",
        ],
        "deferred_full_matrix_boundaries": [
            "broader cross-family and media repetition",
            "retained capacity eviction and corrupt-companion rows",
            "long gateway/network/model-swap soak",
            "remaining JANG conversion and distribution certification",
        ],
        "scope_note": (
            "Clears only the v1.6.17 usable consolidation checkpoint for packaging. "
            "It does not mark the retained full campaign matrix complete."
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
