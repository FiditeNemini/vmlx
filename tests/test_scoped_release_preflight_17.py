from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_scoped_release_preflight_17_passes_retained_checkpoint_artifacts(
    tmp_path: Path,
) -> None:
    package = json.loads(
        (ROOT / "panel/package.json").read_text(encoding="utf-8")
    )
    if package.get("version") != "1.6.17":
        pytest.skip(
            "historical 1.6.17 pre-publication gate applies only while the "
            "source package itself is 1.6.17"
        )
    latest = json.loads((ROOT / "latest.json").read_text(encoding="utf-8"))
    if latest.get("version") == "1.6.17":
        pytest.skip(
            "historical 1.6.17 pre-publication gate no longer applies once "
            "latest.json has advanced to the published checkpoint"
        )
    out = tmp_path / "scoped-release-preflight-17.json"
    result = subprocess.run(
        [
            sys.executable,
            "panel/scripts/scoped-release-preflight-17.py",
            "--out",
            str(out),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["scope"] == "r17_consolidation"
    assert payload["version"] == "1.6.17"
    assert payload["status"] == "pass"
    assert payload["failures"] == []


def test_build_release_dmgs_exposes_r17_consolidation_scope() -> None:
    source = (ROOT / "panel/scripts/build-release-dmgs.sh").read_text(encoding="utf-8")
    assert "r17_consolidation)" in source
    assert "panel/scripts/scoped-release-preflight-17.py" in source
    assert (
        "Supported scoped release values: r19_production, r17_consolidation, "
        "r16_parser_cache, mm3_gemma_vl, codex_ui_only"
    ) in source
