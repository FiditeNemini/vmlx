from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_scoped_release_preflight_16_passes_current_emergency_artifacts(tmp_path: Path) -> None:
    out = tmp_path / "scoped-release-preflight-16.json"
    result = subprocess.run(
        [
            sys.executable,
            "panel/scripts/scoped-release-preflight-16.py",
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
    assert payload["scope"] == "r16_parser_cache"
    assert payload["version"] == "1.6.16"
    assert payload["status"] == "pass"
    assert payload["failures"] == []
    assert "full cross-family media breadth" in payload["deferred_full_matrix_boundaries"]


def test_build_release_dmgs_exposes_r16_parser_cache_scope() -> None:
    source = (ROOT / "panel/scripts/build-release-dmgs.sh").read_text(encoding="utf-8")
    assert "r16_parser_cache)" in source
    assert "panel/scripts/scoped-release-preflight-16.py" in source
    assert "Supported scoped release values: r17_consolidation, r16_parser_cache, mm3_gemma_vl, codex_ui_only" in source
