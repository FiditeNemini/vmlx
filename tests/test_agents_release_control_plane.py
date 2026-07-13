from pathlib import Path


def test_agents_pins_current_vmlx_mlxstudio_release_objective():
    agents = Path("AGENTS.md").read_text(encoding="utf-8")

    required_markers = [
        "Current objective execution contract - vMLX Python engine and MLXStudio app",
        "active Python engine plus Electron/panel app release objective",
        "CLI startup",
        "MLXStudio startup",
        "CLI startup and MLXStudio startup are independent release gates",
        "A CLI `vmlx\nserve` proof does not clear the UI-generated launch/settings path",
        "Chat Completions",
        "Responses API",
        "Tool behavior",
        "Structured output",
        "TurboQuant KV",
        "Installed app",
        "Per-family current proof checklist",
        "MiMo V2.5 JANGTQ_2/JANG_2L",
        "Qwen 3.6 27B/35B MTP",
        "Gemma 4 12B MXFP4/MXFP8/JANG_4M",
        "Step 3.7 Flash",
        "Nemo/Nemotron Omni",
        "MiniMax/JANGTQ_K",
        "DSV4 Flash",
        "ZAYA and other hybrid families",
        "Current no-fake-fix rules",
        "Required progress logging",
    ]

    missing = [marker for marker in required_markers if marker not in agents]
    assert missing == []


def test_agents_forbids_fake_release_clearance_shortcuts():
    agents = Path("AGENTS.md").read_text(encoding="utf-8")

    required_boundaries = [
        "Do not fix model exactness by parser-side value replacement",
        "Do not call media working because weights are preserved",
        "Do not call cache working from `/health`",
        "Do not call release ready from dev-Electron alone",
        "Do not enter signing/notarization/tag/download work",
        "MiMo exactness/media",
        "Step3p7 VLM",
        "MiniMax reporter parity",
        "DSV4 quality/memory",
        "real UI matrix",
        "installed-app parity",
    ]

    missing = [boundary for boundary in required_boundaries if boundary not in agents]
    assert missing == []


def test_release_tracker_points_at_current_control_plane_checklist():
    # The 2026-06-07 execution tracker was pruned in 380bb5274. The current
    # control-plane release tracker lives under the CODEX-REVERIFY-2026-07-11
    # checklist directory.
    tracker = Path(
        "docs/internal/CODEX-REVERIFY-2026-07-11/RELEASE-TRACKER.md"
    ).read_text(encoding="utf-8")

    assert "release tracker" in tracker.lower()
    assert "Status:" in tracker
    assert "all-routes-final.json" in tracker
    assert "full-suite-baseline.xml" in tracker
    assert "release lock" in tracker
