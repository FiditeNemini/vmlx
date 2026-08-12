# SPDX-License-Identifier: Apache-2.0
"""A bundle that DECLARES MTP must never be skipped silently.

MEASURED on the box: Nemotron-3.5-Lightning-30B-A3B-JANG_4M ships 34
``mtp.layers.0.*`` tensors and ``num_nextn_predict_layers=1``, but family
``nemotron_h`` is not in the runtime-supported set, so the loader deactivated
MTP and returned with NOTHING at INFO. The model ran plain autoregressive and
the only surfaces that told the truth were ``/health.mtp`` and the CLI startup
banner. Decode-time ineligibility is DEBUG-only, so nothing mentioned it either.

Silence about a feature the bundle advertises is the defect.
"""

from __future__ import annotations

import re
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "vmlx_engine" / "native_mtp.py"


def test_declared_but_unsupported_mtp_is_logged_not_silent():
    src = SOURCE.read_text(encoding="utf-8")
    # The else-branch that deactivates must log before it does so.
    idx = src.index("        deactivate_native_mtp()\n    return status")
    window = src[max(0, idx - 1400) : idx]
    assert "logger.info(" in window, (
        "the deactivate branch no longer logs — a bundle that declares MTP can "
        "again be skipped in total silence"
    )
    assert "mtp_declared" in window, (
        "the log must be gated on the bundle actually declaring MTP, otherwise "
        "it fires for every non-MTP model"
    )
    assert re.search(r"autoregressive", window), (
        "the message should say what happens instead, so a reader knows the "
        "consequence rather than just the status string"
    )


def test_engine_produces_the_skip_key_the_panel_reads():
    """PerformancePanel reads last_native_mtp_skip; the engine must emit it.

    The panel has always consumed
    ``health.scheduler.batch_generator.last_native_mtp_skip``
    (PerformancePanel.tsx), but NO code in vmlx_engine produced that key — a
    grep found zero producers. The skip tile was dead UI: only positive MTP
    engagement could ever display, and a model silently running without MTP had
    nowhere to say so. A DEBUG log line is not a user-visible surface.
    """
    from vmlx_engine.patches.mlx_lm_mtp.batch_generator import (
        native_mtp_stats_snapshot,
    )

    snapshot = native_mtp_stats_snapshot()
    assert "last_native_mtp_skip" in snapshot, (
        "the engine stopped producing last_native_mtp_skip; the panel's skip "
        "tile is dead UI again"
    )

    panel = (
        Path(__file__).resolve().parents[1]
        / "panel/src/renderer/src/components/sessions/PerformancePanel.tsx"
    )
    if panel.exists():
        assert "last_native_mtp_skip" in panel.read_text(encoding="utf-8"), (
            "the panel no longer reads the skip key — either restore the tile "
            "or stop producing the field"
        )


def test_scheduler_publishes_a_skip_even_with_no_engagement():
    """Gating publication on last_native_mtp hid pure-skip cases entirely."""
    scheduler = (
        Path(__file__).resolve().parents[1] / "vmlx_engine" / "scheduler.py"
    )
    src = scheduler.read_text(encoding="utf-8")
    idx = src.index("native_mtp_stats_snapshot()")
    window = src[idx : idx + 700]
    assert "last_native_mtp_skip" in window, (
        "the scheduler publishes MTP telemetry only when MTP engaged, so a "
        "model that never engaged reports nothing and the reason cannot reach "
        "the UI"
    )
