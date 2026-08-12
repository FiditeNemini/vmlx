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
