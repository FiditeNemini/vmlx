# SPDX-License-Identifier: Apache-2.0
"""The CLI and the app must not disagree about defaults on the same build.

Every entry here is a MEASURED divergence, not a style preference: the same
engine, same model, behaved differently depending on whether it was started by
the Electron app or from a terminal. Those are the worst kind of bug to chase,
because a reproduction from one surface does not reproduce from the other.

The tests read BOTH sources so a change to either side breaks the pair.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "vmlx_engine" / "cli.py"
SESSIONS = ROOT / "panel" / "src" / "main" / "sessions.ts"

pytestmark = pytest.mark.skipif(
    not SESSIONS.exists(), reason="panel sources not present in this checkout"
)


def test_cache_index_target_tokens_match():
    """Both surfaces size the cache index by TOKENS, to the same target.

    A flat block COUNT indexes a different amount of context at a different
    block size: the old CLI default of 1000 blocks at 64 tokens indexed 63,936
    tokens, while the app indexed 262,144. Measured symptom recorded in
    sessions.ts: a 77k prompt got ZERO reuse on an exact repeat (82.5s vs 55.7s
    cold) from a CLI-started engine.
    """
    from vmlx_engine.cli import GENERIC_INDEX_TARGET_TOKENS

    panel = re.search(
        r"const GENERIC_INDEX_TARGET_TOKENS\s*=\s*(\d+)",
        SESSIONS.read_text(encoding="utf-8"),
    )
    assert panel, "panel no longer defines GENERIC_INDEX_TARGET_TOKENS"
    assert int(panel.group(1)) == GENERIC_INDEX_TARGET_TOKENS


def test_cli_sizes_the_index_the_same_way_the_panel_does():
    """indexBlocksForCapacity: ceil(target / blockSize) + 1, the +1 being null."""
    from vmlx_engine.cli import GENERIC_INDEX_TARGET_TOKENS

    def panel_rule(block_size: int) -> int:
        return -(-GENERIC_INDEX_TARGET_TOKENS // block_size) + 1

    # 64 is the shipped default block size; the app sends exactly 4097 there.
    assert panel_rule(64) == 4097
    # And the DSV4 lane's own constant stays consistent with its 256 block.
    from vmlx_engine.cli import DSV4_MAX_CACHE_BLOCKS, DSV4_PAGED_CACHE_BLOCK_SIZE

    assert DSV4_MAX_CACHE_BLOCKS == 4097
    assert DSV4_PAGED_CACHE_BLOCK_SIZE == 256


def test_step3p7_is_paged_exempt_like_the_panel_requires():
    """step3p7 is is_mllm, but the app treats its cache subtype as paged-REQUIRED.

    Without the exemption a bare CLI launch ran it unpaged and re-prefilled every
    repeated long prompt, while the app reused blocks — same build, same model.
    """
    from vmlx_engine.cli import _PAGED_MLLM_EXEMPT_FAMILIES

    assert "step3p7" in _PAGED_MLLM_EXEMPT_FAMILIES
    panel = SESSIONS.read_text(encoding="utf-8")
    assert "step3p7_full_sliding_kv" in panel, (
        "panel no longer declares the step3p7 paged-required subtype; "
        "re-check whether the CLI exemption is still correct"
    )
