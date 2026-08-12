# SPDX-License-Identifier: Apache-2.0
"""The last-resort dropper must be able to reclaim ORDINARY KV blocks.

The candidate filter required ``keep_resident``, so it only ever considered
native/pinned payloads. An ordinary KV block whose L2 admission was rejected
keeps ``keep_resident`` False, the durability fence refuses to release it, and
nothing else reclaims it — so while L2 was full or failing, the byte ceiling was
unenforceable for plain KV and RAM grew for the duration of the outage.

Losing a plain block costs a re-prefill, which is this pass's own doctrine.
"""

from __future__ import annotations

import re
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "vmlx_engine" / "paged_cache.py"


def _candidate_filter_source() -> str:
    src = SOURCE.read_text(encoding="utf-8")
    start = src.index("def _force_drop_undrainable_locked")
    end = src.index("if dropped:", start)
    return src[start:end]


def test_filter_does_not_require_keep_resident():
    window = _candidate_filter_source()
    assert not re.search(r"^\s*and b\.keep_resident\s*$", window, re.MULTILINE), (
        "the force-drop candidate filter requires keep_resident again; ordinary "
        "KV blocks rejected by L2 become permanently un-evictable and the byte "
        "ceiling cannot be enforced during an L2 outage"
    )


def test_the_other_safety_guards_are_still_present():
    """Relaxing one predicate must not relax the chain-safety ones."""
    window = _candidate_filter_source()
    for guard in (
        "b.ref_count == 0",
        "not b.is_null",
        "b.cache_data is not None",
        "b.block_hash is not None",
        "b.block_hash not in live_parents",
        "not drainable_soon(b)",
    ):
        assert guard in window, f"force-drop lost its {guard!r} guard"
