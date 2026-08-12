# SPDX-License-Identifier: Apache-2.0
"""A full pending-write buffer must not silently throw the block away.

`BlockDiskStore` bounds the RAM held by not-yet-serialized block payloads. When
that budget is full, `_reserve_pending_write_bytes` either WAITS for the
background writer to drain (up to the caller's admission timeout) or gives up
and drops the block. Ordinary families passed a hard 0.0, so they never waited:
a block the engine had already paid to prefill was discarded because a 512MB
buffer happened to be full at that instant, leaving only a repeated WARNING.

MEASURED on the box 2026-08-12, identical 4-prompt workload, block disk cache
in a scratch dir, Nanbeige4.2-3B-JANG_4M:

    admission timeout   disk_writes   byte_budget_drops   tokens on disk
    0.0 (old)                   203                 101           12,992
    0.25 (new)                  304                   0           19,260

33% of blocks were being discarded, and a quarter-second bounded wait recovered
all of them: request latencies were 1.83/2.17/2.10/2.10s before and
1.84/2.17/2.13/2.17s after.

The same A/B on Nemotron-3.5-Lightning-30B-A3B-JANG_4M produced ZERO drops in
BOTH arms (624 writes, 39,512 tokens, identical) — the slower model never
saturates the writer, so the changed path is not even reached. That is the
shape of the bug: it bites FAST models hardest, which are exactly the ones whose
TTFT depends most on L2 reuse.

The wait happens on the model-owning thread, so it must stay short. The 30s
granted to path-dependent families is defensible only because a dropped block
BREAKS THE CHAIN for them rather than merely costing a re-prefill.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX_CACHE = ROOT / "vmlx_engine" / "prefix_cache.py"


def test_ordinary_families_get_a_bounded_nonzero_write_admission_wait():
    src = PREFIX_CACHE.read_text(encoding="utf-8")

    idx = src.index('"admission_timeout"')
    window = src[idx : idx + 400]
    assert "_native_block_disk_admission_timeout" in window, (
        "path-dependent families lost their long admission window"
    )
    assert "_block_disk_admission_timeout" in window, (
        "ordinary families are back on a literal admission timeout; a hard 0.0 "
        "here discards 33% of blocks on a fast model (measured)"
    )
    assert not re.search(r"else\s+0\.0\s*\)", window), (
        "the non-native branch went back to dropping instead of waiting"
    )


def test_the_ordinary_wait_defaults_short_and_stays_overridable():
    src = PREFIX_CACHE.read_text(encoding="utf-8")

    m = re.search(
        r'"VMLX_BLOCK_DISK_ADMISSION_TIMEOUT_SECONDS",\s*\n\s*"([0-9.]+)"',
        src,
    )
    assert m, "the ordinary-family admission timeout is no longer configurable"
    default = float(m.group(1))
    assert 0.0 < default <= 1.0, (
        f"default admission wait is {default}s. It blocks the model-owning "
        "thread, so it must stay sub-second; 0 reintroduces the silent drops"
    )

    native = re.search(
        r'"VMLX_NATIVE_BLOCK_DISK_ADMISSION_TIMEOUT_SECONDS",\s*\n\s*"([0-9.]+)"',
        src,
    )
    assert native, "the path-dependent admission timeout knob disappeared"
    assert float(native.group(1)) > default, (
        "path-dependent families must keep the LONGER window — for them a "
        "dropped block breaks the chain, not just a re-prefill"
    )


def test_the_drop_counter_is_still_reported():
    """The only signal this is happening is telemetry, so it must survive.

    The bug was invisible for as long as it was because nothing surfaced it
    except a repeated WARNING in the engine log.
    """
    store = (ROOT / "vmlx_engine" / "block_disk_store.py").read_text(
        encoding="utf-8"
    )
    assert '"byte_budget_drops"' in store
    assert '"max_pending_bytes"' in store
