# SPDX-License-Identifier: Apache-2.0
"""One bad batch must not kill the L2 writer thread for the process lifetime.

``_process_write_batch`` was wrapped in try/finally with no ``except``. A single
unguarded exception escaped the loop, ran the outer ``finally:
write_conn.close()``, and the writer died permanently. Every later block-disk
write then sat in the queue forever — no disk cache, no error surfaced, and the
write fences those items were meant to settle never resolved, so their blocks
stayed pinned and un-evictable.

A failed batch costs a re-prefill. A dead writer costs the entire L2 tier.
"""

from __future__ import annotations

import re
from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[1] / "vmlx_engine" / "block_disk_store.py"
)


def test_write_batch_failure_is_caught_inside_the_loop():
    src = SOURCE.read_text(encoding="utf-8")
    idx = src.index("self._process_write_batch(write_conn, batch)")
    window = src[idx : idx + 1600]
    assert re.search(r"except\s+Exception", window), (
        "_process_write_batch is unguarded again; one exception kills the "
        "writer thread and silently disables the whole L2 tier"
    )
    assert "_complete_write_items" in window, (
        "the finally that settles queued items must still run, or fences hang"
    )


def test_the_failure_is_logged_at_error_not_swallowed():
    src = SOURCE.read_text(encoding="utf-8")
    idx = src.index("self._process_write_batch(write_conn, batch)")
    window = src[idx : idx + 1600]
    assert "logger.error" in window, (
        "a swallowed batch failure is indistinguishable from a working cache; "
        "it must be loud"
    )
