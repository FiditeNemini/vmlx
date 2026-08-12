# SPDX-License-Identifier: Apache-2.0
"""The slow-family timeout lift must reach the STREAMING guard, not just argv.

There are two enforcement points for "how long may this request take":

  * the request timeout, reported at startup as ``Request timeout: 900.0s``
  * ``_stream_timeout`` in the streaming handlers, which falls back to
    ``server._default_timeout`` and raises ``Streaming exceeded {t}s timeout``

``cli.py`` set ``server._default_timeout = args.timeout`` ~350 lines BEFORE it
applied the slow-family lift to ``args.timeout``. So the lift raised argv and
the startup banner while the streaming guard kept the stale 300.

MEASURED on the box: a DSV4-Flash growing-multiturn run died on EVERY turn with
``Streaming exceeded 300.0s timeout``, in a log whose startup lines one screen
earlier read ``timeout 300s -> 900s for slow family=deepseek_v4`` and ``Request
timeout: 900.0s``. Two numbers for one rule, and the one that actually kills the
response was the stale one.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = (ROOT / "vmlx_engine" / "cli.py").read_text(encoding="utf-8")
SERVER = (ROOT / "vmlx_engine" / "server.py").read_text(encoding="utf-8")


def test_the_lift_repropagates_the_server_default():
    lift = CLI.index("for slow family=%s (matches the app ")
    tail = CLI[lift : lift + 1400]
    assert "server._default_timeout = args.timeout" in tail, (
        "the slow-family lift no longer pushes the raised value into "
        "server._default_timeout, so the streaming guard keeps the stale 300s"
    )


def test_the_early_assignment_still_exists_and_is_the_earlier_one():
    """Guards the premise: there are two assignments and order matters."""
    positions = [m.start() for m in re.finditer(
        r"server\._default_timeout = args\.timeout", CLI)]
    assert len(positions) >= 2, (
        "expected the initial assignment plus the post-lift one"
    )
    lift = CLI.index("for slow family=%s (matches the app ")
    assert positions[0] < lift < positions[-1], (
        "the re-propagation must come AFTER the lift; before it, it is a no-op"
    )


def test_the_streaming_guard_really_reads_that_default():
    """If this stops being true, the fix above is aimed at the wrong variable."""
    assert "_stream_timeout = (\n        request.timeout if request.timeout is not None else _default_timeout\n    )" in SERVER
    assert 'f"Streaming exceeded {total_timeout}s timeout"' in SERVER
