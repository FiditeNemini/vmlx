# SPDX-License-Identifier: Apache-2.0
"""The streaming stall guard must not throw away work it already paid for.

`_stream_with_keepalive` enforces `total_timeout` at the TOP of its loop, before
draining `pending`. So when a long prefill finally produces its first token at
the same moment the window expires, the guard raises TimeoutError over a result
that is sitting right there, already computed.

LIVE-REPRODUCED on the box during the 400k growing-multiturn benchmark
(2026-08-12). DSV4-Flash, turn 4, 332k-token prompt with 249k reused and 83k
fresh:

    Request chatcmpl-e8f3b29a: worker reconstructed 973 block(s) from L2
    ERROR: Stream error for chatcmpl-e8f3b29a: Streaming exceeded 900.0s timeout

wall=903.3s against a 900s window — the turn was killed at the boundary, after
~15 minutes of prefill, and the recorded row was ptok=0 ctok=0 ttft=None.

The guard's own docstring says "Only a request with zero progress across a full
window is treated as wedged." A produced item is the strongest possible evidence
of progress, and the guard ignored it, relying solely on an out-of-band probe.

Created by Jinho Jang (eric@jangq.ai).
"""

from __future__ import annotations

import asyncio

import pytest

from vmlx_engine.server import _stream_with_keepalive


async def _drain(agen):
    """Collect real items, dropping keep-alive sentinels."""
    items = []
    async for item in agen:
        if item is not None:
            items.append(item)
    return items


@pytest.mark.asyncio
async def test_a_result_ready_at_the_deadline_is_not_discarded():
    """The turn-4 shape: one very slow item that lands as the window expires."""

    async def slow_then_done():
        # Longer than the timeout below, so the deadline passes while the
        # engine is mid-prefill and the token arrives just after — the live
        # shape was 903.3s of work against a 900s window.
        await asyncio.sleep(0.12)
        yield "first-token"
        yield "second-token"

    items = await _drain(
        _stream_with_keepalive(
            slow_then_done(),
            interval=0.01,
            total_timeout=0.05,
            # A server-default timeout supplies a probe. This one reports
            # nothing — exactly the case that killed turn 4 — so the guard
            # cannot lean on it and must use the produced item instead.
            progress_probe=lambda: None,
        )
    )
    assert items == ["first-token", "second-token"]


@pytest.mark.asyncio
async def test_progress_keeps_a_long_stream_alive_without_a_probe():
    """Steady output past the window must not be killed under the default.

    Each item arrives well inside `interval`, but the total run far exceeds
    `total_timeout`. A stall timeout has to measure the gap between items, not
    the total duration.
    """

    async def steady():
        for index in range(12):
            await asyncio.sleep(0.02)
            yield index

    items = await _drain(
        _stream_with_keepalive(
            steady(),
            interval=0.05,
            total_timeout=0.06,
            progress_probe=lambda: None,
        )
    )
    assert items == list(range(12))


@pytest.mark.asyncio
async def test_a_genuinely_wedged_stream_is_still_killed():
    """The guard must keep doing its job: no output, no probe progress."""

    async def wedged():
        await asyncio.sleep(30)
        yield "never"

    with pytest.raises(TimeoutError):
        await _drain(
            _stream_with_keepalive(
                wedged(),
                interval=0.01,
                total_timeout=0.05,
                progress_probe=lambda: 0,
            )
        )


@pytest.mark.asyncio
async def test_probe_progress_still_extends_the_window():
    """The existing out-of-band extension must keep working."""
    ticks = iter([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    async def slow():
        await asyncio.sleep(0.2)
        yield "done"

    items = await _drain(
        _stream_with_keepalive(
            slow(),
            interval=0.01,
            total_timeout=0.02,
            progress_probe=lambda: next(ticks, 100),
        )
    )
    assert items == ["done"]


@pytest.mark.asyncio
async def test_an_explicit_per_request_timeout_stays_wall_clock():
    """Contract preserved: no probe means the caller asked for a hard budget.

    `request.timeout` passes progress_probe=None, and that path is documented as
    wall-clock. Turning it into a stall timeout would silently ignore a limit the
    caller set deliberately.
    """

    async def steady():
        for index in range(100):
            await asyncio.sleep(0.01)
            yield index

    with pytest.raises(TimeoutError):
        await _drain(
            _stream_with_keepalive(
                steady(),
                interval=0.005,
                total_timeout=0.05,
                progress_probe=None,
            )
        )


@pytest.mark.asyncio
async def test_unknown_progress_grace_is_bounded():
    """An unreadable probe buys time, it does not buy immunity.

    A request that never produces anything must still die, or a wedged stream
    would hold the engine forever.
    """
    from vmlx_engine.server import _UNKNOWN_PROGRESS_GRACE_WINDOWS

    assert _UNKNOWN_PROGRESS_GRACE_WINDOWS >= 1

    async def never():
        await asyncio.sleep(30)
        yield "never"

    with pytest.raises(TimeoutError):
        await _drain(
            _stream_with_keepalive(
                never(),
                interval=0.005,
                total_timeout=0.02,
                progress_probe=lambda: None,
            )
        )
