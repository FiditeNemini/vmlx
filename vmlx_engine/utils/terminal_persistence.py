# SPDX-License-Identifier: Apache-2.0
"""Publish content promptly, but settle a request-owned SSD write before EOF."""

import asyncio
import logging
import time
from dataclasses import replace

from ..persistence_outcome import format_outcome

logger = logging.getLogger(__name__)


async def terminal_persistence_outputs(output, receipt, request_id, on_settled):
    """Keep writer latency separate from production and token delivery.

    The writer owns the receipt. Cancelling a reader must not cancel the write.
    A settled failure is reported as failure, never as durable storage.
    """
    produced_at = time.perf_counter()
    output = replace(output, generated_at=produced_at)
    if output.new_text:
        yield replace(output, finished=False, finish_reason=None)
        output = replace(output, new_text="")
    started = time.perf_counter()
    waited = not receipt.done()
    outcome = await asyncio.shield(asyncio.wrap_future(receipt))
    wait_ms = (time.perf_counter() - started) * 1000
    record = dict(outcome, request_id=request_id, wait_ms=wait_ms, waited=waited)
    on_settled(record)
    logger.info("Terminal durability barrier: request=%s wait_ms=%.3f waited=%s %s",
                request_id, wait_ms, str(waited).lower(), format_outcome(outcome))
    yield output
