"""CPU-only terminal delivery and SSD completion ordering."""
import asyncio
from concurrent.futures import Future
from dataclasses import dataclass

import pytest

from vmlx_engine.utils.terminal_persistence import terminal_persistence_outputs


@dataclass
class Output:
    new_text: str = "last token"
    finished: bool = True
    finish_reason: str | None = "stop"
    generated_at: float | None = None
    completion_tokens: int = 12


@pytest.mark.parametrize("outcome", ["stored", "failed", "refused"])
def test_final_text_precedes_writer_and_terminal_follows_it(outcome):
    async def run():
        receipt = Future()
        records = []
        stream = terminal_persistence_outputs(Output(), receipt, "request-a", records.append)
        text = await anext(stream)
        assert text.new_text == "last token" and not text.finished
        assert text.finish_reason is None and not receipt.done()
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        assert not pending.done() and records == []
        receipt.set_result({"outcome": outcome, "durable": outcome == "stored"})
        terminal = await pending
        assert terminal.new_text == "" and terminal.finished
        assert terminal.completion_tokens == text.completion_tokens == 12
        assert terminal.generated_at == text.generated_at
        assert records[0]["outcome"] == outcome
        assert records[0]["request_id"] == "request-a"
        assert records[0]["waited"]
        await stream.aclose()
    asyncio.run(run())


def test_disconnect_does_not_cancel_owned_write():
    async def run():
        receipt = Future()
        stream = terminal_persistence_outputs(Output(new_text=""), receipt, "request-b", lambda _: None)
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert not receipt.cancelled()
        receipt.set_result({"outcome": "stored", "durable": True})
        await asyncio.sleep(0)
    asyncio.run(run())
