"""Actual SimpleEngine stream bridge, with CPU-only model outputs."""
import asyncio
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import patch

from vmlx_engine.engine.simple import SimpleEngine


def test_tool_terminal_drains_its_own_write_and_preserves_usage():
    async def run():
        receipt = Future()
        chunk = SimpleNamespace(text="<tool_call>done</tool_call>", finish_reason="stop",
                                prompt_tokens=100, completion_tokens=7, cached_tokens=97,
                                cache_detail="dflash2-ssd", persistence_future=receipt)
        with patch("vmlx_engine.engine.simple.is_mllm_model", return_value=True):
            engine = SimpleEngine("fake")
        engine._loaded = True
        engine._model = SimpleNamespace(stream_chat=lambda **_: iter([chunk]))
        async def call(fn):
            return fn()
        engine._run_model_call = call
        assert not await engine.request_graceful_stop(None)
        stream = engine.stream_chat([], request_id="tool-a")
        content = await anext(stream)
        assert not content.finished and content.new_text == chunk.text
        assert await engine.request_graceful_stop("tool-a")
        assert not await engine.request_graceful_stop("another-request")
        assert engine.get_stats()["terminal_cleanup_pending"]
        terminal_task = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        assert not terminal_task.done()
        receipt.set_result({"outcome": "stored", "durable": True, "retained_tokens": 106})
        terminal = await terminal_task
        assert terminal.finished and terminal.new_text == ""
        assert terminal.cached_tokens == 97 and terminal.completion_tokens == 7
        assert terminal.generated_at == content.generated_at
        assert engine.get_stats()["last_durability"]["durable"]
        await stream.aclose()
        assert not await engine.request_graceful_stop("tool-a")
    asyncio.run(run())
