"""Stop must retire synchronous prefill before admitting the next turn."""
import asyncio
import threading
from types import SimpleNamespace

import mlx.core as mx
import pytest

from vmlx_engine.dspark.minicpm_target import MiniCPMDSparkTarget
from vmlx_engine.engine import simple as simple_mod
from vmlx_engine.engine.simple import SimpleEngine


@pytest.mark.parametrize("method", ["prefill", "prefill_segment"])
@pytest.mark.parametrize("matching", [True, False])
def test_stop_ends_at_chunk_boundary_and_next_request_runs(monkeypatch, method, matching):
    monkeypatch.setattr(simple_mod, "is_mllm_model", lambda *args, **kwargs: True)
    started, release = threading.Event(), threading.Event()
    calls = []
    target = MiniCPMDSparkTarget.__new__(MiniCPMDSparkTarget)
    cancel_first = True

    def stock(ids, cache, taps, logits):
        calls.append((ids.tolist(), threading.get_ident()))
        if cancel_first and len(calls) == 1:
            started.set()
            assert release.wait(5), "abort was never delivered"
        return mx.zeros((1, ids.shape[1], 2)), None

    target._stock = stock

    def stream_chat(**kwargs):
        getattr(target, method)(mx.arange(9)[None], [], [], 2)
        yield SimpleNamespace(text="ready", finish_reason="stop", prompt_tokens=9,
                              completion_tokens=1, cached_tokens=0)

    async def run():
        nonlocal cancel_first
        engine = SimpleEngine("prefill-cancel-fixture")
        engine._loaded = True
        engine._model = SimpleNamespace(stream_chat=stream_chat)
        engine._estimate_mllm_chat_prompt_tokens = lambda *args: 9

        async def collect(request_id):
            return [x async for x in engine.stream_chat([], request_id=request_id)]

        task = asyncio.create_task(collect("cancel-first"))
        assert await asyncio.to_thread(started.wait, 5)
        assert await engine.abort_request("cancel-first" if matching else "other") is matching
        release.set()
        outputs = await asyncio.wait_for(task, 5)
        assert outputs[-1].finish_reason == ("abort" if matching else "stop")
        if matching:
            assert not any(x.new_text for x in outputs)
        else:
            assert outputs[-1].text == "ready"
        assert len(calls) == (1 if matching else 5), "incorrect prefill cancellation"
        assert engine._current_request_id is None
        assert not engine._generation_lock.locked()
        cancel_first = False
        following = await collect("following")
        assert following[-1].text == "ready"
        assert following[-1].finish_reason == "stop"
        assert len({thread for _, thread in calls}) == 1

    asyncio.run(run())


def test_cancellation_scope_restores_and_isolates_worker_threads():
    from vmlx_engine.utils.generation_cancellation import (
        GenerationCancelled, generation_cancellation_scope,
        raise_if_generation_cancelled,
    )

    with generation_cancellation_scope(lambda: True):
        with pytest.raises(GenerationCancelled):
            raise_if_generation_cancelled()
        with generation_cancellation_scope(lambda: False):
            raise_if_generation_cancelled()
        # A second worker must not inherit this request's flag.
        result = []
        thread = threading.Thread(target=lambda: result.append(raise_if_generation_cancelled()))
        thread.start()
        thread.join(5)
        assert result == [None]
        with pytest.raises(GenerationCancelled):
            raise_if_generation_cancelled()
    raise_if_generation_cancelled()
