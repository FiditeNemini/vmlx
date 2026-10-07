import asyncio
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from vmlx_engine.engine.simple import SimpleEngine


@pytest.mark.parametrize('exit_kind', ['error', 'close', 'complete'])
def test_mllm_iterator_and_request_cleanup(exit_kind):
    async def run():
        closed = []
        def model_stream(**kwargs):
            try:
                if exit_kind == 'error':
                    raise ValueError('bad media')
                yield SimpleNamespace(text='hello', finish_reason='stop' if exit_kind == 'complete' else None,
                                      prompt_tokens=3, completion_tokens=1)
            finally:
                closed.append(True)
        with patch('vmlx_engine.engine.simple.is_mllm_model', return_value=True):
            engine = SimpleEngine('fake')
        engine._loaded = True
        engine._model = SimpleNamespace(stream_chat=model_stream)
        async def call(fn):
            return fn()
        engine._run_model_call = call
        stream = engine.stream_chat([], request_id='owned')
        if exit_kind == 'error':
            with pytest.raises(ValueError, match='bad media'):
                await anext(stream)
        else:
            await anext(stream)
            await stream.aclose()
        assert engine._current_request_id is None
        assert not engine._generation_lock.locked()
        assert closed == [True]
    asyncio.run(run())


def test_cancel_closes_worker_iterator_before_next_request():
    import threading
    async def run():
        entered, release = threading.Event(), threading.Event()
        order = []
        def model_stream(messages, **kwargs):
            name = messages[0]['content']
            try:
                order.append('start-' + name)
                if name == 'first':
                    entered.set()
                    assert release.wait(5)
                yield SimpleNamespace(text=name, finish_reason='stop', prompt_tokens=3, completion_tokens=1)
            finally:
                order.append('close-' + name)
        with patch('vmlx_engine.engine.simple.is_mllm_model', return_value=True):
            engine = SimpleEngine('fake')
        engine._loaded = True
        engine._model = SimpleNamespace(stream_chat=model_stream)
        first = engine.stream_chat([{'role':'user','content':'first'}], request_id='first')
        second = engine.stream_chat([{'role':'user','content':'second'}], request_id='second')
        task = asyncio.create_task(anext(first))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            following = asyncio.create_task(anext(second))
            await asyncio.sleep(.02)
            assert engine._current_request_id == 'first'
            assert not following.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert (await following).text == 'second'
            await second.aclose()
            assert order == ['start-first', 'close-first', 'start-second', 'close-second']
            assert engine._current_request_id is None
        finally:
            release.set()
            engine._model_executor.shutdown(wait=True)
    asyncio.run(run())
