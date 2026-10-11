"""Keep real terminal token IDs across streaming and non-streamed DFlash chat."""
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vmlx_engine.engine import simple
from vmlx_engine.models.mllm import MLLMOutput


@pytest.mark.asyncio
@pytest.mark.parametrize('terminal_ids', [[10, 7, 8, 11, 9], None])
async def test_terminal_ids_survive_chat_without_assuming_chunks_have_ids(monkeypatch, terminal_ids):
    with patch('vmlx_engine.engine.simple.is_mllm_model', return_value=True):
        engine = simple.SimpleEngine('fake')
    engine._loaded = True
    chunks = [SimpleNamespace(text='<think>x', finish_reason=None, prompt_tokens=4, completion_tokens=2),
              SimpleNamespace(text='</think>y', finish_reason='stop', prompt_tokens=4,
                              completion_tokens=5, tokens=terminal_ids)]
    engine._model = SimpleNamespace(stream_chat=lambda **kwargs: iter(chunks))
    async def worker(fn, *args, **kwargs):
        return fn(*args, **kwargs)
    engine._run_model_call = worker
    monkeypatch.setattr(simple, '_dflash2_enabled', lambda: True)
    stream = [out async for out in engine.stream_chat([])]
    assert stream[0].tokens == []
    assert stream[-1].tokens == (terminal_ids or [])
    assert stream[-1].completion_tokens == 5
    result = await engine.chat([])
    assert result.tokens == stream[-1].tokens
    assert result.completion_tokens == 5
    assert result.raw_text == '<think>x</think>y'


def test_uninstrumented_mllm_outputs_remain_safe_for_legacy_length_fallbacks():
    first, second = MLLMOutput('a'), MLLMOutput('b')
    assert len(first.tokens) == 0
    first.tokens.append(1)
    assert second.tokens == []

@pytest.mark.asyncio
async def test_nonstream_wrapper_tolerates_terminal_without_token_field(monkeypatch):
    with patch('vmlx_engine.engine.simple.is_mllm_model', return_value=True):
        engine = simple.SimpleEngine('fake')
    engine._loaded = True
    async def stream(**kwargs):
        yield SimpleNamespace(text='answer', prompt_tokens=3, completion_tokens=1,
                              finish_reason='stop', cached_tokens=0, cache_detail='', prefill_usage=None)
    engine.stream_chat = stream
    monkeypatch.setattr(simple, '_dflash2_enabled', lambda: True)
    result = await engine.chat([])
    assert result.tokens == []
    assert result.raw_text == 'answer'
