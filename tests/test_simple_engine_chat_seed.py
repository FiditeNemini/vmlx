"""Chat seed must reach the owning MLX worker without leaking into VLM kwargs."""
import asyncio
import threading
from types import SimpleNamespace

import mlx.core as mx
import pytest

from vmlx_engine.engine import simple as simple_mod
from vmlx_engine.engine.simple import SimpleEngine


@pytest.mark.parametrize("mode", ["stream", "dflash_nonstream", "native_nonstream"])
@pytest.mark.parametrize("seed", [None, 0, 123])
def test_chat_seed_applied_once_on_model_worker_before_generation(monkeypatch, mode, seed):
    calls = []
    observed_kwargs = []
    caller_thread = threading.get_ident()
    monkeypatch.setattr(mx.random, "seed", lambda value: calls.append(("seed", value, threading.get_ident())))
    monkeypatch.setattr(simple_mod, "is_mllm_model", lambda _, **kwargs: True)
    monkeypatch.setattr(simple_mod, "_dflash2_enabled", lambda: mode == "dflash_nonstream")

    def chunk():
        calls.append(("generate", None, threading.get_ident()))
        return SimpleNamespace(text="ready", finish_reason="stop", prompt_tokens=10,
                               completion_tokens=1, cached_tokens=7, cache_detail="ssd")

    def stream_chat(**kwargs):
        observed_kwargs.append(kwargs)
        yield chunk()  # Lazy body runs on first next(), not iterator creation.

    def chat(**kwargs):
        observed_kwargs.append(kwargs)
        return chunk()

    async def run():
        engine = SimpleEngine("seed-fixture")
        engine._loaded = True
        engine._model = SimpleNamespace(chat=chat, stream_chat=stream_chat)
        engine._model_family_name = lambda: "qwen3_5"
        kwargs = {} if seed is None else {"seed": seed}
        if mode == "stream":
            out = [x async for x in engine.stream_chat([], request_id="seed-test", **kwargs)][-1]
        else:
            out = await engine.chat([], **kwargs)
        assert out.text == "ready" and out.finish_reason == "stop"

    asyncio.run(run())
    assert all("seed" not in kwargs for kwargs in observed_kwargs)
    assert all(call[2] != caller_thread for call in calls)
    if seed is None:
        assert [call[0] for call in calls] == ["generate"]
    else:
        assert [call[:2] for call in calls] == [("seed", seed), ("generate", None)]
        assert calls[0][2] == calls[1][2]
