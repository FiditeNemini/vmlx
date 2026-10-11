"""Wake must preserve startup drafter selection before target loading."""
import asyncio
from types import SimpleNamespace

from vmlx_engine import server, speculative


def test_two_deep_wakes_keep_drafter_settings_and_startup_order(monkeypatch):
    events = []
    draft_loaded = False

    async def start():
        assert draft_loaded, "target started before its configured drafter"
        events.append("target")
        server._engine._loaded = True

    def load_target(*args, **kwargs):
        # The real load_model replaces this dictionary, losing CLI-only keys.
        server._cli_args = {"use_batching": False}
        server._engine = SimpleNamespace(_loaded=False, start=start)

    def load_draft(config):
        nonlocal draft_loaded
        assert config.model == "/bundle/dflash2"
        assert config.num_tokens == 5
        draft_loaded = True
        events.append("draft")

    async def worker(fn, *args):
        assert server._engine is not None
        return fn(*args)

    async def no_cleanup(*args):
        pass

    for name, value in {
        "_standby_state": "deep", "_pre_sleep_cache_limit": None,
        "_model_type": "mllm", "_model_path": "/bundle", "_model_name": "/bundle",
        "_engine": None, "_enable_jit": False, "_flash_moe_enabled": False,
        "_cli_args": {"speculative_model": "/bundle/dflash2", "num_draft_tokens": 5},
    }.items():
        monkeypatch.setattr(server, name, value)
    monkeypatch.setattr(server, "load_model", load_target)
    monkeypatch.setattr(speculative, "load_draft_model", load_draft)
    monkeypatch.setattr(server, "_run_on_model_executor", worker)
    monkeypatch.setattr(server, "_post_async_engine_teardown_mlx_cleanup", no_cleanup)
    monkeypatch.setattr(server, "_record_metal_ws_model_baseline", lambda *a: None)
    monkeypatch.setattr(server, "_refresh_loaded_max_prompt_tokens", lambda *a: None)

    async def run():
        nonlocal draft_loaded
        for _ in range(2):
            draft_loaded = False
            server._standby_state = "deep"
            assert await server._admin_wake_impl() == {"status": "active"}
            assert server._cli_args["speculative_model"] == "/bundle/dflash2"
            assert server._cli_args["num_draft_tokens"] == 5

    asyncio.run(run())
    assert events == ["draft", "target", "draft", "target"]
