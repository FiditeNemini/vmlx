"""A SimpleEngine session (DFlash2 lanes, non-batched serving) has no scheduler and no prefix cache.

Before: /v1/cache/stats carried no engine block unless DFlash2 was on, the Cache panel waited for a section that
never came and showed "loading" forever (T-PERF wiring map 2026-10-11). Now every engine publishes engine_stats
(type + last durability) and the panel shows an explicit no-prefix-cache state for a loaded empty payload.
"""
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_simple_engine_cache_stats_name_the_engine_and_its_durability(monkeypatch):
    import vmlx_engine.server as server
    import vmlx_engine.speculative as speculative

    durability = {"request_id": "r1", "waited": False, "wait_ms": 0.0}

    class _SimpleEngine:
        is_mllm = False

        def get_cache_stats(self):
            return None

        def get_stats(self):
            return {"engine_type": "simple", "last_durability": durability}

    monkeypatch.setattr(server, "_engine", _SimpleEngine())
    monkeypatch.setattr(server, "_get_scheduler", lambda: None)
    monkeypatch.setattr(speculative, "is_dflash2_enabled", lambda: False)
    payload = await server.cache_stats()
    assert payload["engine_stats"] == {"engine_type": "simple", "last_durability": durability}
    assert "scheduler_stats" not in payload and "dflash2_cache" not in payload


def test_cache_panel_distinguishes_loading_from_no_engine_cache():
    source = Path("panel/src/renderer/src/components/sessions/CachePanel.tsx").read_text(encoding="utf-8")
    assert "{!stats && !error && (" in source                      # loading: only before the first payload
    assert "t('sessions.cachePanel.noEngineCache'" in source
    assert "!schedulerCache && !schedulerStats && !dflash2Cache && !stats?.error && (" not in source
