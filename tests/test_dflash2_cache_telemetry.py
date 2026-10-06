"""DFlash2 telemetry reads metadata, never model arrays or directory contents."""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from vmlx_engine import dflash2_runtime as runtime


def test_session_poll_counts_retained_entries_without_evaluating_arrays(monkeypatch):
    store = runtime._DFlash2SessionStore(max_entries=4)
    store.put({"cache_len": 10, "target_cache": object()})
    store.put({"cache_len": 20, "target_cache": object()})
    monkeypatch.setattr(runtime, "_SESSION_STORE", store)
    monkeypatch.setattr(runtime, "_SESSION_SSD", None)
    stats = runtime.session_cache_stats()
    assert stats["ram"]["entries"] == 2
    assert stats["ram"]["max_entries"] == 4
    assert stats["ram"]["tokens"] == 30
    assert "overlap" in stats["ram"]["tokens_note"]
    assert stats["ssd"] is None
    assert len(store._entries) == 2  # polling never takes ownership


def test_ssd_poll_reports_pool_ledger_not_this_process_written_bytes(monkeypatch):
    @dataclass
    class Budget:
        bytes_after: int = 1234
        max_size_bytes: int = 4096
        accounted: bool = True
        compliant: bool = True
        telemetry_stale: bool = True

    ssd = SimpleNamespace(
        store=SimpleNamespace(root=Path('/pool'), budget=SimpleNamespace(refresh_health=lambda: Budget())),
        stats={"writes": 3, "write_bytes": 9999, "hits": 1, "misses": 2},
        _q=SimpleNamespace(unfinished_tasks=1),
    )
    monkeypatch.setattr(runtime, "_SESSION_SSD", ssd)
    stats = runtime.session_cache_stats()["ssd"]
    assert stats["pending_writes"] == 1
    assert stats["global_budget"]["bytes_after"] == 1234
    assert stats["global_budget"]["telemetry_stale"] is True
    assert stats["global_budget"]["root"] == '/pool'
    assert "pending_writes" not in ssd.stats
