import asyncio
import json
import logging
from types import SimpleNamespace

import pytest

from vmlx_engine import memory_status


def fake(active=96, limit=96, physical=128):
    return SimpleNamespace(
        get_active_memory=lambda: active,
        device_info=lambda: {"max_recommended_working_set_size": limit, "memory_size": physical},
    )


@pytest.fixture(autouse=True)
def no_override(monkeypatch):
    monkeypatch.delenv("VMLX_METAL_WS_MAX_BYTES", raising=False)
    monkeypatch.delenv("VMLX_METAL_WS_MAX_GB", raising=False)


def test_measured_no_estimate_or_allocator_mutation():
    result = memory_status.snapshot(fake())
    assert result["available"] is True
    assert result["active_bytes"] == 96
    assert result["limit_bytes"] == result["device_limit_bytes"] == 96
    assert result["physical_bytes"] == 128
    assert result["pid"] > 0 and result["measured_at_ms"] > 0


@pytest.mark.parametrize("active,limit", [(None, 96), (float("nan"), 96), (-1, 96), (True, 96), (96, 0), (96, float("inf"))])
def test_unknown_measurement_does_not_invent_zero_or_fraction(active, limit):
    result = memory_status.snapshot(fake(active=active, limit=limit))
    assert not result["available"]
    assert "active_bytes" not in result


def test_unknown_physical_ram_keeps_measurement_without_recommendation():
    assert memory_status.snapshot(fake(physical=None))["physical_bytes"] is None


def test_engine_override_is_reported_distinctly(monkeypatch):
    monkeypatch.setenv("VMLX_METAL_WS_MAX_BYTES", "80")
    result = memory_status.snapshot(fake())
    assert result["limit_bytes"] == 80
    assert result["device_limit_bytes"] == 96


def test_legacy_api_and_failure_are_optional():
    assert memory_status.snapshot(SimpleNamespace(metal=fake()))["available"]
    assert not memory_status.snapshot(SimpleNamespace())["available"]


def test_lifecycle_emits_separate_measured_diagnostic(monkeypatch, caplog):
    from vmlx_engine import load_progress
    measured = memory_status.snapshot(fake())
    monkeypatch.setattr(memory_status, "snapshot", lambda: measured)
    with caplog.at_level(logging.INFO):
        load_progress.report_shard(2, 3)
    rows = [r.getMessage() for r in caplog.records]
    progress = json.loads(next(r.split("LOADPROGRESS ")[1] for r in rows if r.startswith("LOADPROGRESS ")))
    memory = json.loads(next(r.split("MEMORYSTATUS ")[1] for r in rows if r.startswith("MEMORYSTATUS ")))
    assert progress["completed"] == 2 and "active_bytes" not in progress
    assert memory == measured


def test_rejection_uses_final_guard_measurement(monkeypatch, caplog):
    measured = memory_status.snapshot(fake())
    monkeypatch.setattr(memory_status, "snapshot", lambda: measured.copy())
    with caplog.at_level(logging.WARNING):
        memory_status.emit_guard_rejection(95.5, 96, 99)
    payload = json.loads(caplog.records[-1].getMessage().split("MEMORYSTATUS ")[1])
    assert payload["active_bytes"] == 95.5
    assert payload["reason"] == "guard_rejection" and payload["threshold_pct"] == 99


def test_health_cached_branch_gets_fresh_memory(monkeypatch):
    from vmlx_engine import server
    monkeypatch.setattr(server, "_engine", object())
    monkeypatch.setattr(server, "_get_scheduler", lambda: SimpleNamespace(running=[1], waiting=[]))
    monkeypatch.setattr(server, "_health_snapshot_cache", {"result": {"status": "healthy", "metal_memory": {"active_bytes": 1}}})
    monkeypatch.setattr(server, "_live_mllm_request_lifecycle_snapshot", lambda _: None)
    monkeypatch.setattr(server, "_health_status_value", lambda: "healthy")
    current = memory_status.snapshot(fake(active=90))
    monkeypatch.setattr(memory_status, "snapshot", lambda: current)
    body = asyncio.run(server.health())
    assert body["health_gauges_cached"]
    assert body["metal_memory"] == current


def test_actual_guard_rejection_logs_after_reclamation(monkeypatch):
    import mlx.core as mx
    from fastapi import HTTPException
    from vmlx_engine import server
    from vmlx_engine.utils import memory_limits
    monkeypatch.setattr(server, "_model_path", "/test/local-bundle")
    monkeypatch.setattr(server, "_model_name", "local-bundle")
    monkeypatch.setattr(server, "_engine", None)
    monkeypatch.setattr(server, "_last_metal_ws_log", 0)
    monkeypatch.setattr(memory_limits, "is_metal_ws_guard_enabled", lambda: True)
    monkeypatch.setattr(memory_limits, "get_metal_ws_guard_threshold", lambda _: 99.0)
    samples = iter([(100, 100), (99, 100)])
    monkeypatch.setattr(memory_limits, "get_effective_metal_working_set_bytes", lambda _: next(samples))
    monkeypatch.setattr(mx, "clear_cache", lambda: None)
    events = []
    monkeypatch.setattr(memory_status, "emit_guard_rejection", lambda *args: events.append(args))
    with pytest.raises(HTTPException) as error:
        asyncio.run(server.check_metal_working_set_pressure(None))
    assert error.value.status_code == 503
    assert events == [(99, 100, 99.0)]
