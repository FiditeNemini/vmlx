"""Engine-owned lifecycle load-progress contract (vmlx_engine/load_progress.py).

The panel's bar is driven by these snapshots — LOADPROGRESS stdout lines
during cold start, /health `load_progress` during wakes. RSS is a separate
diagnostic and never the percentage oracle.
"""

import json
import logging

import pytest

from vmlx_engine import load_progress


@pytest.fixture(autouse=True)
def _reset_state():
    # The module is process-global; make each test hermetic.
    with load_progress._LOCK:
        load_progress._state.update(
            phase=load_progress.PHASE_IDLE,
            completed=0,
            total=0,
            model_loaded=False,
            ready=False,
            generation=0,
        )
    yield


def test_begin_attempt_bumps_generation_and_resets_units():
    first = load_progress.begin_attempt()
    load_progress.report_shard(5, 10)
    second = load_progress.begin_attempt()
    snap = load_progress.snapshot()
    assert second == first + 1
    assert snap["generation"] == second
    assert snap["completed"] == 0 and snap["total"] == 0
    assert snap["model_loaded"] is False and snap["ready"] is False
    assert snap["phase"] == load_progress.PHASE_STARTING


def test_ready_forces_phase_ready():
    load_progress.begin_attempt()
    load_progress.report_shard(10, 10)
    load_progress.report(model_loaded=True, ready=True)
    snap = load_progress.snapshot()
    assert snap["phase"] == load_progress.PHASE_READY
    assert snap["ready"] is True and snap["model_loaded"] is True
    assert snap["completed"] == 10 and snap["total"] == 10


def test_standby_depths():
    load_progress.begin_attempt()
    load_progress.report(model_loaded=True, ready=True)
    load_progress.report_standby("soft")
    soft = load_progress.snapshot()
    assert soft["ready"] is False and soft["model_loaded"] is True
    load_progress.report_standby("deep")
    deep = load_progress.snapshot()
    assert deep["ready"] is False and deep["model_loaded"] is False
    assert deep["phase"] == load_progress.PHASE_IDLE


def test_every_update_emits_a_parseable_loadprogress_line(caplog):
    with caplog.at_level(logging.INFO, logger="vmlx_engine.load_progress"):
        load_progress.begin_attempt()
        load_progress.report_shard(3, 17)
    lines = [r.getMessage() for r in caplog.records if "LOADPROGRESS" in r.getMessage()]
    assert len(lines) == 2
    payload = json.loads(lines[-1].split("LOADPROGRESS ", 1)[1])
    assert payload["phase"] == "loading_weights"
    assert payload["completed"] == 3 and payload["total"] == 17
    assert isinstance(payload["generation"], int)


def test_snapshot_is_a_copy():
    snap = load_progress.snapshot()
    snap["phase"] = "tampered"
    assert load_progress.snapshot()["phase"] != "tampered"


def test_health_payload_carries_the_snapshot():
    # /health must expose the snapshot so an external API-triggered JIT wake
    # is visible to the panel (the server is up during every wake).
    from fastapi.testclient import TestClient

    from vmlx_engine import server

    load_progress.begin_attempt()
    load_progress.report_shard(2, 8)
    client = TestClient(server.app)
    body = client.get("/health").json()
    assert "load_progress" in body, sorted(body.keys())
    assert body["load_progress"]["phase"] == "loading_weights"
    assert body["load_progress"]["completed"] == 2
    assert body["load_progress"]["total"] == 8
