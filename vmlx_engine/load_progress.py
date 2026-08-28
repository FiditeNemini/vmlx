"""Engine-owned model lifecycle progress.

Single source of truth for load/wake progress, replacing the panel's
decorative log-pattern oracle. The engine reports *phase*, unit progress
(*completed/total* — safetensors shards where measurable), *model_loaded*
and *ready*; the panel maps phases to a percentage but never invents one.

Two transports for the same snapshot:

- **stdout**: every update is mirrored as one structured line
  ``LOADPROGRESS {...json...}``. Cold start needs this — uvicorn does not
  serve requests until FastAPI's lifespan startup completes, so /health is
  unreachable for the entire initial model load. The panel owns the spawned
  process's stdout and parses these lines live.
- **/health**: the snapshot is embedded in the health payload, which is how
  an externally-triggered JIT wake (raw API request) becomes visible to the
  panel — the server is already up during every wake.

``generation`` increments once per load/wake attempt so consumers can
discard stale events after a stop/restart/PID replacement.

RSS/Metal residency is deliberately NOT part of this contract — memory is a
separate diagnostic and never the percentage oracle (mmap/SSD-streaming
models never make every bundle byte resident).
"""

from __future__ import annotations

import json
import logging
import threading

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()

PHASE_IDLE = "idle"
PHASE_STARTING = "starting"
PHASE_LOADING_WEIGHTS = "loading_weights"
PHASE_INITIALIZING_ENGINE = "initializing_engine"
PHASE_RESTORING_ACCELERATION = "restoring_acceleration"
PHASE_READY = "ready"

_state: dict = {
    "phase": PHASE_IDLE,
    "completed": 0,
    "total": 0,
    "model_loaded": False,
    "ready": False,
    "generation": 0,
}


def snapshot() -> dict:
    """Current progress snapshot (copy — safe to serialize)."""
    with _LOCK:
        return dict(_state)


def _emit(snap: dict) -> None:
    # One parseable line per update. INFO so it rides the normal engine log
    # stream the panel already consumes.
    logger.info("LOADPROGRESS %s", json.dumps(snap, separators=(",", ":")))


def begin_attempt(phase: str = PHASE_STARTING) -> int:
    """Start a new load/wake attempt: bumps the generation, resets units."""
    with _LOCK:
        _state["generation"] += 1
        _state["phase"] = phase
        _state["completed"] = 0
        _state["total"] = 0
        _state["model_loaded"] = False
        _state["ready"] = False
        snap = dict(_state)
    _emit(snap)
    return snap["generation"]


def report(
    phase: str | None = None,
    completed: int | None = None,
    total: int | None = None,
    model_loaded: bool | None = None,
    ready: bool | None = None,
) -> None:
    """Update any subset of the snapshot and mirror it to stdout."""
    with _LOCK:
        if phase is not None:
            _state["phase"] = phase
        if completed is not None:
            _state["completed"] = int(completed)
        if total is not None:
            _state["total"] = int(total)
        if model_loaded is not None:
            _state["model_loaded"] = bool(model_loaded)
        if ready is not None:
            _state["ready"] = bool(ready)
            if ready:
                _state["phase"] = PHASE_READY
        snap = dict(_state)
    _emit(snap)


def report_shard(completed: int, total: int) -> None:
    """Unit progress from a weight-shard loop (any loader family)."""
    report(phase=PHASE_LOADING_WEIGHTS, completed=completed, total=total)


def report_standby(depth: str) -> None:
    """Entering sleep: not ready; deep sleep also unloads the weights."""
    report(
        phase=PHASE_IDLE,
        ready=False,
        model_loaded=(depth != "deep"),
    )
