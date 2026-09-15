"""Measured Metal diagnostics, separate from load progress and admission policy.

Active allocations are NOT physical residency, model file size, allocator-cache
bytes or SSD usage. Reading these counters never evaluates/synchronizes a graph,
clears a cache, raises a limit or changes whether a request is admitted.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time

from .utils.memory_limits import resolve_working_set_override

logger = logging.getLogger(__name__)


def snapshot(mx_module=None) -> dict:
    result = {
        "version": 1,
        "available": False,
        "pid": os.getpid(),
        "measured_at_ms": int(time.time() * 1000),
        "source": "mlx_active_working_set",
        "reason": "measurement",
    }
    try:
        if mx_module is None:
            import mlx.core as mx_module
        metal = getattr(mx_module, "metal", None)
        get_active = getattr(mx_module, "get_active_memory", None) or metal.get_active_memory
        get_info = getattr(mx_module, "device_info", None) or metal.device_info
        active = get_active()
        info = get_info()
        device_limit = info.get("max_recommended_working_set_size")
        physical = info.get("memory_size")
        # Unknown is not zero. Never manufacture a default RAM fraction.
        if not all(
            isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v) and v >= 0
            for v in (active, device_limit)
        ) or device_limit <= 0:
            return result
        limit = resolve_working_set_override(int(device_limit))
        if limit <= 0:
            return result
        physical = (
            int(physical) if isinstance(physical, (int, float))
            and not isinstance(physical, bool) and math.isfinite(physical)
            and physical > 0 else None
        )
        result.update(
            available=True,
            active_bytes=int(active),
            device_limit_bytes=int(device_limit),
            limit_bytes=limit,
            physical_bytes=physical,
        )
    except Exception:
        # Optional diagnostic must never break a load, health or rejection.
        pass
    return result


def emit_snapshot() -> dict:
    """Cold-load transport; HTTP is not serving during lifespan startup."""
    result = snapshot()
    logger.info("MEMORYSTATUS %s", json.dumps(result, separators=(",", ":")))
    return result


def emit_guard_rejection(active: int, limit: int, threshold_pct: float) -> None:
    """Report the exact final measurement that actually caused a rejection.

    Called only AFTER reclamation and baseline forgiveness, not on a predicted
    token envelope or a recoverable first reading. No new admission rule.
    """
    result = snapshot()
    if not result["available"] or result["limit_bytes"] != limit:
        return
    result.update(
        active_bytes=active, reason="guard_rejection",
        threshold_pct=threshold_pct,
    )
    logger.warning("MEMORYSTATUS %s", json.dumps(result, separators=(",", ":")))
