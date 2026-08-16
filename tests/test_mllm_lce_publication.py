# SPDX-License-Identifier: Apache-2.0
"""Pins for MLLM last_cache_execution publication (campaign #181 telemetry fix).

Live gate runs proved the finish-time response can arrive WITHOUT its
cache_execution dict: cold requests then published nothing to
batch_stats.last_cache_execution, and an earlier record-less call permanently
blocked a later record-carrying one via the once-recorded marker. The fix
makes publication idempotent (runs before the marker short-circuit) and
accepts fallbacks (burst-scan record, then request._cache_execution)."""

from types import SimpleNamespace

from vmlx_engine.mllm_scheduler import MLLMScheduler


def _bare_scheduler():
    sched = object.__new__(MLLMScheduler)
    sched.batch_generator = SimpleNamespace(
        _stats=SimpleNamespace(last_cache_execution=None)
    )
    sched._cache_hit_requests = 0
    sched._cache_hit_tokens = 0
    sched._cache_hit_tokens_by_detail = {}
    return sched


def _response(cache_execution=None, cached_tokens=0):
    return SimpleNamespace(
        cached_tokens=cached_tokens,
        cache_execution=cache_execution,
        cache_detail="",
    )


def test_recordless_finish_response_publishes_via_fallback():
    sched = _bare_scheduler()
    request = SimpleNamespace()
    record = {"request_id": "resp_abc", "cached_tokens": 0, "cache_outcome": "miss"}
    sched._record_cache_hit(
        _response(cache_execution=None), request, execution_fallback=record
    )
    assert sched.batch_generator._stats.last_cache_execution == record


def test_recordless_first_call_does_not_block_later_record():
    sched = _bare_scheduler()
    request = SimpleNamespace()
    # First call: no record anywhere — publishes nothing, marks recorded.
    sched._record_cache_hit(_response(), request)
    assert sched.batch_generator._stats.last_cache_execution is None
    assert request._cache_hit_recorded is True
    # Second call NOW carries the record: publication must still run.
    record = {"request_id": "resp_late", "cached_tokens": 64, "cache_outcome": "hit"}
    sched._record_cache_hit(_response(cache_execution=record, cached_tokens=64), request)
    assert sched.batch_generator._stats.last_cache_execution == record
    # Counters stayed once-only (marker still guards them).
    assert sched._cache_hit_requests == 0


def test_request_side_record_is_last_resort_fallback():
    sched = _bare_scheduler()
    record = {"request_id": "resp_req", "cache_outcome": "miss"}
    request = SimpleNamespace(_cache_execution=record)
    sched._record_cache_hit(_response(), request)
    assert sched.batch_generator._stats.last_cache_execution == record


def test_cold_miss_marks_recorded_and_counts_nothing():
    sched = _bare_scheduler()
    request = SimpleNamespace()
    record = {"request_id": "resp_cold", "cached_tokens": 0, "cache_outcome": "miss"}
    sched._record_cache_hit(_response(cache_execution=record), request)
    assert sched.batch_generator._stats.last_cache_execution == record
    assert request._cache_hit_recorded is True
    assert sched._cache_hit_requests == 0
    assert sched._cache_hit_tokens == 0
