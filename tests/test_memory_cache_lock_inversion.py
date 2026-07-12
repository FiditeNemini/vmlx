"""Issue #233: memory-cache lock inversion + lockless clear/reset_stats.

fetch() used to hold self._lock across _clone_on_stream_owner's hop to the
single-thread llm-worker executor (.result()); the worker's own
step() -> _cleanup_finished() -> store() blocks on the same lock -> deadlock
in the DEFAULT config. fetch() now snapshots the entry under the lock,
clones with the lock RELEASED, and revalidates entry identity afterwards.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from vmlx_engine.memory_cache import MemoryAwarePrefixCache


def _make_cache():
    """Build a cache around a dummy model (ctor: model, config, model_path)."""
    return MemoryAwarePrefixCache(model=object())


class _FakeSSMLayer:
    """Minimal cumulative-state layer accepted by _clone_cache_for_fetch's
    _safe() gate: exposes a `.cache` list (MambaCache/ArraysCache shape).
    Plain Python payload — these tests exercise LOCK behavior only, and real
    MLX ops on bare test threads abort Metal (no engine stream owner here),
    so _truncate_cache is stubbed in every test."""

    def __init__(self):
        self.cache = [0.0, 0.0]


def test_fetch_does_not_hold_lock_across_clone_executor_hop():
    """The worker must be able to take the lock (store/cleanup) WHILE fetch's
    clone runs on it. Pre-#233 this deadlocked; join(timeout) turns a
    regression into a failure instead of a hang."""
    mgr = _make_cache()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-worker")
    mgr.set_clone_executor(executor, "llm-worker")

    tokens = [1, 2, 3, 4]
    assert mgr.store(tokens, [_FakeSSMLayer()])

    clone_entered = threading.Event()
    release_clone = threading.Event()

    def _slow_truncate(cache, length, *a, **kw):
        # Runs ON the llm-worker (via executor). Signal fetch's clone is in
        # flight, then block until the main thread has proven it can take the
        # cache lock concurrently — exactly the interleaving that deadlocked.
        clone_entered.set()
        assert release_clone.wait(timeout=10), "test orchestration stalled"
        return ["clone-sentinel"]

    mgr._truncate_cache = _slow_truncate
    mgr._materialize_cloned_cache = lambda cloned: None

    result: dict = {}

    def _do_fetch():
        result["fetch"] = mgr.fetch(tokens)

    fetcher = threading.Thread(target=_do_fetch, daemon=True)
    fetcher.start()
    assert clone_entered.wait(timeout=10), "fetch never reached the clone"

    # While the clone is parked on the worker, the lock must be FREE:
    # store() (what the worker's _cleanup_finished does) must not block.
    locked_ok = threading.Event()

    def _worker_side_store():
        # store() takes self._lock; pre-fix this blocked forever because
        # fetch held it across the executor hop.
        mgr.store([9, 9, 9], [_FakeSSMLayer()])
        locked_ok.set()

    storer = threading.Thread(target=_worker_side_store, daemon=True)
    storer.start()
    assert locked_ok.wait(timeout=5), (
        "store() blocked while fetch's clone was in flight — lock held "
        "across the executor hop (issue #233 regression)"
    )

    release_clone.set()
    fetcher.join(timeout=10)
    assert not fetcher.is_alive(), "fetch deadlocked (issue #233 regression)"
    cache, remaining = result["fetch"]
    assert cache == ["clone-sentinel"]
    assert remaining == []
    executor.shutdown(wait=False)


def test_fetch_survives_concurrent_eviction_of_hit_entry():
    """Identity revalidation: if the entry is removed while the clone is in
    flight, fetch must still return the clone (data was valid at snapshot)
    and must NOT resurrect LRU bookkeeping for the gone entry."""
    mgr = _make_cache()
    tokens = [5, 6, 7]
    assert mgr.store(tokens, [_FakeSSMLayer()])

    def _evicting_truncate(cache, length, *a, **kw):
        # Simulate a concurrent remove() landing mid-clone (lock is free now).
        mgr.remove(tokens)
        return ["clone-sentinel"]

    mgr._truncate_cache = _evicting_truncate
    mgr._materialize_cloned_cache = lambda cloned: None
    cache, remaining = mgr.fetch(tokens)
    assert cache == ["clone-sentinel"], "clone from a then-valid snapshot must be returned"
    assert remaining == []
    # Entry is gone; its key must not have been re-inserted into any LRU bucket.
    key = tuple(tokens)
    assert key not in mgr._entries
    for bucket in mgr._lru_by_type.values():
        assert key not in bucket


def test_clear_and_reset_stats_take_the_lock():
    """clear()/reset_stats() are called worker-side while fetch() iterates —
    they must serialize on the same lock (was: lockless, live RuntimeError)."""
    mgr = _make_cache()
    mgr.store([1, 2], [_FakeSSMLayer()])
    assert mgr._lock.acquire(timeout=1)
    try:
        blocked = threading.Event()

        def _try_clear():
            mgr.clear()
            mgr.reset_stats()
            blocked.set()

        t = threading.Thread(target=_try_clear, daemon=True)
        t.start()
        # While we hold the lock, clear() must NOT complete.
        assert not blocked.wait(timeout=0.5), "clear()/reset_stats() ran lockless"
    finally:
        mgr._lock.release()
    assert blocked.wait(timeout=5), "clear()/reset_stats() never acquired the lock"
