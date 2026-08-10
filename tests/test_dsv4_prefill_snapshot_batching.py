# SPDX-License-Identifier: Apache-2.0
"""Prefill-side snapshot captures must batch their evals.

Without a collector, ``_copy_delta_tree`` pins every detached leaf with its own
blocking ``mx.eval``. A forced terminal anchor spans ~43 composite layers x ~8
leaves, so the admission prefill serialized a few hundred GPU round-trips onto
the worker thread — the fixed cost inside the DSV4 answer-pass flip and in every
cold/warm TTFT. The decode boundary was already converted to a single batched
eval; these tests pin the same contract for the prefill capture path.
"""

from typing import Any, List

import pytest

from vmlx_engine.utils.dsv4_batch_generator import DSV4BatchGenerator


class _RecordingCache:
    """Minimal stand-in that records whether a collector was threaded in."""

    def __init__(self) -> None:
        self._vmlx_dsv4_delta_next_token = 0
        self._vmlx_dsv4_block_records = []
        self.collector_seen: List[Any] = []

    def export_block_delta(self, *args, **kwargs):
        self.collector_seen.append(kwargs.get("_eval_collector", "MISSING"))
        return {"delta": "x"}


def test_capture_helpers_accept_and_thread_an_eval_collector():
    """All three prefill capture entry points must forward the collector."""
    for name in (
        "_capture_dsv4_completed_blocks",
        "_capture_dsv4_append_safe_checkpoint",
        "_capture_dsv4_terminal_anchor",
    ):
        fn = getattr(DSV4BatchGenerator, name)
        params = fn.__func__.__code__.co_varnames[: fn.__func__.__code__.co_argcount
                                                  + fn.__func__.__code__.co_kwonlyargcount]
        assert "_eval_collector" in params, f"{name} does not accept _eval_collector"


def test_settle_snapshot_evals_batches_and_clears(monkeypatch):
    """One batched submit per capture group, and the collector is drained."""
    import vmlx_engine.utils.dsv4_batch_generator as mod

    calls = {"async": 0, "sync": 0, "n": 0}

    def fake_async_eval(*arrays):
        calls["async"] += 1
        calls["n"] = len(arrays)

    monkeypatch.setattr(mod.mx, "async_eval", fake_async_eval, raising=False)
    collector = ["a", "b", "c"]
    DSV4BatchGenerator._settle_snapshot_evals(collector)
    assert calls["async"] == 1, "leaves were not submitted in a single batch"
    assert calls["n"] == 3
    assert collector == [], "collector must be drained after settling"


def test_settle_snapshot_evals_is_a_noop_when_empty(monkeypatch):
    import vmlx_engine.utils.dsv4_batch_generator as mod

    called = {"n": 0}
    monkeypatch.setattr(
        mod.mx, "async_eval", lambda *a: called.__setitem__("n", called["n"] + 1),
        raising=False,
    )
    DSV4BatchGenerator._settle_snapshot_evals([])
    assert called["n"] == 0


def test_settle_snapshot_evals_falls_back_to_sync_eval(monkeypatch):
    """Older MLX without async_eval must still pin the snapshot."""
    import vmlx_engine.utils.dsv4_batch_generator as mod

    seen = {}
    monkeypatch.setattr(mod.mx, "async_eval", None, raising=False)
    monkeypatch.setattr(
        mod.mx, "eval", lambda *a: seen.__setitem__("n", len(a)), raising=False
    )
    DSV4BatchGenerator._settle_snapshot_evals(["x", "y"])
    assert seen.get("n") == 2


def test_terminal_anchor_threads_collector_into_block_record(monkeypatch):
    """The forced anchor must not fall back to per-leaf eval."""
    seen = {}

    def fake_record(cls, cache_list, start, end, **kwargs):
        seen["collector"] = kwargs.get("_eval_collector", "MISSING")

    monkeypatch.setattr(
        DSV4BatchGenerator, "_capture_dsv4_block_record",
        classmethod(fake_record),
    )
    cache = [_RecordingCache()]
    collector: List[Any] = []
    DSV4BatchGenerator._capture_dsv4_terminal_anchor(
        cache, 128, _eval_collector=collector
    )
    assert seen["collector"] is collector
