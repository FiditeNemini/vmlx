"""Pending-write byte reservations settle per block, not at batch end.

Campaign #181 layer 8, MEASURED on the box 2026-08-15 (qwen3.8 bounded-L2
store-evict-refault): a 383-block store lost 255 blocks because the writer
released the ENTIRE batch's byte reservations only after the batch-end
aggregate accounting/eviction pass. Two ~1.1GB eviction batches kept the 1GB
budget pegged for seconds, every concurrently admitted block timed out at the
0.25s admission wait, and the first drop truncated the rest of the chain
(fence: expected=383 queued=128 completed=128 dropped=255 — 128 blocks x
~8.4MB is exactly the budget).

The budget bounds RAM held by pending payload copies. Once a block's payload
is persisted (or failed), the copy is dead — the reservation must be released
THEN, so admission waiters see budget as soon as the writer makes real
progress, not after the eviction pass.
"""

from __future__ import annotations

import time

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from vmlx_engine.block_disk_store import BlockDiskStore

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


def _cache_data(value: float) -> list[tuple]:
    keys = mx.full((1, 1, 8, 16), value, dtype=mx.float16)
    values = mx.full((1, 1, 8, 16), value + 1, dtype=mx.float16)
    mx.eval(keys, values)  # noqa: S307 - MLX tensor materialization
    return [("kv", keys, values)]


def _wait_for_fence_settled(
    store: BlockDiskStore,
    fence_id: str,
    *,
    timeout: float = 10.0,
) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pipeline = store.get_stats()["write_pipeline"]
        fence = next(
            (
                item
                for item in pipeline["recent_fences"]
                if item["fence_id"] == fence_id
            ),
            None,
        )
        if (
            fence is not None
            and fence["post_eviction_complete"]
            and pipeline["queue_depth"] == 0
            and pipeline["inflight"] == 0
        ):
            return fence
        time.sleep(0.01)
    raise AssertionError(f"write fence {fence_id} did not settle")


def test_reservations_settle_before_batch_accounting(tmp_path):
    """At the moment the writer runs aggregate accounting (which may run the
    multi-second eviction pass), every payload persisted so far must already
    have returned its reservation to the budget.

    Blocks enqueued for the NEXT batch legitimately hold reservations while
    this batch accounts, so the pin is event ordering on the writer thread:
    at each accounting event, releases-so-far >= writes-so-far. The pre-fix
    writer released the whole batch after accounting, making releases lag
    writes at every accounting event.
    """
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    events: list[str] = []
    original_account = store.global_budget.account_finalized_write_locked
    original_write = store._write_block
    original_release = store._release_pending_write_bytes

    def account_spy(*args, **kwargs):
        events.append("account")
        return original_account(*args, **kwargs)

    def write_spy(*args, **kwargs):
        result = original_write(*args, **kwargs)
        events.append("write")
        return result

    def release_spy(reserved):
        events.append("release")
        return original_release(reserved)

    store.global_budget.account_finalized_write_locked = account_spy
    store._write_block = write_spy
    store._release_pending_write_bytes = release_spy
    try:
        fence_id = store.begin_write_fence("resp-batch-release")
        parent = None
        for i in range(4):
            block_hash = bytes([i + 1]) * 32
            assert store.write_block_async(
                block_hash,
                _cache_data(float(i)),
                8,
                parent_hash=parent,
                request_id="resp-batch-release",
                fence_id=fence_id,
            )
            parent = block_hash
        assert store.seal_write_fence(fence_id)
        fence = _wait_for_fence_settled(store, fence_id)
    finally:
        store.global_budget.account_finalized_write_locked = original_account
        store._write_block = original_write
        store._release_pending_write_bytes = original_release
        store.shutdown()

    assert fence["completed"] == 4
    assert fence["dropped"] == 0
    accounts = [i for i, e in enumerate(events) if e == "account"]
    assert accounts, "batch accounting was never invoked"
    lagged = []
    for i in accounts:
        writes = sum(1 for e in events[:i] if e == "write")
        releases = sum(1 for e in events[:i] if e == "release")
        if releases < writes:
            lagged.append((writes, releases))
    assert not lagged, (
        f"accounting ran with unsettled reservations (writes, releases)="
        f"{lagged}; batch-end release starves concurrent admission during "
        "eviction"
    )
