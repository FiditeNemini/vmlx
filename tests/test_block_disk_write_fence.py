"""Request-correlated post-eviction telemetry for block-disk writes."""

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


def _wait_for_fence(
    store: BlockDiskStore,
    fence_id: str,
    *,
    timeout: float = 5.0,
) -> tuple[dict, dict]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        stats = store.get_stats()
        pipeline = stats["write_pipeline"]
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
            return stats, fence
        time.sleep(0.01)
    raise AssertionError(f"write fence {fence_id} did not settle")


def _write_request(
    store: BlockDiskStore,
    *,
    request_id: str,
    block_hash: bytes,
    value: float,
) -> tuple[dict, dict]:
    fence_id = store.begin_write_fence(request_id)
    assert store.write_block_async(
        block_hash,
        _cache_data(value),
        8,
        request_id=request_id,
        fence_id=fence_id,
    )
    assert store.seal_write_fence(fence_id)
    return _wait_for_fence(store, fence_id)


def test_write_fence_correlates_request_without_exposing_hashes(tmp_path):
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    try:
        stats, fence = _write_request(
            store,
            request_id="resp-request-a",
            block_hash=b"a" * 32,
            value=1,
        )
    finally:
        store.shutdown()

    assert fence == {
        "fence_id": fence["fence_id"],
        "request_id": "resp-request-a",
        "expected": 1,
        "queued": 1,
        "completed": 1,
        "failed": 0,
        "dropped": 0,
        "retained": 1,
        "sealed": True,
        "seal_enqueued": True,
        "seal_failed": False,
        "producer_aborted": False,
        "post_eviction_complete": True,
        "completion_generation": 1,
    }
    assert stats["write_pipeline"]["writer_alive"] is True
    assert not any("hash" in key for key in fence)


def test_write_fence_settles_after_full_capacity_replacement(tmp_path):
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    try:
        first_stats, first_fence = _write_request(
            store,
            request_id="resp-old",
            block_hash=b"a" * 32,
            value=1,
        )
        first_size = first_stats["disk_size_bytes"]
        assert first_size > 0
        store.max_size_bytes = int(first_size * 1.5)

        second_stats, second_fence = _write_request(
            store,
            request_id="resp-new",
            block_hash=b"b" * 32,
            value=2,
        )
    finally:
        store.shutdown()

    assert first_fence["retained"] == 1
    assert second_fence["expected"] == 1
    assert second_fence["completed"] == 1
    assert second_fence["failed"] == 0
    assert second_fence["dropped"] == 0
    assert second_fence["retained"] == 1
    assert second_fence["post_eviction_complete"] is True
    assert second_stats["disk_writes"] == 2
    assert second_stats["disk_evictions"] >= 1
    assert second_stats["blocks_on_disk"] == first_stats["blocks_on_disk"] == 1


def test_store_cache_exception_terminates_begun_write_fence(
    tmp_path,
    monkeypatch,
):
    import vmlx_engine.prefix_cache as prefix_cache_module
    from vmlx_engine.paged_cache import PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    manager = PagedCacheManager(
        block_size=4,
        max_blocks=16,
        disk_store=store,
    )
    cache = BlockAwarePrefixCache(model=None, paged_cache_manager=manager)
    cache_data = [
        {
            "state": (
                mx.ones((1, 1, 4, 8), dtype=mx.float16),
                mx.ones((1, 1, 4, 8), dtype=mx.float16),
            ),
            "meta_state": (4,),
            "class_name": "KVCache",
        }
    ]

    def fail_after_fence_begin():
        raise RuntimeError("forced post-begin store failure")

    monkeypatch.setattr(prefix_cache_module.mx, "synchronize", fail_after_fence_begin)
    try:
        with pytest.raises(RuntimeError, match="forced post-begin store failure"):
            cache.store_cache("resp-aborted", [1, 2, 3, 4], cache_data)

        fence_id = store.get_stats()["write_pipeline"]["recent_fences"][-1]["fence_id"]
        stats, fence = _wait_for_fence(store, fence_id)
    finally:
        store.shutdown()

    assert fence["request_id"] == "resp-aborted"
    assert fence["sealed"] is True
    assert fence["seal_enqueued"] is True
    assert fence["producer_aborted"] is True
    assert fence["post_eviction_complete"] is True
    assert fence["expected"] == 0
    assert fence["queued"] == 0
    assert all(item["sealed"] for item in stats["write_pipeline"]["recent_fences"])


def test_write_fence_eviction_failure_is_terminal_and_prunable(
    tmp_path,
    monkeypatch,
):
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)

    def fail_eviction(_conn):
        raise RuntimeError("forced eviction failure")

    monkeypatch.setattr(store, "_maybe_evict", fail_eviction)
    try:
        for index in range(70):
            fence_id = store.begin_write_fence(f"resp-evict-{index}")
            assert store.write_block_async(
                bytes([index % 256]) * 32,
                _cache_data(float(index + 1)),
                8,
                request_id=f"resp-evict-{index}",
                fence_id=fence_id,
            )
            assert store.seal_write_fence(fence_id)
            _stats, fence = _wait_for_fence(store, fence_id)
            assert fence["post_eviction_complete"] is True
            assert "forced eviction failure" in fence["post_eviction_error"]
    finally:
        store.shutdown()


def test_write_fence_finalization_error_does_not_kill_writer(
    tmp_path,
    monkeypatch,
):
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)

    def fail_finalize(_conn, _fence_id):
        raise RuntimeError("forced finalization failure")

    monkeypatch.setattr(store, "_finalize_write_fence", fail_finalize)
    try:
        fence_id = store.begin_write_fence("resp-finalize")
        assert store.write_block_async(
            b"f" * 32,
            _cache_data(9),
            8,
            request_id="resp-finalize",
            fence_id=fence_id,
        )
        assert store.seal_write_fence(fence_id)
        stats, fence = _wait_for_fence(store, fence_id)
    finally:
        store.shutdown()

    assert fence["post_eviction_complete"] is True
    assert "forced finalization failure" in fence["post_eviction_error"]
    assert stats["write_pipeline"]["writer_alive"] is True


def test_write_fence_waits_for_active_producer_before_finalizing(tmp_path):
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    try:
        fence_id = store.begin_write_fence("resp-race")
        assert store._write_fence_expected(fence_id)
        assert store.seal_write_fence(fence_id)

        fence = store.get_stats()["write_pipeline"]["recent_fences"][-1]
        assert fence["sealed"] is True
        assert fence["seal_enqueued"] is False
        assert fence["post_eviction_complete"] is False

        store._write_fence_queue_result(fence_id, block_hash=b"r" * 32)
        _stats, fence = _wait_for_fence(store, fence_id)
    finally:
        store.shutdown()

    assert fence["expected"] == 1
    assert fence["queued"] == 1
    assert fence["retained"] == 0
    assert fence["post_eviction_complete"] is True


def test_clear_terminalizes_unsettled_write_fences(tmp_path):
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    try:
        fence_id = store.begin_write_fence("resp-clear")
        assert store._write_fence_expected(fence_id)
        assert store.seal_write_fence(fence_id)
        store.clear()
        fence = next(
            item
            for item in store.get_stats()["write_pipeline"]["recent_fences"]
            if item["fence_id"] == fence_id
        )
    finally:
        store.shutdown()

    assert fence["sealed"] is True
    assert fence["post_eviction_complete"] is True
    assert "disk cache cleared" in fence["post_eviction_error"]
