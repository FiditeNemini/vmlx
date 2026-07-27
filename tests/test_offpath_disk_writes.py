"""Regression tests for immutable, bounded off-caller disk publication."""

from __future__ import annotations

import asyncio
import queue
import sqlite3
import threading
import time
from types import SimpleNamespace

import mlx.core as mx

from vmlx_engine.block_disk_store import BlockDiskStore
from vmlx_engine.utils.ssm_companion_disk_store import SSMCompanionDiskStore


def _block(value: float = 1.0, *, dtype=mx.float16) -> list[tuple]:
    keys = mx.full((1, 1, 8, 16), value, dtype=dtype)
    values = mx.full((1, 1, 8, 16), value + 1, dtype=dtype)
    mx.eval(keys, values)  # noqa: S307 - MLX tensor materialization
    return [("kv", keys, values)]


class _ArraysState:
    def __init__(self, values: mx.array):
        self.cache = [values]
        self.lengths = mx.array([values.shape[0]], dtype=mx.int32)
        self.left_padding = None


def _wait_for_fence(
    store: BlockDiskStore,
    fence_id: str,
    *,
    timeout: float = 5.0,
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


def test_block_disk_slow_file_io_is_off_caller_and_fence_is_durable(
    monkeypatch,
    tmp_path,
):
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    original_write = store._write_payload_file
    writer_threads: list[str] = []
    payload_types: list[type] = []

    def slow_write(path, payload):
        writer_threads.append(threading.current_thread().name)
        payload_types.append(type(payload))
        time.sleep(0.35)
        return original_write(path, payload)

    monkeypatch.setattr(store, "_write_payload_file", slow_write)
    fence_id = store.begin_write_fence("slow-block")
    try:
        started = time.perf_counter()
        assert store.write_block_async(
            b"b" * 32,
            _block(),
            8,
            request_id="slow-block",
            fence_id=fence_id,
        )
        caller_elapsed = time.perf_counter() - started
        assert caller_elapsed < 0.20
        assert store.seal_write_fence(fence_id)

        fence = _wait_for_fence(store, fence_id)
        stats = store.get_stats()["write_pipeline"]
        assert fence["completed"] == 1
        assert fence["failed"] == 0
        assert fence["retained"] == 1
        assert stats["pending_bytes"] == 0
        assert writer_threads == ["block-disk-writer"]
        assert payload_types == [bytes]
    finally:
        store.shutdown()


def test_block_disk_detaches_bfloat16_then_encodes_off_caller(
    monkeypatch,
    tmp_path,
):
    import numpy as np

    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    save_threads: list[str] = []
    save_inputs: list[dict] = []
    original_save = store._freeze_numpy_safetensors_bytes

    def observed_save(tensors):
        save_threads.append(threading.current_thread().name)
        save_inputs.append(tensors)
        assert all(isinstance(value, np.ndarray) for value in tensors.values())
        assert all(not value.flags.writeable for value in tensors.values())
        time.sleep(0.35)
        return original_save(tensors)

    monkeypatch.setattr(store, "_freeze_numpy_safetensors_bytes", observed_save)
    block_hash = b"f" * 32
    fence_id = store.begin_write_fence("bf16-freeze")
    try:
        source = _block(65536.0, dtype=mx.bfloat16)
        started = time.perf_counter()
        assert store.write_block_async(
            block_hash,
            source,
            8,
            request_id="bf16-freeze",
            fence_id=fence_id,
        )
        assert time.perf_counter() - started < 0.20
        assert store.seal_write_fence(fence_id)
        _wait_for_fence(store, fence_id)

        restored = store.read_block(block_hash)
        assert restored is not None
        assert restored[0][1].dtype == mx.bfloat16
        assert restored[0][1].tolist() == source[0][1].tolist()
        assert save_threads == ["block-disk-writer"]
        assert len(save_inputs) == 1
        pipeline = store.get_stats()["write_pipeline"]
        assert pipeline["offthread_serializations_queued"] == 1
        assert pipeline["offthread_serializations_completed"] == 1
        assert pipeline["offthread_serialization_failures"] == 0
    finally:
        store.shutdown()


def test_block_disk_background_serialization_failure_settles_fence(
    monkeypatch,
    tmp_path,
):
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)

    def fail_serialization(_tensors):
        raise RuntimeError("injected CPU encoding failure")

    monkeypatch.setattr(
        store,
        "_freeze_numpy_safetensors_bytes",
        fail_serialization,
    )
    fence_id = store.begin_write_fence("encoding-failure")
    try:
        assert store.write_block_async(
            b"e" * 32,
            _block(),
            8,
            request_id="encoding-failure",
            fence_id=fence_id,
        )
        assert store.seal_write_fence(fence_id)
        fence = _wait_for_fence(store, fence_id)
        pipeline = store.get_stats()["write_pipeline"]
        assert fence["completed"] == 0
        assert fence["failed"] == 1
        assert fence["retained"] == 0
        assert pipeline["pending_bytes"] == 0
        assert pipeline["offthread_serializations_queued"] == 1
        assert pipeline["offthread_serializations_completed"] == 0
        assert pipeline["offthread_serialization_failures"] == 1
    finally:
        store.shutdown()


def test_block_disk_byte_admission_precedes_serialization(monkeypatch, tmp_path):
    store = BlockDiskStore(
        str(tmp_path),
        max_size_gb=0,
        max_pending_write_bytes=1,
    )
    serialized = False

    def forbidden_serialize(*_args, **_kwargs):
        nonlocal serialized
        serialized = True
        raise AssertionError("serialization must not run after admission failure")

    monkeypatch.setattr(
        "vmlx_engine.block_disk_store._serialize_block",
        forbidden_serialize,
    )
    try:
        assert not store.write_block_async(b"x" * 32, _block(), 8)
        pipeline = store.get_stats()["write_pipeline"]
        assert serialized is False
        assert pipeline["pending_bytes"] == 0
        assert pipeline["byte_budget_drops"] == 1
    finally:
        store.shutdown()


def test_ssm_slow_file_io_is_off_caller_and_wait_is_durable(
    monkeypatch,
    tmp_path,
):
    store = SSMCompanionDiskStore(directory=tmp_path, budget_bytes=0)
    original_write = store._write_payload_file
    writer_threads: list[str] = []
    payload_types: list[type] = []

    def slow_write(path, payload):
        writer_threads.append(threading.current_thread().name)
        payload_types.append(type(payload))
        time.sleep(0.25)
        return original_write(path, payload)

    monkeypatch.setattr(store, "_write_payload_file", slow_write)
    key = "ab" * 32
    state = _ArraysState(mx.array([1.0, 2.0], dtype=mx.float16))
    try:
        started = time.perf_counter()
        assert store.store(key, [state], True, [1, 2], 2)
        caller_elapsed = time.perf_counter() - started
        assert caller_elapsed < 0.20

        assert store.wait_for_write(key, timeout=5.0)
        assert store.wait_for_pending(timeout=5.0)
        restored = store.fetch(key)
        assert restored is not None
        assert restored[0][0].cache[0].tolist() == [1.0, 2.0]
        pipeline = store.stats()["write_pipeline"]
        assert pipeline["pending_jobs"] == 0
        assert pipeline["pending_bytes"] == 0
        assert writer_threads == ["ssm-disk-writer", "ssm-disk-writer"]
        assert payload_types == [bytes, bytes]
    finally:
        assert store.shutdown(timeout=5.0)


def test_ssm_byte_admission_precedes_safetensors_and_reports_failure(
    monkeypatch,
    tmp_path,
):
    import vmlx_engine.utils.ssm_companion_disk_store as ssm_module

    store = SSMCompanionDiskStore(
        directory=tmp_path,
        budget_bytes=0,
        max_pending_write_bytes=1,
    )
    save_called = False

    def forbidden_save(*_args, **_kwargs):
        nonlocal save_called
        save_called = True
        raise AssertionError("save must not run after admission failure")

    monkeypatch.setattr(ssm_module.mx, "save_safetensors", forbidden_save)
    state = _ArraysState(mx.array([1.0, 2.0], dtype=mx.float16))
    try:
        assert not store.store("cd" * 32, [state], True, [1, 2], 2)
        pipeline = store.stats()["write_pipeline"]
        assert save_called is False
        assert pipeline["pending_bytes"] == 0
        assert pipeline["byte_budget_drops"] == 1
    finally:
        assert store.shutdown(timeout=5.0)


def test_ssm_background_publication_failure_is_observable(monkeypatch, tmp_path):
    store = SSMCompanionDiskStore(directory=tmp_path, budget_bytes=0)

    def fail_write(_path, _payload):
        raise OSError("injected disk failure")

    monkeypatch.setattr(store, "_write_payload_file", fail_write)
    key = "ef" * 32
    state = _ArraysState(mx.array([3.0], dtype=mx.float16))
    try:
        assert store.store(key, [state], True, [3], 1)
        assert not store.wait_for_write(key, timeout=5.0)
        pipeline = store.stats()["write_pipeline"]
        assert pipeline["failures"] == 1
        assert pipeline["pending_jobs"] == 0
        assert pipeline["pending_bytes"] == 0
    finally:
        assert store.shutdown(timeout=5.0)


def test_ssm_shutdown_drains_queued_publication(tmp_path):
    store = SSMCompanionDiskStore(directory=tmp_path, budget_bytes=0)
    key = "12" * 32
    state = _ArraysState(mx.array([4.0], dtype=mx.float16))

    assert store.store(key, [state], True, [4], 1)
    assert store.shutdown(timeout=5.0)
    data_path, side_path = store._entry_paths(key)
    assert data_path.is_file()
    assert side_path.is_file()
    assert not store.store(key, [state], True, [4], 1)


def test_block_disk_shutdown_rejects_producer_paused_during_detach(
    monkeypatch,
    tmp_path,
):
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    detach_entered = threading.Event()
    release_detach = threading.Event()
    producer_done = threading.Event()
    producer_result: list[bool] = []
    original_detach = store._detach_safetensors_tensors
    lease_path = store.global_budget._lease_path()

    def paused_detach(tensors):
        detach_entered.set()
        assert release_detach.wait(timeout=5.0)
        return original_detach(tensors)

    def produce() -> None:
        try:
            producer_result.append(
                store.write_block_async(b"z" * 32, _block(), 8)
            )
        finally:
            producer_done.set()

    monkeypatch.setattr(store, "_detach_safetensors_tensors", paused_detach)
    store._writer_shutdown_timeout_seconds = 0.05
    producer = threading.Thread(target=produce, name="paused-block-producer")
    producer.start()
    assert detach_entered.wait(timeout=5.0)

    # Shutdown closes admission before waiting for the already-admitted
    # producer.  It must not stop/release the writer lease and then let this
    # producer enqueue into an abandoned queue.
    store.shutdown()
    assert store._shutdown_started is True
    assert store._accepting_writes is False
    assert lease_path.exists()
    assert not store.write_block_async(b"y" * 32, _block(2.0), 8)

    release_detach.set()
    producer.join(timeout=5.0)
    assert producer_done.is_set()
    assert producer_result == [False]
    assert store._delayed_shutdown_thread is not None
    store._delayed_shutdown_thread.join(timeout=5.0)
    assert store._shutdown_finalized is True
    assert not store._writer_thread.is_alive()
    assert not lease_path.exists()
    assert store._pending_write_bytes == 0
    assert store._pending_write_items == 0

    conn = sqlite3.connect(str(store._db_path))
    try:
        assert conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0] == 0
    finally:
        conn.close()


def test_ssm_queue_full_rejection_does_not_skip_older_pending_job(
    monkeypatch,
    tmp_path,
):
    store = SSMCompanionDiskStore(directory=tmp_path, budget_bytes=0)
    publish_entered = threading.Event()
    release_publish = threading.Event()
    original_publish = store._publish_entry
    key_one = "34" * 32
    key_two = "56" * 32
    state = _ArraysState(mx.array([5.0], dtype=mx.float16))

    def paused_publish(*args, **kwargs):
        publish_entered.set()
        assert release_publish.wait(timeout=5.0)
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(store, "_publish_entry", paused_publish)
    try:
        assert store.store(key_one, [state], True, [5], 1)
        assert publish_entered.wait(timeout=5.0)

        def reject_put(_item):
            raise queue.Full

        monkeypatch.setattr(store._write_queue, "put_nowait", reject_put)
        assert not store.store(key_two, [state], True, [6], 1)

        # Job 2 rejected synchronously, but job 1 is still publishing.  A max()
        # watermark would falsely report both complete here.
        assert store._write_seq == 2
        assert store._last_completed_write == 0
        assert not store.wait_for_pending(timeout=0.05)

        release_publish.set()
        assert store.wait_for_pending(timeout=5.0)
        assert store._last_completed_write == 2
        assert store.wait_for_write(key_one, timeout=5.0)
        assert not store.wait_for_write(key_two, timeout=0.05)
        pipeline = store.stats()["write_pipeline"]
        assert pipeline["pending_jobs"] == 0
        assert pipeline["completion_generation"] == 2
    finally:
        release_publish.set()
        assert store.shutdown(timeout=5.0)


def test_block_disk_clear_waits_for_dequeued_writer_before_deleting(
    monkeypatch,
    tmp_path,
):
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    writer_entered = threading.Event()
    release_writer = threading.Event()
    clear_done = threading.Event()
    clear_errors: list[BaseException] = []
    original_write = store._write_payload_file

    def paused_write(path, payload):
        writer_entered.set()
        assert release_writer.wait(timeout=5.0)
        return original_write(path, payload)

    def run_clear() -> None:
        try:
            store.clear()
        except BaseException as exc:  # pragma: no cover - asserted below
            clear_errors.append(exc)
        finally:
            clear_done.set()

    monkeypatch.setattr(store, "_write_payload_file", paused_write)
    first_hash = b"c" * 32
    second_hash = b"d" * 32
    try:
        assert store.write_block_async(first_hash, _block(), 8)
        assert writer_entered.wait(timeout=5.0)

        clearer = threading.Thread(target=run_clear, name="block-disk-clear")
        clearer.start()
        deadline = time.monotonic() + 5.0
        while store._accepting_writes and time.monotonic() < deadline:
            time.sleep(0.005)
        assert store._accepting_writes is False
        assert not clear_done.wait(timeout=0.05)
        assert not store.write_block_async(second_hash, _block(3.0), 8)

        release_writer.set()
        clearer.join(timeout=5.0)
        assert clear_done.is_set()
        assert clear_errors == []
        assert store._accepting_writes is True
        assert store._pending_write_items == 0
        assert store._write_inflight == 0
        assert not store.has_block(first_hash)

        # Clear reopens a live store only after the old publication is gone.
        assert store.write_block_async(second_hash, _block(4.0), 8)
        assert store.wait_for_blocks([second_hash], timeout=5.0) == {second_hash}
        assert not store.has_block(first_hash)
    finally:
        release_writer.set()
        store.shutdown()


def test_llm_scheduler_drains_ssm_before_shared_block_budget_lease():
    from vmlx_engine.scheduler import Scheduler

    calls: list[str] = []
    ssm_disk = SimpleNamespace(shutdown=lambda **_kwargs: calls.append("ssm"))
    block_disk = SimpleNamespace(shutdown=lambda: calls.append("block"))
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.disk_cache = None
    scheduler._ssm_state_cache = SimpleNamespace(_disk=ssm_disk)
    scheduler.paged_cache_manager = SimpleNamespace(_disk_store=block_disk)

    scheduler.shutdown()

    assert calls == ["ssm", "block"]
    assert scheduler._ssm_state_cache._disk is None


def test_mllm_scheduler_drains_ssm_before_shared_block_budget_lease():
    from vmlx_engine.mllm_scheduler import MLLMScheduler

    calls: list[str] = []
    ssm_disk = SimpleNamespace(shutdown=lambda **_kwargs: calls.append("ssm"))
    block_disk = SimpleNamespace(shutdown=lambda: calls.append("block"))
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler._processing_task = None
    scheduler._running = True
    scheduler.batch_generator = None
    scheduler._ssm_companion_disk_store = ssm_disk
    scheduler.paged_cache_manager = SimpleNamespace(_disk_store=block_disk)
    scheduler._block_disk_l2_enabled = True

    asyncio.run(scheduler.stop())

    assert calls == ["ssm", "block"]
    assert scheduler._ssm_companion_disk_store is None
    assert scheduler.paged_cache_manager._disk_store is None
