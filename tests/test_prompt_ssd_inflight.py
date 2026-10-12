"""Whole-prompt SSD records are readable while their write is still in flight.

openPangu-2.0 (exact typed N-1 snapshots, ~3 GB per ~4.5k-token prompt) stores every terminal through
DiskCacheManager. The scheduler used to block the terminal cleanup on flush_pending_writes, so the next request
waited ~0.5 s at admission for the SSD write and then read the same bytes back. The in-flight map serves the queued
record from memory until its file lands; these tests hold the writer to prove a fetch never needs the file.
"""
import threading

import pytest

mx = pytest.importorskip("mlx.core")


def _cache(value, n=32):
    from mlx_lm.models.cache import KVCache

    state = KVCache()
    state.update_and_fetch(mx.full((1, 1, n, 8), value, dtype=mx.bfloat16),
                           mx.full((1, 1, n, 8), value, dtype=mx.bfloat16))
    return [state]


@pytest.fixture
def held_writer(monkeypatch, tmp_path):
    """A DiskCacheManager whose writer blocks until released."""
    from vmlx_engine.disk_cache import DiskCacheManager

    gate = threading.Event()
    real = DiskCacheManager._write_cache

    def slow_write(self, *args, **kwargs):
        gate.wait(10)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(DiskCacheManager, "_write_cache", slow_write)
    manager = DiskCacheManager(cache_dir=str(tmp_path), max_size_gb=1)
    yield manager, gate
    gate.set()
    manager.shutdown()


def test_exact_fetch_is_served_from_memory_until_the_file_lands(held_writer, tmp_path):
    manager, gate = held_writer
    tokens = list(range(40))
    assert manager.store(tokens, _cache(3))
    assert manager.stats()["inflight_entries"] == 1
    assert not list(tmp_path.glob("cache_*tok.safetensors"))       # nothing on disk yet

    restored = manager.fetch(tokens)
    assert restored is not None
    assert restored[0].offset == 32 and restored[0].keys.dtype == mx.bfloat16
    assert mx.all(restored[0].keys == 3).item()

    gate.set()
    assert manager.flush_pending_writes(tokens)
    assert manager.stats()["inflight_entries"] == 0                # released once written
    from_file = manager.fetch(tokens)
    assert from_file is not None and mx.all(from_file[0].keys == 3).item()


def test_longest_prefix_finds_in_flight_record(held_writer):
    manager, gate = held_writer
    stored = list(range(40))
    assert manager.store(stored, _cache(5))
    # A tool continuation extends the stored prompt; the record is still being written.
    cache, matched = manager.fetch_longest_prefix(stored + [900, 901, 902])
    assert cache is not None and matched == stored
    assert mx.all(cache[0].keys == 5).item()
    # Only the final generation sentinel changed (N-1 payload match).
    cache, matched = manager.fetch_longest_prefix(stored[:-1] + [777, 1, 2])
    assert cache is not None and matched == stored[:-1] + [777]


def test_restored_cache_does_not_alias_the_writers_payload(held_writer):
    manager, gate = held_writer
    tokens = list(range(40))
    assert manager.store(tokens, _cache(7))
    restored = manager.fetch(tokens)
    restored[0].update_and_fetch(mx.zeros((1, 1, 4, 8), dtype=mx.bfloat16),
                                 mx.zeros((1, 1, 4, 8), dtype=mx.bfloat16))
    again = manager.fetch(tokens)                                   # still the stored record
    assert again[0].offset == 32 and mx.all(again[0].keys == 7).item()
    gate.set()
    assert manager.flush_pending_writes(tokens)
    assert mx.all(manager.fetch(tokens)[0].keys == 7).item()       # the file holds the original bytes


def test_same_key_is_not_queued_twice_while_in_flight(held_writer):
    manager, gate = held_writer
    tokens = list(range(40))
    assert manager.store(tokens, _cache(1))
    assert manager.store(tokens, _cache(1))
    assert manager._write_queue.qsize() + (1 if manager.stats()["inflight_entries"] else 0) <= 2
    assert manager.stats()["inflight_entries"] == 1


def test_failed_write_releases_the_entry(monkeypatch, tmp_path):
    from vmlx_engine.disk_cache import DiskCacheManager

    def broken(self, *args, **kwargs):
        raise OSError("disk gone")

    monkeypatch.setattr(DiskCacheManager, "_write_cache", broken)
    manager = DiskCacheManager(cache_dir=str(tmp_path), max_size_gb=1)
    try:
        tokens = list(range(40))
        assert manager.store(tokens, _cache(2))
        manager._write_queue.join()
        assert manager.stats()["inflight_entries"] == 0
        assert manager.fetch(tokens) is None                        # a failed write is a miss, not a phantom hit
    finally:
        manager.shutdown()


def test_terminal_flush_switch_defaults_off(monkeypatch):
    from vmlx_engine.scheduler import _prompt_ssd_terminal_flush

    monkeypatch.delenv("VMLX_PROMPT_SSD_TERMINAL_FLUSH", raising=False)
    assert _prompt_ssd_terminal_flush() is False
    monkeypatch.setenv("VMLX_PROMPT_SSD_TERMINAL_FLUSH", "1")
    assert _prompt_ssd_terminal_flush() is True


def test_store_hands_the_writer_zero_copy_views(held_writer):
    """The standard store path no longer copies the payload on the scheduler thread: the queued arrays are buffer
    views of the evaluated MLX arrays (bf16 as uint16 bits), and the round trip is still exact."""
    manager, gate = held_writer
    tokens = list(range(40))
    assert manager.store(tokens, _cache(6))
    entry = next(iter(manager._inflight.values()))
    arrays = [v for v in entry["arrays"].values() if hasattr(v, "dtype")]
    assert arrays and all(isinstance(v.base, memoryview) for v in arrays)
    gate.set()
    assert manager.flush_pending_writes(tokens)
    restored = manager.fetch(tokens)
    assert restored[0].keys.dtype == mx.bfloat16 and mx.all(restored[0].keys == 6).item()


def test_zero_copy_payload_survives_dropped_refs_and_allocator_churn(held_writer):
    """The queued views must keep their buffers alive and unchanged after the caller drops every reference and the
    allocator recycles same-sized buffers (the scheduler discards its snapshot right after store())."""
    import gc

    from mlx_lm.models.cache import KVCache

    manager, gate = held_writer
    tokens = list(range(40))
    expected = []
    cache = []
    for i in range(4):
        k = (mx.arange(1 * 2 * 256 * 64).reshape(1, 2, 256, 64) % 251 + i).astype(mx.bfloat16)
        c = KVCache()
        c.update_and_fetch(k, k + 1)
        cache.append(c)
        expected.append((mx.array(k), mx.array(k + 1)))
    mx.eval([e for pair in expected for e in pair])
    assert manager.store(tokens, cache)
    del cache, c, k
    gc.collect()
    # Churn: same-shape allocations and in-place-style updates that would reuse any freed buffer.
    for j in range(200):
        junk = mx.full((1, 2, 256, 64), float(j), dtype=mx.bfloat16)
        junk = junk * 2 + 1
        mx.eval(junk)
    restored = manager.fetch(tokens)                                   # served from the in-flight views
    for layer, (k, v) in zip(restored, expected):
        assert mx.array_equal(layer.keys[..., :256, :], k).item()
        assert mx.array_equal(layer.values[..., :256, :], v).item()
    gate.set()
    assert manager.flush_pending_writes(tokens)
    from_disk = manager.fetch(tokens)                                  # the file written from the same views
    for layer, (k, v) in zip(from_disk, expected):
        assert mx.array_equal(layer.keys[..., :256, :], k).item()
        assert mx.array_equal(layer.values[..., :256, :], v).item()
