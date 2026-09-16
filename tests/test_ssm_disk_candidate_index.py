"""Managed native SSD boundary discovery follows durable pool mutations.

Real typed GLM arrays and disk publications exercise the index seam; these
component cases do not claim model-serving or Electron qualification.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import pytest

from tests.test_glm5_companion_disk_codec import native_state, same_state
from vmlx_engine.global_disk_cache_budget import (
    GlobalDiskCacheBudget,
    ensure_managed_block_cache_namespace,
)
from vmlx_engine.utils.ssm_companion_cache import SSMCompanionCache
from vmlx_engine.utils.ssm_companion_disk_store import SSMCompanionDiskStore


TOKENS = list(range(16))
MODEL_KEY = "candidate-index-native-glm"


@pytest.fixture
def managed_stores(tmp_path):
    root = tmp_path / "pool"
    namespace = ensure_managed_block_cache_namespace(root / "aaaaaaaaaaaa")
    stores = []
    budgets = []

    def make(*, cap=10_000_000, model_key=MODEL_KEY):
        budget = GlobalDiskCacheBudget(
            root, cap, reconcile_interval_seconds=3600,
        )
        budgets.append(budget)
        disk = SSMCompanionDiskStore(
            directory=namespace / "ssm_companion",
            budget_bytes=cap,
            global_budget=budget,
        )
        stores.append(disk)
        cache = SSMCompanionCache(
            max_entries=0, max_bytes=0, model_key=model_key, disk_store=disk,
        )
        return budget, disk, cache

    yield make
    for disk in reversed(stores):
        assert disk.shutdown()
    for budget in reversed(budgets):
        budget.close()


def publish(disk, cache, length, states=None):
    states = native_state(length) if states is None else states
    key = cache._key(TOKENS, length)
    assert disk.store(key, states, True, TOKENS, length)
    assert disk.wait_for_write(key)
    assert disk.has_complete(key)
    assert all(path.is_file() for path in disk._entry_paths(key))
    return key, states


def assert_native_hit(cache, length, states, *, max_len):
    hit = cache.fetch_longest_prefix(TOKENS, max_len=max_len)
    assert hit is not None
    boundary, restored, complete = hit
    assert boundary == length
    assert complete is True
    same_state(states, restored)
    assert cache.size == 0  # Restoration must remain SSD-only.


def test_global_clear_retires_primed_native_candidates(managed_stores):
    budget, disk, cache = managed_stores()
    key3, _ = publish(disk, cache, 3)
    key5, _ = publish(disk, cache, 5)
    assert disk.candidate_lengths(6) == [5, 3]

    cleared = budget.clear_eligible()
    assert cleared.evicted_entries == 2
    assert cleared.evicted_bytes > 0
    assert all(not path.exists() for key in (key3, key5)
               for path in disk._entry_paths(key))
    assert disk.stats()["entries"] == 0

    misses_before = disk.stats()["misses"]
    assert cache.fetch_longest_prefix(TOKENS, max_len=6) is None
    lookup = cache.last_prefix_lookup
    assert lookup["candidate_lengths"] == []
    assert lookup["attempted_candidate_lengths"] == [6]
    assert disk.stats()["misses"] - misses_before == 1
    assert disk.candidate_lengths(6) == []


def test_capacity_eviction_keeps_longest_surviving_native_prefix(managed_stores):
    budget, disk, cache = managed_stores()
    key5, _ = publish(disk, cache, 5)
    key3, state3 = publish(disk, cache, 3)
    assert disk.candidate_lengths(6) == [5, 3]
    now = time.time()
    for key, age in ((key5, 200), (key3, 100)):
        for path in disk._entry_paths(key):
            os.utime(path, (now - age, now - age))

    total = budget.enforce(force=True).bytes_after
    survivor_bytes = sum(path.stat().st_size for path in disk._entry_paths(key3))
    # The shared pool trims to 90% of its cap. Keep exactly the newer pair.
    cap = (survivor_bytes * 10 + 8) // 9 + 128
    assert survivor_bytes < cap < total
    trim_budget, _, _ = managed_stores(cap=cap)
    result = trim_budget.enforce(force=True)
    assert result.compliant
    assert result.bytes_after <= cap
    assert result.capacity_evicted_entries_total >= 1
    assert all(not path.exists() for path in disk._entry_paths(key5))
    assert all(path.exists() for path in disk._entry_paths(key3))

    assert disk.candidate_lengths(6) == [3]
    assert_native_hit(cache, 3, state3, max_len=6)
    assert cache.last_prefix_lookup["attempted_candidate_lengths"] == [6, 3]


def test_live_reader_discovers_peer_durable_shorter_boundary(managed_stores):
    _, reader, reader_cache = managed_stores()
    assert reader.candidate_lengths(4) == []
    _, writer, writer_cache = managed_stores()
    key, expected = publish(writer, writer_cache, 3)

    # Exact-key reads are already supported; the missing behavior is discovery
    # at a longer requested boundary in an independently initialized reader.
    assert reader.has_complete(key)
    assert_native_hit(reader_cache, 3, expected, max_len=4)
    assert reader_cache.last_prefix_lookup["attempted_candidate_lengths"] == [4, 3]
    assert reader.candidate_lengths(4) == [3]
    isolated = SSMCompanionCache(
        max_entries=0, max_bytes=0, model_key="other-native-model", disk_store=reader,
    )
    assert isolated.fetch_longest_prefix(TOKENS, max_len=4) is None


def test_reader_discovers_checkpoint_from_separate_process(managed_stores):
    budget, disk, cache = managed_stores()
    assert disk.candidate_lengths(4) == []
    source_root = str(Path(__file__).resolve().parents[1])
    script = f"""
import sys
sys.path.insert(0, {source_root!r})
import mlx.core as mx
mx.set_default_device(mx.cpu)
from pathlib import Path
from tests.test_glm5_companion_disk_codec import native_state
from vmlx_engine.global_disk_cache_budget import GlobalDiskCacheBudget
from vmlx_engine.utils.ssm_companion_disk_store import SSMCompanionDiskStore
budget = GlobalDiskCacheBudget(Path({str(budget.root)!r}), 10_000_000)
disk = SSMCompanionDiskStore(directory=Path({str(disk.directory)!r}), global_budget=budget)
try:
    key = {cache._key(TOKENS, 3)!r}
    assert disk.store(key, native_state(3), True, {TOKENS!r}, 3)
    assert disk.wait_for_write(key)
    assert disk.has_complete(key)
finally:
    assert disk.shutdown()
    assert budget.close()
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert_native_hit(cache, 3, native_state(3), max_len=4)
    assert cache.last_prefix_lookup["attempted_candidate_lengths"] == [4, 3]


def test_unchanged_queries_and_read_touch_reuse_candidate_scan(managed_stores):
    _, disk, cache = managed_stores()
    key, expected = publish(disk, cache, 3)
    assert disk.candidate_lengths(4) == [3]
    scans = disk.stats()["candidate_length_scans"]
    for _ in range(3):
        assert disk.candidate_lengths(4) == [3]
        disk.touch(key)
        assert_native_hit(cache, 3, expected, max_len=4)
    assert disk.stats()["candidate_length_scans"] == scans


def test_same_size_peer_replacement_refreshes_candidate_snapshot(
    managed_stores, monkeypatch,
):
    # A fixed timestamp makes both serialized sidecars exactly the same size;
    # the fresh record IDs and changed tensor values retain their true bytes.
    now = time.time()
    monkeypatch.setattr("vmlx_engine.utils.ssm_companion_disk_store.time.time", lambda: now)
    _, reader, reader_cache = managed_stores()
    _, writer, writer_cache = managed_stores()
    key, _ = publish(writer, writer_cache, 3)
    assert reader.candidate_lengths(4) == [3]
    scans = reader.stats()["candidate_length_scans"]
    sizes_before = tuple(path.stat().st_size for path in writer._entry_paths(key))

    replacement = native_state(3)
    replacement[0].cache[0] = mx.full(
        replacement[0].cache[0].shape, 9.25, dtype=mx.bfloat16,
    )
    replaced_key, _ = publish(writer, writer_cache, 3, replacement)
    assert replaced_key == key
    assert tuple(path.stat().st_size for path in writer._entry_paths(key)) == sizes_before
    assert reader.candidate_lengths(4) == [3]
    assert reader.stats()["candidate_length_scans"] == scans + 1
    assert_native_hit(reader_cache, 3, replacement, max_len=4)


def test_failed_peer_publication_adds_no_native_candidate(managed_stores, monkeypatch):
    _, reader, reader_cache = managed_stores()
    assert reader.candidate_lengths(4) == []
    writer_budget, writer, writer_cache = managed_stores()

    def fail_accounting(*_args, **_kwargs):
        raise OSError("candidate-index controlled publication failure")

    monkeypatch.setattr(writer_budget, "account_finalized_write_locked", fail_accounting)
    key = writer_cache._key(TOKENS, 3)
    assert writer.store(key, native_state(3), True, TOKENS, 3)
    assert writer.wait_for_write(key) is False
    assert writer.has_complete(key) is False
    assert all(not path.exists() for path in writer._entry_paths(key))
    assert reader.candidate_lengths(4) == []
    assert reader_cache.fetch_longest_prefix(TOKENS, max_len=4) is None
    assert reader_cache.last_prefix_lookup["attempted_candidate_lengths"] == [4]


@pytest.mark.parametrize("failure_site", ["sidecar_read", "directory_scan"])
def test_transient_discovery_io_failure_retries_without_ledger_mutation(
    managed_stores, monkeypatch, failure_site,
):
    budget, disk, cache = managed_stores()
    key, expected = publish(disk, cache, 3)
    _, sidecar = disk._entry_paths(key)
    ledger_before = budget._accounting_path().read_bytes()
    failures = 0

    if failure_site == "sidecar_read":
        original_read_text = Path.read_text

        def read_text_once(path, *args, **kwargs):
            nonlocal failures
            if path == sidecar and failures == 0:
                failures += 1
                raise OSError("controlled transient native sidecar read failure")
            return original_read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", read_text_once)
    else:
        original_iterdir = Path.iterdir

        def iterdir_once(path):
            nonlocal failures
            if path == disk.directory and failures == 0:
                failures += 1
                raise OSError("controlled transient native directory scan failure")
            return original_iterdir(path)

        monkeypatch.setattr(Path, "iterdir", iterdir_once)

    # A transient discovery failure may conservatively miss this request.
    # It must not certify that negative result for the unchanged pool revision.
    first = cache.fetch_longest_prefix(TOKENS, max_len=4)
    assert failures == 1
    if first is not None:
        boundary, restored, complete = first
        assert boundary == 3 and complete is True
        same_state(expected, restored)
    assert budget._accounting_path().read_bytes() == ledger_before

    assert_native_hit(cache, 3, expected, max_len=4)
    assert cache.last_prefix_lookup["attempted_candidate_lengths"] == [4, 3]
    assert failures == 1
    assert budget._accounting_path().read_bytes() == ledger_before
