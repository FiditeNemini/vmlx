from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

import vmlx_engine.global_disk_cache_budget as budget_module
from vmlx_engine.global_disk_cache_budget import (
    GlobalDiskCacheBudget,
    ensure_managed_block_cache_namespace,
)
from vmlx_engine.utils.ssm_companion_disk_store import SSMCompanionDiskStore


def _indexed_block(
    namespace: Path,
    name: str,
    *,
    size: int,
    accessed: float,
) -> Path:
    ensure_managed_block_cache_namespace(namespace)
    blocks = namespace / "blocks" / name[:2]
    blocks.mkdir(parents=True, exist_ok=True)
    payload = blocks / f"{name}.safetensors"
    payload.write_bytes(b"x" * size)
    os.utime(payload, (accessed, accessed))
    database = namespace / "block_index.db"
    conn = sqlite3.connect(database)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS blocks ("
            "block_hash TEXT PRIMARY KEY, file_name TEXT NOT NULL, "
            "last_accessed REAL NOT NULL)"
        )
        conn.execute(
            "INSERT INTO blocks(block_hash, file_name, last_accessed) "
            "VALUES (?, ?, ?)",
            (name, str(payload.relative_to(namespace)), accessed),
        )
        conn.commit()
    finally:
        conn.close()
    return payload


def _physical_total(root: Path) -> int:
    budget = GlobalDiskCacheBudget(root, 1024**3, orphan_grace_seconds=0)
    try:
        return budget.enforce(force=True).bytes_after
    finally:
        budget._remove_lease()


def test_global_lru_crosses_model_namespaces(tmp_path: Path) -> None:
    root = tmp_path / "root"
    now = time.time()
    old = _indexed_block(root / "aaaaaaaaaaaa", "aa-old", size=64_000, accessed=now - 100)
    recent = _indexed_block(root / "bbbbbbbbbbbb", "bb-new", size=64_000, accessed=now)
    before = _physical_total(root)

    budget = GlobalDiskCacheBudget(root, before - 32_000, orphan_grace_seconds=0)
    try:
        result = budget.enforce(force=True)
        assert result.compliant is True
        assert result.evicted_entries >= 1
        assert not old.exists()
        assert recent.exists()
    finally:
        budget._remove_lease()


def test_ssm_pair_participates_in_same_global_lru(tmp_path: Path) -> None:
    root = tmp_path / "root"
    namespace = root / "aaaaaaaaaaaa"
    now = time.time()
    recent = _indexed_block(namespace, "aa-block", size=64_000, accessed=now)
    companion = namespace / "ssm_companion" / "cc"
    companion.mkdir(parents=True)
    data = companion / "cc-old.safetensors"
    side = companion / "cc-old.json"
    data.write_bytes(b"s" * 30_000)
    side.write_bytes(b"j" * 10_000)
    os.utime(data, (now - 200, now - 200))
    os.utime(side, (now - 200, now - 200))
    before = _physical_total(root)

    budget = GlobalDiskCacheBudget(root, before - 20_000, orphan_grace_seconds=0)
    try:
        result = budget.enforce(force=True)
        assert result.compliant is True
        assert not data.exists()
        assert not side.exists()
        assert recent.exists()
    finally:
        budget._remove_lease()


def test_recent_orphan_and_live_temp_are_counted_but_protected(tmp_path: Path) -> None:
    root = tmp_path / "custom-root"
    sentinel = root / "do-not-touch.txt"
    root.mkdir()
    sentinel.write_text("unrelated")
    namespace = ensure_managed_block_cache_namespace(root / "aaaaaaaaaaaa")
    blocks = namespace / "blocks"
    blocks.mkdir()
    recent = blocks / "recent.safetensors"
    recent.write_bytes(b"r" * 8_000)

    budget = GlobalDiskCacheBudget(root, 1, orphan_grace_seconds=3600)
    active = blocks / f"active.{budget.lease_id}.1.tmp.safetensors"
    active.write_bytes(b"a" * 8_000)
    stale = blocks / (
        "stale.99999999-0123456789abcdef0123456789abcdef.1.tmp.safetensors"
    )
    stale.write_bytes(b"s" * 8_000)
    legacy_stale = blocks / "legacy.0.tmp.safetensors"
    legacy_stale.write_bytes(b"l" * 8_000)
    old = time.time() - 7200
    os.utime(legacy_stale, (old, old))
    try:
        result = budget.enforce(force=True)
        assert result.compliant is False
        assert result.protected_recent_orphans == 1
        assert recent.exists()
        assert active.exists()
        assert not stale.exists()
        assert not legacy_stale.exists()
        assert sentinel.read_text() == "unrelated"
    finally:
        budget._remove_lease()


def test_old_finalized_orphan_is_evictable(tmp_path: Path) -> None:
    root = tmp_path / "root"
    namespace = ensure_managed_block_cache_namespace(root / "aaaaaaaaaaaa")
    blocks = namespace / "blocks"
    blocks.mkdir()
    orphan = blocks / "old.safetensors"
    orphan.write_bytes(b"o" * 32_000)
    old = time.time() - 3600
    os.utime(orphan, (old, old))
    budget = GlobalDiskCacheBudget(root, 1, orphan_grace_seconds=1)
    try:
        result = budget.enforce(force=True)
        assert result.compliant is True
        assert not orphan.exists()
    finally:
        budget._remove_lease()


def test_zero_is_unlimited_but_forced_reconciliation_reports_physical_truth(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    first = _indexed_block(
        root / "aaaaaaaaaaaa",
        "aa-block",
        size=64_000,
        accessed=time.time(),
    )
    second = _indexed_block(
        root / "bbbbbbbbbbbb",
        "bb-block",
        size=32_000,
        accessed=time.time(),
    )
    budget = GlobalDiskCacheBudget(root, 0)
    try:
        result = budget.enforce(force=True)
        assert result.max_size_bytes == 0
        assert result.compliant is True
        assert result.scan_performed is True
        assert result.accounted is True
        assert result.bytes_after >= first.stat().st_size + second.stat().st_size
        assert first.exists()
        assert second.exists()
        accounted = budget.account_finalized_write(123)
        assert accounted.scan_performed is True
        assert accounted.compliant is True
    finally:
        budget._remove_lease()


def test_strict_unlimited_accounting_advances_physical_reconciliation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    _indexed_block(
        root / "aaaaaaaaaaaa",
        "aa-block",
        size=16_000,
        accessed=time.time(),
    )
    budget = GlobalDiskCacheBudget(root, 0, reconcile_interval_seconds=3600)
    try:
        startup = budget.enforce(force=True)
        with budget.exclusive_mutation_guard() as locked:
            assert locked is True
            strict = budget.account_finalized_write_locked(
                0,
                require_reconciled=True,
            )
        assert strict.scan_performed is True
        assert strict.accounted is True
        assert (
            strict.reconciliation_generation
            > startup.reconciliation_generation
        )
        assert strict.bytes_after == startup.bytes_after
    finally:
        budget.close()


def test_minimum_live_finite_lease_wins_and_unlimited_cannot_relax_it(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    ensure_managed_block_cache_namespace(root / "aaaaaaaaaaaa")
    high = GlobalDiskCacheBudget(root, 1000)
    low = GlobalDiskCacheBudget(root, 500)
    unlimited = GlobalDiskCacheBudget(root, 0)
    try:
        assert len({high.lease_id, low.lease_id, unlimited.lease_id}) == 3
        assert unlimited.enforce(force=True).max_size_bytes == 500
        low._remove_lease()
        assert unlimited.enforce(force=True).max_size_bytes == 1000
    finally:
        high._remove_lease()
        low._remove_lease()
        unlimited._remove_lease()


def test_minimum_cap_is_observed_across_processes(tmp_path: Path) -> None:
    root = tmp_path / "root"
    ensure_managed_block_cache_namespace(root / "aaaaaaaaaaaa")
    code = (
        "import sys; "
        "from vmlx_engine.global_disk_cache_budget import GlobalDiskCacheBudget; "
        "b=GlobalDiskCacheBudget(sys.argv[1], 777); "
        "print(b.lease_id, flush=True); input()"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(root)],
        cwd=Path(__file__).resolve().parents[1],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip()
        local = GlobalDiskCacheBudget(root, 0)
        try:
            assert local.enforce(force=True).max_size_bytes == 777
        finally:
            local._remove_lease()
    finally:
        if child.stdin is not None:
            child.stdin.write("\n")
            child.stdin.flush()
        child.wait(timeout=10)
        assert child.returncode == 0, child.stderr.read() if child.stderr else ""


def test_pid_reuse_birth_mismatch_removes_stale_finite_lease(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "root"
    namespace = ensure_managed_block_cache_namespace(root / "aaaaaaaaaaaa")
    stale_id = f"{os.getpid()}-{'a' * 32}"
    stale_path = root / ".vmlx-global-cache-budget-leases" / f"{stale_id}.json"
    temp_path = namespace / "blocks" / (
        f"orphan.{stale_id}.0.{'b' * 32}.tmp.safetensors"
    )
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(b"x" * 64_000)
    budget = GlobalDiskCacheBudget(root, 1, orphan_grace_seconds=0)
    stale_path.write_text(
        json.dumps(
            {
                "version": 1,
                "max_size_bytes": 1,
                "updated_at_ns": time.time_ns(),
                "pid": os.getpid(),
                "process_birth_identity": "reused-old-process",
            }
        )
    )
    monkeypatch.setattr(
        budget_module,
        "_process_birth_identity",
        lambda _pid: "current-process",
    )
    try:
        budget.enforce(force=True)
        assert not stale_path.exists()
        assert not temp_path.exists()
    finally:
        budget.close()


def test_repeated_accounting_caches_cross_process_birth_probe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "root"
    budget = GlobalDiskCacheBudget(
        root,
        1_000_000,
        reconcile_interval_seconds=3600,
    )
    budget.enforce(force=True)
    fake_path = root / ".vmlx-global-cache-budget-leases" / "fake.json"
    fake_path.write_text(
        json.dumps(
            {
                "version": 1,
                "max_size_bytes": 1_000_000,
                "updated_at_ns": time.time_ns(),
                "pid": 424242,
                "process_birth_identity": "fake-birth",
            }
        )
    )
    calls = 0

    def counted_probe(_pid):
        nonlocal calls
        calls += 1
        return "fake-birth"

    monkeypatch.setattr(budget, "_pid_is_alive", lambda _pid: True)
    monkeypatch.setattr(budget_module, "_process_birth_identity", counted_probe)
    try:
        for _ in range(3):
            with budget.exclusive_mutation_guard() as locked:
                assert locked
                result = budget.account_finalized_write_locked(0)
                assert result.accounted and result.compliant
        assert calls == 1
    finally:
        fake_path.unlink(missing_ok=True)
        budget.close()


def test_refresh_health_observes_other_owner_accounting_generation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    first = GlobalDiskCacheBudget(root, 1_000_000)
    second = GlobalDiskCacheBudget(root, 1_000_000)
    try:
        initial = first.enforce(force=True)
        with second.exclusive_mutation_guard() as locked:
            assert locked
            advanced = second.account_finalized_write_locked(0)
        assert advanced.accounting_generation > initial.accounting_generation
        refreshed = first.refresh_health()
        # Same physical reconciliation generation retains the local proof that
        # a scan occurred while refreshing the other owner's ledger advance.
        assert refreshed.scan_performed is True
        assert refreshed.accounted is True
        assert refreshed.accounting_generation == advanced.accounting_generation
    finally:
        first.close()
        second.close()


def test_accounting_avoids_root_scan_until_crossing_or_strict_fence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "root"
    ensure_managed_block_cache_namespace(root / "aaaaaaaaaaaa")
    budget = GlobalDiskCacheBudget(
        root,
        10_000_000,
        reconcile_interval_seconds=3600,
    )
    try:
        budget.enforce(force=True)
        calls = 0
        original = budget._scan_locked

        def counted_scan(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(budget, "_scan_locked", counted_scan)
        with budget.exclusive_mutation_guard() as locked:
            assert locked is True
            for _ in range(5):
                result = budget.account_finalized_write_locked(100)
                assert result.scan_performed is False
        assert calls == 0

        with budget.exclusive_mutation_guard() as locked:
            assert locked is True
            strict = budget.account_finalized_write_locked(
                0,
                require_reconciled=True,
            )
        assert strict.scan_performed is True
        assert calls >= 2
    finally:
        budget._remove_lease()


def test_negative_delta_forces_reconcile_instead_of_double_subtracting(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    namespace = ensure_managed_block_cache_namespace(root / "aaaaaaaaaaaa")
    blocks = namespace / "blocks"
    blocks.mkdir()
    first = blocks / "first.safetensors"
    second = blocks / "second.safetensors"
    first.write_bytes(b"a" * 10_000)
    second.write_bytes(b"b" * 20_000)
    one = GlobalDiskCacheBudget(root, 10_000_000, orphan_grace_seconds=0)
    two = GlobalDiskCacheBudget(root, 10_000_000, orphan_grace_seconds=0)
    try:
        assert one.enforce(force=True).bytes_after == 30_000
        first.unlink()
        assert two.enforce(force=True).bytes_after == 20_000
        with one.exclusive_mutation_guard() as locked:
            assert locked is True
            result = one.account_finalized_write_locked(-10_000)
        assert result.scan_performed is True
        assert result.bytes_after == 20_000
    finally:
        one._remove_lease()
        two._remove_lease()


def test_custom_root_legacy_cache_is_bounded_without_touching_unrelated_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "custom"
    raw_blocks = root / "blocks"
    raw_blocks.mkdir(parents=True)
    raw_payload = raw_blocks / "legacy.safetensors"
    raw_payload.write_bytes(b"legacy" * 20_000)
    old = time.time() - 1000
    os.utime(raw_payload, (old, old))
    database = root / "block_index.db"
    conn = sqlite3.connect(database)
    try:
        conn.execute(
            "CREATE TABLE blocks (block_hash TEXT PRIMARY KEY, "
            "file_name TEXT NOT NULL, last_accessed REAL NOT NULL)"
        )
        conn.execute(
            "INSERT INTO blocks VALUES (?, ?, ?)",
            ("legacy", "blocks/legacy.safetensors", old),
        )
        conn.commit()
    finally:
        conn.close()
    sentinel = root / "unrelated.txt"
    sentinel.write_text("preserve me")

    namespace = root / "aaaaaaaaaaaa"
    managed = _indexed_block(
        namespace,
        "aa-managed",
        size=64_000,
        accessed=time.time() - 100,
    )
    probe = GlobalDiskCacheBudget(
        root,
        10_000_000,
        orphan_grace_seconds=0,
        allow_legacy_hashed_namespaces=False,
        allow_legacy_direct_namespace=True,
    )
    try:
        before = probe.enforce(force=True).bytes_after
    finally:
        probe._remove_lease()

    budget = GlobalDiskCacheBudget(
        root,
        before - 50_000,
        orphan_grace_seconds=0,
        allow_legacy_hashed_namespaces=False,
        allow_legacy_direct_namespace=True,
    )
    try:
        result = budget.enforce(force=True)
        assert result.compliant is True
        assert not raw_payload.exists()
        assert managed.exists()
        assert sentinel.read_text() == "preserve me"
    finally:
        budget._remove_lease()


def test_ssm_store_replace_and_delete_update_shared_ledger(tmp_path: Path) -> None:
    root = tmp_path / "root"
    namespace = ensure_managed_block_cache_namespace(root / "aaaaaaaaaaaa")
    budget = GlobalDiskCacheBudget(
        root,
        10_000_000,
        reconcile_interval_seconds=3600,
    )
    store = SSMCompanionDiskStore(
        directory=namespace / "ssm_companion",
        budget_bytes=0,
        global_budget=budget,
    )
    try:
        baseline = budget.enforce(force=True).bytes_after
        assert store.store("ab" * 32, [{"state": 1}], True, [1, 2], 2)
        assert store.wait_for_pending()
        first = budget.last_result
        assert first is not None and first.bytes_after > baseline
        assert store.store("ab" * 32, [{"state": "larger"}], True, [1, 2], 2)
        assert store.wait_for_pending()
        second = budget.last_result
        assert second is not None and second.bytes_after >= first.bytes_after
        store.delete("ab" * 32)
        deleted = budget.last_result
        assert deleted is not None and deleted.bytes_after < second.bytes_after
    finally:
        store.shutdown()
        budget._remove_lease()


def test_successful_ssm_fetch_refreshes_cross_namespace_global_lru(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    namespace = ensure_managed_block_cache_namespace(root / "aaaaaaaaaaaa")
    budget = GlobalDiskCacheBudget(root, 10_000_000, orphan_grace_seconds=0)
    store = SSMCompanionDiskStore(
        directory=namespace / "ssm_companion",
        budget_bytes=0,
        global_budget=budget,
    )
    key = "ab" * 32
    trim_budget = None
    try:
        assert store.store(key, [{"state": 1}], True, [1, 2], 2)
        assert store.wait_for_pending()
        data_path, side_path = store._entry_paths(key)
        now = time.time()
        os.utime(data_path, (now - 200, now - 200))
        os.utime(side_path, (now - 200, now - 200))
        older_block = _indexed_block(
            root / "bbbbbbbbbbbb",
            "bb-older-block",
            size=256_000,
            accessed=now - 100,
        )

        # The fully materialized fetch touches both halves of the atomic pair.
        # Global eviction must therefore choose the untouched block that was
        # newer before this fetch, not the now-hot SSM record.
        assert store.fetch(key) is not None
        total = budget.enforce(force=True).bytes_after
        old_block_size = older_block.stat().st_size
        remaining_after_oldest = total - old_block_size
        cap = ((remaining_after_oldest * 10 + 8) // 9) + 1024
        assert cap < total
        trim_budget = GlobalDiskCacheBudget(root, cap, orphan_grace_seconds=0)
        result = trim_budget.enforce(force=True)

        assert result.compliant is True
        assert result.bytes_after <= result.max_size_bytes
        assert not older_block.exists()
        assert data_path.exists()
        assert side_path.exists()
    finally:
        store.shutdown()
        if trim_budget is not None:
            trim_budget.close()
        budget.close()


def test_ssm_torn_data_sidecar_generation_is_a_cache_miss(tmp_path: Path) -> None:
    store = SSMCompanionDiskStore(
        directory=tmp_path / "ssm_companion",
        budget_bytes=0,
    )
    key = "cd" * 32
    assert store.store(key, [{"state": 1}], True, [1, 2], 2)
    assert store.wait_for_pending()
    data_path, side_path = store._entry_paths(key)
    sidecar = json.loads(side_path.read_text())
    sidecar["record_id"] = "different-final-rename-generation"
    side_path.write_text(json.dumps(sidecar))

    assert data_path.exists()
    assert store.fetch(key) is None
    assert store.shutdown()


def test_ssm_aggregate_accounting_failure_stops_later_publications(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import vmlx_engine.utils.ssm_companion_disk_store as ssm_module

    root = tmp_path / "root"
    namespace = ensure_managed_block_cache_namespace(root / "aaaaaaaaaaaa")
    budget = GlobalDiskCacheBudget(root, 10_000_000)
    store = SSMCompanionDiskStore(
        directory=namespace / "ssm_companion",
        budget_bytes=0,
        global_budget=budget,
    )
    save_calls = 0
    original_save = ssm_module.mx.save_safetensors
    original_account = budget.account_finalized_write_locked

    def counted_save(*args, **kwargs):
        nonlocal save_calls
        save_calls += 1
        return original_save(*args, **kwargs)

    def fail_accounting(*_args, **_kwargs):
        raise OSError("forced aggregate accounting failure")

    monkeypatch.setattr(ssm_module.mx, "save_safetensors", counted_save)
    monkeypatch.setattr(
        budget,
        "account_finalized_write_locked",
        fail_accounting,
    )
    try:
        assert store.store("ef" * 32, [{"state": 1}], True, [1, 2], 2)
        assert not store.wait_for_write("ef" * 32)
        first_save_calls = save_calls
        assert first_save_calls == 1
        assert not store.store("fe" * 32, [{"state": 2}], True, [3, 4], 2)
        assert save_calls == first_save_calls
        assert list((namespace / "ssm_companion").rglob("*.safetensors")) == []
        assert list((namespace / "ssm_companion").rglob("*.json")) == []

        # The fail-closed latch is bounded, not permanent.  Once aggregate
        # reconciliation/accounting recovers, a later cache publication works.
        monkeypatch.setattr(
            budget,
            "account_finalized_write_locked",
            original_account,
        )
        store._budget_recovery_interval_ns = 0
        assert store.store("aa" * 32, [{"state": 3}], True, [5, 6], 2)
        assert store.wait_for_pending()
        assert save_calls == first_save_calls + 1
    finally:
        store.shutdown()
        budget.close()


def test_corrupt_derived_accounting_is_rebuilt_from_physical_tree(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    old = _indexed_block(
        root / "aaaaaaaaaaaa",
        "aa-old",
        size=64_000,
        accessed=time.time() - 100,
    )
    budget = GlobalDiskCacheBudget(root, 1, orphan_grace_seconds=0)
    (root / ".vmlx-global-cache-budget-accounting.json").write_text("{")
    try:
        result = budget.enforce(force=True)
        assert result.accounted is True
        assert result.compliant is False  # SQLite metadata remains protected.
        assert not old.exists()
        state = (root / ".vmlx-global-cache-budget-accounting.json").read_text()
        assert '"version": 1' in state
    finally:
        budget._remove_lease()


def test_lease_directory_symlink_is_rejected_without_writing_outside_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / ".vmlx-global-cache-budget-leases").symlink_to(
        outside,
        target_is_directory=True,
    )
    with pytest.raises(OSError):
        GlobalDiskCacheBudget(root, 1000)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("legacy_name", ["blocks", "block_index.db"])
def test_legacy_namespace_symlink_is_rejected_before_claim_or_outside_write(
    tmp_path: Path,
    legacy_name: str,
) -> None:
    namespace = tmp_path / "namespace"
    outside = tmp_path / "outside"
    namespace.mkdir()
    outside.mkdir()
    target = outside / legacy_name
    if legacy_name == "blocks":
        target.mkdir()
    else:
        target.write_text("outside-db-sentinel")
    (namespace / legacy_name).symlink_to(
        target,
        target_is_directory=legacy_name == "blocks",
    )

    with pytest.raises(OSError, match="symlinked block-cache path"):
        ensure_managed_block_cache_namespace(namespace)

    assert not (namespace / ".vmlx-block-cache-namespace-v1").exists()
    if legacy_name == "blocks":
        assert list(target.iterdir()) == []
    else:
        assert target.read_text() == "outside-db-sentinel"


def test_symlinked_namespace_marker_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "root"
    child = root / "aaaaaaaaaaaa"
    blocks = child / "blocks"
    blocks.mkdir(parents=True)
    payload = blocks / "old.safetensors"
    payload.write_bytes(b"x" * 10_000)
    target = tmp_path / "marker-target"
    target.write_text("marker")
    (child / ".vmlx-block-cache-namespace-v1").symlink_to(target)
    budget = GlobalDiskCacheBudget(root, 1)
    try:
        result = budget.enforce(force=True)
        assert result.accounted is False
        assert result.compliant is False
        assert payload.exists()
    finally:
        budget._remove_lease()
