"""LRU touch batches must not launch a managed-root reconciliation scan."""

from __future__ import annotations

import sqlite3
import time

import pytest

from vmlx_engine.block_disk_store import BlockDiskStore


def test_access_only_writer_batch_does_not_reconcile_global_budget(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disk hit's own touches cannot block its remaining block reads.

    On a 178 GB / 23.8k-file managed root, a 32-block Qwen reconstruction
    queued one ``__access__`` item per read.  The writer treated those LRU-only
    updates as finalized cache-byte mutations; once the 30-second interval was
    due, one touch launched a full root reconciliation under the exclusive
    lock and stalled the remaining reads for 3.9 seconds.
    """

    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    conn = sqlite3.connect(str(store._db_path))
    block_hash = "ab" * 32
    now = time.time()
    try:
        conn.execute(
            "INSERT INTO blocks "
            "(block_hash, parent_hash, ancestry_known, file_name, num_tokens, "
            "num_layers, dtype, file_size, created_at, last_accessed, access_count) "
            "VALUES (?, NULL, 1, ?, 64, 1, 'kv', 1, ?, ?, 0)",
            (block_hash, "blocks/ab/payload.safetensors", now, now),
        )
        conn.commit()

        accounting_calls = []
        original_account = store.global_budget.account_finalized_write_locked

        def count_reconcile(*args, **kwargs):
            accounting_calls.append((args, kwargs))
            return original_account(*args, **kwargs)

        monkeypatch.setattr(
            store.global_budget,
            "account_finalized_write_locked",
            count_reconcile,
        )
        store._process_write_batch(
            conn,
            [("__access__", block_hash, now + 1.0)],
        )

        touched = conn.execute(
            "SELECT last_accessed, access_count FROM blocks WHERE block_hash = ?",
            (block_hash,),
        ).fetchone()
        assert touched is not None
        assert touched[0] == pytest.approx(now + 1.0)
        assert touched[1] == 1
        assert accounting_calls == [], (
            "an access-only batch must not enter global budget accounting"
        )
    finally:
        conn.close()
        store.shutdown()
