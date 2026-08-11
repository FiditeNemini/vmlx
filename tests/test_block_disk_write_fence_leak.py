# SPDX-License-Identifier: Apache-2.0
"""An escaping exception must not strand a block-disk write fence.

Every settle path in ``write_block_async`` returns rather than raises, so an
exception that escapes means a producer was registered on the fence and never
accounted for. ``_active`` then sticks above zero: the fence never becomes
ready, never terminalizes, and is never pruned (unfinished fences are retained
by design). At the fence cap ``begin_write_fence`` starts raising — and its
caller catches that, warns, and continues with ``disk_write_fence_id=None``, so
every later request writes UNFENCED for the life of the process, while pins from
the leaked fence's earlier writes keep their bytes un-evictable.

The concrete trigger was the sqlite reconnect inside the index lookups: under fd
exhaustion ``sqlite3.connect`` raises OperationalError("unable to open database
file") and escaped. That is fixed at source too (a failed reconnect degrades to
a cache miss), so this file asserts the general invariant with direct injection.
"""

import sqlite3

import mlx.core as mx
import pytest

from vmlx_engine.block_disk_store import BlockDiskStore


@pytest.fixture
def store(tmp_path):
    return BlockDiskStore(cache_dir=str(tmp_path / "l2"))


def _fence_state(store, fence_id):
    with store._stats_lock:
        return dict(store._write_fences[str(fence_id)])


def test_escaping_exception_settles_the_fence(store):
    fence_id = store.begin_write_fence(request_id="req-1")

    def boom(*_a, **_k):
        raise sqlite3.OperationalError("unable to open database file")

    # Injected at a point unconditionally reached AFTER the fence registers a
    # producer. The real-world trigger was the sqlite reconnect inside the index
    # lookups, but the invariant under test is general: whatever escapes, the
    # fence must not be left owing an accounting entry.
    store._estimate_cache_payload_bytes = boom

    with pytest.raises(sqlite3.OperationalError):
        store.write_block_async(
            bytes.fromhex("aa" * 32),
            [("std", mx.zeros((1, 1, 4, 8)), mx.zeros((1, 1, 4, 8)))],
            16,
            parent_hash=None,
            request_id="req-1",
            fence_id=fence_id,
        )

    state = _fence_state(store, fence_id)
    assert state["_active"] == 0, "producer left registered — fence is stranded"
    assert state["failed"] >= 1


def test_a_stranded_fence_does_not_exhaust_the_cap(store, monkeypatch):
    """The end consequence: later requests silently falling back to UNFENCED."""
    monkeypatch.setattr(
        "vmlx_engine.block_disk_store.MAX_WRITE_FENCES", 3, raising=False
    )

    def boom(*_a, **_k):
        raise sqlite3.OperationalError("unable to open database file")

    store._estimate_cache_payload_bytes = boom

    for i in range(6):
        fence_id = store.begin_write_fence(request_id=f"req-{i}")
        with pytest.raises(sqlite3.OperationalError):
            store.write_block_async(
                bytes.fromhex(f"{i:02x}" * 32),
                [("std", mx.zeros((1, 1, 4, 8)), mx.zeros((1, 1, 4, 8)))],
                16,
                parent_hash=None,
                request_id=f"req-{i}",
                fence_id=fence_id,
            )
        assert _fence_state(store, fence_id)["_active"] == 0
        store.seal_write_fence(fence_id)


def test_index_lookup_degrades_to_a_miss_when_the_reconnect_fails(store, monkeypatch):
    """The concrete trigger, fixed at source: a broken index must not raise."""

    class _DeadConn:
        def execute(self, *_a, **_k):
            raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(
        type(store), "_read_conn", property(lambda _self: _DeadConn())
    )
    monkeypatch.setattr(
        sqlite3,
        "connect",
        lambda *_a, **_k: (_ for _ in ()).throw(
            sqlite3.OperationalError("unable to open database file")
        ),
    )

    # Must answer "not present" rather than raising out of the store.
    assert store.has_block(bytes.fromhex("bb" * 32)) is False
