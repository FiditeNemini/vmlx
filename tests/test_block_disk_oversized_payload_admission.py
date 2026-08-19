# SPDX-License-Identifier: Apache-2.0
"""A payload larger than the pending-write budget must be admissible ALONE.

MEASURED on the box (Laguna-S-2.1 lane, 2026-08-16): a 48-layer Mixed-SWA
deferred clean-store payload at ~19k tokens exceeds the 1GB
``VMLX_BLOCK_DISK_PENDING_WRITE_BYTES`` default, and the old flat cap dropped
it OUTRIGHT — ``if amount > max: return False`` — regardless of how empty the
queue was. A drop under congestion is transient; this one was PERMANENT: the
same payload can never fit under a flat cap, ancestry truncation then kills
every descendant block, and a deep restart-restore finds an L2 hole at that
depth forever. The budget exists to bound the AGGREGATE RAM of detached
payloads awaiting the background writer, not to be a coverage ceiling — so an
oversized single payload is admitted EXCLUSIVELY: wait for the queue to drain
(within the caller's admission timeout), then take the whole budget alone.
Aggregate stays bounded by max(budget, one payload), RAM the process already
holds transiently for the detached copy regardless.

The SSM companion disk store shares the same flat-cap shape (sibling-path
rule) and gets the no-wait variant of the same rule — its lock has no
condition variable, and a lone oversized companion snapshot is the case that
matters.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from vmlx_engine.block_disk_store import BlockDiskStore
from vmlx_engine.utils.ssm_companion_disk_store import SSMCompanionDiskStore

MAX = 1000


def _block_stub(pending: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        _pending_write_condition=threading.Condition(),
        _pending_write_bytes=pending,
        _max_pending_write_bytes=MAX,
        _pending_write_byte_drops=0,
        write_drop_reasons={},
    )


def _ssm_stub(pending: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        _stats_lock=threading.Lock(),
        _pending_write_bytes=pending,
        _max_pending_write_bytes=MAX,
        _pending_write_byte_drops=0,
    )


def test_oversized_payload_admits_exclusively_when_queue_is_empty():
    stub = _block_stub()
    ok = BlockDiskStore._reserve_pending_write_bytes(stub, 2 * MAX, timeout=0.1)
    assert ok, "a lone oversized payload must be admitted, not dropped forever"
    assert stub._pending_write_bytes == 2 * MAX
    assert stub._pending_write_byte_drops == 0


def test_oversized_payload_waits_for_drain_then_admits():
    stub = _block_stub(pending=400)

    def _drain():
        time.sleep(0.05)
        with stub._pending_write_condition:
            stub._pending_write_bytes = 0
            stub._pending_write_condition.notify_all()

    t = threading.Thread(target=_drain)
    t.start()
    ok = BlockDiskStore._reserve_pending_write_bytes(stub, 2 * MAX, timeout=2.0)
    t.join()
    assert ok, "the writer drained within the timeout; the payload must admit"
    assert stub._pending_write_bytes == 2 * MAX


def test_oversized_payload_still_drops_under_sustained_congestion():
    stub = _block_stub(pending=400)
    ok = BlockDiskStore._reserve_pending_write_bytes(stub, 2 * MAX, timeout=0.0)
    assert not ok
    assert stub._pending_write_byte_drops == 1
    assert stub.write_drop_reasons.get("byte_budget") == 1
    assert stub._pending_write_bytes == 400, "a drop must not leak reservation"


def test_oversized_resize_admits_when_reservation_is_exclusive():
    # The estimate was admitted exclusively; the detached payload measured
    # even larger. Dropping at resize would reintroduce the flat ceiling one
    # step later.
    stub = _block_stub(pending=800)
    ok = BlockDiskStore._resize_pending_write_reservation(stub, 800, 3 * MAX)
    assert ok
    assert stub._pending_write_bytes == 3 * MAX


def test_oversized_resize_with_other_writers_drops_and_wakes_waiters():
    stub = _block_stub(pending=900)  # 300 ours + 600 someone else's
    woke = threading.Event()

    def _waiter():
        with stub._pending_write_condition:
            stub._pending_write_condition.wait(timeout=2.0)
            woke.set()

    t = threading.Thread(target=_waiter)
    t.start()
    time.sleep(0.05)
    ok = BlockDiskStore._resize_pending_write_reservation(stub, 300, 3 * MAX)
    t.join()
    assert not ok
    assert stub._pending_write_bytes == 600, "our estimate must be released"
    assert woke.is_set(), (
        "releasing the estimate freed budget; sleeping waiters must be woken "
        "or they stall until their timeout for bytes that are already free"
    )


def test_normal_reservations_unchanged():
    stub = _block_stub()
    assert BlockDiskStore._reserve_pending_write_bytes(stub, 400, timeout=0.0)
    assert BlockDiskStore._reserve_pending_write_bytes(stub, 400, timeout=0.0)
    assert not BlockDiskStore._reserve_pending_write_bytes(stub, 400, timeout=0.0)
    assert stub._pending_write_bytes == 800
    assert BlockDiskStore._resize_pending_write_reservation(stub, 400, 500)
    assert stub._pending_write_bytes == 900


def test_ssm_store_oversized_exclusive_admission():
    stub = _ssm_stub()
    assert SSMCompanionDiskStore._reserve_pending_bytes(stub, 2 * MAX)
    assert stub._pending_write_bytes == 2 * MAX

    busy = _ssm_stub(pending=100)
    assert not SSMCompanionDiskStore._reserve_pending_bytes(busy, 2 * MAX)
    assert busy._pending_write_byte_drops == 1

    resize = _ssm_stub(pending=700)
    assert SSMCompanionDiskStore._resize_pending_reservation(resize, 700, 2 * MAX)
    assert resize._pending_write_bytes == 2 * MAX


def test_oversized_resize_waits_for_drain_within_admission_timeout():
    """Burst-tail durability: dropping at resize the instant other writers
    are pending poisons every descendant of this block for the session, so
    the resize honors the same admission deadline the reservation used."""
    stub = _block_stub(pending=900)  # 300 ours + 600 someone else's

    def _drain():
        time.sleep(0.05)
        with stub._pending_write_condition:
            stub._pending_write_bytes = 300  # the writer finished the others
            stub._pending_write_condition.notify_all()

    t = threading.Thread(target=_drain)
    t.start()
    ok = BlockDiskStore._resize_pending_write_reservation(
        stub, 300, 3 * MAX, timeout=2.0
    )
    t.join()
    assert ok, "the writer drained within the deadline; the resize must admit"
    assert stub._pending_write_bytes == 3 * MAX


def test_oversized_resize_zero_timeout_still_drops_under_congestion():
    stub = _block_stub(pending=900)
    ok = BlockDiskStore._resize_pending_write_reservation(stub, 300, 3 * MAX)
    assert not ok
    assert stub._pending_write_bytes == 600
