# SPDX-License-Identifier: Apache-2.0
"""Yielding a paged table to the warm prefix cache must release its refs ONCE.

`fetch_cache` records the request in BlockAwarePrefixCache._request_tables. The
warm-preference branch released the paged refs and then called
`paged_cache.detach_request`, which pops only the PAGED table — so the
block-aware entry survived and completion cleanup released the SAME block table a
second time. Any block whose ref another request had legitimately taken through
the shared-prefix path would be decremented to zero underneath it and recycled
while still in use.

This models the real bookkeeping the in-file stub omits.
"""


class _PagedCache:
    def __init__(self):
        self.released = []
        self.detached = []
        self.request_tables = {}

    def release_request_refs(self, block_table):
        if block_table is None:
            return 0
        self.released.append(block_table)
        return 0

    def detach_request(self, request_id):
        self.detached.append(request_id)
        self.request_tables.pop(request_id, None)


class _Entry:
    def __init__(self, block_table):
        self.block_table = block_table


class _BlockAware:
    """Mirrors the real class: its own table plus forwarding on detach."""

    def __init__(self):
        self.paged_cache = _PagedCache()
        self._request_tables = {}
        self._entries_by_type = {}
        self._hit_credits = {}

    def detach_request(self, request_id):
        self._hit_credits.pop(request_id, None)
        entry = self._request_tables.pop(request_id, None)
        if entry:
            self.paged_cache.detach_request(request_id)


def _warm_yield(cache, request_id, block_table):
    """The shipped warm-preference sequence (scheduler.py)."""
    cache.paged_cache.release_request_refs(block_table)
    detach = getattr(cache, "detach_request", None)
    if callable(detach):
        detach(request_id)
    cache.paged_cache.detach_request(request_id)


def _completion_cleanup(cache, request_id):
    """The shipped completion path (scheduler.py)."""
    entry = cache._request_tables.pop(request_id, None)
    cache.paged_cache.release_request_refs(entry.block_table if entry else None)
    cache.paged_cache.detach_request(request_id)


def test_block_table_refs_are_released_exactly_once():
    cache = _BlockAware()
    table = object()
    # fetch_cache registered the request in BOTH tables
    cache._request_tables["req"] = _Entry(table)
    cache.paged_cache.request_tables["req"] = table

    _warm_yield(cache, "req", table)
    _completion_cleanup(cache, "req")

    assert cache.paged_cache.released.count(table) == 1, (
        "the same block table was released twice — blocks can be recycled "
        "underneath a live request holding a shared-prefix ref"
    )


def test_paged_table_is_always_detached():
    """Detaching must happen even when no block-aware entry exists."""
    cache = _BlockAware()
    cache.paged_cache.request_tables["req"] = object()
    _warm_yield(cache, "req", None)
    assert "req" in cache.paged_cache.detached
    assert "req" not in cache.paged_cache.request_tables
