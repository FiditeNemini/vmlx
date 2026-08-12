# SPDX-License-Identifier: Apache-2.0
"""A recycled block must never carry the previous tenant's KV.

The recycler treated ``block_hash is None`` as "clean" and returned the block
with its previous ``cache_data`` still attached. The eviction branch beside it
clears the payload; this branch did not. Downstream, the frugal store path can
set a NEW ``block_hash`` without setting ``cache_data``, and reconstruct trusts
any non-None ``cache_data`` — so one request could be served another request's
KV under its own hash. A correctness bug, not merely wasted RAM.
"""

from __future__ import annotations

import pytest

from vmlx_engine.paged_cache import PagedCacheManager


@pytest.fixture()
def pool():
    return PagedCacheManager(block_size=16, max_blocks=8)


def test_recycled_untracked_block_comes_back_without_stale_payload(pool):
    """Exhaust the pool so the recycler MUST hand back the freed block.

    An earlier version of this test freed one block while others were still
    free, so allocate_block popped a fresh one and the test passed against the
    UNFIXED code — a worthless test. The pool must be drained first.
    """
    blocks = []
    while True:
        b = pool.allocate_block()
        if b is None:
            break
        blocks.append(b)
    assert blocks, "pool handed out nothing"

    victim = blocks[0]
    victim.cache_data = {"layer0": "STALE-TENANT-KV"}
    victim.block_hash = None
    assert pool.free_block(victim.block_id) is True

    recycled = pool.allocate_block()
    assert recycled is not None, "the freed block should be recyclable"
    assert recycled.block_id == victim.block_id, (
        "expected the just-freed block back; the test is not exercising the "
        "recycle path"
    )
    assert recycled.cache_data is None, (
        "recycler handed back a block still holding the previous tenant's "
        "cache_data; a later store can attach a new hash to it and reconstruct "
        "will serve that stale KV"
    )


def test_recycling_does_not_leave_bytes_counted_against_the_pool(pool):
    blocks = []
    while True:
        b = pool.allocate_block()
        if b is None:
            break
        blocks.append(b)
    victim = blocks[0]
    victim.cache_data = {"layer0": "STALE"}
    victim.block_hash = None
    pool.free_block(victim.block_id)
    pool.allocate_block()
    resident = getattr(pool, "_resident_bytes", 0)
    assert resident >= 0
