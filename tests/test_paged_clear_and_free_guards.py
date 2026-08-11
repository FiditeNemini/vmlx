# SPDX-License-Identifier: Apache-2.0
"""Two guards that keep one physical block from serving two requests.

1. ``PagedCacheManager.clear()`` rebuilds the pool with block ids 0..N reused.
   A request that survives the clear still holds its old BlockTable, so its
   completion path resolves those ids through ``allocated_blocks`` and operates
   on blocks that now belong to a DIFFERENT request. ``DELETE /v1/cache`` is an
   async handler with no idle gate, so this is reachable mid-generation.

2. ``free_block()`` was the only one of its three siblings without a
   ``ref_count <= 0`` guard. ``release_request_refs`` deliberately leaves a
   released block at ref 0, still in ``allocated_blocks`` AND linked into the
   free queue, so an unpaired free drove it to -1 and relinked a queued block.
   Measured consequences: the counter reports more free blocks than the list can
   reach (so ``popleft`` raises "No free blocks available" while claiming
   capacity), mid-queue it orphans the tail, and the negative ref_count defeats
   the ``ref_count == 0`` revival guards in ``touch``/``increment_ref``.
"""

import pytest

from vmlx_engine.paged_cache import BlockTable, PagedCacheManager


@pytest.fixture
def mgr():
    return PagedCacheManager(max_blocks=8, block_size=16)


def _reachable(manager) -> int:
    """Walk the intrusive free list and count what is actually reachable."""
    queue = manager.free_block_queue
    seen, node = 0, queue.fake_head.next_free_block
    while node is not None and node is not queue.fake_tail:
        seen += 1
        if seen > manager.max_blocks + 2:
            pytest.fail("free list is looped")
        node = node.next_free_block
    return seen


def _own(manager, request_id: str, count: int) -> BlockTable:
    """Allocate `count` blocks and hand them to a request, as the scheduler does."""
    blocks = [manager.allocate_block() for _ in range(count)]
    assert all(b is not None for b in blocks)
    return BlockTable(
        request_id=request_id,
        block_ids=[b.block_id for b in blocks],
        num_tokens=count * manager.block_size,
    )


def test_clear_refuses_while_a_request_holds_blocks(mgr):
    _own(mgr, "req-live", 3)
    assert mgr.blocks_in_use() == 3

    assert mgr.clear() is False
    assert mgr.blocks_in_use() == 3, "state must be untouched after a refusal"


def test_clear_proceeds_once_the_request_is_released(mgr):
    table = _own(mgr, "req-done", 3)
    mgr.release_request_refs(table)

    assert mgr.blocks_in_use() == 0
    assert mgr.clear() is True


def test_forced_clear_still_works_for_teardown(mgr):
    _own(mgr, "req-live", 3)
    assert mgr.blocks_in_use() == 3
    # teardown paths close the batch generator first and rebuild everything
    assert mgr.clear(force=True) is True
    assert mgr.request_tables == {}


def test_free_block_refuses_to_double_free_a_released_block(mgr):
    table = _own(mgr, "req", 3)
    block_id = table.block_ids[0]
    mgr.release_request_refs(table)

    block = mgr.allocated_blocks.get(block_id)
    assert block is not None and block.ref_count == 0

    before = mgr.free_block_queue.num_free_blocks
    assert mgr.free_block(block_id) is False
    assert block.ref_count == 0, "ref_count must not go negative"
    assert mgr.free_block_queue.num_free_blocks == before


def test_free_queue_counter_matches_what_the_list_can_reach(mgr):
    table = _own(mgr, "req", 3)
    block_id = table.block_ids[0]
    mgr.release_request_refs(table)
    mgr.free_block(block_id)  # the erroneous unpaired free

    assert _reachable(mgr) == mgr.free_block_queue.num_free_blocks


def test_every_free_block_is_still_handed_out_exactly_once(mgr):
    table = _own(mgr, "req", 3)
    mgr.release_request_refs(table)
    mgr.free_block(table.block_ids[0])

    handed = []
    for _ in range(mgr.free_block_queue.num_free_blocks):
        handed.append(mgr.free_block_queue.popleft().block_id)
    assert len(handed) == len(set(handed)), f"duplicate handout: {handed}"


def test_a_legitimate_free_is_unchanged(mgr):
    table = _own(mgr, "req", 1)
    block_id = table.block_ids[0]
    assert mgr.allocated_blocks[block_id].ref_count >= 1

    assert mgr.free_block(block_id) is True
    assert block_id not in mgr.allocated_blocks
    assert _reachable(mgr) == mgr.free_block_queue.num_free_blocks
