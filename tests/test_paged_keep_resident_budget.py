# SPDX-License-Identifier: Apache-2.0
"""Byte-budget enforcement must stay bounded when payloads cannot drain to L2.

Native composite payloads are pinned ``keep_resident`` until an L2 copy is
readable. With no disk store (or one that rejects every write) they were
excluded from every eviction candidate list, so ``enforce_byte_budget``
reported zero evictions while the pool stayed over its ceiling — RAM then grew
until the process died. Unreferenced blocks are safe to drop as a last resort:
eviction costs a re-prefill, never correctness.
"""

import mlx.core as mx

from vmlx_engine.paged_cache import PagedCacheManager

_PAYLOAD_BYTES = 4096


def _pool(max_resident_bytes, disk_store=None):
    return PagedCacheManager(
        block_size=16,
        max_blocks=32,
        disk_store=disk_store,
        max_resident_bytes=max_resident_bytes,
    )


def _cache_block(pool, tag, *, keep_resident, ref_count=0, last_access=0.0):
    block = pool.allocate_block()
    assert block is not None
    data = {"k": mx.zeros((8, 8), dtype=mx.float32)}
    mx.eval(data["k"])
    block.cache_data = data
    block.block_hash = tag
    block.token_count = 16
    block.keep_resident = keep_resident
    block.ref_count = ref_count
    block.last_access = last_access
    pool.cached_block_hash_to_block.insert(tag, block)
    pool._note_resident(block, _PAYLOAD_BYTES)
    return block


class _RejectingStore:
    """An L2 store that accepts nothing — the persistent-failure case."""

    max_size_bytes = 1

    def has_block(self, _block_hash):
        return False

    def write_block_async(self, *_args, **_kwargs):
        return False


def test_keep_resident_blocks_cannot_strand_the_byte_budget():
    pool = _pool(max_resident_bytes=_PAYLOAD_BYTES)  # room for exactly one
    assert pool._disk_store is None
    blocks = [
        _cache_block(pool, f"h{i}", keep_resident=True, last_access=float(i))
        for i in range(4)
    ]
    assert pool.resident_bytes == 4 * _PAYLOAD_BYTES

    evicted = pool.enforce_byte_budget()

    assert evicted >= 3, "undrainable keep_resident blocks stranded the budget"
    assert pool.resident_bytes <= _PAYLOAD_BYTES
    # Oldest-first: the most recently touched block is the survivor.
    assert blocks[-1].cache_data is not None
    assert blocks[0].cache_data is None


def test_rejecting_l2_store_also_cannot_strand_the_budget():
    pool = _pool(max_resident_bytes=_PAYLOAD_BYTES, disk_store=_RejectingStore())
    for i in range(3):
        _cache_block(pool, f"r{i}", keep_resident=True, last_access=float(i))
    assert pool.resident_bytes == 3 * _PAYLOAD_BYTES

    pool.enforce_byte_budget()

    assert pool.resident_bytes <= _PAYLOAD_BYTES


def test_referenced_keep_resident_block_is_never_force_dropped():
    pool = _pool(max_resident_bytes=1)
    held = _cache_block(pool, "held", keep_resident=True, ref_count=1)

    pool.enforce_byte_budget()

    assert held.cache_data is not None, "evicted a block with a live reference"


def test_ordinary_candidates_drain_before_forced_drops():
    """The durability-first ordering is preserved; forcing is last resort."""
    pool = _pool(max_resident_bytes=_PAYLOAD_BYTES)
    ordinary = _cache_block(pool, "plain", keep_resident=False, last_access=100.0)
    native = _cache_block(pool, "native", keep_resident=True, last_access=0.0)

    pool.enforce_byte_budget()

    # `native` is older, but ordinary candidates are consumed first, and one
    # eviction already satisfies the ceiling — so the native payload survives.
    assert ordinary.cache_data is None
    assert native.cache_data is not None
