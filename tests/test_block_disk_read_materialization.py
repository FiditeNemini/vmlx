"""Retaining a long SSD chain must not retain one open file per block."""

import hashlib

import mlx.core as mx
import psutil
import pytest

from vmlx_engine.block_disk_store import BlockDiskStore


@pytest.mark.parametrize("selective", [False, True])
def test_returned_blocks_release_lazy_load_descriptors(tmp_path, selective):
    """Real MLX reads, held unevaluated as prefix reconstruction holds them.

    A 4,025-block Flash Next refault exhausted RLIMIT_NOFILE before its final
    concatenation evaluated any loaded arrays. Raising that limit only moves
    the failure to a longer chain. Completed block reads must own their bytes,
    not a deferred file read outside the aggregate cache's shared lock.
    """
    store = BlockDiskStore(str(tmp_path / "l2"), max_size_gb=1.0)
    try:
        keys = mx.arange(16).reshape(1, 1, 4, 4).astype(mx.bfloat16)
        values = keys + 1
        mx.eval(keys, values)
        hashes = [hashlib.sha256(f"fd-block-{i}".encode()).digest() for i in range(24)]
        for block_hash in hashes:
            assert store.write_block_async(block_hash, [("kv", keys, values)], 4)
        assert store.wait_for_blocks(hashes, timeout=10.0) == set(hashes)

        before = psutil.Process().num_fds()
        held = []
        for block_hash in hashes:
            payload = (
                store.read_block_for_reconstruction(block_hash, rotating_target_offset=96)
                if selective else store.read_block(block_hash)
            )
            assert payload is not None
            held.append(payload)
        # Do not evaluate/assert tensor values until after this observation:
        # doing so in the loop would hide the production resource leak.
        growth = psutil.Process().num_fds() - before
        assert growth <= 4, f"24 retained block payloads kept {growth} extra descriptors"
        assert store.get_stats()["disk_hits"] >= len(hashes)
        for payload in held:
            assert payload[0][0] == "kv"
            assert payload[0][1].dtype == mx.bfloat16
            assert bool(mx.array_equal(payload[0][1], keys))
            assert bool(mx.array_equal(payload[0][2], values))
    finally:
        store.shutdown()
