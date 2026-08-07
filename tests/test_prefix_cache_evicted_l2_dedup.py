"""Regression: L1 chain-hash dedup must not honor index rows whose L2
payload was removed by global disk-budget eviction.

Global eviction deletes payload files only; the in-RAM
``cached_block_hash_to_block`` index survives. Before the fix, store_cache's
chain-hash dedup reused those indexed blocks (cache_data=None) and skipped
the disk write, so the chain was permanently unrestorable — every repeat of
an evicted prompt paid a full double prefill (observed live in disk-only
mode on DSV4: 72.5s/69.3s vs 40.4s for a 13.4k-token prompt).
"""

import time
from pathlib import Path

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")

BLOCK_SIZE = 64
NUM_TOKENS = BLOCK_SIZE * 2


class _KVModel:
    class Args:
        num_attention_heads = 32
        num_key_value_heads = 8
        hidden_size = 4096
        head_dim = 128
        kv_lora_rank = 0

    args = Args()

    def make_cache(self):
        return []


def _kv_cache_data(num_tokens: int):
    keys = mx.random.normal((1, 8, num_tokens, 64))
    values = mx.random.normal((1, 8, num_tokens, 64))
    mx.eval(keys, values)
    return [
        {
            "state": (keys, values),
            "meta_state": (str(num_tokens),),
            "class_name": "KVCache",
        }
    ]


def _wait_for_writes(store, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pipeline = store.get_stats()["write_pipeline"]
        if pipeline["queue_depth"] == 0 and pipeline["inflight"] == 0:
            return
        time.sleep(0.02)
    raise AssertionError("disk write pipeline did not settle")


def _payload_files(store) -> list[Path]:
    return sorted(Path(store.cache_dir).rglob("*.safetensors"))


def test_evicted_payload_dedup_falls_through_and_rewrites(tmp_path):
    from vmlx_engine.block_disk_store import BlockDiskStore
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache, PagedCacheManager

    store = BlockDiskStore(str(tmp_path), max_size_gb=1.0)
    try:
        mgr = PagedCacheManager(block_size=BLOCK_SIZE, max_blocks=64)
        mgr._disk_store = store
        # Disk-only / frugal mirror policy: blocks are indexed in L1 with
        # cache_data=None and the payload lives solely on SSD — the exact
        # configuration where the poisoned-index defect manifested.
        mgr.paged_frugal = True
        cache = BlockAwarePrefixCache(model=_KVModel(), paged_cache_manager=mgr)

        tokens = list(range(NUM_TOKENS))
        cache.store_cache("req-original", tokens, _kv_cache_data(NUM_TOKENS))
        _wait_for_writes(store)

        hashes = list(mgr.cached_block_hash_to_block._cache.keys())
        assert len(hashes) == NUM_TOKENS // BLOCK_SIZE
        for h in hashes:
            assert store.has_block(h), "initial L2 payload missing"
            block = mgr.cached_block_hash_to_block.get_block(h)
            assert block is not None
            assert block.cache_data is None, (
                "frugal store should not keep an in-RAM mirror"
            )

        # Simulate global disk-budget eviction: payload files are removed
        # while the per-model SQLite rows and the in-RAM L1 chain-hash
        # index both survive untouched.
        payloads = _payload_files(store)
        assert payloads, "expected payload files on disk"
        for path in payloads:
            path.unlink()
        for h in hashes:
            assert not store.has_block(h)

        # Repeat of the same prompt after eviction: the dedup must detect
        # the unreadable payload, fall through to a fresh allocation, and
        # rewrite the L2 payload instead of skipping the disk write.
        cache.store_cache("req-repeat", tokens, _kv_cache_data(NUM_TOKENS))
        _wait_for_writes(store)

        for h in hashes:
            assert store.has_block(h), (
                "L1 dedup honored a poisoned index row: evicted payload was "
                "never rewritten (permanent double-prefill)"
            )
            assert store.read_block(h) is not None
    finally:
        store.shutdown()


def test_intact_payload_dedup_still_reuses_block(tmp_path):
    from vmlx_engine.block_disk_store import BlockDiskStore
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache, PagedCacheManager

    store = BlockDiskStore(str(tmp_path), max_size_gb=1.0)
    try:
        mgr = PagedCacheManager(block_size=BLOCK_SIZE, max_blocks=64)
        mgr._disk_store = store
        mgr.paged_frugal = True
        cache = BlockAwarePrefixCache(model=_KVModel(), paged_cache_manager=mgr)

        tokens = list(range(NUM_TOKENS))
        cache.store_cache("req-original", tokens, _kv_cache_data(NUM_TOKENS))
        _wait_for_writes(store)

        before = {
            h: mgr.cached_block_hash_to_block.get_block(h).block_id
            for h in mgr.cached_block_hash_to_block._cache
        }
        assert before

        cache.store_cache("req-repeat", tokens, _kv_cache_data(NUM_TOKENS))
        _wait_for_writes(store)

        # Payload intact ⇒ dedup keeps the original blocks (no duplicate
        # allocation, no churn).
        for h, block_id in before.items():
            block = mgr.cached_block_hash_to_block.get_block(h)
            assert block is not None
            assert block.block_id == block_id
    finally:
        store.shutdown()
