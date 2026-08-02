# SPDX-License-Identifier: Apache-2.0
"""Focused DSV4 native block-delta transport and checkpoint tests."""

from __future__ import annotations

import time
import threading
from types import SimpleNamespace

import pytest


mx = pytest.importorskip("mlx.core")


def _pool_delta(
    start: int,
    end: int,
    width: int,
    *,
    absent: bool = False,
    q8: bool = False,
):
    if absent:
        return {
            "schema": "deepseek_v4_pool_delta_v1",
            "storage": "none",
            "start_row": 0,
            "end_row": 0,
        }
    rows = end - start
    if q8:
        assert width % 32 == 0
        groups = width // 32
        q = mx.full((1, rows, groups, 32), end % 255, dtype=mx.uint8)
        scale = mx.full((1, rows, groups, 1), 0.25, dtype=mx.float16)
        minimum = mx.full((1, rows, groups, 1), -1, dtype=mx.float16)
        mx.eval(q, scale, minimum)
        return {
            "schema": "deepseek_v4_pool_delta_v1",
            "storage": "q8",
            "start_row": start,
            "end_row": end,
            "segments": (
                (
                    q,
                    scale,
                    minimum,
                    (1, rows, width),
                    32,
                    8,
                    str(mx.bfloat16),
                ),
            ),
        }
    return {
        "schema": "deepseek_v4_pool_delta_v1",
        "storage": "bf16",
        "start_row": start,
        "end_row": end,
        "value": mx.full((1, rows, width), end, dtype=mx.bfloat16),
    }


def _delta_record(
    start: int,
    end: int,
    ratio: int,
    *,
    periodic=False,
    terminal=False,
    append_safe=False,
    pool_quant=False,
):
    anchor = None
    if periodic or terminal:
        local_rows = min(end, 128)
        local = mx.arange(local_rows, dtype=mx.float32).reshape(1, 1, local_rows, 1)
        anchor = {
            "tokens": end,
            "periodic": bool(periodic),
            "terminal": bool(terminal),
            "append_safe": bool(append_safe),
            "local_state": (local, local + 1000),
            "meta_state": ("0", "128", str(end), str(local_rows)),
            "compressor_buffer_kv": None,
            "compressor_buffer_gate": None,
            "indexer_buffer_kv": None,
            "indexer_buffer_gate": None,
        }
    row_start = start // ratio
    row_end = end // ratio
    return {
        "schema": "deepseek_v4_block_delta_v1",
        "class_name": (
            "PoolQuantizedV4Cache" if pool_quant else "DeepseekV4Cache"
        ),
        "start_token": start,
        "end_token": end,
        "block_size": 256,
        "anchor_interval_blocks": 8,
        "sliding_window": 128,
        "compress_ratio": ratio,
        "compressor_pool": _pool_delta(
            row_start,
            row_end,
            32 if pool_quant else 8,
            q8=pool_quant,
        ),
        "indexer_pool": _pool_delta(
            row_start,
            row_end,
            32 if pool_quant else 4,
            absent=ratio != 4,
            q8=pool_quant,
        ),
        "anchor": anchor,
    }


def _topology_interval(
    start: int,
    end: int,
    *,
    terminal=False,
    append_safe=False,
    pool_quant=False,
):
    if end <= start or end - start > 256:
        raise ValueError(f"invalid DSV4 topology interval [{start}, {end})")
    periodic = end % 2048 == 0
    entries = []
    for layer_index in range(43):
        if layer_index < 2:
            if periodic or terminal:
                local = mx.arange(128, dtype=mx.float32).reshape(1, 1, 128, 1)
                entries.append(
                    ("rotating_kv", local, local + 1000, 128, 0, end, 128)
                )
            else:
                entries.append(("rotating_kv_pending", "RotatingKVCache"))
            continue
        ratio = 4 if (layer_index - 2) % 2 == 0 else 128
        # There are 21 CSA and 20 HCA layers after the two ratio-zero layers.
        entries.append(
            (
                "deepseek_v4_delta_v1",
                _delta_record(
                    start,
                    end,
                    ratio,
                    periodic=periodic,
                    terminal=terminal,
                    append_safe=append_safe,
                    pool_quant=pool_quant,
                ),
                "PoolQuantizedV4Cache" if pool_quant else "DeepseekV4Cache",
                {
                    "schema": "deepseek_v4_v10_delta",
                    "compress_ratio": ratio,
                    "sliding_window": 128,
                    "pool_quant": bool(pool_quant),
                },
            )
        )
    return entries


def _topology_block(
    block_index: int,
    *,
    terminal=False,
    append_safe=False,
    pool_quant=False,
):
    start = block_index * 256
    return _topology_interval(
        start,
        start + 256,
        terminal=terminal,
        append_safe=append_safe,
        pool_quant=pool_quant,
    )


def _write_native_block(store, *, request_id: str, block_hash: bytes, block_index: int):
    fence_id = store.begin_write_fence(request_id)
    assert store.write_block_async(
        block_hash,
        _topology_block(block_index, terminal=True),
        256,
        request_id=request_id,
        fence_id=fence_id,
    )
    assert store.seal_write_fence(fence_id)

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        stats = store.get_stats()
        fence = next(
            (
                item
                for item in stats["write_pipeline"]["recent_fences"]
                if item["fence_id"] == fence_id
            ),
            None,
        )
        if (
            fence is not None
            and fence["post_eviction_complete"]
            and stats["write_pipeline"]["queue_depth"] == 0
            and stats["write_pipeline"]["inflight"] == 0
        ):
            return stats, fence
        time.sleep(0.01)
    raise AssertionError(f"DSV4 disk write fence {fence_id} did not settle")


def test_dsv4_delta_serializer_roundtrip_preserves_43_layer_native_topology():
    from vmlx_engine.block_disk_store import _deserialize_block, _serialize_block
    from vmlx_engine.cache_record_validator import validate_cache_record

    payload = _topology_block(7, terminal=True)
    tensors, dtype, num_layers = _serialize_block(payload)
    restored = _deserialize_block(tensors, dtype)

    assert num_layers == 43
    assert len(restored) == 43
    assert [entry[0] for entry in restored[:2]] == ["rotating_kv", "rotating_kv"]
    assert sum(entry[0] == "deepseek_v4_delta_v1" for entry in restored) == 41
    assert restored[2][1]["anchor"]["tokens"] == 2048
    valid, reason, measured_bytes = validate_cache_record(
        restored,
        expected_num_layers=43,
        source="test-dsv4-delta-roundtrip",
    )
    assert valid, reason
    assert measured_bytes > 0


def test_dsv4_terminal_capture_retains_preceding_aligned_checkpoint():
    from vmlx_engine.utils.dsv4_batch_generator import DSV4BatchGenerator

    class DeepseekV4Cache:
        """Minimal native exporter shaped like the installed JANG cache."""

        def export_block_delta(
            self,
            start,
            end,
            *,
            block_size,
            anchor_interval_blocks,
            force_anchor=False,
        ):
            anchor_interval = block_size * anchor_interval_blocks
            anchored = bool(force_anchor or end % anchor_interval == 0)
            return {
                "start_token": start,
                "end_token": end,
                "anchor": (
                    {
                        "tokens": end,
                        "periodic": end % anchor_interval == 0,
                        "terminal": bool(force_anchor),
                    }
                    if anchored
                    else None
                ),
            }

    cache = DeepseekV4Cache()
    DSV4BatchGenerator._reset_dsv4_block_records([cache], 0)
    DSV4BatchGenerator._capture_dsv4_completed_blocks([cache], 256)
    DSV4BatchGenerator._capture_dsv4_append_safe_checkpoint([cache], 256)
    DSV4BatchGenerator._capture_dsv4_terminal_anchor([cache], 331)

    records = cache._vmlx_dsv4_block_records
    assert [(record["start_token"], record["end_token"]) for record in records] == [
        (0, 256),
        (256, 331),
    ]
    assert records[0]["anchor"]["append_safe"] is True
    assert records[0]["anchor"]["terminal"] is True
    assert records[1]["anchor"]["terminal"] is True
    assert records[1]["anchor"].get("append_safe") is not True


def test_dsv4_append_safe_marker_roundtrips_through_block_disk_transport():
    from vmlx_engine.block_disk_store import _deserialize_block, _serialize_block
    from vmlx_engine.cache_record_validator import validate_cache_record

    payload = _topology_block(0, terminal=True, append_safe=True)
    tensors, dtype, _ = _serialize_block(payload)
    restored = _deserialize_block(tensors, dtype)

    delta_anchors = [
        entry[1]["anchor"]
        for entry in restored
        if entry[0] == "deepseek_v4_delta_v1"
    ]
    assert len(delta_anchors) == 41
    assert all(anchor["append_safe"] is True for anchor in delta_anchors)
    valid, reason, _ = validate_cache_record(
        restored,
        expected_num_layers=43,
        source="test-dsv4-append-safe-roundtrip",
    )
    assert valid, reason


def test_dsv4_append_safe_policy_versions_the_cache_namespace(monkeypatch):
    import vmlx_engine.prefix_cache as prefix_cache

    model = SimpleNamespace(
        args=SimpleNamespace(
            model_type="deepseek_v4",
            num_hidden_layers=43,
            num_attention_heads=64,
            num_key_value_heads=1,
            hidden_size=4096,
        )
    )
    current = prefix_cache.compute_model_cache_key(model)
    monkeypatch.setattr(
        prefix_cache,
        "DSV4_APPEND_SAFE_CHECKPOINT_POLICY",
        "full_block_test_next",
    )

    assert prefix_cache.compute_model_cache_key(model) != current


def test_dsv4_native_delta_disk_store_roundtrip_is_readable_and_bounded(tmp_path):
    from vmlx_engine.block_disk_store import BlockDiskStore
    from vmlx_engine.cache_record_validator import validate_cache_record

    store = BlockDiskStore(
        str(tmp_path),
        max_size_gb=1,
        expected_num_layers=43,
        allow_tq_native=False,
    )
    block_hash = b"d" * 32
    try:
        stats, fence = _write_native_block(
            store,
            request_id="dsv4-native-roundtrip",
            block_hash=block_hash,
            block_index=7,
        )
        restored = store.read_block(block_hash)
    finally:
        store.shutdown()

    assert fence["expected"] == fence["completed"] == fence["retained"] == 1
    assert fence["failed"] == fence["dropped"] == 0
    assert stats["blocks_on_disk"] == 1
    assert stats["disk_writes"] == 1
    assert stats["global_budget"]["compliant"] is True
    assert stats["global_budget"]["bytes_after"] <= stats["global_budget"][
        "max_size_bytes"
    ]
    assert restored is not None
    assert len(restored) == 43
    assert [entry[0] for entry in restored[:2]] == ["rotating_kv", "rotating_kv"]
    assert sum(entry[0] == "deepseek_v4_delta_v1" for entry in restored) == 41
    valid, reason, _ = validate_cache_record(
        restored,
        expected_num_layers=43,
        source="test-dsv4-native-disk-roundtrip",
    )
    assert valid, reason


def test_dsv4_native_delta_disk_store_evicts_oldest_block_at_global_cap(tmp_path):
    from vmlx_engine.block_disk_store import BlockDiskStore

    store = BlockDiskStore(
        str(tmp_path),
        max_size_gb=1,
        expected_num_layers=43,
        allow_tq_native=False,
    )
    old_hash = b"o" * 32
    new_hash = b"n" * 32
    try:
        first_stats, first_fence = _write_native_block(
            store,
            request_id="dsv4-native-old",
            block_hash=old_hash,
            block_index=7,
        )
        first_payload_bytes = first_stats["disk_size_bytes"]
        first_physical_bytes = first_stats["global_budget"]["bytes_after"]
        assert first_payload_bytes > 0
        assert first_physical_bytes >= first_payload_bytes

        cap = int(first_physical_bytes + first_payload_bytes * 0.5)
        store.max_size_bytes = cap
        store.global_budget._requested_max_size_bytes = cap
        store.global_budget._publish_budget(cap)

        second_stats, second_fence = _write_native_block(
            store,
            request_id="dsv4-native-new",
            block_hash=new_hash,
            block_index=15,
        )
        old_present = store.has_block(old_hash)
        new_present = store.has_block(new_hash)
    finally:
        store.shutdown()

    assert first_fence["retained"] == 1
    assert second_fence["expected"] == second_fence["completed"] == 1
    assert second_fence["failed"] == second_fence["dropped"] == 0
    assert second_fence["retained"] == 1
    assert second_fence["post_eviction_complete"] is True
    assert second_stats["disk_writes"] == 2
    assert second_stats["disk_evictions"] >= 1
    assert second_stats["blocks_on_disk"] == 1
    assert second_stats["global_budget"]["compliant"] is True
    assert second_stats["global_budget"]["bytes_after"] <= cap
    assert old_present is False
    assert new_present is True


def test_dsv4_delta_reconstructs_full_43_layer_periodic_anchor():
    from jang_tools.dsv4.mlx_model import DeepseekV4Cache
    from mlx_lm.models.cache import RotatingKVCache
    from vmlx_engine.paged_cache import BlockTable, PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    paged = PagedCacheManager(block_size=256, max_blocks=16)
    cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged)
    table = BlockTable(request_id="dsv4-full-topology")
    for block_index in range(8):
        block = paged.allocate_block()
        assert block is not None
        block.token_count = 256
        block.cache_data = _topology_block(block_index)
        table.block_ids.append(block.block_id)
        table.num_tokens += 256

    rebuilt = cache.reconstruct_cache(table)

    assert rebuilt is not None
    assert len(rebuilt) == 43
    assert all(isinstance(layer, RotatingKVCache) for layer in rebuilt[:2])
    assert all(isinstance(layer, DeepseekV4Cache) for layer in rebuilt[2:])
    assert sum(layer.compress_ratio == 4 for layer in rebuilt[2:]) == 21
    assert sum(layer.compress_ratio == 128 for layer in rebuilt[2:]) == 20
    assert all(layer.offset == 2048 for layer in rebuilt)


def test_dsv4_q8_pool_deltas_roundtrip_disk_and_reconstruct_without_codec_loss(
    tmp_path,
):
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache
    from vmlx_engine.block_disk_store import BlockDiskStore
    from vmlx_engine.paged_cache import BlockTable, PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    store = BlockDiskStore(
        str(tmp_path),
        max_size_gb=1,
        expected_num_layers=43,
        allow_tq_native=False,
    )
    block_hash = b"q" * 32
    try:
        fence_id = store.begin_write_fence("dsv4-q8-native")
        assert store.write_block_async(
            block_hash,
            _topology_block(7, terminal=True, pool_quant=True),
            256,
            request_id="dsv4-q8-native",
            fence_id=fence_id,
        )
        assert store.seal_write_fence(fence_id)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            restored_block = store.read_block(block_hash)
            if restored_block is not None:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("DSV4 q8 native block did not become readable")
    finally:
        store.shutdown()

    q8_delta = restored_block[2][1]["compressor_pool"]
    assert q8_delta["storage"] == "q8"
    assert q8_delta["segments"][0][4:6] == (32, 8)
    assert str(q8_delta["segments"][0][0].dtype).endswith("uint8")

    paged = PagedCacheManager(block_size=256, max_blocks=16)
    cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged)
    table = BlockTable(request_id="dsv4-q8-full-topology")
    for block_index in range(8):
        block = paged.allocate_block()
        assert block is not None
        block.token_count = 256
        block.cache_data = _topology_block(
            block_index,
            pool_quant=True,
        )
        table.block_ids.append(block.block_id)
        table.num_tokens += 256

    rebuilt = cache.reconstruct_cache(table)

    assert rebuilt is not None
    assert len(rebuilt) == 43
    assert all(isinstance(layer, PoolQuantizedV4Cache) for layer in rebuilt[2:])
    assert sum(layer.compress_ratio == 4 for layer in rebuilt[2:]) == 21
    assert sum(layer.compress_ratio == 128 for layer in rebuilt[2:]) == 20
    assert all(layer.offset == 2048 for layer in rebuilt)
    assert all(
        layer.compressor_state._pooled_q_segments
        for layer in rebuilt[2:]
    )


def test_dsv4_paged_on_refault_can_exceed_l1_cap_transiently_then_releases():
    """The RAM slider bounds retained L1, not longest-match SSD recovery."""
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache
    from vmlx_engine.paged_cache import BlockTable, PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    paged = PagedCacheManager(
        block_size=256,
        max_blocks=16,
        max_resident_bytes=1,
    )
    cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged)
    table = BlockTable(request_id="dsv4-transient-l2-refault")
    for block_index in range(8):
        block = paged._promote_from_disk(
            bytes([block_index + 1]) * 32,
            _topology_block(block_index, pool_quant=True),
            256,
        )
        assert block is not None
        assert block.cache_data_transient is True
        table.block_ids.append(block.block_id)
        table.num_tokens += 256

    transient_peak = paged.resident_bytes
    assert transient_peak > paged.max_resident_bytes

    rebuilt = cache.reconstruct_cache(table)

    assert rebuilt is not None
    assert len(rebuilt) == 43
    assert all(isinstance(layer, PoolQuantizedV4Cache) for layer in rebuilt[2:])
    assert paged.transient_disk_promotions == 8
    assert paged.transient_disk_peak_bytes == transient_peak
    assert paged.resident_bytes == 0
    assert all(
        paged.allocated_blocks[block_id].cache_data is None
        for block_id in table.block_ids
    )


def test_dsv4_longest_match_uses_periodic_anchor_and_replays_matched_tail():
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
    cache._expected_num_layers = 43
    blocks = [
        SimpleNamespace(token_count=256, cache_data=_topology_block(i, terminal=i == 8))
        for i in range(9)
    ]
    request_tokens = list(range(2304)) + [90_001, 90_002]

    kept, checkpoint, replayed = cache._normalize_dsv4_delta_candidate(
        request_id="dsv4-partial",
        blocks=blocks,
        matched_tokens=2304,
        request_tokens=request_tokens,
        disk_store=None,
    )

    assert len(kept) == 8
    assert checkpoint == 2048
    assert replayed == 256


@pytest.mark.parametrize("disk_backed", [False, True])
def test_dsv4_short_changed_suffix_uses_preceding_aligned_checkpoint(disk_backed):
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
    cache._expected_num_layers = 43
    payloads = [
        _topology_block(0, terminal=True, append_safe=True),
        _topology_interval(256, 331, terminal=True),
    ]
    hashes = [b"a" * 32, b"b" * 32]
    blocks = [
        SimpleNamespace(
            token_count=token_count,
            block_hash=block_hash,
            cache_data=None if disk_backed else payload,
        )
        for token_count, block_hash, payload in zip(
            (256, 75), hashes, payloads
        )
    ]
    disk_store = (
        SimpleNamespace(
            read_block=lambda block_hash: payloads[hashes.index(block_hash)]
        )
        if disk_backed
        else None
    )
    request_tokens = list(range(331)) + list(range(80_000, 80_086))

    kept, checkpoint, replayed = cache._normalize_dsv4_delta_candidate(
        request_id=f"dsv4-short-aligned-{'disk' if disk_backed else 'ram'}",
        blocks=blocks,
        matched_tokens=331,
        request_tokens=request_tokens,
        disk_store=disk_store,
    )

    assert kept == blocks[:1]
    assert checkpoint == 256
    assert replayed == 75
    assert request_tokens[checkpoint:] == list(range(256, 331)) + list(
        range(80_000, 80_086)
    )


def test_dsv4_short_exact_n_minus_one_still_prefers_terminal_checkpoint():
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
    cache._expected_num_layers = 43
    blocks = [
        SimpleNamespace(
            token_count=256,
            cache_data=_topology_block(0, terminal=True, append_safe=True),
        ),
        SimpleNamespace(
            token_count=75,
            cache_data=_topology_interval(256, 331, terminal=True),
        ),
    ]

    kept, checkpoint, replayed = cache._normalize_dsv4_delta_candidate(
        request_id="dsv4-short-exact-terminal",
        blocks=blocks,
        matched_tokens=331,
        request_tokens=list(range(331)) + [90_004],
        disk_store=None,
    )

    assert kept == blocks
    assert checkpoint == 331
    assert replayed == 0


def test_dsv4_unstamped_terminal_is_not_a_changed_suffix_checkpoint():
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
    cache._expected_num_layers = 43
    blocks = [
        SimpleNamespace(
            token_count=256,
            cache_data=_topology_block(0, terminal=True),
        ),
        SimpleNamespace(
            token_count=75,
            cache_data=_topology_interval(256, 331, terminal=True),
        ),
    ]

    kept, checkpoint, replayed = cache._normalize_dsv4_delta_candidate(
        request_id="dsv4-short-terminal-no-false-hit",
        blocks=blocks,
        matched_tokens=331,
        request_tokens=list(range(331)) + list(range(91_000, 91_086)),
        disk_store=None,
    )

    assert kept == []
    assert checkpoint == 0
    assert replayed == 331


def test_dsv4_append_safe_checkpoint_rejects_stale_rotating_state():
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
    cache._expected_num_layers = 43
    aligned = _topology_block(0, terminal=True, append_safe=True)
    for index in range(2):
        entry = aligned[index]
        aligned[index] = (*entry[:5], 255, entry[6])
    blocks = [
        SimpleNamespace(token_count=256, cache_data=aligned),
        SimpleNamespace(
            token_count=75,
            cache_data=_topology_interval(256, 331, terminal=True),
        ),
    ]

    kept, checkpoint, replayed = cache._normalize_dsv4_delta_candidate(
        request_id="dsv4-short-stale-rotating-no-false-hit",
        blocks=blocks,
        matched_tokens=331,
        request_tokens=list(range(331)) + list(range(92_000, 92_086)),
        disk_store=None,
    )

    assert kept == []
    assert checkpoint == 0
    assert replayed == 331


def _dsv4_pending_tail_fetch_fixture(*, register_chain_hashes: bool):
    """Build a 2,304-token DSV4 match whose ninth block is not restorable.

    Block eight ends at the exact 2,048-token periodic checkpoint. Block nine
    extends the authoritative token match to 2,304 tokens but deliberately
    carries ``rotating_kv_pending``. The fetch path must keep the longer match
    for telemetry while owning only the eight restorable blocks and replaying
    the matched 256-token tail.
    """
    from vmlx_engine.paged_cache import (
        BlockTable,
        PagedCacheManager,
        compute_block_hash,
    )
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    block_size = 256
    matched_tokens = list(range(9 * block_size))
    request_tokens = matched_tokens + [90_101, 90_102]
    paged = PagedCacheManager(block_size=block_size, max_blocks=16)
    cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged)
    cache._expected_num_layers = 43

    blocks = paged.get_new_blocks(9)
    parent_hash = None
    for block_index, block in enumerate(blocks):
        start = block_index * block_size
        end = start + block_size
        block_hash = compute_block_hash(
            parent_hash,
            matched_tokens[start:end],
        )
        block.block_hash = block_hash
        block.parent_hash = parent_hash
        block.token_count = block_size
        # Do not stamp a terminal anchor on block nine. It must remain a
        # pending tail after the exact periodic anchor in block eight.
        block.cache_data = _topology_block(block_index)
        if register_chain_hashes:
            paged.cached_block_hash_to_block.insert(block_hash, block)
        parent_hash = block_hash

    block_ids = [block.block_id for block in blocks]
    assert paged.release_request_refs(
        BlockTable(
            request_id="dsv4-pending-tail-seed",
            block_ids=block_ids,
            num_tokens=len(matched_tokens),
        )
    ) == 9
    assert [block.ref_count for block in blocks] == [0] * 9

    return cache, paged, request_tokens, matched_tokens, blocks


def _assert_dsv4_pending_tail_fetch(
    *,
    cache,
    request_id: str,
    request_tokens,
    blocks,
):
    block_table, remaining = cache.fetch_cache(request_id, request_tokens)

    assert block_table is not None
    assert block_table.block_ids == [block.block_id for block in blocks[:8]]
    assert block_table.num_tokens == 2048
    assert block_table.matched_tokens == 2304
    assert block_table.checkpoint_tokens == 2048
    assert block_table.replayed_tokens == 256
    assert remaining == request_tokens[2048:]
    assert len(remaining) == 258
    assert [block.ref_count for block in blocks[:8]] == [1] * 8
    assert blocks[8].ref_count == 0
    assert cache._hits == 1
    assert cache._tokens_saved == 2048
    assert cache._hit_credits[request_id] == 2048
    assert cache._request_tables[request_id].block_table is block_table


def test_dsv4_chain_hash_fetch_replays_pending_tail_from_periodic_anchor():
    """The authoritative chain path normalizes before the mixed-SWA guard."""
    cache, _paged, request_tokens, _matched_tokens, blocks = (
        _dsv4_pending_tail_fetch_fixture(register_chain_hashes=True)
    )

    _assert_dsv4_pending_tail_fetch(
        cache=cache,
        request_id="dsv4-chain-pending-tail",
        request_tokens=request_tokens,
        blocks=blocks,
    )


def test_dsv4_prefix_index_fetch_replays_pending_tail_from_periodic_anchor():
    """The fallback index path applies the same DSV4 checkpoint accounting."""
    cache, _paged, request_tokens, matched_tokens, blocks = (
        _dsv4_pending_tail_fetch_fixture(register_chain_hashes=False)
    )
    cache._prefix_index[cache._prefix_index_hash(matched_tokens)] = (
        matched_tokens,
        [block.block_id for block in blocks],
        None,
    )

    _assert_dsv4_pending_tail_fetch(
        cache=cache,
        request_id="dsv4-prefix-index-pending-tail",
        request_tokens=request_tokens,
        blocks=blocks,
    )


def test_dsv4_exact_n_minus_one_match_can_use_terminal_anchor():
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
    cache._expected_num_layers = 43
    blocks = [
        SimpleNamespace(token_count=256, cache_data=_topology_block(i, terminal=i == 8))
        for i in range(9)
    ]
    request_tokens = list(range(2304)) + [90_003]

    kept, checkpoint, replayed = cache._normalize_dsv4_delta_candidate(
        request_id="dsv4-exact-terminal",
        blocks=blocks,
        matched_tokens=2304,
        request_tokens=request_tokens,
        disk_store=None,
    )

    assert len(kept) == 9
    assert checkpoint == 2304
    assert replayed == 0


class _ControlledDiskAdmission:
    def __init__(self, *, admitted: bool, readable: bool = False):
        self.admitted = admitted
        self.readable = readable
        self.write_calls = 0

    def has_block(self, _block_hash):
        return self.readable

    def write_block_async(self, *_args, **_kwargs):
        self.write_calls += 1
        return self.admitted


class _FenceResultDisk:
    def __init__(self, *, ready=(), wait_error: Exception | None = None):
        self.ready = set(ready)
        self.wait_error = wait_error
        self.calls = []

    def seal_write_fence(self, fence_id, *, producer_aborted=False):
        self.calls.append(("seal", fence_id, producer_aborted))
        return True

    def wait_for_write_fence_blocks(
        self,
        fence_id,
        hashes,
        *,
        timeout,
        allow_partial=False,
    ):
        self.calls.append(
            ("wait", fence_id, set(hashes), timeout, allow_partial)
        )
        if self.wait_error is not None:
            raise self.wait_error
        return set(hashes) & self.ready


class _NativeHoldPaged:
    def __init__(self):
        self.noted = []
        self.evictable = []
        self.released = []
        self.enforced = 0

    @staticmethod
    def estimate_block_nbytes(_payload):
        return 4096

    def _note_resident(self, block, nbytes):
        self.noted.append((block, nbytes))

    def make_resident_payload_evictable(self, block):
        block.keep_resident = False
        self.evictable.append(block)

    def release_resident_payload(self, block):
        block.cache_data = None
        block.cache_data_from_disk = False
        block.keep_resident = False
        self.released.append(block)

    def enforce_byte_budget(self):
        self.enforced += 1


def test_dsv4_disk_only_fallback_survives_fence_wait_error():
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
    cache.paged_cache = _NativeHoldPaged()
    disk = _FenceResultDisk(wait_error=RuntimeError("forced fence error"))
    block_hash = b"f" * 32
    block = SimpleNamespace(
        cache_data=None,
        cache_data_from_disk=False,
        keep_resident=False,
    )
    payload = _topology_block(7, terminal=True)
    fence = {
        "disk_store": disk,
        "fence_id": "dsv4-fence-error",
        "disk_only_fallbacks": {block_hash: (block, payload)},
    }

    cache._settle_native_write_fence("dsv4-fence-error", fence)

    assert fence["sealed"] is True
    assert [call[0] for call in disk.calls] == ["seal", "wait"]
    assert block.cache_data is payload
    assert block.cache_data_from_disk is False
    assert block.keep_resident is True
    assert cache.paged_cache.noted == [(block, 4096)]


def test_dsv4_native_holds_release_only_for_post_eviction_exact_hashes():
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
    cache.paged_cache = _NativeHoldPaged()
    disk_only_hash = b"d" * 32
    paged_hash = b"p" * 32
    lost_hash = b"l" * 32
    disk = _FenceResultDisk(ready={disk_only_hash, paged_hash})
    disk_only_block = SimpleNamespace(
        cache_data=None,
        cache_data_from_disk=False,
        keep_resident=False,
    )
    paged_block = SimpleNamespace(keep_resident=True)
    lost_block = SimpleNamespace(keep_resident=True)
    payload = _topology_block(8, terminal=True)
    fence = {
        "disk_store": disk,
        "fence_id": "dsv4-fence-retained",
        "disk_only_fallbacks": {
            disk_only_hash: (disk_only_block, payload),
        },
        "native_paged_holds": {
            paged_hash: paged_block,
            lost_hash: lost_block,
        },
    }

    cache._settle_native_write_fence("dsv4-fence-retained", fence)

    assert disk_only_block.cache_data is None
    assert cache.paged_cache.noted == []
    assert paged_block.keep_resident is False
    assert lost_block.keep_resident is True
    assert cache.paged_cache.evictable == [paged_block]


def test_dsv4_disk_only_timeout_releases_fallback_after_eventual_fence_completion():
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    ready = threading.Event()
    block_hash = b"t" * 32

    class _EventuallyReadyDisk:
        def __init__(self):
            self.waits = 0

        @staticmethod
        def seal_write_fence(_fence_id, *, producer_aborted=False):
            assert producer_aborted is False
            return True

        def wait_for_write_fence_blocks(
            self,
            _fence_id,
            hashes,
            *,
            timeout,
            allow_partial=False,
        ):
            assert allow_partial is True
            self.waits += 1
            if self.waits == 1:
                return set()
            assert timeout is None
            assert ready.wait(timeout=2.0)
            return set(hashes)

    cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
    cache.paged_cache = _NativeHoldPaged()
    disk = _EventuallyReadyDisk()
    payload = _topology_block(9, terminal=True)
    block = SimpleNamespace(
        block_hash=block_hash,
        cache_data=None,
        cache_data_from_disk=False,
        keep_resident=False,
    )
    fence = {
        "disk_store": disk,
        "fence_id": "dsv4-fence-eventual",
        "disk_only_fallbacks": {block_hash: (block, payload)},
    }

    cache._settle_native_write_fence("dsv4-fence-eventual", fence)
    assert block.cache_data is payload
    assert block.keep_resident is True

    ready.set()
    deadline = time.monotonic() + 2.0
    while block.cache_data is not None and time.monotonic() < deadline:
        time.sleep(0.005)

    assert disk.waits == 2
    assert block.cache_data is None
    assert block.keep_resident is False
    assert cache.paged_cache.released == [block]


def _register_free_native_block(manager, block_id=1):
    block = manager.blocks[block_id]
    block_hash = bytes([block_id]) * 32
    block.block_hash = block_hash
    block.cache_data = _topology_block(0, terminal=True)
    block.token_count = 256
    block.keep_resident = True
    manager.cached_block_hash_to_block.insert(block_hash, block)
    return block


def test_native_l2_admission_failure_skips_block_without_losing_ancestry():
    from vmlx_engine.paged_cache import PagedCacheManager

    disk = _ControlledDiskAdmission(admitted=False)
    manager = PagedCacheManager(
        block_size=256,
        max_blocks=3,
        disk_store=disk,
    )
    protected = _register_free_native_block(manager)

    allocated = manager.allocate_block()

    assert allocated is manager.blocks[2]
    assert protected.block_hash == b"\x01" * 32
    assert protected.cache_data is not None
    assert protected.keep_resident is True
    assert manager.cached_block_hash_to_block.get_block(protected.block_hash) is protected
    assert disk.write_calls == 1


def test_native_l2_saturation_applies_bounded_allocation_backpressure():
    from vmlx_engine.paged_cache import PagedCacheManager

    disk = _ControlledDiskAdmission(admitted=False)
    manager = PagedCacheManager(
        block_size=256,
        max_blocks=2,
        disk_store=disk,
    )
    protected = _register_free_native_block(manager)

    assert manager.allocate_block() is None
    assert manager.free_block_queue.num_free_blocks == 1
    assert protected.cache_data is not None
    assert protected.block_hash == b"\x01" * 32
    assert disk.write_calls == 1


def test_native_queued_write_is_recycled_only_after_l2_is_readable():
    from vmlx_engine.paged_cache import PagedCacheManager

    disk = _ControlledDiskAdmission(admitted=True)
    manager = PagedCacheManager(
        block_size=256,
        max_blocks=2,
        disk_store=disk,
    )
    protected = _register_free_native_block(manager)

    assert manager.allocate_block() is None
    assert protected.durability_write_pending is True
    assert disk.write_calls == 1
    # A second allocation during the bounded publication grace does not queue
    # another detached copy or recycle the only native checkpoint.
    assert manager.allocate_block() is None
    assert disk.write_calls == 1

    disk.readable = True
    allocated = manager.allocate_block()

    assert allocated is protected
    assert protected.block_hash is None
    assert protected.cache_data is None
    assert protected.keep_resident is False


def test_dsv4_validator_rejects_wrapper_class_and_pool_span_mismatch():
    from vmlx_engine.cache_record_validator import validate_cache_record

    class_bad = _topology_block(0, terminal=True)
    class_bad[2] = (
        class_bad[2][0],
        class_bad[2][1],
        "PoolQuantizedV4Cache",
        class_bad[2][3],
    )
    ok, reason, _ = validate_cache_record(
        class_bad,
        expected_num_layers=43,
        source="dsv4-class-mutation",
    )
    assert ok is False
    assert "cache class mismatch" in reason

    span_bad = _topology_block(0, terminal=True)
    span_bad[2][1]["compressor_pool"]["start_row"] = 1
    ok, reason, _ = validate_cache_record(
        span_bad,
        expected_num_layers=43,
        source="dsv4-span-mutation",
    )
    assert ok is False
    assert "row span does not match token geometry" in reason


def test_dsv4_validator_rejects_rotating_anchor_dimension_mismatch():
    from vmlx_engine.cache_record_validator import validate_cache_record

    malformed = _topology_block(0, terminal=True)
    entry = malformed[2]
    anchor = entry[1]["anchor"]
    anchor["local_state"] = (
        anchor["local_state"][0],
        mx.zeros((1, 1, 128, 2), dtype=mx.float32),
    )

    ok, reason, _ = validate_cache_record(
        malformed,
        expected_num_layers=43,
        source="dsv4-anchor-mutation",
    )

    assert ok is False
    assert "local K/V geometry" in reason


def test_dsv4_validator_rejects_append_safe_marker_on_partial_terminal():
    from vmlx_engine.cache_record_validator import validate_cache_record

    malformed = _topology_interval(256, 331, terminal=True)
    for entry in malformed:
        if entry[0] == "deepseek_v4_delta_v1":
            entry[1]["anchor"]["append_safe"] = True

    ok, reason, _ = validate_cache_record(
        malformed,
        expected_num_layers=43,
        source="dsv4-partial-append-safe-mutation",
    )

    assert ok is False
    assert "append-safe anchor is misaligned" in reason


def test_dsv4_reconstruct_rejects_chain_metadata_change():
    from vmlx_engine.paged_cache import BlockTable, PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    paged = PagedCacheManager(block_size=256, max_blocks=16)
    cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged)
    table = BlockTable(request_id="dsv4-chain-metadata-mutation")
    for block_index in range(8):
        block = paged.allocate_block()
        assert block is not None
        block.token_count = 256
        block.cache_data = _topology_block(block_index)
        if block_index == 1:
            block.cache_data[2][1]["sliding_window"] = 64
            block.cache_data[2][3]["sliding_window"] = 64
        table.block_ids.append(block.block_id)
        table.num_tokens += 256

    assert cache.reconstruct_cache(table) is None
