# SPDX-License-Identifier: Apache-2.0
"""The prefix producer must keep native affine KV packed through SSD restart.

Codec-only tests cannot catch store_cache replacing a quantized_kv entry with
dense kv after constructing its NumPy mirror. These tests enter that producer
with the installed mlx-lm cache class and retain a partial final block.
"""

import json
import struct

import pytest

mx = pytest.importorskip("mlx.core")
from mlx_lm.models.cache import QuantizedKVCache

from vmlx_engine.block_disk_store import BlockDiskStore
from vmlx_engine.paged_cache import PagedCacheManager
from vmlx_engine.prefix_cache import BlockAwarePrefixCache


def _snapshot(bits, dtype):
    native = QuantizedKVCache(group_size=32, bits=bits)
    keys = (mx.arange(7 * 64).reshape(1, 1, 7, 64) / 37 - 2).astype(dtype)
    values = (keys * 0.75 + 1).astype(dtype)
    native.update_and_fetch(keys, values)
    state = native.state
    mx.eval(*state[0], *state[1])
    assert native.offset == 7
    assert all(array.shape[-2] == 7 for side in state for array in side)
    return native, [{
        "class_name": type(native).__name__,
        "state": state,
        "meta_state": native.meta_state,
    }]


def _open_cache(path):
    store = BlockDiskStore(str(path), max_size_gb=0.01, expected_num_layers=1)
    manager = PagedCacheManager(
        block_size=4,
        max_blocks=8,
        disk_store=store,
        max_resident_bytes=0,
        disk_only=True,
    )
    return store, manager, BlockAwarePrefixCache(
        model=None, paged_cache_manager=manager,
    )


def _assert_exact(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert bool(mx.array_equal(actual.view(mx.uint8), expected.view(mx.uint8)))


def _payload_bytes(path):
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    return sum(
        entry["data_offsets"][1] - entry["data_offsets"][0]
        for name, entry in header.items()
        if name not in {"__metadata__", "__vmlx_block_meta__"}
    )


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_quantized_prefix_preserves_packed_disk_payload_after_restart(
    tmp_path, bits, dtype,
):
    native, snapshot = _snapshot(bits, dtype)
    expected_keys, expected_values = native.state
    tokens = list(range(7))
    cache_dir = tmp_path / "packed-prefix"
    writer, manager, cache = _open_cache(cache_dir)
    try:
        table = cache.store_cache("writer", tokens, snapshot)
        assert table is not None
        assert table.num_tokens == 7
        blocks = [manager.allocated_blocks[i] for i in table.block_ids]
        assert [block.token_count for block in blocks] == [4, 3]
        hashes = [block.block_hash for block in blocks]
        assert all(hashes)
        assert writer.wait_for_blocks(hashes, timeout=5.0) == set(hashes)
        assert manager.resident_bytes == 0
        assert all(block.cache_data is None for block in blocks)
    finally:
        writer.shutdown()

    # No live writer/cache objects are used to obtain the restored payload.
    reader, restarted_manager, restarted = _open_cache(cache_dir)
    try:
        for block_hash, start, end in zip(hashes, (0, 4), (4, 7)):
            payload = reader.read_block(block_hash)
            assert payload is not None and len(payload) == 1
            entry = payload[0]
            assert entry[0] == "quantized_kv", (
                "prefix producer expanded packed KV before disk serialization"
            )
            assert tuple(entry[3]) == native.meta_state
            assert len(entry[1]) == len(entry[2]) == 3
            expected = tuple(
                array[..., start:end, :]
                for side in (expected_keys, expected_values) for array in side
            )
            for actual, original in zip(tuple(entry[1]) + tuple(entry[2]), expected):
                _assert_exact(actual, original)
            path = reader._hash_to_path(block_hash.hex())
            assert _payload_bytes(path) == sum(array.nbytes for array in expected)

        assert len(list((cache_dir / "blocks").rglob("*.safetensors"))) == 2
        hit, remaining = restarted.fetch_cache("reader", tokens + [99])
        assert hit is not None and hit.num_tokens == 7
        assert remaining == [99]
        rebuilt = restarted.reconstruct_cache(hit)
        assert rebuilt is not None and len(rebuilt) == 1
        restored = rebuilt[0]
        assert isinstance(restored, QuantizedKVCache)
        assert restored.offset == 7
        assert restored.group_size == 32
        assert restored.bits == bits
        assert restored.meta_state == native.meta_state
        actual_keys, actual_values = restored.state
        for actual, original in zip(
            tuple(actual_keys) + tuple(actual_values),
            tuple(expected_keys) + tuple(expected_values),
        ):
            _assert_exact(actual, original)
        assert restarted_manager.resident_bytes == 0
        assert all(
            restarted_manager.allocated_blocks[i].cache_data is None
            for i in hit.block_ids
        )
    finally:
        reader.shutdown()


def test_quantized_prefix_store_does_not_dequantize_whole_layer(tmp_path, monkeypatch):
    # Install the spy only after native quantization/evaluation has completed.
    # Delegate normally: no synthetic exception/fallback changes the store path.
    _native, snapshot = _snapshot(4, mx.bfloat16)
    original = mx.dequantize
    calls = []

    def record_dequantize(*args, **kwargs):
        calls.append((args[0].shape, kwargs.copy()))
        return original(*args, **kwargs)

    monkeypatch.setattr(mx, "dequantize", record_dequantize)
    store, manager, cache = _open_cache(tmp_path / "no-dense-mirror")
    try:
        table = cache.store_cache("writer", list(range(7)), snapshot)
        assert table is not None
        hashes = [manager.allocated_blocks[i].block_hash for i in table.block_ids]
        assert store.wait_for_blocks(hashes, timeout=5.0) == set(hashes)
    finally:
        store.shutdown()
    assert calls == [], f"packed prefix storage dequantized native KV: {calls}"


@pytest.mark.parametrize("bits", [4, 8])
def test_nested_quantized_prefix_keeps_type_after_restart(tmp_path, bits):
    from mlx_lm.models.cache import CacheList

    native, snapshot = _snapshot(bits, mx.bfloat16)
    nested = [{"class_name": "CacheList", "sub_caches": snapshot, "state": (native.state,)}]
    tokens = list(range(7))
    path = tmp_path / "nested-prefix"
    writer, manager, cache = _open_cache(path)
    try:
        table = cache.store_cache("nested-writer", tokens, nested)
        assert table is not None and table.num_tokens == 7
        hashes = [manager.allocated_blocks[i].block_hash for i in table.block_ids]
        assert writer.wait_for_blocks(hashes, timeout=5.0) == set(hashes)
    finally:
        writer.shutdown()
    reader, _manager, cache = _open_cache(path)
    try:
        hit, suffix = cache.fetch_cache("nested-reader", tokens + [99])
        assert hit is not None and hit.num_tokens == 7 and suffix == [99]
        rebuilt = cache.reconstruct_cache(hit)
        assert rebuilt is not None and isinstance(rebuilt[0], CacheList)
        restored = rebuilt[0].caches[0]
        assert isinstance(restored, QuantizedKVCache)
        assert restored.meta_state == native.meta_state
        for actual, expected in zip(
            tuple(restored.state[0]) + tuple(restored.state[1]),
            tuple(native.state[0]) + tuple(native.state[1]),
        ):
            _assert_exact(actual, expected)
    finally:
        reader.shutdown()
