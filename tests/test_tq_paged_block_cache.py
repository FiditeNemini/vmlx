import mlx.core as mx
import time


class _TinyQwenModel:
    class Args:
        num_attention_heads = 2
        num_key_value_heads = 2
        head_dim = 64
        kv_lora_rank = 0

    args = Args()


def _tq_state(tokens=32, seed=73):
    return {
        "state": (
            mx.random.normal(shape=(1, 2, tokens, 64)).astype(mx.float16),
            mx.random.normal(shape=(1, 2, tokens, 64)).astype(mx.float16),
        ),
        "meta_state": (str(tokens),),
        "class_name": "TurboQuantKVCache",
        "tq_config": {
            "key_bits": 8,
            "value_bits": 8,
            "seed": seed,
        },
    }


def test_tq_paged_blocks_encode_each_slice_and_reconstruct():
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache, PagedCacheManager
    from vmlx_engine.tq_disk_store import decode_tq_block

    manager = PagedCacheManager(block_size=16, max_blocks=16)
    cache = BlockAwarePrefixCache(
        model=_TinyQwenModel(),
        paged_cache_manager=manager,
    )
    tokens = list(range(32))
    cache.store_cache("write", tokens, [_tq_state()])

    table, remaining = cache.fetch_cache("read", tokens)
    assert table is not None
    assert remaining == []
    entries = [
        manager.allocated_blocks[block_id].cache_data[0]
        for block_id in table.block_ids
    ]
    assert [entry[0] for entry in entries] == ["turboquant_kv", "turboquant_kv"]
    assert [entry[3]["seed"] for entry in entries] == [73, 73]
    assert [entry[3]["offset"] for entry in entries] == [16, 16]

    expected_keys = []
    expected_values = []
    for entry in entries:
        keys, values = decode_tq_block(entry)
        expected_keys.append(keys)
        expected_values.append(values)
    expected_keys = mx.concatenate(expected_keys, axis=2)
    expected_values = mx.concatenate(expected_values, axis=2)

    restored = cache.reconstruct_cache(table)
    assert restored is not None
    assert len(restored) == 1
    assert cache._last_reconstruct_tq_blocks == 2
    mx.eval(restored[0].keys, restored[0].values, expected_keys, expected_values)
    assert float(mx.max(mx.abs(restored[0].keys - expected_keys)).item()) == 0.0
    assert float(mx.max(mx.abs(restored[0].values - expected_values)).item()) == 0.0


def test_tq_block_safetensors_record_preserves_seed_and_decodes():
    from vmlx_engine.block_disk_store import _deserialize_block, _serialize_block
    from vmlx_engine.cache_record_validator import validate_cache_record
    from vmlx_engine.tq_disk_store import decode_tq_block, encode_tq_block

    state = _tq_state(tokens=16, seed=91)
    original = encode_tq_block(
        state["state"][0],
        state["state"][1],
        state["tq_config"],
    )
    tensors, dtype, layers = _serialize_block([original])
    assert dtype == "turboquant_kv"
    assert layers == 1
    restored_entries = _deserialize_block(dict(tensors), dtype)
    ok, reason, _ = validate_cache_record(
        restored_entries,
        expected_num_layers=1,
        source="unit-tq-block",
    )
    assert ok, reason
    restored = restored_entries[0]
    assert restored[0] == "turboquant_kv"
    assert restored[3]["seed"] == 91
    assert restored[3]["key_bits"] == 8
    assert restored[3]["value_bits"] == 8
    original_keys, original_values = decode_tq_block(original)
    restored_keys, restored_values = decode_tq_block(restored)
    mx.eval(original_keys, original_values, restored_keys, restored_values)
    assert float(mx.max(mx.abs(restored_keys - original_keys)).item()) == 0.0
    assert float(mx.max(mx.abs(restored_values - original_values)).item()) == 0.0


def test_tq_block_validator_rejects_wrong_seed_or_offset():
    from vmlx_engine.cache_record_validator import validate_cache_record
    from vmlx_engine.tq_disk_store import encode_tq_block

    state = _tq_state(tokens=8, seed=101)
    entry = encode_tq_block(
        state["state"][0],
        state["state"][1],
        state["tq_config"],
    )
    bad_seed = (entry[0], entry[1], entry[2], {**entry[3], "seed": -1})
    ok, reason, _ = validate_cache_record([bad_seed], source="bad-seed")
    assert not ok
    assert "seed" in reason

    bad_offset = (entry[0], entry[1], entry[2], {**entry[3], "offset": 7})
    ok, reason, _ = validate_cache_record([bad_offset], source="bad-offset")
    assert not ok
    assert "shape/config mismatch" in reason


def test_nested_cache_list_tq_block_roundtrip_preserves_seed():
    from vmlx_engine.block_disk_store import _deserialize_block, _serialize_block
    from vmlx_engine.cache_record_validator import validate_cache_record
    from vmlx_engine.tq_disk_store import decode_tq_block, encode_tq_block

    state = _tq_state(tokens=8, seed=137)
    tq_entry = encode_tq_block(
        state["state"][0],
        state["state"][1],
        state["tq_config"],
    )
    tensors, dtype, layers = _serialize_block(
        [("cache_list", [tq_entry, ("skip",)])]
    )
    assert dtype == "cache_list"
    assert layers == 1
    restored = _deserialize_block(dict(tensors), dtype)
    ok, reason, _ = validate_cache_record(
        restored,
        expected_num_layers=1,
        source="unit-cache-list-tq",
    )
    assert ok, reason
    nested_tq = restored[0][1][0]
    assert nested_tq[0] == "turboquant_kv"
    assert nested_tq[3]["seed"] == 137
    keys, values = decode_tq_block(nested_tq)
    mx.eval(keys, values)
    assert keys.shape[-2] == 8
    assert values.shape[-2] == 8


def test_tq_paged_numpy_disk_path_keeps_native_entries(tmp_path):
    from vmlx_engine.block_disk_store import BlockDiskStore
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache, PagedCacheManager

    store = BlockDiskStore(
        str(tmp_path), max_size_gb=0.1, expected_num_layers=1
    )
    manager = PagedCacheManager(block_size=16, max_blocks=16)
    manager._disk_store = store
    cache = BlockAwarePrefixCache(
        model=_TinyQwenModel(),
        paged_cache_manager=manager,
        tq_enabled=True,
    )
    try:
        tokens = list(range(32))
        cache.store_cache("write", tokens, [_tq_state(tokens=32, seed=149)])
        for _ in range(50):
            stats = store.get_stats()
            if stats["blocks_on_disk"] >= 2:
                break
            time.sleep(0.1)

        stats = store.get_stats()
        assert stats["blocks_on_disk"] == 2
        assert stats["tq_native_writes"] == 2
        table = cache._request_tables["write"].block_table
        for block_id in table.block_ids:
            block = manager.allocated_blocks[block_id]
            restored = store.read_block(block.block_hash)
            assert restored is not None
            assert restored[0][0] == "turboquant_kv"
            assert restored[0][3]["seed"] == 149
    finally:
        store.shutdown()
