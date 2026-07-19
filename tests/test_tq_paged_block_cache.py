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


def test_scheduler_prompt_boundary_truncation_preserves_tq_policy():
    from jang_tools.turboquant.cache import TurboQuantKVCache
    from vmlx_engine.scheduler import Scheduler

    cache = TurboQuantKVCache(
        key_dim=64,
        value_dim=64,
        key_bits=3,
        value_bits=4,
        seed=211,
        compress_after=0,
        sink_tokens=2,
    )
    cache.keys = mx.random.normal(shape=(1, 2, 10, 64)).astype(mx.float16)
    cache.values = mx.random.normal(shape=(1, 2, 10, 64)).astype(mx.float16)
    cache.offset = 10

    truncated = Scheduler._truncate_cache_to_prompt_length([cache], prompt_len=7)

    assert truncated is not None
    assert len(truncated) == 1
    restored = truncated[0]
    assert type(restored).__name__ == "TurboQuantKVCache"
    assert restored.key_bits == 3
    assert restored.value_bits == 4
    assert restored._seed == 211
    assert restored.sink_tokens == 2
    assert restored.offset == 6
    keys, values = restored.state
    assert keys.shape == (1, 2, 6, 64)
    assert values.shape == (1, 2, 6, 64)


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
    assert [entry[3]["key_dtype"] for entry in entries] == ["float16", "float16"]
    assert [entry[3]["value_dtype"] for entry in entries] == ["float16", "float16"]

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
    assert restored[0].keys.dtype == mx.float16
    assert restored[0].values.dtype == mx.float16
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
    assert restored[3]["key_dtype"] == "float16"
    assert restored[3]["value_dtype"] == "float16"
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

    bad_dtype = (entry[0], entry[1], entry[2], {**entry[3], "key_dtype": "float64"})
    ok, reason, _ = validate_cache_record([bad_dtype], source="bad-dtype")
    assert not ok
    assert "key_dtype" in reason


def test_tq_block_decode_restores_bfloat16_attention_dtype():
    from vmlx_engine.tq_disk_store import decode_tq_block, encode_tq_block

    keys = mx.random.normal(shape=(1, 2, 8, 64)).astype(mx.bfloat16)
    values = mx.random.normal(shape=(1, 2, 8, 64)).astype(mx.bfloat16)
    entry = encode_tq_block(
        keys,
        values,
        {"key_bits": 8, "value_bits": 8, "seed": 113},
    )
    assert entry[3]["key_dtype"] == "bfloat16"
    assert entry[3]["value_dtype"] == "bfloat16"
    restored_keys, restored_values = decode_tq_block(entry)
    mx.eval(restored_keys, restored_values)
    assert restored_keys.dtype == mx.bfloat16
    assert restored_values.dtype == mx.bfloat16


def test_tq_block_decode_reuses_bounded_decoder_state():
    from vmlx_engine import tq_disk_store

    tq_disk_store._tq_decoder_pair.cache_clear()
    first = _tq_state(tokens=8, seed=127)
    second = _tq_state(tokens=16, seed=127)
    entries = [
        tq_disk_store.encode_tq_block(
            state["state"][0],
            state["state"][1],
            state["tq_config"],
        )
        for state in (first, second)
    ]

    # Measure reconstruction reuse independently of the shared encoder-state
    # reuse exercised while the two entries were serialized above.
    tq_disk_store._tq_decoder_pair.cache_clear()
    decoded = [tq_disk_store.decode_tq_block(entry) for entry in entries]
    mx.eval(*(array for pair in decoded for array in pair))
    info = tq_disk_store._tq_decoder_pair.cache_info()

    assert info.maxsize == 256
    assert info.currsize == 1
    assert info.misses == 1
    assert info.hits == 1
    assert decoded[0][0].shape[-2] == 8
    assert decoded[1][0].shape[-2] == 16


def test_tq_decoder_startup_warmup_materializes_distinct_layer_seeds():
    from jang_tools.turboquant.cache import TurboQuantKVCache
    from vmlx_engine import tq_disk_store

    caches = [
        TurboQuantKVCache(
            key_dim=64,
            value_dim=64,
            key_bits=4,
            value_bits=4,
            seed=seed,
            compress_after=0,
            sink_tokens=0,
        )
        for seed in (211, 212, 213)
    ]
    tq_disk_store._tq_decoder_pair.cache_clear()

    stats = tq_disk_store.warm_tq_decoder_states(caches, probe_heads=2)
    info = tq_disk_store._tq_decoder_pair.cache_info()

    assert stats["configs"] == 3
    assert stats["arrays"] > 0
    assert stats["bytes"] > 0
    assert stats["codec_probes"] == 3
    assert stats["probe_tokens"] == 64
    assert stats["probe_heads"] == 2
    assert info.maxsize == 256
    assert info.currsize == 3


def test_mllm_tq_decoder_warmup_resolves_language_model_owner(monkeypatch):
    from types import SimpleNamespace

    from vmlx_engine import tq_disk_store
    from vmlx_engine.mllm_scheduler import MLLMScheduler

    calls = {}

    def fake_warm(cache_layers, *, probe_heads):
        calls["layers"] = cache_layers
        calls["probe_heads"] = probe_heads
        return {
            "configs": 2,
            "arrays": 4,
            "bytes": 128,
            "probe_tokens": 64,
            "probe_heads": probe_heads,
            "codec_probes": 2,
        }

    monkeypatch.setattr(tq_disk_store, "warm_tq_decoder_states", fake_warm)

    def _hybrid_turboquant_make_cache():
        return ["tq-slot", "ssm-slot"]

    owner = SimpleNamespace(
        args=SimpleNamespace(num_key_value_heads=8),
        make_cache=_hybrid_turboquant_make_cache,
    )
    scheduler = object.__new__(MLLMScheduler)
    scheduler.model = SimpleNamespace(language_model=owner)
    scheduler.config = SimpleNamespace(enable_prefix_cache=True)
    scheduler._tq_active = True
    scheduler._tq_decoder_warmup_stats = None

    stats = scheduler.warm_tq_storage_decoders()

    assert stats["enabled"] is True
    assert stats["configs"] == 2
    assert calls == {
        "layers": ["tq-slot", "ssm-slot"],
        "probe_heads": 8,
    }
    assert scheduler._tq_decoder_warmup_stats == stats


def test_tq_block_storage_encode_does_not_call_live_cache_compress(monkeypatch):
    from jang_tools.turboquant.cache import TurboQuantKVCache
    from vmlx_engine import tq_disk_store

    state = _tq_state(tokens=8, seed=129)
    state["tq_config"].update(key_bits=4, value_bits=4)
    tq_disk_store._tq_decoder_pair.cache_clear()

    def fail_live_compress(*args, **kwargs):
        raise AssertionError("disk block encoding must not call live cache compress()")

    monkeypatch.setattr(TurboQuantKVCache, "compress", fail_live_compress)
    entry = tq_disk_store.encode_tq_block(
        state["state"][0],
        state["state"][1],
        state["tq_config"],
    )
    keys, values = tq_disk_store.decode_tq_block(entry)
    mx.eval(keys, values)

    assert entry[0] == "turboquant_kv"
    assert entry[3]["key_bits"] == 4
    assert entry[3]["value_bits"] == 4
    assert keys.shape == state["state"][0].shape
    assert values.shape == state["state"][1].shape


def test_tq_block_batch_decode_matches_individual_q4_pages_exactly():
    from vmlx_engine import tq_disk_store

    states = [_tq_state(tokens=8, seed=131) for _ in range(3)]
    for state in states:
        state["tq_config"].update(key_bits=4, value_bits=4)
    entries = [
        tq_disk_store.encode_tq_block(
            state["state"][0],
            state["state"][1],
            state["tq_config"],
        )
        for state in states
    ]
    individual = [tq_disk_store.decode_tq_block(entry) for entry in entries]
    expected_keys = mx.concatenate([pair[0] for pair in individual], axis=2)
    expected_values = mx.concatenate([pair[1] for pair in individual], axis=2)

    tq_disk_store._tq_decoder_pair.cache_clear()
    keys, values = tq_disk_store.decode_tq_blocks(entries)
    mx.eval(keys, values, expected_keys, expected_values)
    info = tq_disk_store._tq_decoder_pair.cache_info()

    assert keys.shape[-2] == 24
    assert values.shape[-2] == 24
    assert float(mx.max(mx.abs(keys - expected_keys)).item()) == 0.0
    assert float(mx.max(mx.abs(values - expected_values)).item()) == 0.0
    assert info.misses == 1
    assert info.hits == 0


def test_tq_block_batch_decode_preserves_partial_tail_order():
    from vmlx_engine import tq_disk_store

    states = [_tq_state(tokens=tokens, seed=137) for tokens in (8, 8, 7)]
    for state in states:
        state["tq_config"].update(key_bits=4, value_bits=4)
    entries = [
        tq_disk_store.encode_tq_block(
            state["state"][0],
            state["state"][1],
            state["tq_config"],
        )
        for state in states
    ]
    individual = [tq_disk_store.decode_tq_block(entry) for entry in entries]
    expected_keys = mx.concatenate([pair[0] for pair in individual], axis=2)
    expected_values = mx.concatenate([pair[1] for pair in individual], axis=2)

    tq_disk_store._tq_decoder_pair.cache_clear()
    keys, values = tq_disk_store.decode_tq_blocks(entries)
    mx.eval(keys, values, expected_keys, expected_values)
    info = tq_disk_store._tq_decoder_pair.cache_info()

    assert keys.shape[-2] == 23
    assert values.shape[-2] == 23
    assert float(mx.max(mx.abs(keys - expected_keys)).item()) == 0.0
    assert float(mx.max(mx.abs(values - expected_values)).item()) == 0.0
    assert info.misses == 1
    assert info.hits == 1


def test_tq_block_batch_decode_falls_back_for_mixed_codec_configs():
    from vmlx_engine import tq_disk_store

    states = [_tq_state(tokens=8, seed=seed) for seed in (139, 149)]
    entries = [
        tq_disk_store.encode_tq_block(
            state["state"][0],
            state["state"][1],
            state["tq_config"],
        )
        for state in states
    ]

    tq_disk_store._tq_decoder_pair.cache_clear()
    keys, values = tq_disk_store.decode_tq_blocks(entries)
    mx.eval(keys, values)
    info = tq_disk_store._tq_decoder_pair.cache_info()

    assert keys.shape[-2] == 16
    assert values.shape[-2] == 16
    assert info.currsize == 2
    assert info.misses == 2


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


def test_tq_paged_disk_writes_are_bounded_per_extracted_block(monkeypatch):
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache, PagedCacheManager

    events = []

    class _RecordingDiskStore:
        def write_block_async(self, block_hash, cache_data, token_count):
            events.append(("write", token_count, len(cache_data)))

    manager = PagedCacheManager(block_size=16, max_blocks=16)
    manager._disk_store = _RecordingDiskStore()
    cache = BlockAwarePrefixCache(
        model=_TinyQwenModel(),
        paged_cache_manager=manager,
        tq_enabled=True,
    )
    original_extract = cache._extract_block_tensor_slice

    def record_extract(*args, **kwargs):
        events.append(("extract", args[2] - args[1], None))
        return original_extract(*args, **kwargs)

    monkeypatch.setattr(cache, "_extract_block_tensor_slice", record_extract)
    cache.store_cache(
        "bounded-write",
        list(range(32)),
        [_tq_state(tokens=32, seed=163)],
    )

    assert [event[0] for event in events] == [
        "extract",
        "write",
        "extract",
        "write",
    ]
    assert [event[1] for event in events if event[0] == "write"] == [16, 16]


def test_tq_paged_disk_none_mode_skips_existing_native_blocks(tmp_path):
    from vmlx_engine.block_disk_store import BlockDiskStore
    from vmlx_engine.tq_disk_store import encode_tq_block

    cache_dir = str(tmp_path / "shared")
    state = _tq_state(tokens=8, seed=173)
    tq_entry = encode_tq_block(
        state["state"][0],
        state["state"][1],
        state["tq_config"],
    )
    block_hash = bytes.fromhex("17" * 32)

    writer = BlockDiskStore(cache_dir, max_size_gb=0.1, allow_tq_native=True)
    try:
        writer.write_block_async(block_hash, [tq_entry], token_count=8)
        for _ in range(50):
            if writer.get_stats()["blocks_on_disk"] == 1:
                break
            time.sleep(0.1)
        assert writer.get_stats()["tq_native_writes"] == 1
    finally:
        writer.shutdown()

    disabled = BlockDiskStore(cache_dir, max_size_gb=0.1, allow_tq_native=False)
    try:
        assert disabled.has_block(block_hash) is False
        assert disabled.read_block(block_hash) is None
        stats = disabled.get_stats()
        assert stats["tq_native_enabled"] is False
        assert stats["tq_native_hits"] == 0
        assert stats["disk_misses"] == 1
        assert stats["blocks_on_disk"] == 0
    finally:
        disabled.shutdown()

    restored = BlockDiskStore(cache_dir, max_size_gb=0.1, allow_tq_native=True)
    try:
        assert restored.read_block(block_hash) is None
        assert restored.get_stats()["tq_native_hits"] == 0
    finally:
        restored.shutdown()


def test_block_disk_store_derives_native_tq_disable_from_cli_environment(
    tmp_path, monkeypatch
):
    from vmlx_engine.block_disk_store import BlockDiskStore

    monkeypatch.setenv("VMLX_DISABLE_TQ_KV", "1")
    store = BlockDiskStore(str(tmp_path), max_size_gb=0.1)
    try:
        assert store.get_stats()["tq_native_enabled"] is False
    finally:
        store.shutdown()
