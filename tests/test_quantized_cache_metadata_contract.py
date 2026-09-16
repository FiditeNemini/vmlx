# SPDX-License-Identifier: Apache-2.0
"""Packed KV must never be restored using guessed or mixed quantization metadata."""
# MLX must be optional until importorskip has established this test environment.
# ruff: noqa: E402

import pytest

mx = pytest.importorskip("mlx.core")
from mlx_lm.models.cache import QuantizedKVCache

from vmlx_engine.cache_record_validator import validate_cache_record
from vmlx_engine.paged_cache import BlockTable, PagedCacheManager
from vmlx_engine.prefix_cache import BlockAwarePrefixCache


def _entry(bits=4, group_size=32, steps=4):
    native = QuantizedKVCache(group_size=group_size, bits=bits)
    keys = (mx.arange(steps * 64).reshape(1, 1, steps, 64) / 37 - 2).astype(mx.bfloat16)
    native.update_and_fetch(keys, keys * 0.75 + 1)
    keys, values = native.state
    mx.eval(*keys, *values)
    return ("quantized_kv", keys, values, native.meta_state)


def _reconstruct(entries):
    manager = PagedCacheManager(block_size=4, max_blocks=8)
    blocks = []
    for entry in entries:
        block = manager.allocate_block()
        assert block is not None
        block.cache_data = [entry]
        blocks.append(block.block_id)
    table = BlockTable(
        request_id="metadata", block_ids=blocks, num_tokens=4 * len(entries)
    )
    return BlockAwarePrefixCache(
        model=None, paged_cache_manager=manager
    ).reconstruct_cache(table)


@pytest.mark.parametrize(
    "meta",
    [
        None,
        (),
        ("4",),
        ("4", "32"),
        {},
        {"bits": 4},
        (4, 32, 4.5),
        (4, True, 4),
        (4, 32.5, 4),
        {"group_size": 32, "groupSize": 64, "bits": 4},
    ],
)
def test_ambiguous_quantized_metadata_is_rejected(meta):
    entry = (*_entry()[:3], meta)
    ok, reason, _ = validate_cache_record([entry], expected_num_layers=1)
    assert not ok, (
        "validator admitted packed KV without an explicit group/bits contract"
    )
    assert "meta" in reason
    assert _reconstruct([entry]) is None


def test_omitted_quantized_metadata_is_rejected():
    entry = _entry()[:3]
    assert not validate_cache_record([entry], expected_num_layers=1)[0]
    assert _reconstruct([entry]) is None


@pytest.mark.parametrize(
    "meta",
    [
        {"group_size": 32, "bits": 4},
        {"offset": 4, "groupSize": 32, "bits": 4},
        "[4, 32, 4]",
        "4, 32, 4",
    ],
)
def test_explicit_legacy_metadata_reconstructs_without_defaults(meta):
    entry = (*_entry()[:3], meta)
    assert validate_cache_record([entry], expected_num_layers=1)[0]
    rebuilt = _reconstruct([entry])
    assert rebuilt is not None
    restored = rebuilt[0]
    assert (restored.group_size, restored.bits, restored.offset) == (32, 4, 4)
    for actual, expected in zip(
        restored.state[0] + restored.state[1], entry[1] + entry[2]
    ):
        assert actual.dtype == expected.dtype
        assert bool(mx.array_equal(actual.view(mx.uint8), expected.view(mx.uint8)))


def test_packed_geometry_must_agree_with_declared_bits():
    entry = (*_entry()[:3], ("4", "32", "8"))
    assert not validate_cache_record([entry], expected_num_layers=1)[0]
    assert _reconstruct([entry]) is None


def test_codec_must_agree_across_pages_even_when_each_page_is_valid():
    # These two encodings have identical component shapes but different codecs:
    # q4/group64/width64 versus q8/group32/width32. A shape-only concat succeeds.
    first = _entry(group_size=64)
    keys = mx.ones((1, 1, 4, 32), dtype=mx.bfloat16)
    native = QuantizedKVCache(group_size=32, bits=8)
    native.update_and_fetch(keys, keys)
    state = native.state
    mx.eval(*state[0], *state[1])
    second = ("quantized_kv", state[0], state[1], native.meta_state)
    assert [a.shape for a in first[1]] == [a.shape for a in second[1]]
    assert validate_cache_record([first], expected_num_layers=1)[0]
    assert validate_cache_record([second], expected_num_layers=1)[0]
    assert _reconstruct([first, second]) is None


def test_offset_is_not_the_page_length_and_can_vary_across_pages():
    first = (*_entry()[:3], ("4", "32", "4"))
    second = (*_entry()[:3], ("8", "32", "4"))
    assert validate_cache_record([second], expected_num_layers=1)[0]
    rebuilt = _reconstruct([first, second])
    assert rebuilt is not None
    assert (rebuilt[0].group_size, rebuilt[0].bits, rebuilt[0].offset) == (32, 4, 8)


@pytest.mark.parametrize("nested", [False, True])
def test_invalid_codec_misses_before_concatenate_or_evaluate(monkeypatch, nested):
    entry = (*_entry()[:3], None)
    if nested:
        entry = ("cache_list", [entry])
    calls = []
    for name in ("concatenate", "eval", "dequantize"):
        original = getattr(mx, name)

        def record(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(mx, name, record)
    assert _reconstruct([entry]) is None
    assert not calls


@pytest.mark.parametrize("bits", [4, 8])
def test_legacy_dict_nested_cache_roundtrip_reconstructs_and_continues(tmp_path, bits):
    from mlx_lm.models.cache import CacheList

    from vmlx_engine.block_disk_store import BlockDiskStore

    entry = _entry(bits=bits)
    entry = (*entry[:3], {"group_size": 32, "bits": bits})
    original = QuantizedKVCache(group_size=32, bits=bits)
    original.keys, original.values, original.offset = entry[1], entry[2], 4
    block_hash = bytes([bits]) * 32
    writer = BlockDiskStore(str(tmp_path), max_size_gb=0.01, expected_num_layers=1)
    try:
        assert writer.write_block_async(block_hash, [("cache_list", [entry])], 4)
        assert writer.wait_for_blocks([block_hash], timeout=5) == {block_hash}
    finally:
        writer.shutdown()
    reader = BlockDiskStore(str(tmp_path), max_size_gb=0.01, expected_num_layers=1)
    try:
        payload = reader.read_block(block_hash)
        assert payload is not None
        rebuilt = _reconstruct(payload)
        assert rebuilt is not None and isinstance(rebuilt[0], CacheList)
        restored = rebuilt[0].caches[0]
        assert (restored.group_size, restored.bits) == (32, bits)
        next_keys = mx.arange(64).reshape(1, 1, 1, 64).astype(mx.bfloat16)
        for cache in (original, restored):
            cache.update_and_fetch(next_keys, next_keys / 2)
        for actual, expected in zip(
            restored.state[0] + restored.state[1], original.state[0] + original.state[1]
        ):
            assert bool(mx.array_equal(actual.view(mx.uint8), expected.view(mx.uint8)))
    finally:
        reader.shutdown()


def test_missing_metadata_on_disk_is_a_miss_not_guessed_q8(tmp_path):
    from vmlx_engine.block_disk_store import BlockDiskStore

    entry = (*_entry()[:3], None)
    block_hash = bytes([7]) * 32
    writer = BlockDiskStore(str(tmp_path), max_size_gb=0.01, expected_num_layers=1)
    try:
        assert writer.write_block_async(block_hash, [entry], 4)
        assert writer.wait_for_blocks([block_hash], timeout=5) == {block_hash}
    finally:
        writer.shutdown()
    reader = BlockDiskStore(str(tmp_path), max_size_gb=0.01, expected_num_layers=1)
    try:
        assert reader.read_block(block_hash) is None
    finally:
        reader.shutdown()


@pytest.mark.parametrize("bits", [4, 8])
def test_quantized_zaya_cca_disk_restart_preserves_logits(tmp_path, bits):
    from vmlx_engine.block_disk_store import BlockDiskStore
    from vmlx_engine.models.zaya import Model, ModelArgs
    from vmlx_engine.scheduler import Scheduler

    mx.random.seed(41)
    model = Model(
        ModelArgs(
            hidden_size=128,
            num_hidden_layers=2,
            ffn_hidden_size=128,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_query_groups=1,
            cca_num_q_heads=2,
            kv_channels=64,
            vocab_size=32,
            num_experts=2,
            zaya_mlp_expansion=8,
            max_position_embeddings=128,
        )
    )
    native = model.make_cache()
    tokens = list(range(1, 8))
    mx.eval(model(mx.array([tokens]), cache=native))
    kv, cca = native[0].caches
    native[0].caches = (kv.to_quantized(group_size=32, bits=bits), cca)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model = model
    snapshot = scheduler._extract_cache_states(native)
    # Explicit old dictionary form exercises the corrected ZAYA consumer too.
    snapshot[0]["sub_caches"][0]["meta_state"] = {"group_size": 32, "bits": bits}
    writer = BlockDiskStore(str(tmp_path), max_size_gb=0.01, expected_num_layers=2)
    manager = PagedCacheManager(
        block_size=4, max_blocks=8, disk_store=writer, disk_only=True
    )
    cache = BlockAwarePrefixCache(model=model, paged_cache_manager=manager)
    try:
        table = cache.store_cache("zaya-writer", tokens, snapshot)
        assert table is not None
        hashes = [manager.allocated_blocks[i].block_hash for i in table.block_ids]
        assert writer.wait_for_blocks(hashes, timeout=5) == set(hashes)
    finally:
        writer.shutdown()
    reader = BlockDiskStore(str(tmp_path), max_size_gb=0.01, expected_num_layers=2)
    manager = PagedCacheManager(
        block_size=4, max_blocks=8, disk_store=reader, disk_only=True
    )
    cache = BlockAwarePrefixCache(model=model, paged_cache_manager=manager)
    try:
        table, suffix = cache.fetch_cache("zaya-reader", tokens + [8])
        assert table is not None and suffix == [8]
        rebuilt = cache.reconstruct_cache(table)
        assert rebuilt is not None
        restored = rebuilt[0].caches[0]
        assert (restored.group_size, restored.bits, restored.offset) == (32, bits, 7)
        for actual, expected in zip(
            restored.state[0] + restored.state[1],
            native[0].caches[0].state[0] + native[0].caches[0].state[1],
        ):
            assert bool(mx.array_equal(actual.view(mx.uint8), expected.view(mx.uint8)))
        actual = model(mx.array([[8]]), cache=rebuilt)
        expected = model(mx.array([[8]]), cache=native)
        mx.eval(actual, expected)
        assert bool(mx.array_equal(actual, expected))
    finally:
        reader.shutdown()


@pytest.mark.parametrize(
    "other_tag", ["kv", "rotating_kv", "minimax_m3", "cache_list", "skip"]
)
@pytest.mark.parametrize("reverse", [False, True])
def test_affine_slot_rejects_other_representation_before_concat(
    monkeypatch, other_tag, reverse
):
    entry = _entry()
    k = mx.ones((1, 1, 4, 64), dtype=mx.bfloat16)
    if other_tag == "cache_list":
        other = (other_tag, [entry])
    elif other_tag == "skip":
        other = (other_tag,)
    elif other_tag == "minimax_m3":
        other = (other_tag, k, k, None)
    else:
        other = (other_tag, k, k)
    assert validate_cache_record([other])[0]
    entries = [entry, other] if not reverse else [other, entry]
    calls = []
    original = mx.concatenate

    def record(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(mx, "concatenate", record)
    assert _reconstruct(entries) is None
    assert calls == []


def test_plain_kv_does_not_build_quantized_chain_layouts(monkeypatch):
    from vmlx_engine import cache_record_validator

    calls = []
    original = cache_record_validator.validate_quantized_cache_chain

    def record(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        cache_record_validator, "validate_quantized_cache_chain", record
    )
    keys = mx.ones((1, 1, 4, 64), dtype=mx.bfloat16)
    assert _reconstruct([("kv", keys, keys)]) is not None
    assert calls == []


@pytest.mark.parametrize("bits", [2, 3])
@pytest.mark.parametrize("group", [32, 64, 128])
def test_existing_native_low_bit_codecs_keep_their_packing(bits, group):
    native = QuantizedKVCache(group_size=group, bits=bits)
    keys = (mx.arange(4 * 128).reshape(1, 1, 4, 128) / 37).astype(mx.float16)
    native.update_and_fetch(keys, keys / 2)
    keys, values = native.state
    mx.eval(*keys, *values)
    entry = ("quantized_kv", keys, values, native.meta_state)
    assert validate_cache_record([entry])[0]
    rebuilt = _reconstruct([entry])
    assert rebuilt is not None
    assert rebuilt[0].meta_state == native.meta_state
    for actual, expected in zip(
        rebuilt[0].state[0] + rebuilt[0].state[1], keys + values
    ):
        assert bool(mx.array_equal(actual.view(mx.uint8), expected.view(mx.uint8)))


def test_affine_group_is_supported_by_native_decoder():
    from vmlx_engine.cache_record_validator import (
        CacheValidationError,
        parse_quantized_kv_meta,
    )

    keys = mx.ones((1, 1, 4, 64), dtype=mx.float16)
    with pytest.raises(ValueError):
        mx.quantize(keys, group_size=16, bits=4)
    with pytest.raises(CacheValidationError, match="group"):
        parse_quantized_kv_meta((4, 16, 4))
