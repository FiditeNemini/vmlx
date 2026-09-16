"""Actual SSD writes must preserve each cache tensor's independent dtype."""

import hashlib
import json
import struct

import pytest

mx = pytest.importorskip("mlx.core")

from vmlx_engine.block_disk_store import (
    BlockDiskStore,
    _deserialize_block,
    _serialize_block,
)


def _roundtrip(tmp_path, cache_data):
    cache_dir = tmp_path / "cache"
    block_hash = hashlib.sha256(b"mixed-cache-dtypes").digest()
    writer = BlockDiskStore(str(cache_dir), max_size_gb=0.1)
    try:
        assert writer.write_block_async(block_hash, cache_data, 4)
        assert writer.wait_for_blocks([block_hash], timeout=5.0) == {block_hash}
    finally:
        writer.shutdown()
    # A new reader must hydrate from disk, not the writer's live input objects.
    reader = BlockDiskStore(str(cache_dir), max_size_gb=0.1)
    try:
        restored = reader.read_block(block_hash)
        assert restored is not None
        assert reader.get_stats()["disk_hits"] == 1
        return restored
    finally:
        reader.shutdown()


def _assert_tensor(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    # Byte comparison preserves signed zero and packed integer representations.
    assert bool(mx.array_equal(actual.view(mx.uint8), expected.view(mx.uint8)))


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("nested", ["cache_list", "zaya_cca"])
def test_nested_quantized_scales_roundtrip(tmp_path, bits, dtype, nested):
    k = (mx.arange(256).reshape(1, 1, 4, 64) / 37 - 2).astype(dtype)
    v = (k * 0.75 + 1).astype(dtype)
    keys = mx.quantize(k, group_size=32, bits=bits)
    values = mx.quantize(v, group_size=32, bits=bits)
    entry = ("quantized_kv", keys, values, {"group_size": 32, "bits": bits})
    cache_data = [("cache_list", [entry])] if nested == "cache_list" else [
        ("zaya_cca", entry, None, "", {})
    ]
    restored = _roundtrip(tmp_path, cache_data)[0]
    result = restored[1][0] if nested == "cache_list" else restored[1]
    assert result[0] == "quantized_kv"
    assert result[3] == entry[3]
    for actual, expected in zip(result[1] + result[2], keys + values):
        _assert_tensor(actual, expected)
    # Quantization's packed data is not expanded on disk; BF16 carriers stay2B.
    files = list((tmp_path / "cache" / "blocks").rglob("*.safetensors"))
    assert len(files) == 1
    with files[0].open("rb") as handle:
        header = json.loads(handle.read(struct.unpack("<Q", handle.read(8))[0]))
    tensor_bytes = sum(
        desc["data_offsets"][1] - desc["data_offsets"][0]
        for name, desc in header.items()
        if name not in {"__metadata__", "__vmlx_block_meta__"}
    )
    assert tensor_bytes == sum(array.nbytes for array in keys + values)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_sparse_index_keeps_fp32_coordinates(tmp_path, dtype):
    keys = mx.arange(16).reshape(1, 1, 4, 4).astype(dtype)
    values = (keys + 0.5).astype(dtype)
    # QSA transports raw index features and exact positions in an FP32 lane.
    index = mx.array([257, 1023, 65535, 262143], dtype=mx.float32).reshape(1, 1, 4, 1)
    restored = _roundtrip(tmp_path, [("minimax_m3", keys, values, index)])[0]
    for actual, expected in zip(restored[1:], (keys, values, index)):
        _assert_tensor(actual, expected)


def test_nested_cumulative_and_mixed_kv_dtypes(tmp_path):
    keys = mx.array([0.0, -0.0, 1.0, 1e30], dtype=mx.bfloat16).reshape(1, 1, 1, 4)
    values = mx.array([257, 1023, 65535, 262143], dtype=mx.float32).reshape(1, 1, 1, 4)
    state = mx.array([0.0, -0.0, 1.0, 1e30], dtype=mx.bfloat16).reshape(1, 1, 4)
    restored = _roundtrip(tmp_path, [
        ("kv", keys, values),
        ("cache_list", [("cumulative", [state], "", "ArraysCache")]),
    ])
    _assert_tensor(restored[0][1], keys)
    _assert_tensor(restored[0][2], values)
    _assert_tensor(restored[1][1][0][1][0], state)


def _replace_meta(tensors, update):
    meta = json.loads(bytes(tensors["__vmlx_block_meta__"].tolist()))
    update(meta)
    tensors["__vmlx_block_meta__"] = mx.array(list(json.dumps(meta).encode()), dtype=mx.uint8)


@pytest.mark.parametrize("damage", ["missing", "extra", "unknown", "wrong", "not_dict"])
def test_invalid_tensor_dtype_metadata_fails_closed(damage):
    keys = mx.ones((1, 1, 4, 4), dtype=mx.bfloat16)
    tensors, tag, _ = _serialize_block([("kv", keys, keys)])
    detached = {name: mx.array(value) for name, value in BlockDiskStore._detach_safetensors_tensors(tensors).items()}

    def damage_meta(meta):
        dtypes = meta["__tensor_dtypes__"]
        if damage == "missing":
            del dtypes["layer_0_values"]
        elif damage == "extra":
            dtypes["nonexistent_tensor"] = "float32"
        elif damage == "unknown":
            dtypes["layer_0_values"] = "unknown_native_type"
        elif damage == "wrong":
            dtypes["layer_0_values"] = "float32"
        else:
            meta["__tensor_dtypes__"] = []

    _replace_meta(detached, damage_meta)
    assert _deserialize_block(detached, tag) == []


@pytest.mark.parametrize("nested", ["cache_list", "zaya_cca"])
def test_legacy_undeclared_nested_bf16_carrier_is_a_miss(nested):
    x = mx.ones((1, 1, 4, 64), dtype=mx.bfloat16)
    packed = mx.quantize(x, group_size=32, bits=4)
    entry = ("quantized_kv", packed, packed, {"group_size": 32, "bits": 4})
    payload = [("cache_list", [entry])] if nested == "cache_list" else [
        ("zaya_cca", entry, None, "", {})
    ]
    tensors, tag, _ = _serialize_block(payload)
    _replace_meta(tensors, lambda meta: meta.pop("__tensor_dtypes__"))
    detached = {name: mx.array(value) for name, value in BlockDiskStore._detach_safetensors_tensors(tensors).items()}
    assert _deserialize_block(detached, tag) == []


@pytest.mark.parametrize("carrier", ["uint16", "float32"])
def test_legacy_declared_bf16_kv_remains_readable(carrier):
    keys = mx.array([0.0, -0.0, 1.0, 1e30], dtype=mx.bfloat16).reshape(1, 1, 1, 4)
    tensors, tag, _ = _serialize_block([("kv", keys, keys)])
    _replace_meta(tensors, lambda meta: meta.pop("__tensor_dtypes__"))
    for name in ("layer_0_keys", "layer_0_values"):
        tensors[name] = keys.view(mx.uint16) if carrier == "uint16" else keys.astype(mx.float32)
    restored = _deserialize_block(tensors, tag)
    _assert_tensor(restored[0][1], keys)
    _assert_tensor(restored[0][2], keys)


def test_conflicting_native_tree_dtype_is_rejected():
    state = mx.array([257, 1023], dtype=mx.float32)
    tensors, tag, _ = _serialize_block([
        ("deepseek_v4", {"native": state}, (), "DeepseekV4Cache", {})
    ])

    def damage_meta(meta):
        meta["0"]["state_tree"]["items"]["native"]["orig_dtype"] = "mlx.core.bfloat16"

    _replace_meta(tensors, damage_meta)
    assert _deserialize_block(tensors, tag) == []
