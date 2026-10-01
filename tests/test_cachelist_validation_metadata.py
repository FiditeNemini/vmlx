"""CacheList candidate selection must not hydrate plain KV payloads twice."""

from types import SimpleNamespace
import json
import numpy as np
import mlx.core as mx
import pytest
from safetensors.numpy import save_file
from vmlx_engine.block_disk_store import (
    BlockDiskStore,
    _serialize_block,
    _deserialize_block,
    _load_block_validation_entries,
)
from vmlx_engine.prefix_cache import BlockAwarePrefixCache


def serialized(tmp_path, entries):
    tensors, dtype, _ = _serialize_block(entries)
    path = tmp_path / "block.safetensors"
    save_file(BlockDiskStore._detach_safetensors_tensors(tensors), str(path))
    return (path, _deserialize_block(dict(tensors), dtype))


def test_plain_cachelist_selection_uses_metadata_and_preserves_topology(tmp_path):
    entries = [
        (
            "cache_list",
            [
                (
                    "kv",
                    mx.zeros((1, 4, 64, 192), dtype=mx.bfloat16),
                    mx.ones((1, 4, 64, 128), dtype=mx.bfloat16),
                ),
                ("kv", mx.zeros((1, 1, 64, 128)), mx.ones((1, 1, 64, 0))),
            ],
        ),
        ("rotating_kv_pending", "RotatingKVCache"),
    ]
    path, full = serialized(tmp_path, entries)
    reads = []

    class Store:

        def read_block_validation_entries(self, block_hash):
            return _load_block_validation_entries(path)

        def read_block(self, block_hash):
            reads.append(block_hash)
            return full

    block = SimpleNamespace(cache_data=None, block_hash=b"a")
    selected = list(BlockAwarePrefixCache._iter_terminal_check_entries(block, Store()))
    assert reads == [], "candidate validation must not materialize the KV payload"
    assert selected[0][0] == full[0][0] == "cache_list"
    for observed, expected in zip(selected[0][1], full[0][1], strict=True):
        assert observed[0] == expected[0] == "kv"
        assert tuple(observed[1].shape) == tuple(expected[1].shape)
        assert tuple(observed[2].shape) == tuple(expected[2].shape)
        assert observed[1].dtype == str(expected[1].dtype)
    full_store = SimpleNamespace(read_block=lambda _: full)
    assert BlockAwarePrefixCache._rotating_l2_chain_missing_terminal_state(
        [block], disk_store=Store(), target_tokens=64
    ) == BlockAwarePrefixCache._rotating_l2_chain_missing_terminal_state(
        [block], disk_store=full_store, target_tokens=64
    )
    assert reads == []


def test_cumulative_cachelist_retains_full_reader_fallback(tmp_path):
    path, _ = serialized(
        tmp_path,
        [("cache_list", [("cumulative", [mx.ones((1, 4))], "", "ArraysCache")])],
    )
    assert _load_block_validation_entries(path) is None


@pytest.mark.parametrize("damage", ["missing_value", "unknown_child", "sparse_child"])
def test_incomplete_or_unknown_children_fall_back(tmp_path, damage):
    from safetensors.numpy import load_file

    path, _ = serialized(
        tmp_path,
        [("cache_list", [("kv", mx.ones((1, 1, 2, 4)), mx.ones((1, 1, 2, 4)))])],
    )
    tensors = load_file(str(path))
    if damage == "missing_value":
        tensors.pop("layer_0_sub_0_values")
    elif damage == "unknown_child":
        tensors["layer_0_sub_0_future_state"] = tensors["layer_0_sub_0_keys"]
    else:
        tensors["layer_0_sub_1_values"] = tensors.pop("layer_0_sub_0_values")
    save_file(tensors, str(path))
    assert _load_block_validation_entries(path) is None


@pytest.mark.parametrize(
    "damage", ["missing_dtype", "unknown_dtype", "count_mismatch", "nested_type"]
)
def test_unsupported_metadata_falls_back(tmp_path, damage):
    from safetensors.numpy import load_file

    path, _ = serialized(
        tmp_path,
        [("cache_list", [("kv", mx.ones((1, 1, 2, 4)), mx.ones((1, 1, 2, 4)))])],
    )
    tensors = load_file(str(path))
    meta = json.loads(tensors["__vmlx_block_meta__"].tobytes())
    if damage in ("missing_dtype", "unknown_dtype"):
        meta.pop("__tensor_dtypes__")  # Exercise the legacy fallback schema.
    if damage == "missing_dtype":
        meta["__orig_dtypes__"].pop("0_sub_0")
    elif damage == "unknown_dtype":
        meta["__orig_dtypes__"]["0_sub_0"] = "future_dtype"
    elif damage == "count_mismatch":
        meta["0"]["sub_count"] = 2
    else:
        meta["0"]["subs"] = {"0": {"type": "quantized_kv"}}
    tensors["__vmlx_block_meta__"] = np.frombuffer(
        json.dumps(meta).encode(), dtype=np.uint8
    ).copy()
    save_file(tensors, str(path))
    assert _load_block_validation_entries(path) is None


def test_mixed_plain_kv_sibling_does_not_invent_missing_rotating_state(tmp_path):
    terminal, full = serialized(
        tmp_path,
        [
            ("cache_list", [("kv", mx.ones((1, 1, 2, 4)), mx.ones((1, 1, 2, 4)))]),
            ("kv", mx.ones((1, 1, 2, 4)), mx.ones((1, 1, 2, 4))),
        ],
    )
    previous = SimpleNamespace(
        cache_data=[("rotating_kv_pending", "RotatingKVCache")], block_hash=b"p"
    )
    block = SimpleNamespace(cache_data=None, block_hash=b"t")
    full_store = SimpleNamespace(read_block=lambda _: full)
    metadata_store = SimpleNamespace(
        read_block_validation_entries=lambda _: _load_block_validation_entries(
            terminal
        ),
        read_block=lambda _: full,
    )
    check = BlockAwarePrefixCache._rotating_l2_chain_missing_terminal_state
    assert check([previous, block], disk_store=full_store, target_tokens=2) is False
    assert check([previous, block], disk_store=metadata_store, target_tokens=2) is False


def test_plain_kv_only_keeps_full_reader_path(tmp_path):
    path, _ = serialized(
        tmp_path, [("kv", mx.ones((1, 1, 2, 4)), mx.ones((1, 1, 2, 4)))]
    )
    assert _load_block_validation_entries(path) is None


@pytest.mark.parametrize("sibling_type", ["cumulative", "quantized_kv", "future_cache"])
def test_unsupported_sibling_keeps_consumer_full_reader(tmp_path, sibling_type):
    from safetensors.numpy import load_file

    path, full = serialized(
        tmp_path,
        [
            ("cache_list", [("kv", mx.ones((1, 1, 2, 4)), mx.ones((1, 1, 2, 4)))]),
            ("cumulative", [mx.ones((1, 4))], "", "ArraysCache"),
        ],
    )
    tensors = load_file(str(path))
    meta = json.loads(tensors["__vmlx_block_meta__"].tobytes())
    meta["__layer_types__"]["1"] = sibling_type
    tensors["__vmlx_block_meta__"] = np.frombuffer(
        json.dumps(meta).encode(), dtype=np.uint8
    ).copy()
    save_file(tensors, str(path))
    reads = []

    def read_full(block_hash):
        reads.append(block_hash)
        return full

    store = SimpleNamespace(
        read_block_validation_entries=lambda _: _load_block_validation_entries(path),
        read_block=read_full,
    )
    block = SimpleNamespace(cache_data=None, block_hash=b"s")
    assert (
        list(BlockAwarePrefixCache._iter_terminal_check_entries(block, store)) == full
    )
    assert reads == [b"s"]


def rewrite_metadata(path, mutate):
    from safetensors.numpy import load_file

    tensors = load_file(str(path))
    metadata = json.loads(tensors["__vmlx_block_meta__"].tobytes())
    mutate(metadata)
    tensors["__vmlx_block_meta__"] = np.frombuffer(
        json.dumps(metadata).encode(), dtype=np.uint8
    ).copy()
    save_file(tensors, str(path))


def read_full_fixture(path):
    return _deserialize_block(mx.load(str(path)), "bfloat16")


def plain_nested(tmp_path):
    return serialized(
        tmp_path,
        [
            (
                "cache_list",
                [
                    (
                        "kv",
                        mx.ones((1, 1, 2, 4), dtype=mx.bfloat16),
                        mx.ones((1, 1, 2, 4), dtype=mx.float32),
                    )
                ],
            )
        ],
    )[0]


def test_cachelist_runtime_mismatch_uses_full_reader_rejection(tmp_path):
    path = plain_nested(tmp_path)
    rewrite_metadata(
        path,
        lambda meta: meta.update(__runtime_cache_fingerprint__="different-runtime"),
    )
    assert read_full_fixture(path) == []
    assert _load_block_validation_entries(path) is None


def test_modern_independent_dtypes_override_legacy_grouped_dtype(tmp_path):
    path = plain_nested(tmp_path)
    rewrite_metadata(
        path,
        lambda meta: meta["__orig_dtypes__"].update({"0_sub_0": "mlx.core.float16"}),
    )
    full = read_full_fixture(path)
    observed = _load_block_validation_entries(path)
    assert observed is not None
    for index in (1, 2):
        assert observed[0][1][0][index].dtype == str(full[0][1][0][index].dtype)


@pytest.mark.parametrize("damage", ["conflict", "missing", "unknown", "tree_conflict"])
def test_modern_dtype_admission_matches_full_reader(tmp_path, damage):
    path = plain_nested(tmp_path)

    def mutate(meta):
        declarations = meta["__tensor_dtypes__"]
        if damage == "conflict":
            declarations["layer_0_sub_0_values"] = "float16"
        elif damage == "missing":
            declarations.pop("layer_0_sub_0_values")
        elif damage == "unknown":
            declarations["layer_0_sub_0_values"] = "future_dtype"
        else:
            meta["extra_tree"] = {
                "kind": "tensor",
                "key": "layer_0_sub_0_values",
                "orig_dtype": "mlx.core.float16",
            }

    rewrite_metadata(path, mutate)
    assert read_full_fixture(path) == []
    assert _load_block_validation_entries(path) is None


def test_out_of_range_cachelist_tag_does_not_admit_plain_kv(tmp_path):
    path, _ = serialized(
        tmp_path, [("kv", mx.ones((1, 1, 2, 4)), mx.ones((1, 1, 2, 4)))]
    )
    rewrite_metadata(
        path, lambda meta: meta["__layer_types__"].update({"99": "cache_list"})
    )
    assert _load_block_validation_entries(path) is None
