"""Packed PLE ownership, bit-exact imports and atomic history updates."""
from types import SimpleNamespace
import gc
import mlx.core as mx
import numpy as np
import pytest

from vmlx_engine.models.qwen4_exp.deferred_ple import Declined, pending
from vmlx_engine.models.qwen4_exp.packed_ple import PackedDeferredPLE, PackedPlan, PackedOwnedLeaf


class Hasher:
    ngram_heads = 2
    head_offsets = [0, 100]
    head_vocab_sizes = [100, 100]
    context_len = 2
    eos_token_id = 0

    def hash_tokens(self, ids, previous):
        base = int(ids[0, 0]) % 100
        return np.array([[[base, 100 + base]]], dtype=np.int64)


class Shard:
    mode = "affine"
    layout_signature = (4, 4, 64, "affine", 64, "U32", (8,), "F16", (1,), "F16", (1,))
    weight = SimpleNamespace(shape=(200, 8), dtype_tag="U32", mlx_dtype=mx.uint32)
    scales = SimpleNamespace(shape=(200, 1), dtype_tag="F16", mlx_dtype=mx.float16)
    biases = SimpleNamespace(shape=(200, 1), dtype_tag="F16", mlx_dtype=mx.float16)
    output_dtype = mx.float16

    def _dequantize_mlx(self, weight, scale, bias, *, profile):
        assert profile is None
        return mx.dequantize(weight, scale, bias, bits=4, group_size=64,
                             mode="affine", dtype=self.output_dtype)


def fixture(*, bf16_metadata=False, output_dtype=mx.float16):
    shard = Shard()
    hosts = (np.full((2, 8), 0x12345678, dtype=np.uint32),
             np.array([[0.5], [0.25]], dtype=np.float16),
             np.array([[1.0], [-1.0]], dtype=np.float16))
    if bf16_metadata:
        shard.scales = SimpleNamespace(shape=(200, 1), dtype_tag="BF16", mlx_dtype=mx.bfloat16)
        shard.biases = SimpleNamespace(shape=(200, 1), dtype_tag="BF16", mlx_dtype=mx.bfloat16)
        shard.output_dtype = mx.bfloat16
        shard.layout_signature = (4, 4, 64, "affine", 64, "U32", (8,),
                                  "BF16", (1,), "BF16", (1,))
        hosts = (hosts[0], np.array([[0x3F00], [0x3E80]], dtype=np.uint16),
                 np.array([[0x3F80], [0xBF80]], dtype=np.uint16))
    table = SimpleNamespace(
        _host_assembly=True, _closed=False, shards=[shard], total_rows=200,
        head_dim=64, host_gather_stats={k: 0 for k in
                                      ("calls", "rows", "unique_rows", "shards", "layout_groups")},
    )
    table._read_host_assembled = lambda rows: (
        np.arange(2), 2, 1, [(shard, np.arange(2), hosts)]
    )
    layer = SimpleNamespace(hasher=Hasher(), ngram_embedding=SimpleNamespace(
        _file_backed=table, output_dtype=output_dtype))
    cache = [None, None, None, None]
    return layer, cache, table, shard, hosts


def test_packed_graph_exactly_matches_original_dequant():
    layer, cache, table, shard, hosts = fixture()
    reference = shard._dequantize_mlx(*(mx.array(x) for x in hosts), profile=None)
    with PackedDeferredPLE([(layer, cache)]) as scope:
        token = mx.array([[17]], dtype=mx.int32)
        embedding = scope.bind_packed(layer, token, cache)
        graph = embedding + mx.array(1, dtype=mx.float16)
        assert cache[2] is None
        scope.flush()
    assert mx.array_equal(graph, reference.reshape(1, 1, 128) + 1).item()
    assert np.array_equal(np.asarray(cache[2]), [[0, 17]])
    assert table.host_gather_stats["layout_groups"] == 1


def test_heterogeneous_unselected_shard_declines_before_build():
    layer, cache, table, shard, _ = fixture()
    table.shards.append(SimpleNamespace(layout_signature=("different",)))
    with pytest.raises(Declined, match="heterogeneous"):
        PackedPlan(layer, cache)


def test_overlap_in_head_ranges_declines_before_build():
    layer, cache, *_ = fixture()
    layer.hasher.head_offsets = [0, 99]
    with pytest.raises(Declined, match="disjoint"):
        PackedPlan(layer, cache)


def test_unexpected_dedup_aborts_without_history_commit():
    layer, cache, table, shard, hosts = fixture()
    table._read_host_assembled = lambda rows: (
        np.array([0, 0]), 1, 1, [(shard, np.array([0]), hosts)]
    )
    with pytest.raises(RuntimeError, match="cohort/dedup"):
        with PackedDeferredPLE([(layer, cache)]) as scope:
            scope.bind_packed(layer, mx.array([[17]]), cache)
            scope.flush()
    assert cache[2] is None


def test_bf16_metadata_bind_is_bit_alias_not_numeric_cast():
    leaf = PackedOwnedLeaf([2, 1], mx.bfloat16)
    raw = np.array([[0x3F80], [0x8000]], dtype=np.uint16)
    graph = mx.view(leaf.value, mx.uint16)
    leaf.fill(raw)
    assert np.array_equal(np.asarray(graph), raw)


def test_bf16_table_then_fp16_embedding_cast_matches_native_boundary():
    layer, cache, _, shard, hosts = fixture(bf16_metadata=True, output_dtype=mx.float16)
    reference = shard._dequantize_mlx(mx.array(hosts[0]),
        mx.array(hosts[1]).view(mx.bfloat16), mx.array(hosts[2]).view(mx.bfloat16),
        profile=None).astype(mx.float16).reshape(1, 1, 128)
    with PackedDeferredPLE([(layer, cache)]) as scope:
        embedding = scope.bind_packed(layer, mx.array([[17]]), cache)
        assert embedding.dtype == mx.float16
        scope.flush()
    assert mx.array_equal(embedding.view(mx.uint16), reference.view(mx.uint16)).item()


def test_filled_packed_backing_outlives_python_transaction_and_plan():
    layer, cache, _, shard, hosts = fixture()
    reference = shard._dequantize_mlx(*(mx.array(x) for x in hosts), profile=None)
    with PackedDeferredPLE([(layer, cache)]) as scope:
        embedding = scope.bind_packed(layer, mx.array([[17]]), cache)
        graph = embedding + 1
        scope.flush()
    del embedding, scope, layer, cache
    gc.collect()
    assert mx.array_equal(graph, reference.reshape(1, 1, 128) + 1).item()


def test_two_steps_keep_private_leaf_bytes_and_advance_exact_history():
    layer, cache, _, shard, hosts = fixture()
    graphs = []
    references = []
    for token_id in (17, 23):
        hosts[0].fill(np.uint32(token_id * 0x01010101))
        references.append(shard._dequantize_mlx(
            *(mx.array(x) for x in hosts), profile=None).reshape(1, 1, 128) + token_id)
        previous = cache[2]
        with PackedDeferredPLE([(layer, cache)]) as scope:
            embedding = scope.bind_packed(layer, mx.array([[token_id]]), cache)
            graphs.append(embedding + token_id)
            assert cache[2] is previous
            scope.flush()
        assert not pending()
    assert np.array_equal(np.asarray(cache[2]), [[17, 23]])
    for graph, reference in zip(graphs, references):
        assert mx.array_equal(graph, reference).item()


def test_multilayer_partial_fill_failure_commits_no_history(monkeypatch):
    a, ca, *_ = fixture()
    b, cb, *_ = fixture()
    scope = PackedDeferredPLE([(a, ca), (b, cb)])
    second = scope.plans[id(b)].leaves[1]

    def fail(_host):
        raise IOError("injected second-layer metadata fill failure")

    monkeypatch.setattr(second, "fill", fail)
    with pytest.raises(IOError):
        with scope:
            token = mx.array([[17]])
            scope.bind_packed(a, token, ca)
            scope.bind_packed(b, token, cb)
            scope.flush()
    assert ca[2] is None and cb[2] is None and not pending()
    assert scope.plans[id(a)].leaves[0].filled  # failure occurred AFTER fill
    assert all(leaf.aborted for plan in scope.plans.values() for leaf in plan.leaves)
    # Do NOT evaluate an abandoned partially-filled graph.
