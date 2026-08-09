"""Cross-layer batched TQ block-restore decode (vmlx#91).

The paged/L2 restore path used to build one independent codec graph per KV
layer; decode_tq_layer_groups stacks signature-compatible layers (per-layer
seeds included) into one lazy graph. These tests pin bit-exactness against
the per-layer decode_tq_blocks reference and the single-eval restore
invariant of reconstruct_cache.
"""

import mlx.core as mx


def _encode_layer_pages(
    layer_seeds,
    page_tokens,
    *,
    heads=1,
    dim=32,
    key_bits=4,
    value_bits=4,
):
    """Encode per-layer page chains with one distinct codec seed per layer."""
    from vmlx_engine.tq_disk_store import encode_tq_block

    groups = {}
    for layer_idx, seed in enumerate(layer_seeds):
        entries = []
        for page_idx, tokens in enumerate(page_tokens):
            mx.random.seed(100_000 + layer_idx * 1_000 + page_idx)
            keys = mx.random.normal(shape=(1, heads, tokens, dim)).astype(
                mx.float16
            )
            values = mx.random.normal(shape=(1, heads, tokens, dim)).astype(
                mx.float16
            )
            entries.append(
                encode_tq_block(
                    keys,
                    values,
                    {
                        "key_bits": key_bits,
                        "value_bits": value_bits,
                        "seed": seed,
                    },
                )
            )
        groups[("layer", layer_idx)] = entries
    return groups


def _assert_exact(actual, expected):
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert float(mx.max(mx.abs(actual - expected)).item()) == 0.0


def test_layer_group_decode_matches_per_layer_decode_exactly():
    """Realistic restore shape: 16 layers x 32 pages x 256 tokens.

    Per-layer seeds are all distinct (native TQ policies seed each attention
    layer independently), so the batched path must apply stacked per-layer
    rotation signs and QJL projections — the output has to be bit-identical
    to decode_tq_blocks run on each layer alone.
    """
    from vmlx_engine.tq_disk_store import (
        decode_tq_blocks,
        decode_tq_layer_groups,
    )

    layer_seeds = [977 + 13 * i for i in range(16)]
    groups = _encode_layer_pages(layer_seeds, [256] * 32)

    reference = {
        key: decode_tq_blocks(entries) for key, entries in groups.items()
    }
    batched = decode_tq_layer_groups(groups)

    assert set(batched) == set(reference)
    for key in reference:
        _assert_exact(batched[key][0], reference[key][0])
        _assert_exact(batched[key][1], reference[key][1])


def test_layer_group_decode_preserves_partial_tail_runs():
    """A shorter tail page splits the chain into two cross-layer runs."""
    from vmlx_engine.tq_disk_store import (
        decode_tq_blocks,
        decode_tq_layer_groups,
    )

    layer_seeds = [311, 331, 353, 373]
    groups = _encode_layer_pages(layer_seeds, [16, 16, 16, 11], heads=2)

    reference = {
        key: decode_tq_blocks(entries) for key, entries in groups.items()
    }
    batched = decode_tq_layer_groups(groups)

    assert set(batched) == set(reference)
    for key in reference:
        _assert_exact(batched[key][0], reference[key][0])
        _assert_exact(batched[key][1], reference[key][1])


def test_layer_group_decode_isolates_incompatible_layer():
    """A layer with a different codec keeps its own exact fallback decode."""
    from vmlx_engine.tq_disk_store import (
        decode_tq_blocks,
        decode_tq_layer_groups,
    )

    groups = _encode_layer_pages([421, 431, 433], [16, 16])
    outlier = _encode_layer_pages([439], [16, 16], key_bits=8, value_bits=8)
    groups[("layer", 3)] = outlier[("layer", 0)]

    reference = {
        key: decode_tq_blocks(entries) for key, entries in groups.items()
    }
    batched = decode_tq_layer_groups(groups)

    assert set(batched) == set(reference)
    for key in reference:
        _assert_exact(batched[key][0], reference[key][0])
        _assert_exact(batched[key][1], reference[key][1])


def test_layer_group_decode_bfloat16_dtype_restore():
    """Stacked decode restores the recorded attention dtype per family."""
    from vmlx_engine.tq_disk_store import (
        decode_tq_blocks,
        decode_tq_layer_groups,
        encode_tq_block,
    )

    groups = {}
    for layer_idx, seed in enumerate((11, 17)):
        mx.random.seed(9_000 + layer_idx)
        keys = mx.random.normal(shape=(1, 2, 16, 64)).astype(mx.bfloat16)
        values = mx.random.normal(shape=(1, 2, 16, 64)).astype(mx.bfloat16)
        groups[layer_idx] = [
            encode_tq_block(
                keys,
                values,
                {"key_bits": 4, "value_bits": 4, "seed": seed},
            )
        ]

    batched = decode_tq_layer_groups(groups)
    for layer_idx, entries in groups.items():
        expected_keys, expected_values = decode_tq_blocks(entries)
        _assert_exact(batched[layer_idx][0], expected_keys)
        _assert_exact(batched[layer_idx][1], expected_values)
        assert batched[layer_idx][0].dtype == mx.bfloat16
        assert batched[layer_idx][1].dtype == mx.bfloat16


def test_reconstruct_cache_multi_layer_tq_exact_with_single_eval(monkeypatch):
    """Full store→fetch→reconstruct round trip at multi-layer TQ shapes.

    The restore must reproduce the per-block decode_tq_block reference
    bit-exactly AND issue exactly ONE mx.eval (the deferred combined eval) —
    per-layer eval sync points were the measured 6x TTFT penalty (vmlx#91).
    """
    import vmlx_engine.prefix_cache as prefix_cache_mod
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache, PagedCacheManager
    from vmlx_engine.tq_disk_store import decode_tq_block

    # VMLX_TQ_DECODE_TIMING deliberately forces evals inside the decode for
    # wall-time attribution; the invariant under test is the DEFAULT path.
    monkeypatch.delenv("VMLX_TQ_DECODE_TIMING", raising=False)

    class _TinyModel:
        class Args:
            num_attention_heads = 2
            num_key_value_heads = 2
            head_dim = 64
            kv_lora_rank = 0

        args = Args()

    layer_seeds = [503, 509, 521, 523]
    tokens = list(range(64))
    states = []
    for layer_idx, seed in enumerate(layer_seeds):
        mx.random.seed(42_000 + layer_idx)
        states.append(
            {
                "state": (
                    mx.random.normal(shape=(1, 2, 64, 64)).astype(mx.float16),
                    mx.random.normal(shape=(1, 2, 64, 64)).astype(mx.float16),
                ),
                "meta_state": ("64",),
                "class_name": "TurboQuantKVCache",
                "tq_config": {
                    "key_bits": 4,
                    "value_bits": 4,
                    "seed": seed,
                },
            }
        )

    manager = PagedCacheManager(block_size=16, max_blocks=32)
    cache = BlockAwarePrefixCache(
        model=_TinyModel(),
        paged_cache_manager=manager,
    )
    cache.store_cache("write", tokens, states)
    table, remaining = cache.fetch_cache("read", tokens)
    assert table is not None
    assert remaining == []

    expected = []
    for layer_idx in range(len(layer_seeds)):
        layer_keys = []
        layer_values = []
        for block_id in table.block_ids:
            entry = manager.allocated_blocks[block_id].cache_data[layer_idx]
            assert entry[0] == "turboquant_kv"
            assert entry[3]["seed"] == layer_seeds[layer_idx]
            keys, values = decode_tq_block(entry)
            layer_keys.append(keys)
            layer_values.append(values)
        expected.append(
            (
                mx.concatenate(layer_keys, axis=2),
                mx.concatenate(layer_values, axis=2),
            )
        )
    mx.eval(*[tensor for pair in expected for tensor in pair])

    real_eval = mx.eval
    eval_calls = {"count": 0}

    def _counting_eval(*args, **kwargs):
        eval_calls["count"] += 1
        return real_eval(*args, **kwargs)

    monkeypatch.setattr(prefix_cache_mod.mx, "eval", _counting_eval)
    restored = cache.reconstruct_cache(table)
    monkeypatch.undo()

    assert restored is not None
    assert len(restored) == len(layer_seeds)
    assert cache._last_reconstruct_tq_blocks == len(layer_seeds) * len(
        table.block_ids
    )
    assert eval_calls["count"] == 1
    for layer_idx, layer_cache in enumerate(restored):
        assert layer_cache.offset == 64
        _assert_exact(layer_cache.keys, expected[layer_idx][0])
        _assert_exact(layer_cache.values, expected[layer_idx][1])
