"""Selected blocks keep exact AR partitions, native state, and fallback rules."""
import mlx.core as mx
import numpy as np
import pytest

from vmlx_engine.metal import qwen4_sparse_decode as sparse


def assert_exact(a, b):
    mx.eval(a, b)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("tokens", [65537, 76800, 76801, 76802, 76803, 131072])
def test_selected_blocks_equal_dense_mask_and_stock_with_strides(tokens):
    # Capacity and last-dimension strides must remain logical views, not force
    # hidden copies of the whole context before the sparse consumer.
    q = mx.random.normal((1, 24, 1, 512), key=mx.random.key(12)).astype(mx.float16)[..., ::2]
    k = mx.random.normal((1, 2, tokens + 9, 512), key=mx.random.key(13)).astype(mx.float16)[:, :, :tokens, ::2]
    v = mx.random.normal((1, 2, tokens + 13, 512), key=mx.random.key(14)).astype(mx.float16)[:, :, :tokens, ::2]
    blocks = tokens // 4
    # Unsorted, strided block IDs; masked entries are ignored.
    selected = ((mx.arange(1024) * 37) % blocks)[::-2][None].astype(mx.int32)
    valid = (mx.arange(1024)[::2] % 7 != 0)[None]
    # Use valid selected indices without relying on out-of-bounds scatter
    # semantics in the reference mask.
    indices = np.asarray(selected)[0][np.asarray(valid)[0]]
    expected_bits = np.zeros(tokens, dtype=np.bool_)
    for block in indices:
        expected_bits[int(block) * 4:int(block) * 4 + 4] = True
    expected_bits[blocks * 4:] = True
    mask = mx.where(mx.array(expected_bits), 0., -mx.inf).astype(mx.float16)[None, None, None]
    actual = sparse.attention_from_blocks(q, k, v, selected, valid, scale=.0625, enabled=True)
    assert actual is not None
    expected = mx.fast.scaled_dot_product_attention(q, k, v, scale=.0625, mask=mask)
    assert_exact(actual, expected)


@pytest.mark.parametrize("change", ["off", "dtype", "count", "valid", "rows", "override", "context"])
def test_block_consumer_admission(change, monkeypatch):
    q = mx.zeros((1, 24, 1, 256), dtype=mx.float16)
    k = v = mx.zeros((1, 2, 76800, 256), dtype=mx.float16)
    selected = mx.arange(512, dtype=mx.int32)[None]
    valid = mx.ones((1, 512), dtype=mx.bool_)
    enabled = True
    if change == "off": enabled = False
    if change == "dtype": selected = selected.astype(mx.uint32)
    if change == "count": selected = selected[:, :128]
    if change == "valid": valid = valid.astype(mx.int32)
    if change == "rows": q = mx.zeros((1, 24, 2, 256), dtype=mx.float16)
    if change == "override": monkeypatch.setenv("MLX_SDPA_BLOCKS", "128")
    if change == "context": k = v = k[:, :, :65536]
    assert sparse.attention_from_blocks(q, k, v, selected, valid, scale=.0625, enabled=enabled) is None


def test_direct_blocks_default_off(monkeypatch):
    from vmlx_engine.models.qwen4_exp.language import QSAAttention, Qwen4ExpTextArgs
    monkeypatch.delenv("VMLX_QWEN4_SPARSE_AR_DIRECT_BLOCKS", raising=False)
    assert not QSAAttention(Qwen4ExpTextArgs(hidden_size=32))._sparse_ar_direct_blocks
    monkeypatch.setenv("VMLX_QWEN4_SPARSE_AR_DIRECT_BLOCKS", "1")
    assert QSAAttention(Qwen4ExpTextArgs(hidden_size=32))._sparse_ar_direct_blocks


def test_qsa_single_append_trim_restore_and_verify_fallback(monkeypatch):
    from vmlx_engine.models.qwen4_exp.language import QSAAttention, QSACache, Qwen4ExpTextArgs
    monkeypatch.setenv("VMLX_QWEN4_SPARSE_AR", "1")
    monkeypatch.setenv("VMLX_QWEN4_QSA_MASK", "0")
    monkeypatch.setenv("VMLX_QWEN4_QSA_MERGE_SELECT", "0")
    monkeypatch.setenv("VMLX_QWEN4_QSA_SCORE_REDUCE", "0")
    mx.random.seed(123)
    layer = QSAAttention(Qwen4ExpTextArgs(hidden_size=32))
    layer.set_dtype(mx.float16)
    layer.eval()
    prior = 65536
    keys = mx.random.normal((1, 2, prior, 256)).astype(mx.float16)
    values = mx.random.normal((1, 2, prior, 256)).astype(mx.float16)
    raw = mx.random.normal((1, 1, prior, 128)).astype(mx.float32)
    # Distinct media coordinates ride in the raw indexer lane unchanged.
    positions = mx.stack([mx.arange(prior), mx.arange(prior) + 5, mx.arange(prior) + 17], axis=-1)
    payload = mx.concatenate([raw, positions[None, None].astype(mx.float32)], axis=-1)
    caches = [QSACache(), QSACache()]
    for cache in caches:
        cache.state = (keys, values, payload)
    mx.eval(layer.parameters(), keys, values, payload)
    observed = []
    original = sparse.attention_from_blocks

    def capture(*args, **kwargs):
        observed.append(args[0].shape[2])
        return original(*args, **kwargs)

    monkeypatch.setattr(sparse, "attention_from_blocks", capture)
    snapshots = None
    for step, rows in enumerate([1, 1, 1, 1, 2, 1]):
        if step == 3:
            for cache in caches:
                assert cache.trim(2) == 2
        if step == 5:
            for cache, snapshot in zip(caches, snapshots):
                cache.state = snapshot
                assert not cache.derived
        before = caches[0].offset
        x = mx.random.normal((1, rows, 32)).astype(mx.float16)
        outputs = []
        for direct, cache in zip((False, True), caches):
            layer._sparse_ar_direct_blocks = direct
            outputs.append(layer(x, cache=cache))
        assert_exact(*outputs)
        assert caches[0].offset == caches[1].offset == before + rows
        for cache in caches:
            assert cache._idx_offset == cache.offset
        for a, b in zip(caches[0].state, caches[1].state):
            assert_exact(a, b)
        if step == 1:
            snapshots = [cache.state for cache in caches]
    # Multi-row verification uses the unchanged full-mask consumer.
    assert observed == [1, 1, 1, 1, 1]
