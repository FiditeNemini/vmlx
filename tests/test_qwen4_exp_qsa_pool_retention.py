"""QSA completed-pool retention (plan item A) must be bitwise-identical to the
full recompute under every cache mutation the serving path performs."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from vmlx_engine.models.minimax_m3.cache import (
    BatchMiniMaxM3SparseCache,
    MiniMaxM3SparseCache,
    clone_minimax_m3_sparse,
    restore_minimax_m3_sparse,
    truncate_minimax_m3_cache,
)
from vmlx_engine.models.qwen4_exp.language import (
    QSAIndexer,
    Qwen4ExpTextArgs,
    _QSAPooledFrontier,
)

RATIO = 4
BUDGET = 8  # block_topk = 2 → sparse path from the 3rd complete block on
HIDDEN = 32


def _indexer() -> QSAIndexer:
    args = Qwen4ExpTextArgs(
        hidden_size=HIDDEN,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=BUDGET,
        indexer_compress_ratio=RATIO,
        head_dim=32,
        rope_theta=10_000.0,
        partial_rotary_factor=0.25,
        mrope_section=[2, 1, 1],
    )
    mx.random.seed(7)
    ix = QSAIndexer(args)
    mx.eval(ix.parameters())
    return ix


def _hidden(n: int, seed: int, batch: int = 1) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((batch, n, HIDDEN))


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return bool(mx.array_equal(a, b))


def _frontier(cache) -> _QSAPooledFrontier | None:
    return cache.derived.get("qsa_pooled")


def _run(ix, cache, x, *, retain: bool, monkeypatch, position_ids=None):
    """Mirror QSAAttention: append K/V first (that advances ``cache.offset``),
    then call the indexer with the saved pre-append offset."""
    monkeypatch.setenv("VMLX_QWEN4_QSA_POOL_RETAIN", "1" if retain else "0")
    b, s = x.shape[0], x.shape[1]
    pre = cache.offset
    kv = mx.zeros((b, 1, s, 4))
    cache.update_and_fetch(kv, kv)
    out = ix(x, cache, offset=pre, position_ids=position_ids)
    if out is not None:
        mx.eval(out)
    assert cache.offset == cache._idx_offset
    return out


def _lockstep(ix, steps, monkeypatch, *, batch: int = 1, positions=False):
    """Drive a retained cache and a recompute cache through the same token
    stream; assert every mask (and the retained pooled frontier) is bitwise
    equal to the recompute path. Returns the two caches."""
    ret, ref = MiniMaxM3SparseCache(), MiniMaxM3SparseCache()
    for i, s in enumerate(steps):
        x = _hidden(s, 100 + i, batch)
        pos = None
        if positions:
            base = ref.offset
            t = mx.arange(base, base + s)
            # three distinct M-RoPE coordinates (temporal / h / w)
            pos = mx.stack([t, t + 17, t + 5])[:, None, :]
            pos = mx.broadcast_to(pos, (3, batch, s))
        a = _run(ix, ret, x, retain=True, monkeypatch=monkeypatch, position_ids=pos)
        b = _run(ix, ref, x, retain=False, monkeypatch=monkeypatch, position_ids=pos)
        assert _same(a, b), f"step {i} (S={s}, T={ref.offset}) mask mismatch"
        assert ref.derived == {}, "recompute path must not retain"
        fr = _frontier(ret)
        if a is not None and fr is not None and fr.evicted:
            continue  # size-capped: exactness already asserted above
        if a is not None:
            assert fr is not None and fr.blocks == ret.offset // RATIO
            full = ix._pool_blocks(
                ret.idx_keys[:, 0, :, : ix.head_dim],
                ret.idx_keys[:, 0, :, ix.head_dim :],
                0,
                fr.blocks,
                batch,
            )
            assert _same(fr.pooled, full), f"step {i} pooled frontier mismatch"
    return ret, ref


def test_retention_default_on_and_kill_switch(monkeypatch):
    monkeypatch.delenv("VMLX_QWEN4_QSA_POOL_RETAIN", raising=False)
    from vmlx_engine.models.qwen4_exp.language import _qsa_pool_retention_enabled

    assert _qsa_pool_retention_enabled()
    for v in ("0", "false", "OFF"):
        monkeypatch.setenv("VMLX_QWEN4_QSA_POOL_RETAIN", v)
        assert not _qsa_pool_retention_enabled()


@pytest.mark.parametrize("prefill", [11, 12, 13, 15, 16, 17])
def test_prefill_then_decode_every_remainder(prefill, monkeypatch):
    """Sparse admission sits at 3 complete blocks (12 tokens) here, the
    analogue of 2048 tokens in the shipped 512-block budget: cover lengths
    on both sides and every four-token remainder, then single-token decode."""
    ix = _indexer()
    ret, _ = _lockstep(ix, [prefill] + [1] * 9, monkeypatch)
    fr = _frontier(ret)
    assert fr is not None and fr.reused > 0


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_mtp_verify_row_counts(rows, monkeypatch):
    """S = depth + 1 rows per verify forward (S=1 is AR)."""
    ix = _indexer()
    _lockstep(ix, [14] + [rows] * 8, monkeypatch)


def test_media_mrope_coordinates(monkeypatch):
    ix = _indexer()
    _lockstep(ix, [13, 1, 3, 1, 2], monkeypatch, positions=True)


@pytest.mark.parametrize("accepted", [0, 1, 2, 3])
def test_rejected_draft_trim_every_accepted_prefix(accepted, monkeypatch):
    """Verify appends depth+1 rows, then trims back to the accepted prefix.
    Repeat several cycles so the frontier is truncated mid-block, on a block
    boundary, and after a whole-block rollback."""
    ix = _indexer()
    rows = 4
    ret, ref = MiniMaxM3SparseCache(), MiniMaxM3SparseCache()
    x0 = _hidden(15, 3)
    _run(ix, ret, x0, retain=True, monkeypatch=monkeypatch)
    _run(ix, ref, x0, retain=False, monkeypatch=monkeypatch)
    for cycle in range(6):
        x = _hidden(rows, 40 + cycle)
        a = _run(ix, ret, x, retain=True, monkeypatch=monkeypatch)
        b = _run(ix, ref, x, retain=False, monkeypatch=monkeypatch)
        assert _same(a, b)
        reject = rows - accepted
        if reject:
            truncate_minimax_m3_cache([ret], ret.offset - reject)
            truncate_minimax_m3_cache([ref], ref.offset - reject)
        assert ret.offset == ref.offset
        fr = _frontier(ret)
        assert fr is not None and fr.blocks <= ret.offset // RATIO
        # a probe token after the trim must see identical selection
        xp = _hidden(1, 80 + cycle)
        a = _run(ix, ret, xp, retain=True, monkeypatch=monkeypatch)
        b = _run(ix, ref, xp, retain=False, monkeypatch=monkeypatch)
        assert _same(a, b), f"cycle {cycle} accepted={accepted}"


def test_trim_to_zero_and_regrow(monkeypatch):
    ix = _indexer()
    ret, ref = _lockstep(ix, [14, 1, 1], monkeypatch)
    n = ret.offset
    assert ret.trim(n) == n and ref.trim(n) == n
    assert _frontier(ret).blocks == 0 and _frontier(ret).pooled is None
    _run(ix, ret, _hidden(13, 9), retain=True, monkeypatch=monkeypatch)
    _run(ix, ref, _hidden(13, 9), retain=False, monkeypatch=monkeypatch)
    a = _run(ix, ret, _hidden(1, 10), retain=True, monkeypatch=monkeypatch)
    b = _run(ix, ref, _hidden(1, 10), retain=False, monkeypatch=monkeypatch)
    assert _same(a, b)


def test_same_length_different_history_via_restore(monkeypatch):
    """Offset equality is not content identity: a restore of a different
    history at the same length must drop the frontier and re-derive."""
    ix = _indexer()
    ret, _ = _lockstep(ix, [13, 1], monkeypatch)
    alt = MiniMaxM3SparseCache()
    _run(ix, alt, _hidden(14, 555), retain=False, monkeypatch=monkeypatch)
    assert alt.offset == ret.offset
    keys, values, idx = alt.state
    ret.state = (keys, values, idx)
    assert ret.derived == {}
    a = _run(ix, ret, _hidden(1, 11), retain=True, monkeypatch=monkeypatch)
    b = _run(ix, alt, _hidden(1, 11), retain=False, monkeypatch=monkeypatch)
    assert _same(a, b)


def test_clone_restore_and_truncate_helpers_start_clean(monkeypatch):
    ix = _indexer()
    ret, _ = _lockstep(ix, [13, 1, 1], monkeypatch)
    assert _frontier(ret) is not None
    c = clone_minimax_m3_sparse(ret, ret.offset - 1)
    assert c is not None and c.derived == {}
    r = restore_minimax_m3_sparse(*ret.state)
    assert r.derived == {}
    ref = MiniMaxM3SparseCache()
    ref.state = ret.state
    a = _run(ix, r, _hidden(2, 12), retain=True, monkeypatch=monkeypatch)
    b = _run(ix, ref, _hidden(2, 12), retain=False, monkeypatch=monkeypatch)
    assert _same(a, b)


def test_cross_request_isolation(monkeypatch):
    """Two requests through the same layer never share a frontier."""
    ix = _indexer()
    a_cache, _ = _lockstep(ix, [13, 1], monkeypatch)
    b_cache, _ = _lockstep(ix, [15, 1], monkeypatch)
    assert _frontier(a_cache) is not _frontier(b_cache)
    assert _frontier(a_cache).blocks == a_cache.offset // RATIO
    assert _frontier(b_cache).blocks == b_cache.offset // RATIO


def test_padded_batch_cache_takes_recompute_path(monkeypatch):
    """The batched sparse cache carries per-row left padding; retention is
    single-sequence only, and the batch path stays the exact full recompute."""
    ix = _indexer()
    monkeypatch.setenv("VMLX_QWEN4_QSA_POOL_RETAIN", "1")
    bc = BatchMiniMaxM3SparseCache([0, 2])
    x = _hidden(14, 21, batch=2)
    kv = mx.zeros((2, 1, 14, 4))
    bc.update_and_fetch(kv, kv)
    out = ix(x, bc, offset=0, position_ids=mx.broadcast_to(mx.arange(14)[None], (2, 14)))
    if out is not None:
        mx.eval(out)
    assert bc.derived == {}


def test_index_payload_stays_f32_and_authoritative(monkeypatch):
    ix = _indexer()
    ret, _ = _lockstep(ix, [13, 1], monkeypatch)
    assert ret.idx_keys.dtype == mx.float32
    assert ret.idx_keys.shape[-1] == ix.head_dim + 3
    # derived state is not part of the persisted tuple
    assert len(ret.state) == 3


def test_size_cap_evicts_but_stays_exact(monkeypatch):
    ix = _indexer()
    monkeypatch.setenv("VMLX_QWEN4_QSA_POOL_RETAIN_MAX_MB", "0.0001")  # ~105 bytes
    ret, ref = _lockstep(ix, [13, 1, 1, 1], monkeypatch)
    fr = _frontier(ret)
    assert fr is not None and fr.evicted > 0 and fr.pooled is None and fr.nbytes == 0
    monkeypatch.delenv("VMLX_QWEN4_QSA_POOL_RETAIN_MAX_MB")
    a = _run(ix, ret, _hidden(1, 77), retain=True, monkeypatch=monkeypatch)
    b = _run(ix, ref, _hidden(1, 77), retain=False, monkeypatch=monkeypatch)
    assert _same(a, b)
    assert _frontier(ret).nbytes > 0
