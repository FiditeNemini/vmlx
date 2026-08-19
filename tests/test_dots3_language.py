# SPDX-License-Identifier: Apache-2.0
"""dots3_note language model — absorbed/materialized equivalence pins.

The absorbed (latent-cache) path must compute the SAME attention as the
materialized stage-A path: q_nope' = q_nope @ W_kb_nope scores against the
latent exactly as q·k, and W_kb_v applies after the weights. These tests pin
that equivalence through prefill, chunked prefill, decode, the sliding
window, and the hysteresis trim — in float32 so any divergence is a math
bug, not quantization noise.
"""

import os

import mlx.core as mx
import mlx.nn as nn
import pytest

from vmlx_engine.models.dots3_note.config import TextConfig
from vmlx_engine.models.dots3_note.language import (
    Dots3LatentCache,
    LanguageModel,
)


def _tiny_config(**overrides):
    kwargs = dict(
        hidden_size=48,
        num_hidden_layers=4,
        vocab_size=96,
        intermediate_size=64,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=24,
        num_attention_heads=4,
        q_lora_rank=24,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        rope_theta=10000.0,
        swa_num_attention_heads=2,
        swa_q_lora_rank=24,
        swa_kv_lora_rank=20,
        swa_qk_nope_head_dim=12,
        swa_qk_rope_head_dim=4,
        swa_v_head_dim=8,
        swa_rope_theta=10000.0,
        sliding_window_size=6,
        index_topk=512,
        layer_types=[
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
    )
    kwargs.update(overrides)
    return TextConfig(**kwargs)


@pytest.fixture
def model():
    mx.random.seed(7)
    lm = LanguageModel(_tiny_config())
    # Randomize away from zero-init so the comparison is meaningful, and
    # keep float32 so absorbed-vs-materialized differences are math bugs.
    params = lm.parameters()
    import mlx.utils as u

    randomized = u.tree_map(
        lambda a: mx.random.normal(a.shape).astype(mx.float32) * 0.05, params
    )
    lm.update(randomized)
    mx.eval(lm.parameters())
    return lm


def _materialized_caches(lm):
    from mlx_lm.models.cache import KVCache

    return [KVCache() for _ in range(lm.text_config.num_hidden_layers)]


def _latent_caches(lm):
    cfg = lm.text_config
    return [
        Dots3LatentCache(
            window=cfg.sliding_window_size if cfg.is_sliding(i) else None
        )
        for i in range(cfg.num_hidden_layers)
    ]


def test_prefill_equivalence(model):
    ids = mx.array([[3, 17, 42, 9, 55, 20, 31, 8]])
    a = model(ids, cache=_materialized_caches(model))
    b = model(ids, cache=_latent_caches(model))
    assert float(mx.abs(a - b).max()) < 1e-4
    assert (mx.argmax(a, -1) == mx.argmax(b, -1)).all()


def test_decode_equivalence_past_window(model):
    # 20 tokens stepwise with window 6: exercises decode masks, the sliding
    # trim (trim_step lowered to force it), and full-layer growth.
    Dots3LatentCache.trim_step = 4
    try:
        mat = _materialized_caches(model)
        lat = _latent_caches(model)
        ids = [3, 17, 42, 9, 55, 20, 31, 8, 11, 4, 61, 2, 90, 33, 27, 5, 44, 70, 13, 6]
        prefix = mx.array([ids[:4]])
        a = model(prefix, cache=mat)
        b = model(prefix, cache=lat)
        assert float(mx.abs(a - b).max()) < 1e-4
        for tok in ids[4:]:
            step = mx.array([[tok]])
            a = model(step, cache=mat)
            b = model(step, cache=lat)
            assert float(mx.abs(a - b).max()) < 1e-4, f"diverged at token {tok}"
    finally:
        Dots3LatentCache.trim_step = 256


def test_chunked_prefill_matches_single_shot(model):
    ids = mx.array([[3, 17, 42, 9, 55, 20, 31, 8, 11, 4, 61, 2]])
    one = model(ids, cache=_latent_caches(model))
    chunked_cache = _latent_caches(model)
    out = None
    for start in range(0, 12, 4):
        out = model(ids[:, start : start + 4], cache=chunked_cache)
    assert float(mx.abs(one[:, -4:] - out).max()) < 1e-4


def test_context_bound_refuses_loudly_on_materialized_path(model):
    # The absorbed path handles >index_topk via the DSA scorer; only the
    # materialized (VMLX_DOTS3_MLA_ABSORB=0) path lacks an indexer stream
    # and must refuse rather than silently attend wrong.
    cfg = model.text_config
    ids = mx.array([[1] * (cfg.index_topk + 1)])
    with pytest.raises(ValueError, match="DSA"):
        model(ids, cache=_materialized_caches(model))


def test_sliding_cache_trim_keeps_window(model):
    cache = Dots3LatentCache(window=6)
    Dots3LatentCache.trim_step = 4
    _old_overhang = Dots3LatentCache.retain_overhang
    Dots3LatentCache.retain_overhang = 3
    try:
        for i in range(24):
            latent = mx.ones((1, 1, 1, 3)) * i
            k_pe = mx.ones((1, 1, 1, 2)) * i
            fetched_latent, _ = cache.update_and_fetch(latent, k_pe)
            # The fetch must always include at least window-1 past + self
            # (until fewer exist), regardless of trim timing.
            assert fetched_latent.shape[2] >= min(i + 1, 6)
        assert cache.offset == 24
        # Stored state is bounded by window-1 + retain_overhang + trim_step
        # (the overhang keeps recent block boundaries checkpointable,
        # ledger row 152).
        assert cache.latent.shape[2] <= 5 + 3 + 4
        # ... and never trims below window-1 + retain_overhang.
        assert cache.latent.shape[2] >= 5 + 3
    finally:
        Dots3LatentCache.trim_step = 256
        Dots3LatentCache.retain_overhang = _old_overhang


def test_dsa_selection_prefix_matches_dense(model):
    # index_topk=8 on a 12-token prompt: queries at positions 0..7 have <= 8
    # causal keys, so selection covers ALL of them and their rows must equal
    # the dense result exactly — through every layer (causality bounds the
    # reachable set). Later rows drop their lowest-scoring key and may
    # legitimately differ.
    cfg = model.text_config
    old = cfg.index_topk
    for layer in model.model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "indexer"):
            attn.indexer.index_topk = 8
    cfg.index_topk = 8
    try:
        ids = mx.array([[3, 17, 42, 9, 55, 20, 31, 8, 11, 4, 61, 2]])
        sparse = model(ids, cache=_latent_caches(model))
        cfg.index_topk = 4096
        for layer in model.model.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "indexer"):
                attn.indexer.index_topk = 4096
        dense = model(ids, cache=_latent_caches(model))
        assert float(mx.abs(sparse[:, :8] - dense[:, :8]).max()) < 1e-4
        assert bool(mx.isfinite(sparse).all())
    finally:
        cfg.index_topk = old
        for layer in model.model.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "indexer"):
                attn.indexer.index_topk = old


def test_dsa_decode_past_bound_runs(model):
    cfg = model.text_config
    old = cfg.index_topk
    cfg.index_topk = 6
    for layer in model.model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "indexer"):
            attn.indexer.index_topk = 6
    try:
        caches = _latent_caches(model)
        out = model(mx.array([[3, 17, 42, 9, 55, 20]]), cache=caches)
        for tok in (31, 8, 11, 4):
            out = model(mx.array([[tok]]), cache=caches)
            assert bool(mx.isfinite(out).all())
        # The full-layer indexer stream covered every token.
        full_layers = [
            i for i in range(cfg.num_hidden_layers) if not cfg.is_sliding(i)
        ]
        assert caches[full_layers[0]].idx_k.shape[1] == 10
    finally:
        cfg.index_topk = old
        for layer in model.model.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "indexer"):
                attn.indexer.index_topk = old


def test_chunked_prefill_past_window_with_trim(model):
    # The serve-level failure class: chunked prefill (S>1) after the sliding
    # cache trimmed — a global-position mask cannot broadcast against the
    # shorter physical key length. 30 tokens in chunks of 5, window 6,
    # trim_step 4, compared against the materialized path stepwise.
    Dots3LatentCache.trim_step = 4
    try:
        ids = list(range(3, 33))
        mat = _materialized_caches(model)
        lat = _latent_caches(model)
        a = b = None
        for start in range(0, 30, 5):
            chunk = mx.array([ids[start : start + 5]])
            a = model(chunk, cache=mat)
            b = model(chunk, cache=lat)
        assert float(mx.abs(a - b).max()) < 1e-4
    finally:
        Dots3LatentCache.trim_step = 256


def test_restored_generic_kvcache_is_adopted(model):
    # The prefix-cache positional reconstruction returns plain KVCache
    # objects (class identity is not stored per kv block). The model must
    # re-type them before attention, or packed latent streams get misread
    # as per-head K/V — plausible-but-wrong output, no exception.
    from mlx_lm.models.cache import KVCache

    cfg = model.text_config
    ref = _latent_caches(model)
    ids = mx.array([[3, 17, 42, 9, 55, 20, 31, 8]])
    model(ids, cache=ref)

    mixed = []
    for i, c in enumerate(ref):
        if not cfg.is_sliding(i):
            fake = KVCache()
            packed, latent = c.state
            fake.keys = packed
            fake.values = latent
            fake.offset = c.offset
            mixed.append(fake)
        else:
            mixed.append(Dots3LatentCache.from_state(c.state, c.meta_state))

    step = mx.array([[11]])
    ref_step_caches = ref
    a = model(step, cache=ref_step_caches)
    b = model(step, cache=mixed)
    assert isinstance(mixed[0], Dots3LatentCache), "adoption did not happen"
    assert float(mx.abs(a - b).max()) < 1e-4


def test_restored_empty_sliding_shells_are_adopted_with_window(model):
    # The SSD-only hit lane hands SLIDING slots back as EMPTY generic
    # KVCache shells (the sliding cumulative tuple only restores on exact
    # boundaries). Left un-adopted they run the materialized lane from the
    # restore point on — 64-head K/V, window enforced only by the mask so
    # nothing ever trims: measured live as ~1.7MB of retained Metal per
    # prefilled token across 33 sliding layers (active 96.8->118.7GB over
    # one 12.7k-token span). Adoption must re-enter the window-bounded
    # latent lane and keep the logical offset for RoPE.
    from mlx_lm.models.cache import KVCache

    cfg = model.text_config
    ref = _latent_caches(model)
    ids = mx.array([[3, 17, 42, 9, 55, 20, 31, 8]])
    model(ids, cache=ref)

    mixed = []
    for i, c in enumerate(ref):
        fake = KVCache()
        if not cfg.is_sliding(i):
            packed, latent = c.state
            fake.keys = packed
            fake.values = latent
        fake.offset = c.offset
        mixed.append(fake)

    out = model(mx.array([[11]]), cache=mixed)
    for i, c in enumerate(mixed):
        assert isinstance(c, Dots3LatentCache), f"layer {i} not adopted"
        if cfg.is_sliding(i):
            assert c.window == cfg.sliding_window_size
    # The adopted sliding cache must continue at the logical position, not
    # restart at zero (RoPE would otherwise rotate the wrong angles).
    sliding_idx = next(i for i in range(cfg.num_hidden_layers) if cfg.is_sliding(i))
    assert mixed[sliding_idx].offset == ref[sliding_idx].offset + 1
    assert bool(mx.isfinite(out).all())


def test_padding_mask_all_ones_equals_causal(model):
    # The MLLM generator forwards the processor's [B, S] all-ones PADDING
    # mask as `mask=` on every media forward. Consumed verbatim as an
    # ADDITIVE mask it cancels in softmax and deletes causality (measured on
    # the real bundle: temp-0 media answers changed). All-valid 2-D masks
    # must produce EXACTLY the causal output.
    ids = mx.array([[3, 17, 42, 9, 55, 20, 31, 8]])
    a = model(ids, cache=_latent_caches(model))
    b = model(
        ids,
        mask=mx.ones((1, ids.shape[1]), dtype=mx.int32),
        cache=_latent_caches(model),
    )
    assert float(mx.abs(a - b).max()) == 0.0


def test_padding_mask_with_zeros_refuses_loudly(model):
    # Real padding needs a physical-layout-aware merge with the sliding
    # lane; this single-sequence engine never produces it. Attending wrong
    # is worse than erroring.
    ids = mx.array([[3, 17, 42, 9]])
    mask = mx.array([[1, 1, 1, 0]], dtype=mx.int32)
    with pytest.raises(ValueError, match="padding"):
        model(ids, mask=mask, cache=_latent_caches(model))


def test_orphan_media_placeholders_refuse_loudly():
    # Placeholders in input_ids with NO payload forwarded: _scatter_at never
    # runs, so without the guard the model reads pad-token embeddings as
    # media and confabulates fluently (live: 38 orphaned audio pads made a
    # 440 Hz sine "pure unbroken silence" with zero errors).
    from vmlx_engine.models.dots3_note.config import ModelConfig
    from vmlx_engine.models.dots3_note.dots3_note import Model

    cfg = ModelConfig(text_config=_tiny_config())
    outer = Model(cfg)
    with pytest.raises(ValueError, match="audio placeholder"):
        outer.get_input_embeddings(
            input_ids=mx.array([[3, cfg.audio_token_id, 9]])
        )
    with pytest.raises(ValueError, match="image/video placeholder"):
        outer.get_input_embeddings(
            input_ids=mx.array([[3, cfg.image_token_id, 9]])
        )


# ---- partial block-restore lane (ledger row 152) -------------------------


def _block_store_roundtrip_env(n_tokens, seed, block_size=64, window=129):
    """Store a 4-layer dots3-shaped stack through the REAL block machinery."""
    import numpy as np

    from vmlx_engine.paged_cache import PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache
    from vmlx_engine.models.dots3_note.language import Dots3LatentCache
    from vmlx_engine.models.dots3_note_register import (
        register_dots3_note_runtime,
    )

    register_dots3_note_runtime()
    mx.random.seed(seed)
    rope, idx_dim, rank_full, rank_swa = 16, 32, 48, 96
    raw, originals = [], []
    for li in range(4):
        sliding = li in (1, 3)
        c = Dots3LatentCache(window=window if sliding else None)
        # Live truth: sliding (SWA-geometry) layers use a DIFFERENT latent
        # rank than full layers and never run the indexer — the first live
        # failure of this lane was a latent/k_pe SWAP that only a
        # rank-asymmetric, indexer-less sliding layer exposes.
        rank = rank_swa if sliding else rank_full
        latent = mx.random.normal((1, 1, n_tokens, rank)).astype(mx.bfloat16)
        k_pe = mx.random.normal((1, 1, n_tokens, rope)).astype(mx.bfloat16)
        c.update_and_fetch(latent, k_pe)
        idx_k = None
        if not sliding:
            idx_k = mx.random.normal((1, n_tokens, idx_dim)).astype(mx.bfloat16)
            c.update_indexer(idx_k)
        raw.append(c)
        originals.append((latent, k_pe, idx_k))
    mgr = PagedCacheManager(block_size=block_size, max_blocks=600)
    bac = BlockAwarePrefixCache(object(), mgr)
    tokens = list(range(1000, 1000 + n_tokens))
    states = [
        {
            "state": c.state,
            "meta_state": c.meta_state,
            "class_name": type(c).__name__,
        }
        for c in raw
    ]
    table = bac.store_cache("store-req", tokens, states)
    assert table is not None
    return np, bac, tokens, originals, raw


def test_partial_block_restore_covered_boundary_is_exact():
    # A divergent prompt sharing a block-aligned prefix within the retained
    # overhang restores a FULL-LENGTH cache list whose sliding state is
    # byte-equal to the original keys ending at the boundary (the live
    # failure was an EMPTY reconstruction -> forced cold prefill).
    np, bac, tokens, originals, raw = _block_store_roundtrip_env(
        n_tokens=512, seed=7
    )
    divergent = tokens[:256] + list(range(50000, 50064))
    table, _ = bac.fetch_cache("fetch-req", divergent)
    assert table is not None and table.num_tokens == 256
    rec = bac.reconstruct_cache(table)
    assert rec is not None and len(rec) == 4
    assert all(int(c.offset) == 256 for c in rec)
    # sliding layer (window 129, rank 96): boundary 256 needs keys 128..255;
    # the latent stream must keep ITS rank (a latent/k_pe swap is the live
    # failure mode this pins)
    slid = np.array(rec[1].latent.astype(mx.float32))
    assert slid.shape[-1] == 96
    ref = np.array(originals[1][0][:, :, 128:256].astype(mx.float32))
    assert np.array_equal(slid, ref)
    kpe = np.array(rec[1].k_pe.astype(mx.float32))
    assert kpe.shape[-1] == 16
    kref = np.array(originals[1][1][:, :, 128:256].astype(mx.float32))
    assert np.array_equal(kpe, kref)
    # sliding layers run no indexer: the restored stream must be absent
    assert rec[1].idx_k is None
    # full layer values = latent, positional slice 0..255
    full = np.array(rec[0].values.astype(mx.float32))
    fref = np.array(originals[0][0][:, :, :256].astype(mx.float32))
    assert np.array_equal(full, fref)


def test_partial_block_restore_uncovered_boundary_is_honest_miss():
    # A boundary deeper than the retained overhang can NEVER be rebuilt
    # exactly - the reconstruction must not hand back full-layer state with
    # empty sliding windows (silent drift). It comes back compacted (or
    # None), which the generator declines.
    np, bac, tokens, originals, raw = _block_store_roundtrip_env(
        n_tokens=2048, seed=11
    )
    slide_phys = int(raw[1].latent.shape[2])
    assert slide_phys < 2048  # trim engaged, deep history physically gone
    divergent = tokens[:256] + list(range(70000, 70064))
    table, _ = bac.fetch_cache("fetch-deep", divergent)
    if table is None:
        return  # even better: no hit claimed at all
    rec = bac.reconstruct_cache(table)
    assert rec is None or len(rec) < 4


def test_exact_restore_still_roundtrips_with_overhang():
    # The retention overhang must not disturb the exact-match lane.
    np, bac, tokens, originals, raw = _block_store_roundtrip_env(
        n_tokens=512, seed=13
    )
    table, _ = bac.fetch_cache("fetch-exact", tokens)
    assert table is not None
    rec = bac.reconstruct_cache(table)
    assert rec is not None and len(rec) == 4
    assert all(int(c.offset) == int(table.num_tokens) for c in rec)


def test_trim_to_boundary_semantics():
    from vmlx_engine.models.dots3_note.language import Dots3LatentCache

    mx.random.seed(3)
    c = Dots3LatentCache(window=129)
    latent = mx.random.normal((1, 1, 400, 48)).astype(mx.bfloat16)
    k_pe = mx.random.normal((1, 1, 400, 16)).astype(mx.bfloat16)
    c.update_and_fetch(latent, k_pe)
    assert c.trim_to_boundary(400) is True  # no-op at own offset
    assert c.trim_to_boundary(300) is True
    assert int(c.offset) == 300 and int(c.latent.shape[2]) == 128
    # keys 0..49 were dropped by the 300-rewind (128-key window) — a further
    # rewind to 50 is NOT reconstructible and must refuse.
    assert c.trim_to_boundary(50) is False
    assert c.trim_to_boundary(400) is False  # cannot go forward

    # A fresh untrimmed sliding cache CAN rewind below the window: all keys
    # are still physically present, and boundary < window keeps all of them.
    c2 = Dots3LatentCache(window=129)
    c2.update_and_fetch(
        mx.random.normal((1, 1, 400, 48)).astype(mx.bfloat16),
        mx.random.normal((1, 1, 400, 16)).astype(mx.bfloat16),
    )
    assert c2.trim_to_boundary(50) is True
    assert int(c2.offset) == 50 and int(c2.latent.shape[2]) == 50

    full = Dots3LatentCache()
    full.update_and_fetch(
        mx.random.normal((1, 1, 100, 48)).astype(mx.bfloat16),
        mx.random.normal((1, 1, 100, 16)).astype(mx.bfloat16),
    )
    assert full.trim_to_boundary(64) is True
    assert int(full.offset) == 64 and int(full.latent.shape[2]) == 64


def test_retention_overhang_does_not_change_outputs(model):
    # Longer sliding retention is a memory policy: the physical-layout mask
    # is (q - k) < window, so retained-but-out-of-window keys must be
    # invisible. Prefill+decode outputs must match a zero-overhang control.
    from vmlx_engine.models.dots3_note.language import Dots3LatentCache

    lm = model
    mx.random.seed(21)
    ids = mx.random.randint(0, 128, (1, 96))
    step = mx.random.randint(0, 128, (1, 1))

    old = Dots3LatentCache.retain_overhang
    try:
        Dots3LatentCache.retain_overhang = 0
        c0 = lm.make_cache()
        out0 = lm(ids, cache=c0)
        d0 = lm(step, cache=c0)
        Dots3LatentCache.retain_overhang = 64
        c1 = lm.make_cache()
        out1 = lm(ids, cache=c1)
        d1 = lm(step, cache=c1)
    finally:
        Dots3LatentCache.retain_overhang = old
    assert mx.array_equal(out0, out1).item()
    assert mx.array_equal(d0, d1).item()


# ---- prefill materialization from the latent (DSV4 split, ledger row 156) --


def _prefill_materialize_env(value):
    """Context-managed override of the prefill materialization ceiling."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        key = "VMLX_DOTS3_PREFILL_MATERIALIZE_MAX_KEYS"
        old = os.environ.get(key)
        os.environ[key] = str(value)
        try:
            yield
        finally:
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old

    return _ctx()


def test_prefill_materialization_engages_and_is_equivalent(model):
    # A guard that silently declines makes an A/B compare stock to stock, so
    # prove ENGAGEMENT first: the materialized branch must actually be taken
    # for the ceiling-on arm and skipped for the ceiling-off arm.
    from vmlx_engine.models.dots3_note import language as lang

    ids = mx.array([[3, 17, 42, 9, 55, 20, 31, 8, 61, 4, 77, 12]])
    seen = {"materialize": [], "absorb": []}
    real_einsum = lang.mx.einsum

    def _spy(spec, *args, **kwargs):
        if spec == "bltr,hnr->bhtn":
            seen["materialize"].append(spec)
        return real_einsum(spec, *args, **kwargs)

    lang.mx.einsum = _spy
    try:
        with _prefill_materialize_env(4096):
            hot = model(ids, cache=_latent_caches(model))
        engaged = len(seen["materialize"])
        seen["materialize"].clear()
        with _prefill_materialize_env(0):
            cold = model(ids, cache=_latent_caches(model))
        declined = len(seen["materialize"])
    finally:
        lang.mx.einsum = real_einsum

    assert engaged > 0, "materialized prefill branch never ran with the ceiling on"
    assert declined == 0, "materialized branch ran with the ceiling at 0"
    # Same keys, two representations of the same math: argmax must match and
    # the residual stays inside the bf16 association-order noise floor.
    assert (mx.argmax(hot, -1) == mx.argmax(cold, -1)).all()
    assert float(mx.abs(hot - cold).max()) < 1e-3


def test_prefill_materialization_respects_the_key_ceiling(model):
    # Above the ceiling the long-context latent path must stay in charge —
    # the 512K economics depend on never materializing a huge key block.
    from vmlx_engine.models.dots3_note import language as lang

    ids = mx.array([[3, 17, 42, 9, 55, 20, 31, 8]])
    hits = []
    real_einsum = lang.mx.einsum

    def _spy(spec, *args, **kwargs):
        if spec == "bltr,hnr->bhtn":
            hits.append(spec)
        return real_einsum(spec, *args, **kwargs)

    lang.mx.einsum = _spy
    try:
        with _prefill_materialize_env(4):  # 8 tokens > ceiling 4
            model(ids, cache=_latent_caches(model))
    finally:
        lang.mx.einsum = real_einsum
    assert hits == [], "materialization ignored the key ceiling"


def test_decode_never_materializes(model):
    # S == 1 must stay on the absorbed path (that is what the latent cache
    # exists for, and the fp32 S==1 SDPA trap lives there).
    from vmlx_engine.models.dots3_note import language as lang

    caches = _latent_caches(model)
    model(mx.array([[3, 17, 42, 9]]), cache=caches)
    hits = []
    real_einsum = lang.mx.einsum

    def _spy(spec, *args, **kwargs):
        if spec == "bltr,hnr->bhtn":
            hits.append(spec)
        return real_einsum(spec, *args, **kwargs)

    lang.mx.einsum = _spy
    try:
        with _prefill_materialize_env(1_000_000):
            model(mx.array([[55]]), cache=caches)
    finally:
        lang.mx.einsum = real_einsum
    assert hits == [], "decode step materialized K/V"


# ---- DSA gather-attention: O(S*topk) instead of O(S*total) -----------------


def _dsa_gather_env(value):
    """Context-managed override of the DSA gather-attention gate."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        key = "VMLX_DOTS3_DSA_GATHER"
        old = os.environ.get(key)
        os.environ[key] = str(value)
        try:
            yield
        finally:
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old

    return _ctx()


def _set_topk(model, k):
    model.text_config.index_topk = k
    for layer in model.model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "indexer"):
            attn.indexer.index_topk = k


def test_dsa_gather_matches_dense_mask_prefill(model):
    # Same selection, two attention representations: outputs must match on
    # EVERY row (unlike sparse-vs-dense, which only pins fully-covered
    # rows — here BOTH arms use the same selected keys). topk=8 on a
    # 12-token single shot also exercises the causal guard: rows 0..6 have
    # fewer than 8 valid keys, so the indexer returns future positions the
    # gather must re-mask -inf.
    old = model.text_config.index_topk
    _set_topk(model, 8)
    try:
        ids = mx.array([[3, 17, 42, 9, 55, 20, 31, 8, 11, 4, 61, 2]])
        dense = model(ids, cache=_latent_caches(model))
        with _dsa_gather_env("1"):
            gather = model(ids, cache=_latent_caches(model))
        assert float(mx.abs(dense - gather).max()) < 1e-4
        assert (mx.argmax(dense, -1) == mx.argmax(gather, -1)).all()
        assert bool(mx.isfinite(gather).all())
    finally:
        _set_topk(model, old)


def test_dsa_gather_matches_dense_mask_decode(model):
    # Stepwise decode past the bound. Cache content is path-independent
    # (writes happen before the attention-path split), so per-step logits
    # must match between the dense-mask and gather arms.
    old = model.text_config.index_topk
    _set_topk(model, 6)
    try:
        prefix = mx.array([[3, 17, 42, 9, 55, 20]])
        dense_caches = _latent_caches(model)
        gather_caches = _latent_caches(model)
        a = model(prefix, cache=dense_caches)
        with _dsa_gather_env("1"):
            b = model(prefix, cache=gather_caches)
        assert float(mx.abs(a - b).max()) < 1e-4
        for tok in (31, 8, 11, 4, 90, 33):
            step = mx.array([[tok]])
            a = model(step, cache=dense_caches)
            with _dsa_gather_env("1"):
                b = model(step, cache=gather_caches)
            assert float(mx.abs(a - b).max()) < 1e-4, f"diverged at {tok}"
    finally:
        _set_topk(model, old)


def test_dsa_gather_engages_and_declines(model):
    # ENGAGEMENT proof: the gather method must actually run when the gate
    # is on and DSA engages, and must NOT run when the gate is off or DSA
    # is not engaged — a silently-declining guard makes an A/B compare
    # stock to stock.
    from vmlx_engine.models.dots3_note.language import Dots3MLAAttention

    calls = []
    real = Dots3MLAAttention._dsa_gather_attention

    def _spy(self, *args, **kwargs):
        calls.append(1)
        return real(self, *args, **kwargs)

    Dots3MLAAttention._dsa_gather_attention = _spy
    old = model.text_config.index_topk
    _set_topk(model, 8)
    try:
        ids = mx.array([[3, 17, 42, 9, 55, 20, 31, 8, 11, 4, 61, 2]])
        with _dsa_gather_env("1"):
            model(ids, cache=_latent_caches(model))
        assert calls, "gather path never engaged with the gate on"
        calls.clear()
        with _dsa_gather_env("0"):
            model(ids, cache=_latent_caches(model))
        assert not calls, "gather path ran with the gate off"
        _set_topk(model, 4096)  # total <= topk: DSA itself must not engage
        with _dsa_gather_env("1"):
            model(ids, cache=_latent_caches(model))
        assert not calls, "gather path ran without DSA engagement"
    finally:
        Dots3MLAAttention._dsa_gather_attention = real
        _set_topk(model, old)


def test_dsa_gather_query_tiling_matches_single_tile(model):
    # Force multiple tiles (tiny element budget -> tile of ~2 queries) and
    # pin against the one-tile result: tiling is a memory policy, never a
    # math change.
    from vmlx_engine.models.dots3_note.language import Dots3MLAAttention

    old = model.text_config.index_topk
    _set_topk(model, 8)
    budget = Dots3MLAAttention.gather_element_budget
    ids = mx.array([[3, 17, 42, 9, 55, 20, 31, 8, 11, 4, 61, 2]])
    try:
        with _dsa_gather_env("1"):
            one = model(ids, cache=_latent_caches(model))
            # K=8, D=kv_lora_rank(16)+rope(4)=20 -> tile = 320//160 = 2.
            Dots3MLAAttention.gather_element_budget = 320
            tiled = model(ids, cache=_latent_caches(model))
        assert float(mx.abs(one - tiled).max()) < 1e-4
    finally:
        Dots3MLAAttention.gather_element_budget = budget
        _set_topk(model, old)


def test_dsa_gather_defaults_on_for_depth():
    """The gather path is DEFAULT ON, and that default is deliberate.

    2026-08-16 (ledger 173): it shipped OFF for a day because it was judged
    on speed, where the gain is percent-level. The real value is DEPTH — the
    dense [B,1,S,total] mask transient grows with context and is what
    exhausts prefill headroom. Matched A/B: dense 413'd at 12,608 context on
    two independent runs while gather served 16,384 (and was 8.7% faster at
    8k). Pin the default so a future edit cannot quietly hand users back the
    hard 413 at 16k.
    """
    import importlib

    from vmlx_engine.models.dots3_note import language as lang

    key = "VMLX_DOTS3_DSA_GATHER"
    old = os.environ.pop(key, None)
    try:
        importlib.reload(lang)
        assert lang._dsa_gather_enabled() is True
        os.environ[key] = "0"
        assert lang._dsa_gather_enabled() is False  # the escape hatch still works
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old

