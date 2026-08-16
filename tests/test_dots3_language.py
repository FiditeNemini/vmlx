# SPDX-License-Identifier: Apache-2.0
"""dots3_note language model — absorbed/materialized equivalence pins.

The absorbed (latent-cache) path must compute the SAME attention as the
materialized stage-A path: q_nope' = q_nope @ W_kb_nope scores against the
latent exactly as q·k, and W_kb_v applies after the weights. These tests pin
that equivalence through prefill, chunked prefill, decode, the sliding
window, and the hysteresis trim — in float32 so any divergence is a math
bug, not quantization noise.
"""

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
    try:
        for i in range(20):
            latent = mx.ones((1, 1, 1, 3)) * i
            k_pe = mx.ones((1, 1, 1, 2)) * i
            fetched_latent, _ = cache.update_and_fetch(latent, k_pe)
            # The fetch must always include at least window-1 past + self
            # (until fewer exist), regardless of trim timing.
            assert fetched_latent.shape[2] >= min(i + 1, 6)
        assert cache.offset == 20
        # Stored state is bounded by window-1 + trim_step.
        assert cache.latent.shape[2] <= 5 + 4
    finally:
        Dots3LatentCache.trim_step = 256


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
