"""Hybrid (SSM/GDN + attention) companion checkpoints every N prompt tokens (VMLX_HYBRID_BOUNDARY_CHECKPOINT_TOKENS).

Without them the companion exists only at request ends, so a later prompt diverging before the end of a stored one
found its KV blocks but no companion and re-prefilled everything (Qwen3.8 Flash-Next, 37.7k tokens: 33.6 s). The
vmlx#91 resume restores the longest stored checkpoint below the KV hit; resuming a GDN hybrid from any boundary is
bit-identical to a cold prefill (tests/test_hybrid_chunked_prefill_equivalence.py). These tests pin the planning
(absolute block-aligned multiples, warm offsets, the clean boundary, media) and the routing (a periodic checkpoint is
stored to the companion cache at capture time and is never carried on the request).
"""
from types import MethodType, SimpleNamespace

import mlx.core as mx
import pytest

from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator


class _Store:
    def __init__(self):
        self.stored = []

    def store(self, tokens, n, layers, is_complete=True, cache_extra_keys=None):
        self.stored.append((n, len(tokens), len(layers), is_complete, cache_extra_keys))


def _gen(media_limit=None):
    g = SimpleNamespace(
        _is_hybrid=True,
        block_aware_cache=SimpleNamespace(block_size=64),
        _ssm_state_cache=_Store(),
        _hybrid_kv_positions=[1],
        _ssm_companion_enabled=True,
    )
    g._ssm_block_aligned_boundary = MethodType(MLLMBatchGenerator._ssm_block_aligned_boundary, g)
    g._media_safe_capture_limit = (lambda toks: len(toks)) if media_limit is None else (lambda toks: media_limit)
    return g


def _plan(g, req, seq_len, clean):
    return MLLMBatchGenerator._ssm_capture_boundaries_for(g, req, seq_len, False, clean)


def test_cold_prompt_plans_absolute_multiples_below_the_clean_boundary(monkeypatch):
    monkeypatch.delenv("VMLX_HYBRID_BOUNDARY_CHECKPOINT_TOKENS", raising=False)
    g = _gen()
    req = SimpleNamespace(_cached_tokens=0, _original_token_ids=list(range(13001)), request_id="cold")
    got = _plan(g, req, 13000, 12990)
    assert req._periodic_ssm_boundaries == {4096, 8192, 12288}
    assert {4096, 8192, 12288, 12990} <= set(got) and got == sorted(got)


def test_warm_request_plans_in_absolute_positions_and_skips_planned_ones(monkeypatch):
    monkeypatch.setenv("VMLX_HYBRID_BOUNDARY_CHECKPOINT_TOKENS", "1000")         # rounded up to 1024 (block 64)
    g = _gen()
    req = SimpleNamespace(_cached_tokens=5000, _original_token_ids=list(range(9001)), request_id="warm")
    got = _plan(g, req, 4000, 0)
    assert req._periodic_ssm_boundaries == {5120, 6144, 7168, 8192}
    assert [5000 + b for b in got] == [5120, 6144, 7168, 8192]
    # a required checkpoint that IS a multiple stays a carried (non-periodic) checkpoint
    req2 = SimpleNamespace(_cached_tokens=0, _original_token_ids=list(range(5000)), request_id="req",
                           _ssm_required_checkpoint_tokens=2048)
    _plan(g, req2, 4999, 0)
    assert 2048 not in req2._periodic_ssm_boundaries and 1024 in req2._periodic_ssm_boundaries


def test_off_switch_and_media_limit(monkeypatch):
    monkeypatch.setenv("VMLX_HYBRID_BOUNDARY_CHECKPOINT_TOKENS", "0")
    req = SimpleNamespace(_cached_tokens=0, _original_token_ids=list(range(9000)), request_id="off")
    assert _plan(_gen(), req, 9000, 0) == [] and req._periodic_ssm_boundaries == set()
    monkeypatch.setenv("VMLX_HYBRID_BOUNDARY_CHECKPOINT_TOKENS", "4096")
    req = SimpleNamespace(_cached_tokens=0, _original_token_ids=list(range(20000)), request_id="media")
    _plan(_gen(media_limit=9000), req, 20000, 0)          # first media placeholder at 9000: nothing past it
    assert req._periodic_ssm_boundaries == {4096, 8192}


def _cache():
    from mlx_lm.models.cache import ArraysCache, KVCache
    ssm = ArraysCache(size=2)
    ssm[0] = mx.ones((1, 3, 8))
    ssm[1] = mx.ones((1, 2, 4, 4))
    kv = KVCache()
    kv.update_and_fetch(mx.zeros((1, 2, 4, 8)), mx.zeros((1, 2, 4, 8)))
    return [ssm, kv]


def test_periodic_capture_is_stored_now_and_not_carried(monkeypatch):
    monkeypatch.delenv("VMLX_HYBRID_BOUNDARY_CHECKPOINT_TOKENS", raising=False)
    g = _gen()
    req = SimpleNamespace(_cached_tokens=0, _original_token_ids=list(range(13001)), request_id="cap",
                          _cache_extra_keys=("gp", 7))
    _plan(g, req, 13000, 12990)
    tokens = req._original_token_ids
    assert MLLMBatchGenerator._maybe_capture_clean_ssm_boundary(g, req, _cache(), tokens, 4096)
    assert g._ssm_state_cache.stored == [(4096, 4096, 1, True, ("gp", 7))]
    assert not getattr(req, "_inline_ssm_checkpoints", None)
    # the clean boundary is still carried for the scheduler's clean-boundary assembly
    assert MLLMBatchGenerator._maybe_capture_clean_ssm_boundary(g, req, _cache(), tokens, 12990)
    assert [cp[0] for cp in req._inline_ssm_checkpoints] == [12990]
    assert len(g._ssm_state_cache.stored) == 1


def test_bypass_request_stores_nothing(monkeypatch):
    monkeypatch.delenv("VMLX_HYBRID_BOUNDARY_CHECKPOINT_TOKENS", raising=False)
    g = _gen()
    req = SimpleNamespace(_cached_tokens=0, _original_token_ids=list(range(9001)), request_id="bypass",
                          _bypass_prefix_cache=True)
    _plan(g, req, 9000, 0)
    assert MLLMBatchGenerator._maybe_capture_clean_ssm_boundary(g, req, _cache(), req._original_token_ids, 4096)
    assert g._ssm_state_cache.stored == [] and not getattr(req, "_inline_ssm_checkpoints", None)
