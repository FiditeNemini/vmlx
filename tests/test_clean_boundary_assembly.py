"""The clean prompt-boundary cache is assembled, not re-prefilled.

A path-dependent store needs the cache as it stood at the prompt boundary.
Because recurrent state and rotating windows cannot be recovered from
post-generation state, the store used to re-prefill the ENTIRE prompt — a
second full forward pass, profiled at 40.8% of engine time (~28s at 15.4k
tokens) and running after the response was dispatched, so it blocked the next
request. Message 2 of a fresh conversation cost ~36s despite being a full
cache hit; message 3 cost 1.8s. With the assembly: 2.36s.

For the pure-hybrid layout the second pass is unnecessary — append-only
attention KV slices back to the boundary exactly, and the recurrent layers
were deep-copied at the boundary before generation advanced them.

These tests pin the two things that make it safe: it must assemble exactly,
and it must DECLINE (returning None, so the caller re-prefills) whenever the
layout is one where slicing would be wrong.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from vmlx_engine.mllm_scheduler import (
    _assemble_clean_hybrid_boundary_cache,
    _assemble_mixed_swa_boundary_cache,
)


class _Gen:
    def __init__(self, kv_positions):
        self._hybrid_kv_positions = list(kv_positions)


class _Req:
    def __init__(self, checkpoints=None, request_id="r1"):
        self.request_id = request_id
        if checkpoints is not None:
            self._clean_boundary_recurrent = checkpoints


class _Recurrent:
    """A non-KV layer (SSM/GatedDelta state)."""

    def __init__(self, tag):
        self.tag = tag


class RotatingKVCache:
    """Minimal exact-name rotating cache used by the capture contract."""

    def __init__(self, *, boundary=10, max_size=8):
        # End-of-prefill state includes the final prompt token. The concat
        # overhang retains one extra row, so the N-1 boundary is still exact.
        self.max_size = max_size
        self.keep = 0
        self.offset = boundary + 1
        physical = max_size + 1
        values = mx.arange(physical * 4).reshape(1, 1, physical, 4)
        self.keys = values.astype(mx.float16)
        self.values = self.keys
        self._idx = physical


def _kv(n_tokens, physical=None, dim=8, heads=2):
    from mlx_lm.models.cache import KVCache

    c = KVCache()
    physical = physical or n_tokens
    real = mx.ones((1, heads, n_tokens, dim))
    if physical > n_tokens:
        pad = mx.zeros((1, heads, physical - n_tokens, dim))
        c.keys = mx.concatenate([real, pad], axis=2)
        c.values = mx.concatenate([real, pad], axis=2)
    else:
        c.keys, c.values = real, real
    c.offset = n_tokens
    return c


def _layout(boundary, generated=5, physical_pad=0):
    """2 attention layers (positions 0,2) + 2 recurrent layers."""
    live = [
        _kv(boundary + generated, physical=boundary + generated + physical_pad),
        _Recurrent("live-recurrent-CONTAMINATED"),
        _kv(boundary + generated, physical=boundary + generated + physical_pad),
        _Recurrent("live-recurrent-CONTAMINATED"),
    ]
    snapshot = [_Recurrent("boundary-A"), _Recurrent("boundary-B")]
    return live, snapshot


def test_assembles_attention_sliced_to_the_boundary():
    live, snap = _layout(boundary=100)
    out = _assemble_clean_hybrid_boundary_cache(
        _Gen([0, 2]), _Req([(100, list(range(100)), snap)]), live, 100
    )
    assert out is not None, "should have assembled"
    assert len(out) == 4
    for idx in (0, 2):
        assert out[idx].keys.shape[2] == 100
        assert out[idx].offset == 100
        assert float(mx.min(out[idx].keys)) == 1.0, "a pad/generated row leaked in"


def test_uses_the_boundary_snapshot_not_the_contaminated_live_state():
    """The whole point: generation advanced the live recurrent layers."""
    live, snap = _layout(boundary=100)
    out = _assemble_clean_hybrid_boundary_cache(
        _Gen([0, 2]), _Req([(100, list(range(100)), snap)]), live, 100
    )
    assert [out[1].tag, out[3].tag] == ["boundary-A", "boundary-B"]


def test_slices_past_allocation_slack():
    """A live KVCache allocates in step chunks; the slack is not tokens."""
    live, snap = _layout(boundary=100, physical_pad=151)
    out = _assemble_clean_hybrid_boundary_cache(
        _Gen([0, 2]), _Req([(100, list(range(100)), snap)]), live, 100
    )
    assert out[0].keys.shape[2] == 100
    assert float(mx.min(out[0].keys)) == 1.0


def test_declines_without_a_snapshot_at_that_boundary():
    live, snap = _layout(boundary=100)
    req = _Req([(64, list(range(64)), snap)])  # wrong boundary
    assert _assemble_clean_hybrid_boundary_cache(_Gen([0, 2]), req, live, 100) is None


def test_declines_on_rotating_attention():
    """Hybrid assembly must defer rotating layers to the snapshot path."""
    from mlx_lm.models.cache import RotatingKVCache

    live, snap = _layout(boundary=100)
    live[0] = RotatingKVCache(max_size=64)
    assert _assemble_clean_hybrid_boundary_cache(
        _Gen([0, 2]), _Req([(100, list(range(100)), snap)]), live, 100
    ) is None


def test_declines_on_quantized_attention():
    from mlx_lm.models.cache import QuantizedKVCache

    live, snap = _layout(boundary=100)
    live[2] = QuantizedKVCache(group_size=64, bits=4)
    assert _assemble_clean_hybrid_boundary_cache(
        _Gen([0, 2]), _Req([(100, list(range(100)), snap)]), live, 100
    ) is None


def test_declines_when_recurrent_count_mismatches():
    live, snap = _layout(boundary=100)
    assert _assemble_clean_hybrid_boundary_cache(
        _Gen([0, 2]), _Req([(100, [], [snap[0]])]), live, 100
    ) is None


def test_declines_when_the_live_cache_is_shorter_than_the_boundary():
    live, snap = _layout(boundary=100, generated=0)
    assert _assemble_clean_hybrid_boundary_cache(
        _Gen([0, 2]), _Req([(200, list(range(200)), snap)]), live, 200
    ) is None


@pytest.mark.parametrize("bad", [0, -1])
def test_declines_on_a_nonsense_boundary(bad):
    live, snap = _layout(boundary=100)
    assert _assemble_clean_hybrid_boundary_cache(
        _Gen([0, 2]), _Req([(100, [], snap)]), live, bad
    ) is None


def test_declines_without_kv_positions():
    live, snap = _layout(boundary=100)
    assert _assemble_clean_hybrid_boundary_cache(
        _Gen([]), _Req([(100, [], snap)]), live, 100
    ) is None


def test_keyed_handoff_is_consumed_from_the_generator():
    """The scheduler holds a different request wrapper than the generator, so
    the snapshot is handed over by request_id."""
    live, snap = _layout(boundary=100)
    gen = _Gen([0, 2])
    gen._clean_boundary_snapshots = {"rX": [(100, list(range(100)), snap)]}
    req = _Req(checkpoints=None, request_id="rX")
    out = _assemble_clean_hybrid_boundary_cache(gen, req, live, 100)
    assert out is not None
    assert gen._clean_boundary_snapshots == {}, "snapshot must be consumed"


def _mixed_swa_generator(*, has_media=False, media_allowed=True):
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen._prefix_cache_enabled = True
    gen._mixed_attention_cache_model = True
    gen._request_has_media_cache_context = lambda _request, _tokens: has_media
    gen._media_prefix_cache_allowed = (
        lambda _request, _tokens: media_allowed
    )
    return gen


def test_text_mixed_swa_captures_exact_boundary_without_media_gate():
    """Text Step/Gemma turns must not pay a second clean prompt forward."""
    gen = _mixed_swa_generator(has_media=False, media_allowed=False)
    req = SimpleNamespace(request_id="text", _original_token_ids=list(range(11)))
    rotating = RotatingKVCache(boundary=10)

    assert gen._maybe_capture_mixed_swa_boundary(req, [rotating])
    boundary, snapshots = req._mixed_swa_boundary
    assert boundary == 10
    assert list(snapshots) == [0]
    snap = snapshots[0]
    assert snap is not rotating
    assert snap.offset == 10
    assert snap._idx == 8
    assert snap.keys.shape[2] == 8


def test_mixed_swa_capture_is_skipped_when_no_store_can_consume_it():
    req = SimpleNamespace(request_id="off", _original_token_ids=list(range(11)))
    disabled = _mixed_swa_generator(has_media=False)
    disabled._prefix_cache_enabled = False
    assert not disabled._maybe_capture_mixed_swa_boundary(
        req, [RotatingKVCache(boundary=10)]
    )

    complete = _mixed_swa_generator(has_media=False)
    req._cached_tokens = 10
    assert not complete._maybe_capture_mixed_swa_boundary(
        req, [RotatingKVCache(boundary=10)]
    )
    assert req._mixed_swa_boundary is None


def test_media_mixed_swa_still_fails_closed_when_media_cache_is_disallowed():
    gen = _mixed_swa_generator(has_media=True, media_allowed=False)
    req = SimpleNamespace(request_id="media", _original_token_ids=list(range(11)))

    assert not gen._maybe_capture_mixed_swa_boundary(
        req, [RotatingKVCache(boundary=10)]
    )
    assert req._mixed_swa_boundary is None


def test_mixed_swa_assembly_uses_snapshot_and_slices_full_attention_kv():
    """Rotating state comes from prefill; append-only KV comes from finish."""
    gen = _mixed_swa_generator(has_media=False)
    req = SimpleNamespace(request_id="handoff", _original_token_ids=list(range(11)))
    prefill_rotating = RotatingKVCache(boundary=10)
    assert gen._maybe_capture_mixed_swa_boundary(req, [prefill_rotating])

    captured = req._mixed_swa_boundary
    gen._mixed_swa_boundary_snapshots = {"handoff": captured}
    req._mixed_swa_boundary = None
    finish_rotating = RotatingKVCache(boundary=14)
    finish_kv = _kv(14)
    out = _assemble_mixed_swa_boundary_cache(
        gen,
        req,
        [finish_rotating, finish_kv],
        10,
    )

    assert out is not None
    assert out[0].offset == 10
    assert out[0].keys.shape[2] == 8
    assert out[1].offset == 10
    assert out[1].keys.shape[2] == 10
    assert gen._mixed_swa_boundary_snapshots == {}, "handoff must be consumed"


def test_mixed_swa_assembly_declines_rolled_finish_cache_without_capture():
    gen = _mixed_swa_generator(has_media=False)
    req = SimpleNamespace(request_id="missing")
    assert _assemble_mixed_swa_boundary_cache(
        gen,
        req,
        [RotatingKVCache(boundary=14), _kv(14)],
        10,
    ) is None


def test_scheduler_prefers_boundary_assembly_and_retires_unconsumed_handoff():
    """Pin the owning cleanup wiring, including the zero-retained-RAM exit."""
    import inspect

    from vmlx_engine.mllm_scheduler import MLLMScheduler

    source = inspect.getsource(MLLMScheduler._cleanup_finished)
    assembly = source.index("_assemble_mixed_swa_boundary_cache(")
    fallback = source.index('"_prefill_for_clean_path_dependent_cache"')
    assert assembly < fallback
    assert '_swa_snapshots.pop(str(request_id), None)' in source
