"""Periodic exact Naive checkpoints (I-28): window-only capture + assembly from the N-1 snapshot.

A boundary checkpoint must equal, leaf by leaf, a cache that was only ever fed the boundary's tokens: append-only
full-attention leaves are slices of the later N-1 snapshot, sliding-window leaves are the copies captured when the
prefill stood exactly at the boundary.
"""
import pytest

mx = pytest.importorskip("mlx.core")
from mlx_lm.models.cache import CacheList, KVCache, RotatingKVCache  # noqa: E402

from vmlx_engine.models.naive_n05_flash import cache_snapshot as CS  # noqa: E402

H, D, WINDOW, LAYERS = 2, 8, 8, 2


def _cache():
    return [CacheList(KVCache(), RotatingKVCache(max_size=WINDOW)) for _ in range(LAYERS)]


def _feed(cache, chunks):
    for keys in chunks:
        for layer in cache:
            for leaf in layer.caches:
                leaf.update_and_fetch(keys, keys * 2.0)
    mx.eval([layer.state for layer in cache])


def _chunk(n, seed):
    mx.random.seed(seed)
    return mx.random.normal((1, H, n, D))


def _leaves(cache):
    for layer in cache:
        for leaf in layer.caches:
            yield leaf


def test_boundary_checkpoint_equals_a_cache_fed_only_to_the_boundary():
    a, b = _chunk(16, 1), _chunk(20, 2)
    live = _cache()
    _feed(live, [a])
    windows = CS.clone_rotating_windows(live)          # prefill stands exactly at 16
    _feed(live, [b])
    snapshot = CS.clone_prompt_cache(live)             # the N-1 snapshot at 36
    got = CS.assemble_boundary_checkpoint(snapshot, windows, 16)

    ref = _cache()
    _feed(ref, [a])
    want = CS.clone_prompt_cache(ref)
    assert CS.snapshot_size(got, expected_tokens=16) == CS.snapshot_size(want, expected_tokens=16)
    for g, w in zip(_leaves(got), _leaves(want)):
        assert type(g) is type(w) and int(g.offset) == int(w.offset) == 16
        for ga, wa in zip(g.state, w.state):
            assert ga.shape == wa.shape
            assert mx.array_equal(ga, wa).item()
        if type(g) is RotatingKVCache:
            assert tuple(map(str, g.meta_state)) == tuple(map(str, w.meta_state))


def test_window_capture_copies_only_the_sliding_leaves():
    live = _cache()
    _feed(live, [_chunk(16, 3)])
    windows = CS.clone_rotating_windows(live)
    for layer in windows:
        assert layer[0] == "list"
        kv, rot = layer[1]
        assert kv is None and type(rot) is RotatingKVCache and int(rot.offset) == 16


def test_assembly_rejects_mismatched_offsets_and_structures():
    live = _cache()
    _feed(live, [_chunk(16, 4)])
    windows = CS.clone_rotating_windows(live)
    _feed(live, [_chunk(4, 5)])
    snapshot = CS.clone_prompt_cache(live)
    with pytest.raises(ValueError):
        CS.assemble_boundary_checkpoint(snapshot, windows, 12)       # windows were taken at 16
    with pytest.raises(ValueError):
        CS.assemble_boundary_checkpoint(snapshot[:1], windows, 16)   # structure mismatch
