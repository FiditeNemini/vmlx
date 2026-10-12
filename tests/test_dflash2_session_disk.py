"""DFlash2 session SSD tier: exact round trip of target/draft caches, prefix-cut lookup, misses never restore wrong state."""
import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

from vmlx_engine.dflash2_session_disk import DFlash2SessionSSD, candidate_cuts

IM_START, IM_END = 900, 901


def _target(n):
    kv, gdn = KVCache(), ArraysCache(size=2)
    kv.update_and_fetch(mx.random.normal((1, 2, n, 8)), mx.random.normal((1, 2, n, 8)))
    gdn[0] = mx.random.normal((1, 3, 16))
    gdn[1] = mx.random.normal((1, 2, 4, 4))
    return [kv, gdn]


def _draft(n):
    c = RotatingKVCache(max_size=4)
    for _ in range(n):
        c.update_and_fetch(mx.random.normal((1, 1, 1, 8)), mx.random.normal((1, 1, 1, 8)))
    return [c]


@pytest.fixture
def ssd(tmp_path):
    s = DFlash2SessionSSD(root=tmp_path, max_size_bytes=1 << 30, model_key="unit")
    yield s
    s.store.close()


def _take(ssd, prompt):
    return ssd.take_matching(prompt, im_start_id=IM_START, eos_ids=[IM_END],
                             make_target=lambda: [KVCache(), ArraysCache(size=2)],
                             make_draft=lambda: [RotatingKVCache(max_size=4)])


def test_candidate_cuts_are_tag_and_after_eos_positions_longest_first():
    p = [IM_START, 1, 2, IM_END, 3, IM_START, 4, IM_END, 5]
    assert candidate_cuts(p, IM_START, [IM_END]) == [8, 5, 4]


def test_turn_entry_round_trips_exactly_and_matches_a_longer_prompt(ssd):
    tokens = [IM_START, 1, 2, IM_START, 7, 8, IM_END]
    target, draft, gap = _target(6), _draft(5), mx.random.normal((1, 1, 5))
    ssd.put({"kind": "turn", "tokens": tokens, "cache_len": 6, "target_cache": target,
             "draft_cache": draft, "draft_hidden_gap": gap})
    assert ssd.flush()
    got = _take(ssd, tokens + [10, IM_START, 11])
    assert got is not None and got["kind"] == "turn" and got["cache_len"] == 6 and got["tokens"] == tokens
    for a, b in zip(target[0].state, got["target_cache"][0].state):
        assert mx.array_equal(a, b)
    assert got["target_cache"][0].offset == 6
    for a, b in zip(target[1].state, got["target_cache"][1].state):
        assert mx.array_equal(a, b)
    d0, d1 = draft[0], got["draft_cache"][0]
    assert d1.meta_state == d0.meta_state and d1.offset == d0.offset
    assert all(mx.array_equal(a, b) for a, b in zip(d0.state, d1.state))
    assert mx.array_equal(got["draft_hidden_gap"], gap)


def test_boundary_entry_without_draft_matches_at_the_tag(ssd):
    prefix = [IM_START, 1, 2, IM_END, 5]
    ssd.put({"kind": "boundary", "tokens": prefix, "cache_len": 5, "target_cache": _target(5),
             "draft_cache": None, "draft_hidden_gap": None})
    assert ssd.flush()
    got = _take(ssd, prefix + [IM_START, 6, 7])
    assert got is not None and got["draft_cache"] is None and got["cache_len"] == 5


def test_diverged_prompt_and_entry_covering_whole_prompt_are_misses(ssd):
    tokens = [IM_START, 1, 2, IM_END]
    ssd.put({"kind": "turn", "tokens": tokens, "cache_len": 3, "target_cache": _target(3),
             "draft_cache": None, "draft_hidden_gap": None})
    assert ssd.flush()
    assert _take(ssd, [IM_START, 1, 3, IM_END, IM_START, 4]) is None   # different history
    assert _take(ssd, tokens) is None                                   # nothing left to prefill


def test_cache_layout_mismatch_is_a_miss(ssd):
    tokens = [IM_START, 1, IM_END]
    ssd.put({"kind": "turn", "tokens": tokens, "cache_len": 2, "target_cache": _target(2),
             "draft_cache": None, "draft_hidden_gap": None})
    assert ssd.flush()
    got = ssd.take_matching(tokens + [IM_START], im_start_id=IM_START, eos_ids=[IM_END],
                            make_target=lambda: [KVCache(), KVCache()], make_draft=lambda: [])
    assert got is None


def test_clear_removes_entries(ssd):
    tokens = [IM_START, 1, IM_END]
    ssd.put({"kind": "turn", "tokens": tokens, "cache_len": 2, "target_cache": _target(2),
             "draft_cache": None, "draft_hidden_gap": None})
    assert ssd.flush()
    assert ssd.clear() > 0
    assert _take(ssd, tokens + [IM_START]) is None


def test_empty_draft_cache_is_stored_without_draft_state(ssd):
    tokens = [IM_START, 1, IM_END]
    ssd.put({"kind": "turn", "tokens": tokens, "cache_len": 2, "target_cache": _target(2),
             "draft_cache": [RotatingKVCache(max_size=4)], "draft_hidden_gap": None})   # never ran
    assert ssd.flush()
    got = _take(ssd, tokens + [IM_START])
    assert got is not None and got["draft_cache"] is None and got["cache_len"] == 2


def test_boundary_draft_context_survives_ssd_without_draft_kv(ssd):
    prefix = [IM_START, 1, 2, IM_END, 5]
    context = mx.arange(12).reshape(1, 4, 3).astype(mx.bfloat16)
    ssd.put({"kind": "boundary", "tokens": prefix, "cache_len": 5,
             "target_cache": _target(5), "draft_cache": None,
             "draft_hidden_gap": None, "draft_context": context})
    assert ssd.flush()
    got = _take(ssd, prefix + [IM_START, 6, 7])
    assert got["draft_cache"] is None
    assert got["draft_context"].dtype == context.dtype
    assert mx.array_equal(got["draft_context"], context)


def test_boundary_snapshot_keeps_bounded_window_and_absolute_resume_position():
    from vmlx_engine.dflash2_runtime import _snapshot_draft_state, _rebuild_draft_hidden
    old = mx.arange(10).reshape(1, 10, 1)
    state = _snapshot_draft_state(None, None, None, [old], 10, 0, 4)
    assert state['draft_context'].tolist() == [[[6], [7], [8], [9]]]
    suffix = mx.array([[[10], [11]]])
    hidden, offset = _rebuild_draft_hidden(suffix, state, 10, 2, 0, 4)
    assert hidden.tolist() == [[[8], [9], [10], [11]]]
    assert offset == 8
    # A large delta already contains the whole sliding window: no stale
    # historical rows may be prepended across the omitted delta positions.
    hidden, offset = _rebuild_draft_hidden(suffix, state, 10, 5, 3, 2)
    assert hidden.tolist() == suffix.tolist() and offset == 13


def test_boundary_after_turn_preserves_draft_kv_and_contiguous_gap():
    from types import SimpleNamespace
    from vmlx_engine.dflash2_runtime import _snapshot_draft_state
    draft = _draft(3)
    gap = mx.array([[[3], [4]]])
    prior = {'draft_cache': draft, 'draft_hidden_gap': gap, 'cache_len': 5}
    runtime = SimpleNamespace(make_prompt_cache=lambda _: [RotatingKVCache(max_size=4)])
    state = _snapshot_draft_state(runtime, None, prior, [mx.array([[[5], [6]]])], 7, 5, 4)
    assert state['draft_cache'][0] is not draft[0]
    assert state['draft_cache'][0].offset == 3
    assert state['draft_hidden_gap'].tolist() == [[[3], [4], [5], [6]]]
    assert state['draft_context'] is None


def test_legacy_partial_suffix_is_not_certified_as_full_draft_context():
    from vmlx_engine.dflash2_runtime import _snapshot_draft_state
    state = _snapshot_draft_state(None, None, {}, [mx.array([[[8], [9]]])], 10, 8, 4)
    assert state['draft_context'] is None


@pytest.mark.parametrize("legacy", ["dflash2_session_v1", "dflash2_session_v2", "dflash2_session_v3"])
def test_v4_schema_clears_legacy_entries(tmp_path, legacy):
    """v1 entries were written while snapshots aliased live GDN state; v2 drafter caches rebuilt on resume could hold
    zero K/V rows (forced absolute offset); a v3 reader would restore a v4 delta entry as a whole cache. Never read
    them, clear them at startup."""
    import mlx.core as mx
    from vmlx_engine.dflash2_session_disk import DFlash2SessionSSD, SCHEMA
    from vmlx_engine.utils.omni_session_disk_store import OmniSessionDiskStore

    assert SCHEMA == "dflash2_session_v4"
    old = OmniSessionDiskStore(root=tmp_path, model_key="m", max_size_bytes=1 << 30, schema=legacy)
    old.save("sig", lambda tmp: mx.save_safetensors(str(tmp), {"a": mx.ones((4,))}, {"dflash2": "{}"}))
    old.close()
    assert any(old.directory.glob("*.safetensors"))
    ssd = DFlash2SessionSSD(root=tmp_path, max_size_bytes=1 << 30, model_key="m")
    assert ssd.legacy_cleared > 0
    assert not any(old.directory.glob("*.safetensors"))


# ---- delta entries (v4, 2026-10-12) ---------------------------------------------------------------------------
def _grow(target, n_more):
    """Append n_more rows to the KV layer and advance the GDN state, as a continued prefill/decode would."""
    kv, gdn = target
    kv.update_and_fetch(mx.random.normal((1, 2, n_more, 8)), mx.random.normal((1, 2, n_more, 8)))
    gdn[0] = gdn[0] * 0.5 + 1.0
    gdn[1] = gdn[1] * 0.25 - 1.0
    return target


def test_delta_chain_restores_bit_identical_and_writes_only_new_rows(ssd):
    system = [IM_START, 1, 2, 3]
    boundary = system + [IM_START, 4, 5, 6]
    turn = boundary + [IM_START, 7, 8, IM_END]
    target = _target(len(system))
    r1 = ssd.put({"kind": "system", "tokens": system, "cache_len": len(system), "target_cache": target})
    snap_sys = [mx.array(a) for a in target[0].state]
    _grow(target, len(boundary) - len(system))
    r2 = ssd.put({"kind": "boundary", "tokens": boundary, "cache_len": len(boundary), "target_cache": target,
                  "ssd_parent": dict(r1.vmlx_ssd_ref, receipt=r1)})
    _grow(target, len(turn) - len(boundary))
    r3 = ssd.put({"kind": "turn", "tokens": turn, "cache_len": len(turn), "target_cache": target,
                  "ssd_parent": dict(r2.vmlx_ssd_ref, receipt=r2)})
    assert ssd.flush()
    assert all(r.result()["durable"] for r in (r1, r2, r3))
    assert r3.vmlx_ssd_ref["depth"] == 2 and ssd.stats["delta_writes"] == 2
    got = _take(ssd, turn + [IM_START, 9])
    assert got is not None and got["cache_len"] == len(turn) and got["ssd_ref"]["depth"] == 2
    kv = got["target_cache"][0]
    assert kv.offset == len(turn) and kv.keys.shape[2] > len(turn)          # headroom: the next append is in place
    for a, b in zip(target[0].state, kv.state):
        assert mx.array_equal(a, b)
    for a, b in zip(target[1].state, got["target_cache"][1].state):        # GDN state from the leaf, not the root
        assert mx.array_equal(a, b)
    # the root entry still restores alone, unchanged by its children
    got_sys = _take(ssd, system + [IM_START, 42])
    assert got_sys is not None and all(mx.array_equal(a, b) for a, b in zip(snap_sys, got_sys["target_cache"][0].state))
    # the delta files hold only the new rows: much smaller than the root
    sizes = {}
    for tokens in (system, boundary, turn):
        from vmlx_engine.dflash2_session_disk import signature_for
        sizes[len(tokens)] = ssd.store.paths(signature_for(tokens))[0].stat().st_size
    assert sizes[len(turn)] < sizes[len(system)] + 4096


def test_delta_with_missing_parent_is_a_miss_never_wrong_state(ssd):
    from vmlx_engine.dflash2_session_disk import signature_for
    system = [IM_START, 1, 2, 3]
    boundary = system + [IM_START, 4, 5, 6]
    target = _target(len(system))
    r1 = ssd.put({"kind": "system", "tokens": system, "cache_len": len(system), "target_cache": target})
    _grow(target, 4)
    r2 = ssd.put({"kind": "boundary", "tokens": boundary, "cache_len": len(boundary), "target_cache": target,
                  "ssd_parent": dict(r1.vmlx_ssd_ref, receipt=r1)})
    assert ssd.flush() and r2.result()["durable"]
    for path in ssd.store.paths(signature_for(system)):                    # the pool evicted the parent
        path.unlink()
    assert _take(ssd, boundary + [IM_START, 9]) is None
    assert ssd.stats["chain_misses"] == 1


def test_child_of_a_failed_parent_is_refused_and_a_refused_parent_gives_a_full_child(ssd):
    from concurrent.futures import Future
    system = [IM_START, 1, 2, 3]
    boundary = system + [IM_START, 4, 5, 6]
    target = _target(len(system))
    failed = Future(); failed.set_result({"outcome": "failed", "durable": False})
    from vmlx_engine.dflash2_session_disk import signature_for
    ref = {"sig": signature_for(system), "len": len(system), "depth": 0}
    _grow(target, 4)
    r = ssd.put({"kind": "boundary", "tokens": boundary, "cache_len": len(boundary), "target_cache": target,
                 "ssd_parent": dict(ref, receipt=failed)})
    assert ssd.flush() and r.result()["durable"] and r.vmlx_ssd_ref["depth"] == 0     # written whole
    pending = Future()                                                       # parent "queued" but never written
    turn = boundary + [IM_START, 7, IM_END]
    _grow(target, 3)
    ssd._parent_durable = lambda sig, receipt: False
    r2 = ssd.put({"kind": "turn", "tokens": turn, "cache_len": len(turn), "target_cache": target,
                  "ssd_parent": dict(r.vmlx_ssd_ref, receipt=pending)})
    assert ssd.flush() and not r2.result()["durable"]


def _big_target(n):
    kv, gdn = KVCache(), ArraysCache(size=2)
    kv.update_and_fetch(mx.random.normal((1, 8, n, 256)), mx.random.normal((1, 8, n, 256)))
    gdn[0] = mx.random.normal((1, 3, 16))
    gdn[1] = mx.random.normal((1, 2, 4, 4))
    mx.eval(kv.keys, kv.values)
    return [kv, gdn]


def _append_peak(target):
    """MLX peak-memory increase of a one-row in-place append to the live KV layer."""
    mx.eval(target[0].keys, target[0].values)
    mx.reset_peak_memory()
    base = mx.get_active_memory()
    k, v = target[0].update_and_fetch(mx.zeros((1, 8, 1, 256)), mx.zeros((1, 8, 1, 256)))
    mx.eval(k, v)
    return mx.get_peak_memory() - base


@pytest.mark.parametrize("delta", [True, False])
def test_delta_pack_does_not_make_the_next_append_copy_the_live_buffer(tmp_path, delta):
    """A queued SSD write that holds a VIEW of the live KV buffer makes MLX copy the whole buffer on the next append
    (copy-on-write). Delta rows are copied when packed, so a queued delta write leaves the append in place; a whole
    entry (the control) still holds the view."""
    import threading
    gate = threading.Event()
    ssd = DFlash2SessionSSD(root=tmp_path, max_size_bytes=1 << 30, model_key="cow")
    try:
        system = [IM_START] + list(range(1, 1000))
        boundary = system + [IM_START] + list(range(2000, 2999))
        target = _big_target(len(system))
        r1 = ssd.put({"kind": "system", "tokens": system, "cache_len": len(system), "target_cache": target})
        assert ssd.flush() and r1.result()["durable"]
        k, v = target[0].update_and_fetch(mx.random.normal((1, 8, 1000, 256)), mx.random.normal((1, 8, 1000, 256)))
        mx.eval(k, v)
        del k, v
        save = ssd.store.save
        ssd.store.save = lambda sig, writer: (gate.wait(30), save(sig, writer))[1]     # hold the write in flight
        entry = {"kind": "boundary", "tokens": boundary, "cache_len": len(boundary), "target_cache": target}
        if delta:
            entry["ssd_parent"] = dict(r1.vmlx_ssd_ref, receipt=r1)
        r2 = ssd.put(entry)
        del entry
        buffer_bytes = target[0].keys.nbytes
        grew = _append_peak(target)
        gate.set()
        assert ssd.flush() and r2.result()["durable"]
        if delta:
            assert grew < buffer_bytes / 4, (grew, buffer_bytes)
        else:
            assert grew >= buffer_bytes, (grew, buffer_bytes)   # the control: the view forced a whole-buffer copy
    finally:
        gate.set()
        ssd.store.close()
