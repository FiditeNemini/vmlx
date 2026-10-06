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
