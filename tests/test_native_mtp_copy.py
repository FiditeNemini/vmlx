"""Copy drafts for the native-MTP verify cycle (vmlx_engine/native_mtp_copy.py)."""
from vmlx_engine.native_mtp_copy import NGRAM, SuffixCopyProposer


def _proposer(prompt, **kw):
    p = SuffixCopyProposer.from_prompt(prompt)
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def test_long_suffix_match_proposes_the_continuation():
    span = list(range(100, 120))                       # 20 distinct tokens
    prompt = [1, 2, 3] + span + [7, 8] + span[:10]     # suffix repeats span[:10]
    p = _proposer(prompt, max_width=3)
    assert p.propose(room=50) == span[10:13]
    assert p.last_match >= 10


def test_short_matches_are_not_proposed():
    prompt = [5, 6, 7, 8, 9, 50, 51, 52, 53, 54, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # suffix matches only 5 tokens earlier
    p = _proposer(prompt, min_match=8)
    assert p.propose(room=50) == []


def test_room_caps_the_window_and_zero_room_proposes_nothing():
    span = list(range(200, 230))
    p = _proposer(span + span[:12], max_width=3)
    assert p.propose(room=2) == span[12:14]
    assert p.propose(room=0) == []


def test_sync_appends_reply_and_queue_and_rebuilds_after_a_rewind():
    span = list(range(300, 330))
    p = _proposer(span)
    p.sync(span[:6], span[6:12])                       # emitted + queued reply repeats the prompt
    assert p.tokens == span + span[:12]
    assert p.propose(room=10) == span[12:15]
    p.sync(span[:6], span[6:13])                       # one more queued token: incremental
    assert p.tokens[-1] == span[12]
    p.sync(span[:4], [])                               # shrank: rebuilt from the prompt
    assert p.tokens == span + span[:4]


def test_two_first_token_misses_silence_the_proposer():
    span = list(range(400, 440))
    p = _proposer(span + span[:12], silence_cycles=3)
    assert p.propose(room=10)
    p.observe(3, 0)
    p.observe(3, 0)
    assert [p.propose(room=10) for _ in range(3)] == [[], [], []]
    assert p.propose(room=10)                          # back after the silence
    assert p.stats.silenced == 1 and p.stats.first_misses == 2


def test_self_position_is_never_a_candidate():
    p = _proposer(list(range(500, 500 + NGRAM + 10)))   # no repeat anywhere
    assert p.propose(room=10) == []


def test_copy_emits_are_counted_apart_from_head_drafts_and_rewound_like_drafts():
    import inspect
    from types import SimpleNamespace
    import vmlx_engine.mllm_batch_generator as g

    stats = g.MLLMNativeMTPStats()
    state = SimpleNamespace(stats=stats)
    for source in ("init", "draft", "copy", "copy", "bonus", "verify"):
        g._native_mtp_bump_emit(state, source)
    assert (stats.draft_emits, stats.copy_emits) == (1, 2)
    # the terminal-boundary rewind must treat copied tokens as accepted drafts
    src = inspect.getsource(g.MLLMBatchGenerator._rewind_native_mtp_terminal_boundary)
    assert src.count('source in ("draft", "copy")') == 2
