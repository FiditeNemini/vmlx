# SPDX-License-Identifier: Apache-2.0
"""Hand the first pass's prompt-boundary KV to the visible-answer pass.

When a soft-capped first pass emits no visible content, the server re-issues the
same prompt as `<base>:visible-answer`. That pass gets a perfect cache hit and
still re-reads the entire prefix from L2, which is the reasoning-to-content
stall. Measured on DSV4-Flash, reproducible to ~0.3s over four runs:

    83k prefix  / 325 blocks -> max decode gap 7.22, 7.47, 7.63s
    166k prefix / 649 blocks -> max decode gap 21.33, 21.34, 21.67, 21.82s

The existing reconstruct memo cannot cover it. The memo is armed inside the
reconstruct path, so it captures the state at the START of the first pass — the
prefix reused from the PREVIOUS turn — while the answer pass asks for the full
prompt boundary including the fresh span the first pass just prefilled. Live:

    Reconstruct memo MISS: held 324 blocks/82944 tokens, asked for
    649 blocks/166065 tokens (different blocks)

The first pass already produces the right object: the clean prompt-boundary
snapshot it stores into the prefix cache. This retains that and lets the answer
pass claim it.

DEFAULT OFF. It moves path-dependent DSV4/ZAYA KV between two requests, and a
wrong hand-off does not raise — it answers from the wrong state.

Created by Jinho Jang (eric@jangq.ai).
"""

from __future__ import annotations

import pytest

from vmlx_engine.scheduler import Scheduler


class _Req:
    def __init__(self, request_id, *, soft_cap=True, num_prompt_tokens=0):
        self.request_id = request_id
        self._dsv4_thinking_soft_cap = soft_cap
        self.num_prompt_tokens = num_prompt_tokens


class _Sched:
    """The hand-off surface only, lifted off Scheduler."""

    _ANSWER_PASS_SUFFIX = Scheduler._ANSWER_PASS_SUFFIX
    # staticmethod: re-wrap, else lifting it makes it an instance method.
    _answer_pass_snapshot_reuse_enabled = staticmethod(
        Scheduler._answer_pass_snapshot_reuse_enabled
    )
    _retain_answer_pass_snapshot = Scheduler._retain_answer_pass_snapshot
    _take_answer_pass_snapshot = Scheduler._take_answer_pass_snapshot
    _drop_answer_pass_snapshot = Scheduler._drop_answer_pass_snapshot

    def __init__(self):
        self._answer_pass_snapshots = {}


@pytest.fixture
def sched():
    return _Sched()


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("VMLX_ANSWER_PASS_SNAPSHOT_REUSE", "1")


def test_disabled_by_default_retains_and_returns_nothing(sched, monkeypatch):
    monkeypatch.delenv("VMLX_ANSWER_PASS_SNAPSHOT_REUSE", raising=False)
    sched._retain_answer_pass_snapshot(_Req("abc"), ["layer"] * 43, 166068)
    assert sched._answer_pass_snapshots == {}
    assert sched._take_answer_pass_snapshot(_Req("abc:visible-answer")) is None


def test_the_answer_pass_claims_what_the_first_pass_retained(sched, enabled):
    snapshot = [{"k": i} for i in range(43)]
    sched._retain_answer_pass_snapshot(_Req("abc"), snapshot, 166068)

    got = sched._take_answer_pass_snapshot(
        _Req("abc:visible-answer", num_prompt_tokens=166065)
    )
    assert got is not None
    assert len(got) == 43
    # A deep copy, so the first pass decoding on into its own state cannot
    # mutate what the answer pass will read.
    assert got is not snapshot
    assert got[0] is not snapshot[0]


def test_it_is_single_use(sched, enabled):
    sched._retain_answer_pass_snapshot(_Req("abc"), ["l"] * 43, 100)
    assert sched._take_answer_pass_snapshot(_Req("abc:visible-answer")) is not None
    # Anything still held after the pass that wanted it is stale, and holding it
    # would pin a full KV copy.
    assert sched._take_answer_pass_snapshot(_Req("abc:visible-answer")) is None


def test_only_a_soft_capped_pass_retains(sched, enabled):
    """A soft cap is the signal that a second pass is coming."""
    sched._retain_answer_pass_snapshot(_Req("abc", soft_cap=False), ["l"] * 43, 100)
    assert sched._answer_pass_snapshots == {}


def test_a_first_pass_never_claims_its_own_snapshot(sched, enabled):
    sched._retain_answer_pass_snapshot(_Req("abc"), ["l"] * 43, 100)
    assert sched._take_answer_pass_snapshot(_Req("abc")) is None
    # ...and it is still there for the pass it was meant for.
    assert sched._take_answer_pass_snapshot(_Req("abc:visible-answer")) is not None


def test_a_longer_ask_is_refused(sched, enabled):
    """Shorter is expected; longer means this is not the pass we saved for.

    The answer pass replays the same prompt minus the generation-prompt tail, so
    it asks for slightly FEWER tokens (166,065 against 166,068 live). A larger
    ask cannot be served from this state.
    """
    sched._retain_answer_pass_snapshot(_Req("abc"), ["l"] * 43, 166068)
    assert (
        sched._take_answer_pass_snapshot(
            _Req("abc:visible-answer", num_prompt_tokens=200000)
        )
        is None
    )


def test_the_live_shape_is_accepted(sched, enabled):
    sched._retain_answer_pass_snapshot(_Req("abc"), ["l"] * 43, 166068)
    assert (
        sched._take_answer_pass_snapshot(
            _Req("abc:visible-answer", num_prompt_tokens=166065)
        )
        is not None
    )


def test_dropping_releases_the_copy(sched, enabled):
    sched._retain_answer_pass_snapshot(_Req("abc"), ["l"] * 43, 100)
    sched._drop_answer_pass_snapshot("abc:visible-answer")
    assert sched._answer_pass_snapshots == {}
    sched._retain_answer_pass_snapshot(_Req("xyz"), ["l"] * 43, 100)
    sched._drop_answer_pass_snapshot("xyz")
    assert sched._answer_pass_snapshots == {}


def test_snapshots_do_not_cross_requests(sched, enabled):
    sched._retain_answer_pass_snapshot(_Req("one"), [{"id": 1}], 100)
    sched._retain_answer_pass_snapshot(_Req("two"), [{"id": 2}], 100)
    assert sched._take_answer_pass_snapshot(_Req("two:visible-answer"))[0]["id"] == 2
    assert sched._take_answer_pass_snapshot(_Req("one:visible-answer"))[0]["id"] == 1
