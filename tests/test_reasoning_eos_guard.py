"""Sampler-level reasoning EOS guard.

Live-measured defect: Qwen3.8-Flash-Next (template-default xhigh effort,
temp 1.0/top_k 20) sampled EOS inside an open <think> block on ~25% of
tool-result continuations — the turn finalized reasoning-only (chat 502
reasoning_only_no_content, silently empty /v1/responses message). The guard
bans EOS while a think block is open, so the model must close the block
before it may stop.
"""

import mlx.core as mx
import pytest

from vmlx_engine import sampling

OPEN, CLOSE, EOS, WORD = 3, 4, 5, 6
VOCAB = 8


@pytest.fixture(autouse=True)
def _clean_guard():
    sampling.clear_reasoning_eos_guard()
    yield
    sampling.clear_reasoning_eos_guard()


def _logits_pref(token: int) -> mx.array:
    """Logits preferring `token`, with EOS a close second."""
    row = [0.0] * VOCAB
    row[token] = 10.0
    if token != EOS:
        row[EOS] = 8.0
    return mx.array([row])


def _greedy_base(logits: mx.array) -> mx.array:
    return mx.argmax(logits, axis=-1)


def _guarded():
    sampling.set_reasoning_eos_guard(OPEN, CLOSE, [EOS])
    return sampling._ReasoningEosGuardedSampler(
        _greedy_base, sampling._reasoning_eos_guard
    )


def test_eos_banned_inside_think_and_allowed_after_close():
    s = _guarded()
    assert int(s(_logits_pref(OPEN)).item()) == OPEN        # think opens
    # Model wants EOS inside think — the guard must pick something else.
    inside = int(s(_logits_pref(EOS)).item())
    assert inside != EOS
    assert int(s(_logits_pref(CLOSE)).item()) == CLOSE      # think closes
    assert int(s(_logits_pref(WORD)).item()) == WORD        # visible answer token
    assert int(s(_logits_pref(EOS)).item()) == EOS          # EOS now legal


def test_eos_allowed_when_think_never_opened():
    s = _guarded()
    assert int(s(_logits_pref(WORD)).item()) == WORD
    assert int(s(_logits_pref(EOS)).item()) == EOS


def test_state_resets_after_eos_for_sampler_reuse():
    s = _guarded()
    assert int(s(_logits_pref(OPEN)).item()) == OPEN
    assert int(s(_logits_pref(CLOSE)).item()) == CLOSE
    assert int(s(_logits_pref(WORD)).item()) == WORD
    assert int(s(_logits_pref(EOS)).item()) == EOS
    # A generator-owned sampler may serve the next request: EOS must not be
    # banned by stale think state.
    assert int(s(_logits_pref(EOS)).item()) == EOS


def test_batched_rows_pass_through_unguarded():
    s = _guarded()
    batched = mx.concatenate([_logits_pref(OPEN), _logits_pref(EOS)], axis=0)
    out = s(batched)
    # Row 1 prefers EOS and must keep it: closure state cannot track
    # continuous-batching row reassignment, so multi-row logits are untouched.
    assert int(out[1].item()) == EOS


def test_make_sampler_wraps_only_when_guard_active():
    plain = sampling.make_sampler(temp=1.0, top_k=5)
    assert not isinstance(plain, sampling._ReasoningEosGuardedSampler)
    sampling.set_reasoning_eos_guard(OPEN, CLOSE, [EOS])
    wrapped = sampling.make_sampler(temp=1.0, top_k=5)
    assert isinstance(wrapped, sampling._ReasoningEosGuardedSampler)
    # Greedy fast path stays unwrapped (never observed to fail; keeps argmax hot path).
    greedy = sampling.make_sampler(temp=0.0)
    assert not isinstance(greedy, sampling._ReasoningEosGuardedSampler)


def test_guard_env_kill_switch(monkeypatch):
    sampling.set_reasoning_eos_guard(OPEN, CLOSE, [EOS])
    monkeypatch.setenv("VMLX_REASONING_EOS_GUARD", "0")
    assert not sampling.reasoning_eos_guard_active()
    plain = sampling.make_sampler(temp=1.0, top_k=5)
    assert not isinstance(plain, sampling._ReasoningEosGuardedSampler)


def test_wrapper_delegates_attributes():
    def base(logits):
        return mx.argmax(logits, axis=-1)

    base._vmlx_accepts_logits = True
    base.min_tokens_to_keep = 7
    sampling.set_reasoning_eos_guard(OPEN, CLOSE, [EOS])
    s = sampling._ReasoningEosGuardedSampler(base, sampling._reasoning_eos_guard)
    assert s._vmlx_accepts_logits is True
    assert s.min_tokens_to_keep == 7


def test_exact_live_failure_sequence_now_completes():
    """The measured failure: think opens, model tries to EOS at 'final.' —
    the guard forces continuation; the model closes think and answers."""
    s = _guarded()
    seq = [OPEN, WORD, WORD, EOS, CLOSE, WORD, EOS]  # EOS mid-think attempt
    out = [int(s(_logits_pref(t)).item()) for t in seq]
    assert EOS not in out[:4]           # the mid-think EOS was refused
    assert out[4] == CLOSE
    assert out[-1] == EOS               # normal stop after visible content


def test_priming_from_pre_opened_prompt_tail():
    """qwen4_exp's template appends <think>\n to the generation prompt — the
    guard must start INSIDE the block even though it never samples OPEN."""
    s = _guarded()
    sampling.prime_reasoning_guard(s, [1, 2, OPEN])
    banned = int(s(_logits_pref(EOS)).item())
    assert banned != EOS                                     # engaged from token 0
    assert int(s(_logits_pref(CLOSE)).item()) == CLOSE
    assert int(s(_logits_pref(WORD)).item()) == WORD
    assert int(s(_logits_pref(EOS)).item()) == EOS


def test_priming_ignores_closed_blocks_in_history():
    s = _guarded()
    sampling.prime_reasoning_guard(s, [OPEN, WORD, CLOSE, WORD])
    assert int(s(_logits_pref(EOS)).item()) == EOS


def test_close_then_immediate_eos_is_refused_until_visible_token():
    """Second measured failure shape: the model closes think and EOSes with
    zero visible answer. At least one visible token must follow the close."""
    s = _guarded()
    assert int(s(_logits_pref(OPEN)).item()) == OPEN
    assert int(s(_logits_pref(CLOSE)).item()) == CLOSE
    refused = int(s(_logits_pref(EOS)).item())
    assert refused != EOS
    assert int(s(_logits_pref(WORD)).item()) == WORD
    assert int(s(_logits_pref(EOS)).item()) == EOS


def test_primed_close_then_eos_also_refused():
    s = _guarded()
    sampling.prime_reasoning_guard(s, [OPEN])
    assert int(s(_logits_pref(CLOSE)).item()) == CLOSE
    assert int(s(_logits_pref(EOS)).item()) != EOS
    assert int(s(_logits_pref(WORD)).item()) == WORD
    assert int(s(_logits_pref(EOS)).item()) == EOS
