# SPDX-License-Identifier: Apache-2.0
"""Byte-identical A/B for the opt-in compact top-p sampler.

2026-08-17. `make_sampler`'s compact path was guarded on `top_k > 0`, so the
common bundle shape -- `top_k: 0` with a `top_p` -- fell through to MLX-LM's
generic nucleus, which builds full-vocabulary logprobs and sorts the entire
vocabulary every token. dots3-note ships `top_p 0.95, top_k 0` over a 152,064
token vocab, and measured live in the app the sampler costs ~30% of decode
(greedy 26.5 t/s vs sampled 18.9 t/s, same model and prompt).

Bounding the candidate set is exact only while the nucleus mass falls inside
the bound, so the optimisation ships OPT-IN (`VMLX_COMPACT_TOP_P=1`). These
tests are the gate that has to go green before it could ever default on: the
compact sampler must select the SAME token as the full-vocabulary nucleus, on
realistic vocab-sized distributions, at a fixed seed.
"""

from __future__ import annotations

import os

import mlx.core as mx
import pytest

from vmlx_engine.sampling import make_sampler


VOCAB = 152_064  # dots3-note


def _logits(seed: int, vocab: int = VOCAB, peaky: bool = True) -> mx.array:
    """A realistic decode-step logit row.

    Real logits are sharply peaked: a handful of plausible continuations carry
    almost all the mass. `peaky=False` produces a deliberately flat row to
    exercise the case where the nucleus is WIDE and the compact bound matters.
    """
    key = mx.random.key(seed)
    base = mx.random.normal((1, vocab), key=key)
    if peaky:
        # concentrate mass: scale up so softmax is dominated by a few tokens
        base = base * 6.0
    return base


def _sample_with(env: dict, logits: mx.array, *, top_p: float, temp: float, seed: int) -> int:
    old = {k: os.environ.get(k) for k in env}
    os.environ.update({k: v for k, v in env.items()})
    try:
        sampler = make_sampler(temp=temp, top_p=top_p, top_k=0, min_p=0.0, seed=seed)
        out = sampler(logits)
        mx.eval(out)
        return int(out.reshape(-1)[0])
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_compact_top_p_is_opt_in_and_off_by_default():
    """Default must be the generic path -- no silent behaviour change."""
    os.environ.pop("VMLX_COMPACT_TOP_P", None)
    os.environ.pop("VMLINUX_COMPACT_TOP_P", None)
    sampler = make_sampler(temp=0.8, top_p=0.95, top_k=0, min_p=0.0)
    # the compact wrapper advertises logits input; the generic one does not
    assert getattr(sampler, "_vmlx_accepts_logits", False) is False


def test_compact_path_engages_only_when_enabled():
    """A/B is meaningless unless the new path actually ENGAGED."""
    os.environ["VMLX_COMPACT_TOP_P"] = "1"
    try:
        sampler = make_sampler(temp=0.8, top_p=0.95, top_k=0, min_p=0.0)
        assert getattr(sampler, "_vmlx_accepts_logits", False) is True
    finally:
        os.environ.pop("VMLX_COMPACT_TOP_P", None)


@pytest.mark.parametrize("seed", [0, 1, 7, 13, 99])
@pytest.mark.parametrize("top_p", [0.9, 0.95, 0.99])
def test_compact_top_p_selects_the_same_token_as_full_vocab_nucleus(seed, top_p):
    """THE GATE: identical token, fixed seed, realistic peaked logits."""
    logits = _logits(seed)
    generic = _sample_with({}, logits, top_p=top_p, temp=0.8, seed=seed)
    compact = _sample_with(
        {"VMLX_COMPACT_TOP_P": "1"}, logits, top_p=top_p, temp=0.8, seed=seed
    )
    assert compact == generic, (
        f"compact top-p diverged at seed={seed} top_p={top_p}: "
        f"{compact} != {generic}"
    )


def test_greedy_and_top_k_paths_are_untouched():
    """The change must not reach the paths that already had fast routes."""
    greedy = make_sampler(temp=0.0, top_p=0.95, top_k=0)
    assert getattr(greedy, "_vmlx_is_greedy", False) is True

    os.environ["VMLX_COMPACT_TOP_P"] = "1"
    try:
        # top_k > 0 keeps using the pre-existing compact top-k route
        topk = make_sampler(temp=0.8, top_p=0.95, top_k=40, min_p=0.0)
        assert getattr(topk, "_vmlx_accepts_logits", False) is True
        # min_p set must NOT take the compact route (it cannot express min_p)
        minp = make_sampler(temp=0.8, top_p=0.95, top_k=0, min_p=0.05)
        assert getattr(minp, "_vmlx_accepts_logits", False) is False
    finally:
        os.environ.pop("VMLX_COMPACT_TOP_P", None)
