# SPDX-License-Identifier: Apache-2.0
"""Sampling helpers for vMLX generation paths."""

from __future__ import annotations

import os
from functools import partial
from typing import Callable

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler as _mlx_make_sampler

# ─── Reasoning EOS guard ────────────────────────────────────────────────
#
# Some reasoning bundles (measured live: Qwen3.8-Flash-Next at its template
# default xhigh effort, temp 1.0 / top_k 20) intermittently sample EOS
# INSIDE an unclosed <think> block — ~25% of tool-result continuations
# ended reasoning-only with no visible answer and no tool call, surfacing
# as a chat 502 reasoning_only_no_content or a silently empty /v1/responses
# message. The think markers are single tokens for these families, so the
# structural fix lives at the sampler: while a think block is open, EOS is
# banned; the model must close the block before it may stop. The state
# machine is built from pure MLX ops on the sampler's own outputs — no
# host synchronization is added to the decode hot path.
#
# Registered per loaded model by the server (single-token markers only).
# Guarded requests are single-sequence; batched logits rows pass through
# unguarded because closure state cannot track continuous-batching row
# reassignment. VMLX_REASONING_EOS_GUARD=0 disables (diagnostics only —
# the guard removes a failure mode, it never blocks output).

_reasoning_eos_guard: dict | None = None


def set_reasoning_eos_guard(
    think_open_id: int, think_close_id: int, eos_ids: list[int]
) -> None:
    global _reasoning_eos_guard
    if not eos_ids:
        _reasoning_eos_guard = None
        return
    _reasoning_eos_guard = {
        "open": int(think_open_id),
        "close": int(think_close_id),
        "eos": [int(t) for t in eos_ids],
    }


def clear_reasoning_eos_guard() -> None:
    global _reasoning_eos_guard
    _reasoning_eos_guard = None


def reasoning_eos_guard_active() -> bool:
    return _reasoning_eos_guard is not None and (
        os.environ.get("VMLX_REASONING_EOS_GUARD", "1") != "0"
    )


class _ReasoningEosGuardedSampler:
    """Per-request stateful wrapper banning EOS inside an open think block.

    Delegates every attribute to the wrapped sampler so decode-path
    introspection (``_vmlx_accepts_logits``, ``_vmlx_categorical``,
    ``min_tokens_to_keep`` …) keeps working. All state transitions are pure
    MLX ops on the sampler's own outputs — no ``.item()`` on the hot path.
    """

    def __init__(self, sampler: Callable[[mx.array], mx.array], guard: dict):
        self._sampler = sampler
        self._open_id = guard["open"]
        self._close_id = guard["close"]
        self._eos_ids = guard["eos"]
        self._in_think = mx.array(False)
        self._ban_vectors: dict[tuple[int, str], mx.array] = {}

    def _ban_vector(self, vocab: int, dtype: mx.Dtype) -> mx.array:
        key = (vocab, str(dtype))
        vec = self._ban_vectors.get(key)
        if vec is None:
            eos = mx.array([t for t in self._eos_ids if 0 <= t < vocab])
            positions = mx.arange(vocab)
            mask = mx.any(positions[:, None] == eos[None, :], axis=-1)
            vec = mx.where(mask, mx.array(-float("inf")), mx.array(0.0)).astype(dtype)
            self._ban_vectors[key] = vec
        return vec

    def __call__(self, logits: mx.array) -> mx.array:
        single = logits.shape[0] == 1 if logits.ndim >= 2 else True
        if single:
            ban = self._ban_vector(logits.shape[-1], logits.dtype)
            logits = logits + mx.where(
                self._in_think, ban, mx.zeros_like(ban)
            )
        tokens = self._sampler(logits)
        if single:
            flat = tokens.reshape(-1)
            saw_open = mx.any(flat == self._open_id)
            saw_close = mx.any(flat == self._close_id)
            saw_stop = saw_close
            for eos in self._eos_ids:
                saw_stop = mx.logical_or(saw_stop, mx.any(flat == eos))
            self._in_think = mx.logical_and(
                mx.logical_or(self._in_think, saw_open),
                mx.logical_not(saw_stop),
            )
        return tokens

    def __getattr__(self, name):
        return getattr(self._sampler, name)


class _SamplerWrapper:
    def __init__(
        self,
        sampler: Callable[[mx.array], mx.array],
        *,
        accepts_logits: bool = False,
        is_greedy: bool = False,
    ):
        self._sampler = sampler
        if accepts_logits:
            self._vmlx_accepts_logits = True
        if is_greedy:
            self._vmlx_is_greedy = True

    def __call__(self, logits: mx.array) -> mx.array:
        return self._sampler(logits)

    def __getattr__(self, name):
        return getattr(self._sampler, name)


def make_sampler(
    *,
    temp: float = 0.0,
    top_p: float = 0.0,
    min_p: float = 0.0,
    top_k: int = 0,
    seed: int | None = None,
) -> Callable[[mx.array], mx.array]:
    """Build a sampler for vMLX decode loops.

    MLX-LM's default top-k sampler masks a full-vocabulary logits vector and
    then calls categorical sampling across that full vocabulary. That is
    correct but expensive for vMLX's common app defaults on large-vocab models
    such as ZAYA (top_k=40, vocab > 260k). For bounded top-k sampling, gather
    the compact top-k logits and sample there, then map the sampled local index
    back to the original token id.
    """
    base = _make_base_sampler(
        temp=temp, top_p=top_p, min_p=min_p, top_k=top_k, seed=seed
    )
    # Greedy decoding is deterministic and was never observed to EOS inside
    # a think block; leaving it unwrapped keeps the argmax fast path intact.
    if getattr(base, "_vmlx_is_greedy", False):
        return base
    if reasoning_eos_guard_active():
        return _ReasoningEosGuardedSampler(base, _reasoning_eos_guard)
    return base


def _make_base_sampler(
    *,
    temp: float = 0.0,
    top_p: float = 0.0,
    min_p: float = 0.0,
    top_k: int = 0,
    seed: int | None = None,
) -> Callable[[mx.array], mx.array]:
    if float(temp or 0.0) == 0.0:
        return _SamplerWrapper(
            lambda x: mx.argmax(x, axis=-1),
            accepts_logits=True,
            is_greedy=True,
        )

    if seed is not None:
        if int(top_k or 0) > 0 and float(min_p or 0.0) == 0.0:
            return _SamplerWrapper(
                _make_seeded_compact_top_k_sampler(
                    temp=float(temp),
                    top_p=float(top_p or 0.0),
                    top_k=int(top_k),
                    seed=int(seed),
                ),
                accepts_logits=True,
            )
        return _make_seeded_logprob_sampler(
            temp=float(temp),
            top_p=float(top_p or 0.0),
            min_p=float(min_p or 0.0),
            top_k=int(top_k or 0),
            seed=int(seed),
        )

    if int(top_k or 0) > 0 and float(min_p or 0.0) == 0.0:
        return _SamplerWrapper(
            _make_compact_top_k_sampler(
                temp=float(temp),
                top_p=float(top_p or 0.0),
                top_k=int(top_k),
            ),
            accepts_logits=True,
        )

    return _mlx_make_sampler(temp=temp, top_p=top_p, min_p=min_p, top_k=top_k)


def make_minimax_m3_sampler(
    *,
    temp: float = 0.0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    top_k: int = 0,
    seed: int | None = None,
) -> Callable[[mx.array], mx.array]:
    """Build the MiniMax-M3 server sampler to match the clean runtime path.

    The standalone M3 runtime samples from raw logits after casting them to
    fp32.  The generic MLX-LM sampler expects log-probabilities and applies a
    different nucleus implementation.  M3's low-bit routed experts make that
    difference visible, so the server path must use the same logit-space sampler
    as ``models/minimax_m3/runtime.py``.
    """

    temp_f = float(temp or 0.0)
    top_p_f = float(top_p if top_p is not None else 1.0)
    top_k_i = int(top_k or 0)

    key = mx.random.key(int(seed)) if seed is not None else None

    def _categorical(logits: mx.array) -> mx.array:
        nonlocal key
        if key is None:
            return mx.random.categorical(logits)
        keys = mx.random.split(key)
        key = keys[0]
        return mx.random.categorical(logits, key=keys[1])

    def _sample(logits: mx.array) -> mx.array:
        logits = logits.astype(mx.float32)
        if temp_f == 0.0:
            return mx.argmax(logits, axis=-1)

        logits = logits / temp_f
        if top_k_i > 0:
            vocab_size = logits.shape[-1]
            k = min(top_k_i, vocab_size)
            top_indices = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]
            top_logits = mx.take_along_axis(logits, top_indices, axis=-1)
            order = mx.argsort(-top_logits, axis=-1)
            logits = mx.take_along_axis(top_logits, order, axis=-1)
            top_indices = mx.take_along_axis(top_indices, order, axis=-1)
        else:
            top_indices = None

        if top_p_f > 0.0 and top_p_f < 1.0:
            probs = mx.softmax(logits, axis=-1)
            order = mx.argsort(-probs, axis=-1)
            sorted_probs = mx.take_along_axis(probs, order, axis=-1)
            cumulative = mx.cumsum(sorted_probs, axis=-1)
            keep = (cumulative - sorted_probs) <= top_p_f
            sorted_probs = mx.where(keep, sorted_probs, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(axis=-1, keepdims=True)
            sampled_sorted = _categorical(mx.log(sorted_probs + 1e-12))
            sampled = mx.take_along_axis(
                order,
                sampled_sorted[..., None],
                axis=-1,
            ).squeeze(-1)
        else:
            sampled = _categorical(logits)

        if top_indices is not None:
            sampled = mx.take_along_axis(top_indices, sampled[..., None], axis=-1).squeeze(-1)
        return sampled

    wrapper = _SamplerWrapper(_sample, accepts_logits=True)
    wrapper._vmlx_sampler_kind = "minimax_m3_runtime"
    wrapper._vmlx_ignores_min_p = bool(float(min_p or 0.0))
    return wrapper


def _make_seeded_logprob_sampler(
    *, temp: float, top_p: float, min_p: float, top_k: int, seed: int
) -> Callable[[mx.array], mx.array]:
    """Create a request-local sampler backed by an independent MLX PRNG key."""
    from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p

    key = mx.random.key(seed)

    def _next_key():
        nonlocal key
        keys = mx.random.split(key)
        key = keys[0]
        return keys[1]

    def _categorical(logits: mx.array) -> mx.array:
        return mx.random.categorical(logits, key=_next_key())

    def _uniform() -> float:
        return float(mx.random.uniform(key=_next_key()).item())

    def _sample(logprobs: mx.array) -> mx.array:
        nonlocal key
        if 0.0 < top_p < 1.0:
            logprobs = apply_top_p(logprobs, top_p)
        if min_p != 0.0:
            logprobs = apply_min_p(logprobs, min_p)
        if top_k > 0:
            logprobs = apply_top_k(logprobs, top_k)
        return _categorical(logprobs * (1.0 / temp))

    _sample._vmlx_categorical = _categorical
    _sample._vmlx_random_uniform = _uniform
    return _sample


def _make_seeded_compact_top_k_sampler(
    *, temp: float, top_p: float, top_k: int, seed: int
) -> Callable[[mx.array], mx.array]:
    """Seeded variant of the compact logit-space top-k fast path."""
    key = mx.random.key(seed)

    def _next_key():
        nonlocal key
        keys = mx.random.split(key)
        key = keys[0]
        return keys[1]

    def _categorical(logits: mx.array) -> mx.array:
        return mx.random.categorical(logits, key=_next_key())

    def _uniform() -> float:
        return float(mx.random.uniform(key=_next_key()).item())

    def _sample(logits: mx.array) -> mx.array:
        nonlocal key
        vocab_size = logits.shape[-1]
        k = min(top_k, vocab_size)
        top_indices = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]
        top_logits = mx.take_along_axis(logits, top_indices, axis=-1)
        order = mx.argsort(-top_logits, axis=-1)
        top_logits = mx.take_along_axis(top_logits, order, axis=-1)
        top_indices = mx.take_along_axis(top_indices, order, axis=-1)
        if 0.0 < top_p < 1.0:
            probs = mx.exp(top_logits - mx.logsumexp(logits, axis=-1, keepdims=True))
            cumulative = mx.cumsum(probs, axis=-1)
            keep = (cumulative - probs) < top_p
            top_logits = mx.where(keep, top_logits, -float("inf"))
        sampled_local = _categorical(top_logits * (1.0 / temp))
        return mx.take_along_axis(
            top_indices, sampled_local[..., None], axis=-1
        ).squeeze(-1)

    _sample._vmlx_categorical = _categorical
    _sample._vmlx_random_uniform = _uniform
    return _sample


def _make_compact_top_k_sampler(
    *,
    temp: float,
    top_p: float,
    top_k: int,
) -> Callable[[mx.array], mx.array]:
    @partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
    def _sample(logits: mx.array) -> mx.array:
        vocab_size = logits.shape[-1]
        k = min(top_k, vocab_size)
        top_indices = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]
        top_logits = mx.take_along_axis(logits, top_indices, axis=-1)

        order = mx.argsort(-top_logits, axis=-1)
        top_logits = mx.take_along_axis(top_logits, order, axis=-1)
        top_indices = mx.take_along_axis(top_indices, order, axis=-1)

        if top_p > 0.0 and top_p < 1.0:
            probs = mx.exp(top_logits - mx.logsumexp(logits, axis=-1, keepdims=True))
            cumulative = mx.cumsum(probs, axis=-1)
            # Keep the token that crosses top_p as well as all earlier tokens.
            # This matches the full-vocabulary nucleus cutoff, then applies the
            # top-k bound without sampling over masked-out vocabulary entries.
            keep = (cumulative - probs) < top_p
            top_logits = mx.where(keep, top_logits, -float("inf"))

        sampled_local = mx.random.categorical(top_logits * (1.0 / temp))
        return mx.take_along_axis(top_indices, sampled_local[..., None], axis=-1).squeeze(-1)

    return _sample
