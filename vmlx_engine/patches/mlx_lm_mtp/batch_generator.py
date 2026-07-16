# SPDX-License-Identifier: Apache-2.0
"""Conditional MTP dispatch inside ``mlx_lm.generate.GenerationBatch``.

This is the integration point that lets the existing oMLX scheduler /
paged cache / prefix cache / SSD cache stack drive MTP without touching any
of those layers. ``GenerationBatch`` is mlx-lm's per-step decoder for the
active set of sequences in continuous batching. We patch:

- ``GenerationBatch.__init__`` — after the standard ``_step()`` has run
  the prompt's last token through the backbone, we add an MTP "post-init"
  step that runs one more 1-token backbone forward (with hidden) and one
  MTP-head forward. Two confirmed tokens are queued for emission and a
  draft is stashed for the first verify cycle.

- ``GenerationBatch.next`` — when the batch holds exactly one MTP-capable
  sequence we emit from the per-batch queue first; once empty, we run a
  2-token verify forward over ``[next_main, draft]`` with
  ``n_confirmed=1`` and a single MTP-head forward at the bonus position
  (accept) or confirmed position (reject), refilling the queue from the
  verify outputs.

The throughput math (greedy, accept rate p):
  - Tokens per cycle: 1 + p (accept emits draft+bonus; reject emits verify_pred only)
  - Cost per cycle: 1× backbone (n+1-token verify) + 1× MTP head.

  The win depends ENTIRELY on how much cheaper the (n+1)-token verify forward is
  than (n+1) single-token forwards. On a DENSE model the marginal verify token
  is nearly free (weights read once) so the ratio is ~1.05× and MTP wins big.
  On an MoE it is NOT: measured on Hy3-JANG_2L a 2-token verify costs 1.36× a
  1-token forward (43.6ms vs 31.5ms) because each routed expert does per-row
  2-bit dequant+matmul — the cost is per-row COMPUTE, not shared expert-read
  memory (proven: a 2-token forward whose rows pick the SAME experts costs the
  same as one whose rows pick DIFFERENT experts). Break-even acceptance is
  therefore ~59%; Hy3's affine-8 head drafts against a 2-bit backbone accept at
  ~58%, so MTP straddles break-even and nets ~-3%. See `native_mtp_blocked` in
  the JANG_2L bundle. It only becomes a win with a higher-bit routed backbone
  (less-damaged → higher head agreement → acceptance clears break-even).

Greedy identity (sampler is None): the patched dispatch must produce the same
tokens as the standard step. The oMLX-side equivalent is pinned in
``tests/test_mlx_lm_mtp_patch.py``.

Stochastic acceptance (sampler is not None): we use ``min(1, p_target / p_draft)``
(Leviathan & Chen 2023). On rejection we sample from the residual
``max(p_target - p_draft, 0) / Z`` so the marginal output distribution
equals the target distribution exactly.

PagedCacheManager interaction
-----------------------------
``cache.trim(1)`` on a ``BatchKVCache`` only updates ``self._idx``; the
underlying paged blocks are untouched. ``ArraysCache.rollback_state``
holds ``(conv_snap, ssm_snap)`` snapshots produced by the patched
``GatedDeltaNet.__call__`` and is restored on reject. Because both code
paths only mutate cache *length* (not block ownership), oMLX's
``PagedCacheManager`` is oblivious to the trim — its block_table is
unaffected and prefix-cache lookups continue to work normally.

TokenBuffer interaction
-----------------------
``GenerationBatch._token_context[0]`` is a ``TokenBuffer`` accumulating
the prompt + every forward-input token. We update it in lock-step with
each forward-input position so that ``logits_processors`` see the same
token sequence the standard step would see. On reject we shrink the
buffer's ``_size`` by 1 to discard the rejected draft.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, List, Optional, Tuple

logger = logging.getLogger(__name__)

_PATCHED = False

# Barrier-based per-phase profiling. Off by default (adds an mx.eval after each
# GPU phase, which serializes the pipeline — measurement only). When on, the
# _MtpStats timings become real GPU-phase costs instead of enqueue times.
_MTP_PROFILE = bool(os.environ.get("VMLX_MTP_PROFILE"))

# Measurement bypass: keep the MTP head LOADED (so the memory footprint and
# residency are identical to an MTP-on run) but decode with the standard
# autoregressive step. This isolates the MTP algorithm from the head's memory
# cost, giving a footprint-controlled baseline for A/B on a variance-prone box.
_MTP_BYPASS = bool(os.environ.get("VMLX_MTP_BYPASS"))

# Process-local, read-only telemetry surface for /health. The generation loop
# runs on the scheduler worker while health is served from the API loop, so
# publish complete snapshots under a lock instead of exposing the mutable
# per-request dataclass. A process restart intentionally resets these totals.
_MTP_TELEMETRY_LOCK = threading.Lock()
_LAST_NATIVE_MTP: Optional[dict] = None
_NATIVE_MTP_TOTALS = {
    "requests": 0,
    "cycles": 0,
    "drafted_tokens": 0,
    "accepted_tokens": 0,
}


def _pbar(*arrays) -> None:
    """Profiling-only barrier: resolve `arrays` iff VMLX_MTP_PROFILE is set."""
    if _MTP_PROFILE:
        import mlx.core as mx

        mx.eval(*arrays)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply() -> bool:
    """Wrap ``GenerationBatch.__init__`` + ``GenerationBatch.next``."""
    global _PATCHED
    if _PATCHED:
        return True

    try:
        from mlx_lm.generate import GenerationBatch
    except ImportError:
        logger.debug("mlx_lm.generate.GenerationBatch not importable")
        return False

    if hasattr(GenerationBatch, "_omlx_mtp_patched"):
        _PATCHED = True
        return True

    original_init = GenerationBatch.__init__
    original_next = GenerationBatch.next
    original_filter = GenerationBatch.filter
    original_extend = GenerationBatch.extend

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if _MTP_BYPASS:
            return  # head stays loaded; decode via the standard step
        if _is_mtp_eligible(self):
            try:
                _post_init_mtp(self)
                logger.info(
                    "MTP path activated for uid=%s (model has mtp_forward, batch=1)",
                    getattr(self, "uids", ["?"])[0],
                )
            except _MtpStepFallback as exc:
                logger.warning("MTP post-init fallback: %s", exc)
        else:
            # The empty-batch case is BatchGenerator.__init__ pre-creating
            # ``self._generation_batch = GenerationBatch.empty(...)`` and is
            # always part of normal startup — silence it. Only log when the
            # batch is genuinely populated (e.g. continuous batching with
            # batch>1) so the message points at a real misconfiguration.
            uids = getattr(self, "uids", None)
            if uids:
                reason = _ineligibility_reason(self)
                if reason:
                    logger.debug("MTP path not active: %s", reason)

    def patched_next(self, *args, **kwargs):
        if _is_mtp_eligible(self):
            state = getattr(self, "_omlx_mtp_state", None)
            if state is not None:
                try:
                    return _mtp_next(self, state)
                except _MtpStepFallback as exc:
                    logger.debug(
                        "MTP next() fallback to standard step: %s", exc
                    )
                    # Best-effort: drop state so subsequent calls don't try
                    # to resume a half-built MTP cycle from a stale snapshot.
                    if hasattr(self, "_omlx_mtp_state"):
                        try:
                            delattr(self, "_omlx_mtp_state")
                        except AttributeError:
                            pass
        return original_next(self, *args, **kwargs)

    def patched_extend(self, batch, *args, **kwargs):
        # ``BatchGenerator._next()`` builds a fresh single-sequence
        # ``GenerationBatch`` via ``prompt_batch.split(...).generate(...)``
        # then merges it into ``self._generation_batch`` via extend(). The
        # MTP post-init runs on the fresh batch (since that's the one whose
        # __init__ fires with uids=[0]); without this transfer the state
        # would die with the donor instance.
        donor_state = getattr(batch, "_omlx_mtp_state", None)
        result = original_extend(self, batch, *args, **kwargs)
        if donor_state is not None and not hasattr(self, "_omlx_mtp_state"):
            self._omlx_mtp_state = donor_state
            try:
                delattr(batch, "_omlx_mtp_state")
            except AttributeError:
                pass
            logger.debug(
                "MTP state transferred from donor batch to host batch (uid=%s)",
                getattr(self, "uids", ["?"])[0] if getattr(self, "uids", None) else "?",
            )
        return result

    def patched_filter(self, keep, *args, **kwargs):
        # When the outer scheduler retires this sequence (e.g. EOS detected
        # outside our finish path), it calls filter([]) to drop everything.
        # Surface stats here so the user sees them even when the standard
        # _emit_response finish path doesn't fire.
        state = getattr(self, "_omlx_mtp_state", None)
        result = original_filter(self, keep, *args, **kwargs)
        if state is not None and not getattr(self, "uids", None):
            # Batch is now empty — log + drop state.
            try:
                _log_mtp_stats(
                    "?", state.stats, getattr(state, "_finish_reason", "external")
                )
            except Exception:
                pass
            try:
                delattr(self, "_omlx_mtp_state")
            except AttributeError:
                pass
        return result

    GenerationBatch.__init__ = patched_init
    GenerationBatch.next = patched_next
    GenerationBatch.filter = patched_filter
    GenerationBatch.extend = patched_extend
    GenerationBatch._omlx_mtp_patched = True
    _PATCHED = True
    return True


def _model_has_mtp_module(model: Any) -> bool:
    """Check whether the model actually has an MTP head attached.

    The ``mtp_forward`` method is added to the class unconditionally by
    the patch, but the per-instance ``mtp`` module is only attached when
    ``mtp_enabled`` was True at load time (see qwen35_model._patch_model
    and deepseek_v4_model._patch_model). Without the inner module the
    ``mtp_forward`` call would AttributeError, so we gate eligibility on
    the actual module's presence.
    """
    inner = getattr(model, "language_model", model)
    return hasattr(inner, "mtp") and getattr(inner, "mtp", None) is not None


def _is_mtp_eligible(gen_batch: Any) -> bool:
    """``__init__`` and ``next`` only engage MTP for single-sequence batches
    when the model exposes ``mtp_forward`` *and* has an attached MTP head."""
    if not hasattr(gen_batch, "model"):
        return False
    if not hasattr(gen_batch.model, "mtp_forward"):
        return False
    if not _model_has_mtp_module(gen_batch.model):
        return False
    uids = getattr(gen_batch, "uids", None)
    if uids is None or len(uids) != 1:
        return False
    return True


def _ineligibility_reason(gen_batch: Any) -> str:
    """Return a short human-readable reason for why the MTP path isn't active.

    Only used for debug logging — the patched_init / patched_next paths
    don't act on this string.
    """
    if not hasattr(gen_batch, "model"):
        return "GenerationBatch has no .model attribute"
    if not hasattr(gen_batch.model, "mtp_forward"):
        return (
            f"model {type(gen_batch.model).__module__}.{type(gen_batch.model).__name__} "
            "has no mtp_forward (qwen35 patch may not have applied to this class)"
        )
    if not _model_has_mtp_module(gen_batch.model):
        return "model has no attached mtp head (mtp_enabled was False at load time)"
    uids = getattr(gen_batch, "uids", None)
    if uids is None:
        return "GenerationBatch has no uids"
    if len(uids) != 1:
        return f"batch size {len(uids)} != 1 (continuous batching, MTP off by design)"
    return ""


class _MtpStepFallback(RuntimeError):
    """Raised inside the MTP path to signal a clean fallback to the standard step."""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class _MtpStats:
    """Acceptance / throughput counters for one MTP-active sequence.

    Logged at INFO when the sequence finishes (length / stop / filter)
    so the operator can see whether the draft+verify cycle is actually
    productive on this model + sampler combo.
    """

    cycles: int = 0  # number of verify cycles run
    accepts: int = 0  # cycles where the FULL draft chain was accepted
    rejects: int = 0  # cycles with at least one rejected draft position
    init_emits: int = 0  # tokens emitted from the post-init queue (always 2)
    draft_emits: int = 0  # tokens emitted as accepted drafts
    bonus_emits: int = 0  # tokens emitted as bonus (accepted + emit_bonus)
    verify_emits: int = 0  # tokens emitted as verify-position correction (reject path)
    # Depth-aware counters (depth > 1 engages only when every cache layer is
    # trimmable — pure-KV families like hy_v3; hybrids stay depth-1).
    depth: int = 1  # resolved draft depth for this sequence
    draft_tokens_proposed: int = 0  # sum of chain lengths across cycles
    draft_tokens_accepted: int = 0  # sum of accepted prefix lengths
    accepted_by_depth: List[int] = field(default_factory=lambda: [0, 0, 0])
    drafted_by_depth: List[int] = field(default_factory=lambda: [0, 0, 0])
    seed_main_forwards: int = 0
    verify_main_forwards: int = 0
    mtp_forwards: int = 0
    # Component-level timings. Help diagnose where MTP overhead comes from
    # when accept rate is healthy but wall-clock throughput isn't.
    backbone_ms: float = 0.0  # cumulative time inside the 2-token verify forward
    mtp_head_ms: float = 0.0  # cumulative time inside MTP-head forwards
    sample_ms: float = 0.0  # cumulative time in sampling + acceptance check
    cache_ops_ms: float = 0.0  # cumulative time in trim / rollback restore


@dataclass
class _MtpState:
    """Per-batch MTP state stashed on the GenerationBatch instance."""

    # Pending tokens to emit in upcoming next() calls. Each entry is
    # (token_id_int, logprobs_1d, source_label). source_label is one of
    # "init", "draft", "bonus", "verify" — used to bucket stats correctly
    # when the queue is drained.
    queue: Deque[Tuple[int, Any, str]] = field(default_factory=deque)

    # Cache for the MTP head (separate from gen_batch.prompt_cache).
    mtp_cache: Optional[List[Any]] = None

    # First input token of the next verify forward. Tracked as a 1-element
    # mx.array (uint32) so it can be concatenated with the drafts cheaply.
    next_main: Optional[Any] = None

    # Draft chain (depth >= 1). Parallel lists, one entry per chained draft:
    # d1 predicts the token after next_main, d2 the one after d1, ...
    draft_toks: List[Any] = field(default_factory=list)  # each (1,) uint32
    draft_lps: List[Any] = field(default_factory=list)  # each (vocab,) float
    # Filtered (sampler-applied) draft logprobs reused by the next cycle's
    # acceptance ratio + residual sampling.
    draft_accept_lps: List[Any] = field(default_factory=list)  # each (vocab,)
    # Host-side int copies cached at draft creation time so the verify cycle
    # compares draft vs verify ids without a GPU→CPU sync per position.
    draft_ids: List[int] = field(default_factory=list)

    # Resolved draft depth for this sequence (1..3).
    depth: int = 1

    # Accept-rate / throughput counters. Surfaced via logger.info on finish.
    stats: _MtpStats = field(default_factory=_MtpStats)


def _mtp_rate(accepted: int, drafted: int) -> Optional[float]:
    if drafted <= 0:
        return None
    return accepted / drafted


def _native_mtp_payload(
    uid: Any, stats: _MtpStats, finish_reason: str
) -> dict:
    timings = {
        "verify": float(stats.backbone_ms),
        "sample": float(stats.sample_ms),
        "draft": float(stats.mtp_head_ms),
        "cache": float(stats.cache_ops_ms),
    }
    timing_total = sum(timings.values())
    timings["total"] = timing_total
    timings["avg_cycle"] = timing_total / max(1, int(stats.cycles or 0))
    depth_rates = {
        label: _mtp_rate(
            int(stats.accepted_by_depth[index]),
            int(stats.drafted_by_depth[index]),
        )
        for index, label in enumerate(("d1", "d2", "d3"))
    }
    return {
        "request_id": str(uid),
        "finish_reason": finish_reason,
        "final_depth": int(stats.depth or 1),
        "cycles": int(stats.cycles),
        "accepts": int(stats.accepts),
        "rejects": int(stats.rejects),
        "init_emits": int(stats.init_emits),
        "draft_emits": int(stats.draft_emits),
        "bonus_emits": int(stats.bonus_emits),
        "verify_emits": int(stats.verify_emits),
        "drafted_tokens": int(stats.draft_tokens_proposed),
        "accepted_tokens": int(stats.draft_tokens_accepted),
        "acceptance_rate": _mtp_rate(
            int(stats.draft_tokens_accepted),
            int(stats.draft_tokens_proposed),
        ),
        "accepted_by_depth": list(stats.accepted_by_depth),
        "drafted_by_depth": list(stats.drafted_by_depth),
        "depth_acceptance_rates": depth_rates,
        "forwards": {
            "seed_main": int(stats.seed_main_forwards),
            "verify_main": int(stats.verify_main_forwards),
            "replay_main": 0,
            "mtp": int(stats.mtp_forwards),
        },
        "timings_ms": timings,
        "profiled_phase_timing": bool(_MTP_PROFILE),
        "published_at": time.time(),
        "fallback_reason": None,
    }


def _publish_native_mtp_stats(
    uid: Any, stats: _MtpStats, finish_reason: str
) -> dict:
    global _LAST_NATIVE_MTP

    payload = _native_mtp_payload(uid, stats, finish_reason)
    with _MTP_TELEMETRY_LOCK:
        _LAST_NATIVE_MTP = payload
        _NATIVE_MTP_TOTALS["requests"] += 1
        _NATIVE_MTP_TOTALS["cycles"] += int(stats.cycles)
        _NATIVE_MTP_TOTALS["drafted_tokens"] += int(
            stats.draft_tokens_proposed
        )
        _NATIVE_MTP_TOTALS["accepted_tokens"] += int(
            stats.draft_tokens_accepted
        )
    return payload


def native_mtp_stats_snapshot() -> dict:
    """Return immutable process-local BatchGenerator MTP acceptance telemetry."""
    with _MTP_TELEMETRY_LOCK:
        totals = dict(_NATIVE_MTP_TOTALS)
        totals["acceptance_rate"] = _mtp_rate(
            int(totals["accepted_tokens"]), int(totals["drafted_tokens"])
        )
        return {
            "last_native_mtp": copy.deepcopy(_LAST_NATIVE_MTP),
            "native_mtp_totals": totals,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_generation_stream():
    """Return the ``mlx_lm.generate`` module-level generation stream.

    The standard ``GenerationBatch._step`` runs all forward passes inside
    ``mx.stream(generation_stream)``; the MTP cycle does the same so the
    paged cache writes land on the same stream and ordering is preserved.
    The stream lives on the *outer* ``BatchGenerator``, not on
    ``GenerationBatch``, so we read it from the module.

    Note: ``mlx_lm.__init__`` re-exports a ``generate`` *function*, so
    ``import mlx_lm.generate as mlg`` resolves to the function, not the
    module. We use ``sys.modules`` to grab the actual module.
    """
    import sys

    return sys.modules["mlx_lm.generate"].generation_stream


def _resolve_sampler(gen_batch: Any):
    """Match ``GenerationBatch._step``'s per-sequence sampler resolution (batch=1)."""
    if gen_batch.samplers and gen_batch.samplers[0] is not None:
        return gen_batch.samplers[0]
    return gen_batch.fallback_sampler


def _is_greedy(gen_batch: Any) -> bool:
    """Return whether the active batch uses the deterministic sampler path."""
    if gen_batch.samplers and gen_batch.samplers[0] is not None:
        # Seeded requests install a request-local sampler even at temperature
        # zero. Object presence therefore cannot distinguish stochastic from
        # greedy: recognize the explicit argmax contract marker.
        return bool(getattr(gen_batch.samplers[0], "_vmlx_is_greedy", False))
    return bool(getattr(gen_batch.fallback_sampler, "_vmlx_is_greedy", True))


def _proc_list(gen_batch: Any) -> Optional[List[Any]]:
    if gen_batch.logits_processors and gen_batch.logits_processors[0]:
        return gen_batch.logits_processors[0]
    return None


def _apply_processors(processors, prev_tokens, logits_2d):
    if not processors:
        return logits_2d
    for proc in processors:
        logits_2d = proc(prev_tokens, logits_2d)
    return logits_2d


def _logprobs(logits_2d):
    import mlx.core as mx

    return logits_2d - mx.logsumexp(logits_2d, axis=-1, keepdims=True)


def _accept_lp_for(sampler, lp):
    """Reproduce the sampler's filter+temperature pipeline on `lp` so the
    acceptance ratio (and residual distribution) match the distribution the
    sampler actually drew from.

    Reads sampling params off the callable as function attributes. For stock
    mlx-lm/vMLX samplers without reusable filter helpers, returns `lp`
    unchanged so behavior matches the pre-PR-990 raw-lp acceptance.
    """
    import mlx.core as mx

    try:
        from omlx.utils.sampling import apply_min_p, apply_top_k, apply_top_p
    except Exception:
        return lp

    temp = float(getattr(sampler, "temp", 0.0) or 0.0)
    if temp == 0.0:
        # Greedy / unknown sampler — raw lp is the acceptance distribution.
        return lp

    out = lp
    top_p = float(getattr(sampler, "top_p", 0.0) or 0.0)
    if 0.0 < top_p < 1.0:
        out = apply_top_p(out, top_p)
    min_p = float(getattr(sampler, "min_p", 0.0) or 0.0)
    if min_p != 0.0:
        min_keep = int(getattr(sampler, "min_tokens_to_keep", 1) or 1)
        out = apply_min_p(out, min_p, min_keep)
    top_k = int(getattr(sampler, "top_k", 0) or 0)
    if top_k > 0:
        out = apply_top_k(out, top_k)

    # Temperature scale + renormalize so the output is a proper logprob
    # distribution that can be indexed by token id for the acceptance check.
    scaled = out * (1.0 / temp)
    return scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)


def _trim_token_buffer(gen_batch: Any, n: int) -> None:
    """Shrink ``_token_context[0]`` by ``n`` speculative tokens."""
    if n <= 0:
        return
    procs = _proc_list(gen_batch)
    if procs is None:
        return
    buf = gen_batch._token_context[0]
    buf._size = max(0, buf._size - n)


def _restore_or_trim_caches(prompt_cache: List[Any], n: int = 1) -> bool:
    """Roll back ``n`` speculative tokens from each layer cache after a
    (possibly partial) draft-chain rejection.

    SSM / linear-attention layers expose ``rollback_state`` populated by the
    patched ``GatedDeltaNet.__call__``; that snapshot restores to the
    confirmed prefix, i.e. it drops ALL speculative tokens of the last
    forward. Partial rollback (n < chain length) is therefore only valid on
    trimmable KV layers — which is why depth > 1 is gated on every cache
    layer being trimmable (see ``_effective_depth``). Standard KV cache
    layers trim by ``n``. Layers that support neither cause the entire MTP
    step to fall back to the standard path.
    """
    for c in prompt_cache:
        rollback = getattr(c, "rollback_state", None)
        if rollback is not None:
            conv_snap, ssm_snap = rollback
            c[0] = conv_snap
            c[1] = ssm_snap
            c.rollback_state = None
            continue
        if hasattr(c, "is_trimmable") and c.is_trimmable():
            c.trim(n)
            continue
        return False
    return True


def _effective_depth(gen_batch: Any) -> int:
    """Resolve the draft-chain depth (1..3) for this sequence.

    Sources, in order: VMLINUX/VMLX_NATIVE_MTP_DEPTH env, the bundle's
    validated ``vmlx_mtp_tuning.json``, default (via
    ``native_mtp_effective_depth``). Depth > 1 additionally requires:
      - every prompt-cache layer trimmable (partial rollback is a KV trim;
        SSM rollback_state can only restore to the confirmed prefix), and
      - ``mtp_forward`` supporting ``return_hidden`` (chained drafting feeds
        the head's hidden back as the next step's previous-hidden).
    Hybrid families (qwen3.5/3.6) fail the trimmable check and keep the
    proven depth-1 behavior byte-for-byte.
    """
    try:
        from vmlx_engine.native_mtp import native_mtp_effective_depth

        depth, _source = native_mtp_effective_depth(None)
    except Exception:
        depth = 1
    depth = max(1, min(3, int(depth or 1)))
    if depth <= 1:
        return 1
    for c in gen_batch.prompt_cache:
        if getattr(c, "rollback_state", None) is not None:
            return 1
        if not (hasattr(c, "is_trimmable") and c.is_trimmable()):
            return 1
    return depth


def _clear_rollback(prompt_cache: List[Any]) -> None:
    """Drop ``rollback_state`` snapshots after a draft is accepted."""
    for c in prompt_cache:
        if hasattr(c, "rollback_state") and c.rollback_state is not None:
            c.rollback_state = None


def _ensure_uint32(arr):
    """Ensure a 1-element mx.array is uint32 (cache update_and_fetch expects it)."""
    import mlx.core as mx

    if arr.dtype == mx.uint32:
        return arr
    return arr.astype(mx.uint32)


# ---------------------------------------------------------------------------
# Post-init: run one extra backbone forward + MTP forward; queue the two
# emitted tokens; stash a draft for the first verify cycle.
# ---------------------------------------------------------------------------

def _post_init_mtp(gen_batch: Any) -> None:
    """Bridge from standard ``__init__``'s ``_step()`` into vMLX's MTP cycle.

    State on entry (after standard ``__init__``):
      - cache contains the prompt up to ``prompt[-1]`` inclusive
      - ``_next_tokens`` = ``main_tok`` (token sampled from ``prompt[-1]``'s logits)
      - ``_next_logprobs[0]`` = main_tok's distribution
      - ``tokens[0]`` = original prompt list

    We perform one more 1-token backbone forward (so the cache also includes
    ``main_tok`` and we obtain the hidden state at that position), run the
    MTP head to produce a draft for the next verify cycle, and seed
    ``state.queue`` with two confirmed tokens — ``main_tok`` and the
    standard-sample at the next position. After this, the queue handles
    the first two emit calls and the third call enters the verify cycle.

    If the batch was empty when ``__init__`` ran, ``_next_tokens`` is
    ``None`` — we leave MTP inactive and the standard path runs unchanged.
    """
    import mlx.core as mx

    if gen_batch._next_tokens is None or not gen_batch.uids:
        # Nothing was sampled in the standard _step (empty batch). The
        # next() call will be a no-op anyway; leave the patch inert.
        return

    sampler = _resolve_sampler(gen_batch)
    procs = _proc_list(gen_batch)

    main_tok = _ensure_uint32(gen_batch._next_tokens)  # (1,)
    main_lp = gen_batch._next_logprobs[0]  # (vocab,)

    if procs is not None:
        prev_buf = gen_batch._token_context[0].update_and_fetch(main_tok)
    else:
        prev_buf = None

    # 1-token backbone forward at main_tok with hidden state.
    with mx.stream(_get_generation_stream()):
        logits, hidden = gen_batch.model(
            main_tok[:, None], cache=gen_batch.prompt_cache, return_hidden=True
        )

    next_main_logits = logits[:, -1, :]  # (1, vocab) — distribution after main_tok
    next_main_logits = _apply_processors(procs, prev_buf, next_main_logits)
    next_main_lp = _logprobs(next_main_logits)
    next_main_tok = sampler(next_main_lp)  # (1,)

    mx.eval(main_tok, next_main_tok)

    state = _MtpState()
    state.mtp_cache = gen_batch.model.make_mtp_cache()
    state.depth = _effective_depth(gen_batch)
    state.stats.depth = state.depth
    state.stats.seed_main_forwards = 1
    state.next_main = _ensure_uint32(next_main_tok)
    state.queue.append((int(main_tok.tolist()[0]), main_lp, "init"))
    state.queue.append(
        (int(next_main_tok.tolist()[0]), next_main_lp.squeeze(0), "init")
    )
    gen_batch._omlx_mtp_state = state

    # Draft chain: the head sees (hidden_at_main, next_main_tok) and proposes
    # d1..dN for the first verify cycle forward([next_main, d1..dN]).
    hidden_at_main = hidden[:, -1:, :]  # (1, 1, H)
    _draft_chain(
        gen_batch,
        state,
        hidden_at_main,
        state.next_main,
        prev_buf=prev_buf,
    )


# ---------------------------------------------------------------------------
# next() dispatch
# ---------------------------------------------------------------------------

def _mtp_next(gen_batch: Any, state: _MtpState) -> Any:
    """Emit one token; run a verify cycle if the queue is empty."""
    if state.queue:
        token_id, logprobs_1d, source = state.queue.popleft()
        _bump_emit_stat(state, source)
        return _emit_response(gen_batch, token_id, logprobs_1d, state.stats)

    _run_verify_cycle(gen_batch, state)
    if not state.queue:
        # Verify cycle should always populate the queue with at least the
        # rejected-verify token; if it didn't, fall back to the standard
        # step rather than yield an undefined response.
        raise _MtpStepFallback("verify cycle produced no emit tokens")

    token_id, logprobs_1d, source = state.queue.popleft()
    _bump_emit_stat(state, source)
    return _emit_response(gen_batch, token_id, logprobs_1d, state.stats)


def _log_mtp_stats(uid: Any, stats: "_MtpStats", finish_reason: str) -> None:
    """Emit a one-line summary of MTP draft/verify activity for a finished sequence.

    Format chosen to make wall-clock vs. accept-rate gaps debuggable:
      MTP[<uid>] finish=<reason> tokens=<N> cycles=<C> accept=<A>/<C> (<rate>%)
        emits[init=<i>,draft=<d>,bonus=<b>,verify=<v>]
        timing[backbone=<X>ms mtp=<Y>ms sample=<S>ms cache=<C>ms]
    """
    payload = _publish_native_mtp_stats(uid, stats, finish_reason)
    total_emits = (
        stats.init_emits + stats.draft_emits + stats.bonus_emits + stats.verify_emits
    )
    if stats.cycles > 0:
        rate_str = f"{stats.accepts / stats.cycles * 100:.1f}%"
    else:
        rate_str = "n/a"
    if payload["acceptance_rate"] is not None:
        tok_rate_str = f"{payload['acceptance_rate'] * 100:.1f}%"
    else:
        tok_rate_str = "n/a"
    logger.info(
        "MTP[%s] finish=%s depth=%d tokens=%d cycles=%d full-accept=%d/%d (%s) "
        "draft-tokens=%d/%d (%s) "
        "emits[init=%d,draft=%d,bonus=%d,verify=%d] "
        "timing[backbone=%.1fms mtp=%.1fms sample=%.1fms cache=%.1fms]",
        uid,
        finish_reason,
        stats.depth,
        total_emits,
        stats.cycles,
        stats.accepts,
        stats.cycles,
        rate_str,
        stats.draft_tokens_accepted,
        stats.draft_tokens_proposed,
        tok_rate_str,
        stats.init_emits,
        stats.draft_emits,
        stats.bonus_emits,
        stats.verify_emits,
        stats.backbone_ms,
        stats.mtp_head_ms,
        stats.sample_ms,
        stats.cache_ops_ms,
    )


def _bump_emit_stat(state: _MtpState, source: str) -> None:
    if source == "init":
        state.stats.init_emits += 1
    elif source == "draft":
        state.stats.draft_emits += 1
    elif source == "bonus":
        state.stats.bonus_emits += 1
    elif source == "verify":
        state.stats.verify_emits += 1


# ---------------------------------------------------------------------------
# Verify cycle: 2-token forward + accept/reject + MTP forward for next draft.
# ---------------------------------------------------------------------------

def _run_verify_cycle(gen_batch: Any, state: _MtpState) -> None:
    """Run one verify cycle over the draft chain ``d1..dN``.

    One backbone forward over ``[next_main, d1..dN]`` (N+1 tokens,
    ``n_confirmed=1``), longest-prefix acceptance, then one new draft chain
    from the last confirmed position. Populates ``state.queue`` with
    ``k`` accepted drafts + 1 correction/bonus token (k in 0..N), rolls the
    cache back by ``N - k`` on partial/full rejection, and refreshes
    ``state.next_main`` / draft-chain lists for the next cycle.

    Depth 1 reproduces the original 2-token draft+bonus cycle exactly.
    """
    import time

    import mlx.core as mx

    if state.next_main is None or not state.draft_toks:
        raise _MtpStepFallback("verify cycle entered without next_main / drafts")

    sampler = _resolve_sampler(gen_batch)
    procs = _proc_list(gen_batch)
    is_greedy = _is_greedy(gen_batch)
    n = len(state.draft_toks)

    inputs = mx.concatenate([state.next_main] + list(state.draft_toks))  # (n+1,)

    # Update the token buffer per position so logits processors see the same
    # history shape as standard autoregressive decode. prev_bufs[i] is the
    # buffer state after consuming inputs[0..i].
    prev_bufs: List[Any] = []
    if procs is not None:
        prev_bufs.append(
            gen_batch._token_context[0].update_and_fetch(state.next_main)
        )
        for d in state.draft_toks:
            prev_bufs.append(gen_batch._token_context[0].update_and_fetch(d))

    # --- backbone forward + sample (single eval point) ---
    # Dispatch backbone, processors, logprobs, and sampler all on stream
    # without forcing intermediate evaluation. The single ``mx.eval`` after
    # sampling resolves the whole graph in one stall instead of two.
    t0 = time.perf_counter()
    with mx.stream(_get_generation_stream()):
        logits, hidden = gen_batch.model(
            inputs[None, :],
            cache=gen_batch.prompt_cache,
            return_hidden=True,
            n_confirmed=1,
        )
    _pbar(logits, hidden)  # profiling: isolate real verify-forward GPU cost
    state.stats.backbone_ms += (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    if procs is None:
        # The (n+1, vocab) slab is already contiguous — skip n+1 slices + a
        # concatenate. (Measured neutral on Hy3; kept because it is simply
        # less work. The per-cycle cost that makes MTP a loss on this MoE is
        # the verify forward itself, not this bookkeeping — see below.)
        combined_logits = logits[0]
    else:
        combined_logits = mx.concatenate(
            [
                _apply_processors(procs, prev_bufs[i], logits[:, i, :])
                for i in range(n + 1)
            ],
            axis=0,
        )
    combined_lp = combined_logits - mx.logsumexp(
        combined_logits, axis=-1, keepdims=True
    )
    # Sample per row: a single batched sampler() call measured identical on
    # Hy3 and would change RNG consumption for stochastic samplers, so keep
    # the row-wise draws the depth-1 path has always used.
    sampled = [sampler(combined_lp[i : i + 1]) for i in range(n + 1)]
    mx.eval(*sampled)
    sampled_ids = [int(t.tolist()[0]) for t in sampled]

    # Longest-prefix acceptance. Position i's logits verify draft i+1
    # (0-indexed: combined_lp[i] is the target distribution for d_{i+1}).
    k = 0
    while k < n:
        d_id = state.draft_ids[k]
        if is_greedy:
            ok = sampled_ids[k] == d_id
        else:
            verify_accept_lp = _accept_lp_for(sampler, combined_lp[k : k + 1])
            log_accept = (
                verify_accept_lp[0, d_id].item()
                - state.draft_accept_lps[k][d_id].item()
            )
            _uniform = getattr(sampler, "_vmlx_random_uniform", None)
            draw = _uniform() if _uniform is not None else random.random()
            ok = log_accept >= 0 or draw < math.exp(log_accept)
        if not ok:
            break
        k += 1
    state.stats.sample_ms += (time.perf_counter() - t0) * 1000

    state.stats.cycles += 1
    state.stats.verify_main_forwards += 1
    state.stats.draft_tokens_proposed += n
    state.stats.draft_tokens_accepted += k
    for index in range(n):
        state.stats.drafted_by_depth[index] += 1
        if index < k:
            state.stats.accepted_by_depth[index] += 1
    full_accept = k == n
    if full_accept:
        state.stats.accepts += 1
    else:
        state.stats.rejects += 1

    # --- cache rollback / cleanup (timed) ---
    t0 = time.perf_counter()
    if full_accept:
        _clear_rollback(gen_batch.prompt_cache)
    else:
        if not _restore_or_trim_caches(gen_batch.prompt_cache, n - k):
            if procs is not None:
                _trim_token_buffer(gen_batch, n - k)
            raise _MtpStepFallback("cache layer rejects rollback")
        if procs is not None:
            _trim_token_buffer(gen_batch, n - k)
    state.stats.cache_ops_ms += (time.perf_counter() - t0) * 1000

    # --- queue emits: k accepted drafts + 1 correction/bonus ---
    for i in range(k):
        state.queue.append((state.draft_ids[i], state.draft_lps[i], "draft"))

    if full_accept:
        emit_id = sampled_ids[n]
        emit_lp = combined_lp[n : n + 1].squeeze(0)
        source = "bonus"
    elif is_greedy:
        emit_id = sampled_ids[k]
        emit_lp = combined_lp[k : k + 1].squeeze(0)
        source = "verify"
    else:
        # Residual sample on the *filtered* distributions so the sample
        # comes from `max(p_target_filt - p_draft_filt, 0)`. emit_lp stays
        # the raw verify lp so downstream logprobs reporting is consistent
        # with non-MTP paths.
        verify_accept_lp = _accept_lp_for(sampler, combined_lp[k : k + 1])
        emit_id, _ = _residual_sample(
            verify_accept_lp, state.draft_accept_lps[k], sampler=sampler
        )
        emit_lp = combined_lp[k : k + 1].squeeze(0)
        source = "verify"
    state.queue.append((emit_id, emit_lp, source))

    # --- new draft chain from the last confirmed position ---
    emit_tok = mx.array([emit_id], dtype=mx.uint32)
    state.next_main = emit_tok
    _draft_chain(
        gen_batch,
        state,
        hidden[:, k : k + 1, :],
        emit_tok,
        prev_buf=prev_bufs[k] if procs is not None else None,
    )


# ---------------------------------------------------------------------------
# Helpers used by the verify cycle.
# ---------------------------------------------------------------------------

def _draft_chain(
    gen_batch: Any,
    state: _MtpState,
    anchor_hidden: Any,
    t0_tok: Any,
    prev_buf: Optional[Any],
) -> None:
    """Draft ``state.depth`` chained tokens with the MTP head.

    d1 = head(anchor_hidden, t0); d_{i+1} = head(h_i, d_i) where h_i is the
    head's own (post-final-norm) hidden from step i — the same recursion
    vLLM uses for multi-step MTP with a single head. Replaces the previous
    single-draft ``_step_mtp``. The head cache is never rolled back (loose
    history by design — verify guarantees correctness; the cache only
    shapes draft quality).

    Fills ``state.draft_toks / draft_lps / draft_accept_lps / draft_ids``.
    """
    import time

    import mlx.core as mx

    sampler = _resolve_sampler(gen_batch)
    procs = _proc_list(gen_batch)

    state.draft_toks = []
    state.draft_lps = []
    state.draft_accept_lps = []
    state.draft_ids = []

    t_start = time.perf_counter()
    hidden = anchor_hidden
    cur_tok = t0_tok
    proc_ctx = prev_buf
    for _ in range(max(1, state.depth)):
        state.stats.mtp_forwards += 1
        next_ids = cur_tok.reshape(1, 1)
        with mx.stream(_get_generation_stream()):
            mtp_logits, mtp_hidden = gen_batch.model.mtp_forward(
                hidden, next_ids, state.mtp_cache, return_hidden=True
            )
            mtp_logits_2d = mtp_logits[:, -1, :]
        if procs is not None and proc_ctx is not None:
            proc_ctx = mx.concatenate([proc_ctx, _ensure_uint32(cur_tok)])
            mtp_logits_2d = _apply_processors(procs, proc_ctx, mtp_logits_2d)
        new_lp = _logprobs(mtp_logits_2d)
        new_tok = sampler(new_lp)
        # Filtered draft lp — what the sampler actually drew from; the verify
        # cycle's acceptance ratio uses this so the math matches the sampling
        # distribution rather than raw softmax.
        new_accept_lp = _accept_lp_for(sampler, new_lp)
        # ``.tolist()`` forces evaluation and doubles as the host-side copy.
        new_id = int(new_tok.tolist()[0])

        state.draft_toks.append(_ensure_uint32(new_tok))
        state.draft_lps.append(new_lp.squeeze(0))
        state.draft_accept_lps.append(new_accept_lp.squeeze(0))
        state.draft_ids.append(new_id)

        hidden = mtp_hidden[:, -1:, :]
        cur_tok = state.draft_toks[-1]
    _pbar(*state.draft_toks)  # profiling: isolate real MTP-head GPU cost
    state.stats.mtp_head_ms += (time.perf_counter() - t_start) * 1000


def _residual_sample(
    verify_lp_2d: Any, draft_lp_1d: Any, *, sampler=None
) -> Tuple[int, Any]:
    """Sample from ``max(p_target - p_draft, 0)`` (Leviathan et al. 2022).

    On degenerate input (residual all zero) falls back to the target
    distribution rather than the verify-position argmax — keeps the sample
    drawn from a proper distribution and stays in-graph (no host sync).
    This is vMLX's local fallback for speculative rejection at temperature.

    Returns ``(token_id_int, verify_lp_1d)``.
    """
    import mlx.core as mx

    p_target = mx.exp(verify_lp_2d.squeeze(0))
    p_draft = mx.exp(draft_lp_1d)
    residual = mx.maximum(p_target - p_draft, 0.0)
    # Keep z in graph; mx.where switches to the target distribution when
    # the residual mass is zero. ``categorical`` treats log(0) = -inf as
    # p=0 so no safety epsilon is needed.
    z = residual.sum(keepdims=True)
    dist = mx.where(z > 0, residual, p_target)
    logits = mx.log(dist).reshape(1, -1)
    categorical = getattr(sampler, "_vmlx_categorical", None)
    sample = categorical(logits) if categorical is not None else mx.random.categorical(logits)
    return int(sample.item()), verify_lp_2d.squeeze(0)


# ---------------------------------------------------------------------------
# Response builder — mirrors GenerationBatch.next()'s per-sequence epilogue.
# ---------------------------------------------------------------------------

def _emit_response(
    gen_batch: Any,
    token_id: int,
    logprobs_1d: Any,
    stats: Optional["_MtpStats"] = None,
) -> List[Any]:
    """Produce a single-element response list, applying the standard
    epilogue (token append + max_tokens / matcher checks) so external
    callers (BatchGenerator, scheduler, response stream) see the same
    contract as the unmodified next().
    """
    Response = type(gen_batch).Response

    finish_reason: Optional[str] = None
    match_sequence = None

    gen_batch.tokens[0].append(token_id)
    gen_batch._num_tokens[0] += 1
    if gen_batch._num_tokens[0] >= gen_batch.max_tokens[0]:
        finish_reason = "length"

    new_state, match_sequence, current_state = gen_batch.state_machines[0].match(
        gen_batch._matcher_states[0], token_id
    )
    gen_batch._matcher_states[0] = new_state
    if match_sequence is not None and current_state is None:
        finish_reason = "stop"

    if finish_reason is not None:
        prompt_cache = gen_batch.extract_cache(0)
        all_tokens = gen_batch.tokens[0]
        response = Response(
            uid=gen_batch.uids[0],
            token=token_id,
            logprobs=logprobs_1d,
            finish_reason=finish_reason,
            current_state=current_state,
            match_sequence=match_sequence,
            prompt_cache=prompt_cache,
            all_tokens=all_tokens,
        )
        if stats is not None:
            _log_mtp_stats(gen_batch.uids[0], stats, finish_reason)
        # Drop state *before* filter([]) so the patched_filter epilogue
        # doesn't double-log when the standard finish path already logged.
        if hasattr(gen_batch, "_omlx_mtp_state"):
            try:
                delattr(gen_batch, "_omlx_mtp_state")
            except AttributeError:
                pass
        gen_batch.filter([])
        return [response]

    return [
        Response(
            uid=gen_batch.uids[0],
            token=token_id,
            logprobs=logprobs_1d,
            finish_reason=None,
            current_state=current_state,
            match_sequence=match_sequence,
            prompt_cache=None,
            all_tokens=None,
        )
    ]
