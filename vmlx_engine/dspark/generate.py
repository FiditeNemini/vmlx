"""DSpark speculative decoding loop (greedy or sample-then-match) and its AR reference.

Each round: the drafter proposes a block from the anchor (the last committed,
not yet forwarded token) and the target's tap features; the target verifies
``[anchor, d_1..d_k]`` in one row-exact forward; drafts are accepted while
they equal the target's own pick at each position; the first mismatch (or the
bonus position) contributes the target's token. Every committed token is the
target's own choice given the committed prefix, so with greedy picks the
output equals AR decoding token for token (verify rows are bit-equal to
decode steps, see ``minicpm_target``). With a sampler each committed token is
a draw from the target distribution ("sample-then-match"); acceptance is then
P_target(draft) instead of 1[argmax = draft].
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional, Sequence

import mlx.core as mx

from .minicpm_target import MAX_VERIFY_ROWS


@dataclass
class DSparkStats:
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    ar_tokens: int = 0
    tokens: int = 0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    widths: dict = field(default_factory=dict)

    @property
    def tokens_per_round(self) -> float:
        return (self.tokens - 1 - self.ar_tokens) / self.rounds if self.rounds else 0.0

    @property
    def acceptance(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    def as_dict(self) -> dict:
        return {
            "rounds": self.rounds, "drafted": self.drafted, "accepted": self.accepted, "ar_tokens": self.ar_tokens,
            "tokens": self.tokens, "tokens_per_round": round(self.tokens_per_round, 3),
            "acceptance": round(self.acceptance, 4), "prefill_s": round(self.prefill_s, 3),
            "decode_s": round(self.decode_s, 3),
            "decode_tok_s": round((self.tokens - 1) / self.decode_s, 2) if self.decode_s > 0 else None,
            "widths": dict(sorted(self.widths.items())),
        }


def choose_width(conf: Sequence[float], max_draft: int, cost_per_row: float = 0.12) -> int:
    """Draft count k maximizing E[tokens | k] / cost(k + 1 rows).

    E[tokens | k] = 1 + sum_j prod_{i<=j} conf_i (confidence head = predicted
    acceptance); cost(T) = 1 + cost_per_row * (T - 1) decode-step units.
    Exactness does not depend on k; only speed does.
    """
    best, best_k, run, expected = 0.0, 1, 1.0, 1.0
    for k in range(1, min(max_draft, len(conf)) + 1):
        run *= float(conf[k - 1])
        expected += run
        value = expected / (1.0 + cost_per_row * k)
        if value > best:
            best, best_k = value, k
    return best_k


class ARFallback:
    """When to stop drafting (the DFlash2 lane's AdaptiveSpec policy, measured costs only).

    A draft round costs the drafter (fixed, ~5.4 ms on the 35B) plus a verify whose sparse-MoE cost grows with width,
    and pays only when enough drafts are accepted: on low-acceptance text (multilingual, thinking at T1.0) DSpark ran
    x0.62-0.86 of AR on the 35B even after the stock-block verify. The loop times every draft round and every AR step
    (synced wall time, EMA); when an AR step's measured rate beats the recent draft-round rate by ``margin`` it runs an
    AR burst of ``burst`` tokens, and every ``probe_every`` AR tokens one draft round re-measures the text. The first
    action is a draft round; the second is one AR step to seed its cost. Exactness is unaffected: AR steps and verify
    rows both equal decode steps; only the schedule changes.
    """

    alpha = 0.3
    margin = 0.05
    burst = 8
    probe_every = 24

    def __init__(self):
        self.draft_rate: Optional[float] = None     # committed tokens per second of draft rounds (EMA)
        self.ar_seconds: Optional[float] = None     # seconds per AR step (EMA)
        self.ar_left = 0
        self.since_probe = 0
        self.rounds = 0
        self.probe = False                           # the next action must be a draft round

    def next_is_ar(self) -> bool:
        if self.probe:
            self.probe = False
            return False
        if self.ar_left > 0:
            return True
        if self.rounds >= 1 and self.ar_seconds is None:
            self.ar_left = 1                         # seed the AR cost once
            return True
        if self.draft_rate is not None and self.ar_seconds is not None \
                and 1.0 / self.ar_seconds > self.draft_rate * (1.0 + self.margin):
            self.ar_left = self.burst
            return True
        return False

    def record_draft(self, tokens: int, seconds: float) -> None:
        self.rounds += 1
        self.since_probe = 0
        rate = tokens / max(seconds, 1e-9)
        self.draft_rate = rate if self.draft_rate is None else self.draft_rate + self.alpha * (rate - self.draft_rate)

    def record_ar(self, seconds: float) -> None:
        self.record_ar_burst(1, seconds)

    def burst_size(self) -> int:
        """AR steps to run now: the rest of the current burst, cut at the next draft probe."""
        return max(1, min(self.ar_left, self.probe_every - self.since_probe))

    def record_ar_burst(self, steps: int, seconds: float) -> None:
        per_step = seconds / max(1, steps)
        self.ar_seconds = per_step if self.ar_seconds is None else self.ar_seconds + self.alpha * (per_step - self.ar_seconds)
        self.ar_left = max(0, self.ar_left - steps)
        self.since_probe += steps
        if self.since_probe >= self.probe_every:
            self.ar_left = 0
            self.probe = True                        # one draft round re-measures the text


def _pick(logits: mx.array, sampler: Optional[Callable]) -> mx.array:
    if sampler is None:
        return mx.argmax(logits, axis=-1)
    lp = logits.astype(mx.float32)
    return sampler(lp - mx.logsumexp(lp, axis=-1, keepdims=True)).reshape(-1)


def iter_dspark(
    target,
    drafter,
    prompt_ids: Sequence[int],
    max_new: int,
    eos: set,
    *,
    prefill_step: Optional[int] = None,
    sampler: Optional[Callable] = None,
    max_draft: int = MAX_VERIFY_ROWS - 1,
    cost_per_row: float = 0.12,
    fixed_width: Optional[int] = None,
    stats: Optional[DSparkStats] = None,
    ar_fallback: Optional[ARFallback] = None,
) -> Iterator[List[int]]:
    """Yield the token ids committed by each round (prefill yields the first). ``ar_fallback`` (a policy instance,
    or None for draft rounds only) interleaves AR steps when drafting measures slower than plain decoding."""
    max_draft = max(1, min(int(max_draft), MAX_VERIFY_ROWS - 1, drafter.block))
    stats = stats if stats is not None else DSparkStats()
    taps = drafter.target_layer_ids
    cache = target.make_cache()
    drafter.reset()
    t0 = time.perf_counter()
    logits, feats = target.prefill(mx.array([list(prompt_ids)]), cache, taps, prefill_step)
    anchor = _pick(logits[None], sampler).reshape(1)
    drafter.add_context(feats)
    mx.eval(anchor, drafter._k, drafter._v)
    stats.prefill_s = time.perf_counter() - t0
    first = int(anchor.item())
    stats.tokens = 1
    yield [first]
    if first in eos or max_new <= 1:
        return
    mask_embed = target.embed_tokens(mx.array([[drafter.mask_token_id]]))
    t1 = time.perf_counter()
    produced = 1
    while produced < max_new:
        room = max_new - produced
        if ar_fallback is not None and fixed_width is None and ar_fallback.next_is_ar():
            # Pipelined AR burst: step i+1 is queued (async_eval) before step i's token is read, as in the AR lane,
            # and the burst's tap rows reach the drafter in one call. Same values as one-at-a-time steps.
            n_steps = max(1, min(int(ar_fallback.burst_size()), room))
            t_step = time.perf_counter()
            logits, feats = target.ar_step(anchor, cache, taps)
            pick = _pick(logits[None], sampler).reshape(1)
            mx.async_eval(pick, feats)
            burst_feats, new, stop = [], [], False
            for i in range(n_steps):
                nxt = None
                if i + 1 < n_steps:
                    logits2, feats2 = target.ar_step(pick.astype(anchor.dtype), cache, taps)
                    nxt = (_pick(logits2[None], sampler).reshape(1), feats2)
                    mx.async_eval(*nxt)
                tok = int(pick.item())
                burst_feats.append(feats)
                new.append(tok)
                if tok in eos:
                    stop = True
                    break
                if nxt is not None:
                    pick, feats = nxt
            drafter.add_context(mx.concatenate(burst_feats, axis=1) if len(burst_feats) > 1 else burst_feats[0])
            anchor = mx.array([new[-1]], dtype=anchor.dtype)
            ar_fallback.record_ar_burst(len(new), time.perf_counter() - t_step)
            stats.ar_tokens += len(new)
            produced += len(new)
            stats.tokens = produced
            stats.decode_s = time.perf_counter() - t1
            yield new
            if stop:
                return
            continue
        t_round = time.perf_counter()
        block = mx.concatenate(
            [target.embed_tokens(anchor.reshape(1, 1)),
             mx.broadcast_to(mask_embed, (1, drafter.block - 1, mask_embed.shape[-1]))], axis=1)
        hidden = drafter(block)
        n = min(max_draft, room - 1) if room > 1 else 0
        if n >= 1:
            drafts, conf = drafter.draft(hidden, anchor, target.lm_head, max_draft,
                                         sampler=sampler)
            mx.eval(drafts, conf)
            k = fixed_width if fixed_width is not None else choose_width(conf.tolist(), max_draft, cost_per_row)
            k = max(1, min(int(k), n))
            proposal = drafts[:k]
        else:
            k, proposal = 0, mx.array([], dtype=anchor.dtype)
        rows = mx.concatenate([anchor, proposal]).reshape(1, k + 1)
        logits, feats, sink = target.verify(rows, cache, taps)
        picks = _pick(logits, sampler)
        mx.eval(picks, feats)
        drafted = proposal.tolist() if k else []
        target_picks = picks.tolist()
        accepted = 0
        while accepted < k and drafted[accepted] == target_picks[accepted]:
            accepted += 1
        new = drafted[:accepted] + [target_picks[accepted]]
        target.rollback(cache, sink, kept=accepted + 1, total=k + 1)
        drafter.add_context(feats[:, :accepted + 1])
        anchor = mx.array([new[-1]], dtype=anchor.dtype)
        stats.rounds += 1
        stats.drafted += k
        stats.accepted += accepted
        if ar_fallback is not None:
            ar_fallback.record_draft(len(new), time.perf_counter() - t_round)
        stats.widths[k + 1] = stats.widths.get(k + 1, 0) + 1
        cut = next((i for i, t in enumerate(new) if t in eos), None)
        if cut is not None:
            new = new[:cut + 1]
        new = new[:max_new - produced]
        produced += len(new)
        stats.tokens = produced
        stats.decode_s = time.perf_counter() - t1
        yield new
        if cut is not None:
            return


def generate_dspark(target, drafter, prompt_ids, max_new, eos, **kwargs) -> List[int]:
    out: List[int] = []
    for chunk in iter_dspark(target, drafter, prompt_ids, max_new, eos, **kwargs):
        out.extend(chunk)
    return out


def generate_ar(target, prompt_ids, max_new, eos, *, prefill_step=None, stats: Optional[dict] = None) -> List[int]:
    """Greedy AR reference with the same prefill structure and the served decode call."""
    cache = target.make_cache()
    t0 = time.perf_counter()
    logits, _ = target.prefill(mx.array([list(prompt_ids)]), cache, (), prefill_step)
    tok = mx.argmax(logits).reshape(1)
    out = [int(tok.item())]
    t1 = time.perf_counter()
    while len(out) < max_new and out[-1] not in eos:
        tok = mx.argmax(target.decode(tok, cache)).reshape(1)
        out.append(int(tok.item()))
    if stats is not None:
        stats.update(tokens=len(out), prefill_s=t1 - t0, decode_s=time.perf_counter() - t1,
                     decode_tok_s=(len(out) - 1) / max(time.perf_counter() - t1, 1e-9))
    return out
