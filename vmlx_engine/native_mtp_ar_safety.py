# SPDX-License-Identifier: Apache-2.0
"""Policy-independent on-the-fly MTP -> AR safety governor (shared by lanes).

Both native-MTP generators (the multimodal lane in ``mllm_batch_generator``
and the text lane in ``patches/mlx_lm_mtp/batch_generator``) run this every
verify cycle, for FIXED and ADAPTIVE depth alike.  The depth-adapt gates are
what a fixed policy disables; this valve is not a depth policy, it is the
guarantee that a request never keeps speculating while losing to plain AR.

Mechanism (wall-clock + counters only, no device sync added anywhere):
  * one ring sample per cycle: (cycles, emitted, perf_counter)
  * skip while cycles < warmup (x restore_scale) or the ring is not full
  * per-cycle wall = dt; per-cycle ms/tok = dt / d_emitted
  * anchor = median per-cycle wall + context length at the FIRST full window
  * scale = clamp(median_now / anchor, 1, context_now / context_anchor)
    (a forward cannot grow faster than linearly in context, so an abrupt
    MTP-only slowdown cannot hide behind the scale; the seed is a floor)
  * trip iff window-mean ms/tok AND per-cycle MEDIAN ms/tok exceed
    seed_ar_ms x scale x margin (a single stalled cycle cannot trip an
    irreversible demotion)

Known limitation: the first request after a model load measures an inflated
seed (compile / cold pages), so the valve is lax for that one request.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

DEFAULT_WARMUP_CYCLES = 12
DEFAULT_WINDOW_CYCLES = 16
DEFAULT_MARGIN = 1.25


def _env_raw(*names: str) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return None


def env_flag(default: bool, *names: str) -> bool:
    raw = _env_raw(*names)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def env_int(default: int, *names: str, minimum: int = 1) -> int:
    raw = _env_raw(*names)
    try:
        value = int(raw) if raw is not None else int(default)
    except (TypeError, ValueError):
        value = int(default)
    return max(minimum, value)


def env_float(default: float, *names: str) -> float:
    raw = _env_raw(*names)
    try:
        return float(raw) if raw is not None else float(default)
    except (TypeError, ValueError):
        return float(default)


def ar_safety_enabled() -> bool:
    return env_flag(True, "VMLINUX_NATIVE_MTP_AR_SAFETY", "VMLX_NATIVE_MTP_AR_SAFETY")


def ar_safety_warmup_cycles() -> int:
    return env_int(
        DEFAULT_WARMUP_CYCLES,
        "VMLINUX_NATIVE_MTP_ADAPTIVE_WARMUP_CYCLES",
        "VMLX_NATIVE_MTP_ADAPTIVE_WARMUP_CYCLES",
        minimum=1,
    )


def ar_safety_window_cycles() -> int:
    return env_int(
        DEFAULT_WINDOW_CYCLES,
        "VMLINUX_NATIVE_MTP_AR_SAFETY_WINDOW",
        "VMLX_NATIVE_MTP_AR_SAFETY_WINDOW",
        minimum=4,
    )


def ar_safety_margin() -> float:
    return env_float(
        DEFAULT_MARGIN,
        "VMLINUX_NATIVE_MTP_RUNTIME_COST_MARGIN",
        "VMLX_NATIVE_MTP_RUNTIME_COST_MARGIN",
    )


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    return ordered[len(ordered) // 2]


def windowed_ar_verdict(
    *,
    ar_step_ms: float,
    anchor_cycle_ms: float,
    cur_cycle_ms: float,
    context_ratio: float,
    delta_emitted: int,
    delta_wall_ms: float,
    margin: float,
    per_cycle_ms_per_tok: Optional[Sequence[float]] = None,
) -> Optional[Tuple[float, float]]:
    """Pure windowed MTP-vs-AR decision. Returns (mtp_ms_per_tok, ar_baseline)
    when MTP is slower than the context-scaled AR baseline by the margin,
    else None.  Unit-testable without a model."""
    if ar_step_ms <= 0.0:
        return None
    if delta_emitted <= 0 or delta_wall_ms <= 0.0:
        return None
    mtp_ms_per_tok = delta_wall_ms / delta_emitted
    scale = 1.0
    if anchor_cycle_ms > 0.0 and cur_cycle_ms > 0.0:
        scale = cur_cycle_ms / anchor_cycle_ms
        scale = min(scale, max(1.0, float(context_ratio)))
        scale = max(1.0, scale)
    ar_baseline = ar_step_ms * scale
    threshold = ar_baseline * margin
    if mtp_ms_per_tok <= threshold:
        return None
    if per_cycle_ms_per_tok:
        if median(per_cycle_ms_per_tok) <= threshold:
            return None
    return mtp_ms_per_tok, ar_baseline


@dataclass
class ArSafetyState:
    """Per-request governor state, embedded in each lane's MTP state."""

    anchor_cycle_ms: float = 0.0
    anchor_context_tokens: int = 0
    prompt_tokens: int = 0
    ring: List[Tuple[int, int, float]] = field(default_factory=list)


@dataclass(frozen=True)
class ArSafetyTrip:
    cycles: int
    mtp_ms_per_tok: float
    ar_baseline: float
    seed_ar_ms: float
    margin: float
    window: int
    cycle_median_ms_per_tok: float
    cycle_max_ms_per_tok: float
    anchor_cycle_ms: float
    cur_cycle_ms: float
    anchor_context_tokens: int
    context_now: int

    def reason(self, prior_depth: int) -> str:
        return (
            f"windowed_ar_safety d{prior_depth} "
            f"mtp_ms_per_tok={self.mtp_ms_per_tok:.1f}"
            f">ar_baseline={self.ar_baseline:.1f}(seed={self.seed_ar_ms:.1f})"
            f"x{self.margin:.2f}"
        )

    def log_text(self, prior_depth: int) -> str:
        return (
            f"windowed AR safety D{prior_depth} -> AR at cycle={self.cycles} "
            f"({self.mtp_ms_per_tok:.1f}ms/tok vs ctx-scaled AR "
            f"{self.ar_baseline:.1f}ms, seed {self.seed_ar_ms:.1f}ms, "
            f"window={self.window}, cycle median "
            f"{self.cycle_median_ms_per_tok:.1f} max "
            f"{self.cycle_max_ms_per_tok:.1f} ms/tok, cycle wall anchor "
            f"{self.anchor_cycle_ms:.1f} -> {self.cur_cycle_ms:.1f} ms, "
            f"context {self.anchor_context_tokens} -> {self.context_now})"
        )


def ar_safety_step(
    st: ArSafetyState,
    *,
    cycles: int,
    emitted: int,
    now: float,
    seed_ar_ms: float,
    restored_prefix: bool = False,
) -> Optional[ArSafetyTrip]:
    """Record this cycle's sample and decide.  Call once per verify cycle,
    AFTER the cycle's device round-trip, BEFORE any depth adaptation.  The
    caller owns the demotion (depth=1 + ar_fallback_pending) and logging."""
    if seed_ar_ms <= 0.0:
        return None
    if not ar_safety_enabled():
        return None
    restore_scale = 4 if restored_prefix else 1
    warmup = restore_scale * ar_safety_warmup_cycles()
    if cycles < warmup:
        return None
    window = restore_scale * ar_safety_window_cycles()
    ring = st.ring
    ring.append((int(cycles), int(emitted), float(now)))
    if len(ring) > window + 1:
        del ring[0 : len(ring) - (window + 1)]
    if len(ring) <= window:
        return None  # window not full yet — never judge a cold window

    c0, e0, t0 = ring[0]
    per_cycle_ms: List[float] = []
    per_cycle_ms_per_tok: List[float] = []
    for prev, cur in zip(ring, ring[1:]):
        d_ms = (cur[2] - prev[2]) * 1000.0
        d_e = cur[1] - prev[1]
        per_cycle_ms.append(d_ms)
        if d_e > 0:
            per_cycle_ms_per_tok.append(d_ms / d_e)
    cur_cycle_ms = median(per_cycle_ms)
    context_now = int(st.prompt_tokens) + int(emitted)
    if st.anchor_cycle_ms <= 0.0:
        if cur_cycle_ms <= 0.0:
            return None
        st.anchor_cycle_ms = cur_cycle_ms
        st.anchor_context_tokens = max(1, context_now)
    context_ratio = context_now / max(1, st.anchor_context_tokens)
    margin = ar_safety_margin()
    verdict = windowed_ar_verdict(
        ar_step_ms=float(seed_ar_ms),
        anchor_cycle_ms=float(st.anchor_cycle_ms),
        cur_cycle_ms=cur_cycle_ms,
        context_ratio=context_ratio,
        delta_emitted=int(emitted) - e0,
        delta_wall_ms=(float(now) - t0) * 1000.0,
        margin=margin,
        per_cycle_ms_per_tok=per_cycle_ms_per_tok,
    )
    if verdict is None:
        return None
    mtp_ms_per_tok, ar_baseline = verdict
    return ArSafetyTrip(
        cycles=int(cycles),
        mtp_ms_per_tok=mtp_ms_per_tok,
        ar_baseline=ar_baseline,
        seed_ar_ms=float(seed_ar_ms),
        margin=margin,
        window=window,
        cycle_median_ms_per_tok=median(per_cycle_ms_per_tok),
        cycle_max_ms_per_tok=(
            max(per_cycle_ms_per_tok) if per_cycle_ms_per_tok else 0.0
        ),
        anchor_cycle_ms=float(st.anchor_cycle_ms),
        cur_cycle_ms=cur_cycle_ms,
        anchor_context_tokens=int(st.anchor_context_tokens),
        context_now=context_now,
    )
