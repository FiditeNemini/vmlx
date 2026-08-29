"""Session-scoped learned depth profiles for adaptive native MTP.

The per-request wall-value controller (native_mtp_adaptive) starts every
request at the configured depth and needs tens of verify cycles before its
runtime-cost and acceptance gates can conclude that MTP is slower than AR.
Because ``MLLMNativeMTPState`` is request-local, every request repeated that
whole failed experiment: short and first turns finished before the controller
learned anything, and later turns relearned the same lesson from scratch.

This module remembers what the controller concluded, per workload shape, for
the lifetime of one loaded model:

- The store lives on the generator instance, so a model/bundle/config reload
  or a new engine process starts clean (invalidation for free).
- Profiles are keyed by (sampler class, restored-prefix, context bucket) so
  greedy/sampled, cold/restored, and short/long observations cannot poison
  one another.
- An unseen profile starts at depth 1 — a bounded, near-AR probe — never at
  an optimistic D2/D3 for a whole short request. The rolling controller
  promotes from there only on measured profitable evidence.
- A profile that learned "AR wins" (cost/acceptance fallback) skips MTP
  activation entirely on later requests, with a bounded re-probe every
  ``REPROBE_EVERY_REQUESTS`` requests so a workload shift can re-enable it.
- Nothing here changes sampling, verification, or emitted tokens: it only
  chooses the STARTING draft depth (or AR) that the existing controller and
  safety gates then adjust with their own measurements.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

# After this many consecutive AR-profiled requests, allow one D1 probe again.
REPROBE_EVERY_REQUESTS = 8

# Context buckets are deliberately coarse so evidence accumulates quickly.
_CTX_BUCKETS = ((2048, "short"), (16384, "medium"))

ProfileKey = Tuple[str, bool, str]


def profile_key(
    *,
    temperature: float,
    restored_prefix: bool,
    prompt_tokens: int,
) -> ProfileKey:
    sampler_class = "greedy" if float(temperature or 0.0) <= 0.0 else "sampled"
    bucket = "long"
    for limit, name in _CTX_BUCKETS:
        if int(prompt_tokens or 0) < limit:
            bucket = name
            break
    return (sampler_class, bool(restored_prefix), bucket)


@dataclass
class NativeMTPSessionProfile:
    """Learned starting depth for one workload shape. 0 means AR."""

    learned_depth: Optional[int] = None
    requests_observed: int = 0
    requests_since_ar_probe: int = 0
    last_finish_reason: str = ""
    last_fallback_reason: Optional[str] = None
    last_values_tok_s: Dict[str, Any] = field(default_factory=dict)


class NativeMTPProfileStore:
    """Per-generator (per loaded model, per process) profile registry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._profiles: Dict[ProfileKey, NativeMTPSessionProfile] = {}

    def start_depth(
        self,
        key: ProfileKey,
        *,
        configured_depth: int,
    ) -> Tuple[int, str]:
        """Return (starting depth, seed source). Depth 0 means stay AR.

        The configured depth stays a ceiling for the seed only; the
        controller's own depth_ceiling still governs later promotion.
        """

        ceiling = max(1, min(3, int(configured_depth or 1)))
        with self._lock:
            profile = self._profiles.get(key)
            if profile is None or profile.learned_depth is None:
                return min(1, ceiling), "unseen_probe_d1"
            if profile.learned_depth <= 0:
                profile.requests_since_ar_probe += 1
                if profile.requests_since_ar_probe >= REPROBE_EVERY_REQUESTS:
                    profile.requests_since_ar_probe = 0
                    return min(1, ceiling), "ar_reprobe_d1"
                return 0, "profile_ar_skip"
            depth = max(1, min(profile.learned_depth, ceiling))
            return depth, f"profile_learned_d{depth}"

    def observe(
        self,
        key: ProfileKey,
        *,
        final_depth: int,
        fallback_to_ar: bool,
        fallback_reason: Optional[str],
        finish_reason: str,
        values_tok_s: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record what one finished request's controller concluded."""

        if finish_reason == "error":
            # A crashed request proves nothing about depth value.
            return
        with self._lock:
            profile = self._profiles.setdefault(key, NativeMTPSessionProfile())
            profile.requests_observed += 1
            profile.last_finish_reason = str(finish_reason)
            profile.last_fallback_reason = fallback_reason
            if values_tok_s:
                profile.last_values_tok_s = dict(values_tok_s)
            if fallback_to_ar:
                profile.learned_depth = 0
                profile.requests_since_ar_probe = 0
            else:
                profile.learned_depth = max(1, min(3, int(final_depth or 1)))

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "|".join(
                    (key[0], "restored" if key[1] else "cold", key[2])
                ): {
                    "learned_depth": profile.learned_depth,
                    "requests_observed": profile.requests_observed,
                    "last_finish_reason": profile.last_finish_reason,
                    "last_fallback_reason": profile.last_fallback_reason,
                    "last_values_tok_s": dict(profile.last_values_tok_s),
                }
                for key, profile in self._profiles.items()
            }
