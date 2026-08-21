"""Logical vs physical extent of a KV cache.

A live `KVCache` allocates its key/value buffers in `step` (256) token chunks,
so `keys.shape[seq_axis] > offset` is the NORMAL steady state, not an anomaly:
the trailing rows are uninitialised/zero and are not tokens. `offset` is the
only authority on how many tokens the cache actually holds.

Two rules follow, and violating either has already cost a shipped release:

1. Never treat a buffer extent as a token count. dots3 adopted restored
   `.keys`/`.values` verbatim, so its sparse-attention indexer counted pad rows
   as real keys and generation ran to `max_tokens` with no visible answer
   (fixed in `models/dots3_note/language.py`).
2. Never let a buffer extent become an `offset`. That laundering is what makes
   rule 1 invisible downstream — and in a truncation path it persists zero rows
   into the prefix cache as though they were tokens.

Use `logical_truncate_target()` anywhere a cache is sliced to a length, so the
clamp includes `offset` and not just the physical shape.
"""

from __future__ import annotations

from typing import Any


def cache_offset(cache_obj: Any) -> int:
    """`cache_obj.offset` as a non-negative int, or 0 if absent/garbage."""
    try:
        offset = int(getattr(cache_obj, "offset", 0) or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, offset)


def logical_truncate_target(cache_obj: Any, target_len: int, physical: int) -> int:
    """Clamp a requested truncation length to what the cache LOGICALLY holds.

    `min(target_len, physical)` is not enough: when `target_len` exceeds the
    cache's `offset`, the extra rows are allocation slack, and slicing them in
    turns zeros into tokens. Including `offset` in the clamp is the difference
    between truncating a cache and corrupting it.

    `offset == 0` is treated as "unknown" rather than "empty" — some cache
    types never populate it, and clamping those to zero would delete a cache
    that is perfectly valid. Advise, never refuse.
    """
    bound = min(int(target_len), int(physical))
    offset = cache_offset(cache_obj)
    if offset > 0:
        bound = min(bound, offset)
    return bound
