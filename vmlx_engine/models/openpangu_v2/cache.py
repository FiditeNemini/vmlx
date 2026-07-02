# SPDX-License-Identifier: Apache-2.0
"""Per-layer cache for openPangu-2.0-Flash (openpangu_v2).

Each decoder layer is EITHER a DSA full-attention layer (unbounded KVCache +
lightning-indexer key cache) OR a SWA sliding-window layer (RotatingKVCache,
per-layer window 512 or 2048). BOTH additionally carry the 3 causal-conv
states (qa / compresskv / o) — path-dependent, Mamba-style: the conv state
MUST round-trip with the KV or a KV-only prefix-cache reuse is a silent false
hit (garbled turn-2). Mirrors the proven vmlx-swift OpenPanguV2Cache contract.

is_trimmable() is False on purpose: trimming KV without re-deriving the conv
states corrupts the path-dependent state, so speculative/trim-based reuse must
re-prefill instead. detect_cache_type() resolves this class to UNKNOWN, which
makes the prefix/paged stores skip it safely until a typed lane exists.

Created by Jinho Jang (eric@jangq.ai).
"""

from __future__ import annotations

from typing import Any, List, Optional

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache

# Conv-state slot order (matches the Swift OpenPanguConvKind enum).
CONV_QA = 0
CONV_COMPRESS_KV = 1
CONV_O = 2

_SENTINEL_SHAPE = (1, 0, 1)


def _serializable(a: Optional[mx.array]) -> mx.array:
    return a if a is not None else mx.zeros(_SENTINEL_SHAPE)


def _nullable(a: mx.array) -> Optional[mx.array]:
    if a.ndim == 3 and a.shape[1] == 0:
        return None
    return a


class OpenPanguV2LayerCache:
    """One layer's cache: inner KV (+ indexer KV on DSA) + 3 conv states."""

    def __init__(self, window: int = 0, is_dsa: bool = False):
        self.is_dsa = is_dsa
        self.window = int(window)
        if is_dsa or not window:
            self.kv: Any = KVCache()
        else:
            self.kv = RotatingKVCache(max_size=int(window), keep=0)
        # Indexer key cache (DSA layers only): append-only positional keys.
        self.indexer_kv: Optional[KVCache] = KVCache() if is_dsa else None
        # (B, kernel-1, channels) trailing inputs for qa / compresskv / o convs.
        self.conv_states: List[Optional[mx.array]] = [None, None, None]

    # ---- KV-cache protocol surface used by the engine -------------------
    @property
    def offset(self) -> int:
        return self.kv.offset

    def is_trimmable(self) -> bool:
        # Path-dependent conv state cannot survive a trim.
        return False

    def trim(self, n: int) -> int:
        return 0

    @property
    def state(self):
        kv_state = list(self.kv.state)
        idx_state = list(self.indexer_kv.state) if self.indexer_kv is not None else []
        return (
            kv_state
            + idx_state
            + [_serializable(s) for s in self.conv_states]
        )

    @state.setter
    def state(self, value):
        value = list(value)
        conv = value[-3:]
        rest = value[: len(value) - 3]
        if self.indexer_kv is not None:
            # Inner KVCache state is (keys, values) pairs of fixed arity 2.
            self.kv.state = rest[:2]
            self.indexer_kv.state = rest[2:]
        else:
            self.kv.state = rest
        self.conv_states = [_nullable(a) for a in conv]

    @property
    def meta_state(self):
        inner = getattr(self.kv, "meta_state", ())
        return tuple(inner) + (
            "openpangu_v2_cache_v1",
            "dsa" if self.is_dsa else "swa",
            str(self.window),
        )

    @meta_state.setter
    def meta_state(self, value):
        value = tuple(value)
        if len(value) >= 3 and value[-3] == "openpangu_v2_cache_v1":
            value = value[:-3]
        if hasattr(type(self.kv), "meta_state"):
            try:
                self.kv.meta_state = value
            except Exception:
                pass
