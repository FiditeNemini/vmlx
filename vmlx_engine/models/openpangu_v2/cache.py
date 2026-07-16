# SPDX-License-Identifier: Apache-2.0
"""Per-layer cache for openPangu-2.0-Flash (openpangu_v2).

Each decoder layer is EITHER a DSA full-attention layer (unbounded KVCache +
lightning-indexer key cache) OR a SWA sliding-window layer (RotatingKVCache,
per-layer window 512 or 2048). BOTH additionally carry the 3 causal-conv
states (qa / compresskv / o) — path-dependent, Mamba-style: the conv state
MUST round-trip with the KV or a KV-only prefix-cache reuse is a silent false
hit (garbled turn-2). Mirrors the proven vmlx-swift OpenPanguV2Cache contract.

is_trimmable() is False on purpose: trimming KV without re-deriving the conv
states corrupts the path-dependent state.  Exact prompt-boundary reuse is safe
only through the typed clone/from_state lane below, which round-trips the inner
KV, DSA indexer, rotating-SWA metadata, and all three convolution states as one
unit.  Generic paged/block truncation remains unsupported.

Created by Jinho Jang (eric@jangq.ai).
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

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

    def size(self) -> int:
        return int(self.offset)

    def empty(self) -> bool:
        return int(self.offset) == 0

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
        if len(value) < 3:
            raise ValueError(
                "openPangu cache state is missing the three causal-conv slots"
            )
        conv = value[-3:]
        rest = value[: len(value) - 3]
        if self.indexer_kv is not None:
            # Inner KVCache state is (keys, values) pairs of fixed arity 2.
            self.kv.state = rest[:2]
            self.indexer_kv.state = rest[2:]
        else:
            self.kv.state = rest
        self.conv_states = [_nullable(a) for a in conv]

    # ---- single-sequence BatchGenerator protocol -------------------------
    # mlx_lm's BatchGenerator calls filter/extract/prepare/finalize on every
    # layer cache during split/finish handling EVEN for a single sequence.
    # Without these, serving openPangu with --continuous-batching aborted
    # every request: AttributeError 'OpenPanguV2LayerCache' object has no
    # attribute 'filter' from generate.py split→filter (sweep 2026-07-05,
    # 66 aborts, empty responses with usage 0/0/0). The conv states are
    # path-dependent and single-sequence native, so only the [0] fast path
    # is supported — multi-sequence batching must queue serially (see the
    # scheduler single-sequence guard for this cache type).
    def filter(self, batch_indices) -> None:
        keep = (
            list(batch_indices)
            if isinstance(batch_indices, (list, tuple))
            else [batch_indices]
        )
        if keep != [0]:
            raise NotImplementedError(
                "OpenPanguV2LayerCache.filter() only supports the "
                "single-sequence fast path. openPangu conv states are "
                "path-dependent; multi-sequence batching must run serially."
            )

    def extract(self, idx: int) -> "OpenPanguV2LayerCache":
        if idx != 0:
            raise IndexError(
                f"OpenPanguV2LayerCache.extract({idx}) is invalid for the "
                "single-sequence fast path"
            )
        return self

    def prepare(self, *args, **kwargs) -> None:
        right_padding = kwargs.get("right_padding")
        if right_padding is not None and any(int(x) != 0 for x in right_padding):
            raise NotImplementedError(
                "OpenPanguV2LayerCache.prepare() cannot right-pad the "
                "single-sequence fast path"
            )

    def finalize(self) -> None:
        return None

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

    @classmethod
    def from_state(cls, state, meta_state) -> "OpenPanguV2LayerCache":
        """Rebuild the exact typed composite from prompt-cache metadata.

        mlx-lm's generic implementation constructs cache classes via ``__new__``.
        That cannot work here because the state setter needs to know whether the
        layer owns a DSA indexer and whether its inner KV is rotating.  The typed
        trailer supplies those constructor invariants before any tensors land.
        """

        meta = tuple(meta_state or ())
        if len(meta) < 3 or meta[-3] != "openpangu_v2_cache_v1":
            raise ValueError("openPangu cache metadata is missing its typed trailer")
        mode = str(meta[-2])
        if mode not in {"dsa", "swa"}:
            raise ValueError(f"invalid openPangu cache mode: {mode}")
        try:
            window = int(meta[-1])
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid openPangu cache window metadata") from exc

        rebuilt = cls(window=window, is_dsa=(mode == "dsa"))
        rebuilt.state = state
        rebuilt.meta_state = meta
        return rebuilt


def clone_openpangu_layer_cache(
    source: OpenPanguV2LayerCache,
    *,
    copy_fn: Callable[[mx.array], mx.array],
) -> OpenPanguV2LayerCache:
    """Return a non-aliasing exact-boundary copy of one composite layer.

    There is intentionally no length argument: shortening a causal-convolution
    state is undefined.  Callers must prove they are cloning the full logical
    boundary before invoking this helper.
    """

    if type(source).__name__ != "OpenPanguV2LayerCache":
        raise TypeError(f"unexpected cache class: {type(source).__name__}")
    copied_state = [copy_fn(value) for value in source.state]
    return OpenPanguV2LayerCache.from_state(copied_state, source.meta_state)
