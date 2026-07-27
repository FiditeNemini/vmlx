# SPDX-License-Identifier: Apache-2.0
"""Behavior-neutral native-MTP head-cache telemetry helpers.

The MTP head owns a cache separate from the backbone prompt cache.  Runtime
health should be able to show whether that cache is growing, retained, or
recreated without evaluating MLX arrays merely to answer a status request.

Only plain Python integer fields already stored on cache objects are read.
Tensor-valued offsets, properties, and callable ``size()`` methods are
deliberately ignored because inspecting them could synchronize or materialize
device work.  The scan is capped and emits aggregate scalars so an unexpected
cache container cannot make health payloads grow without bound.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any, Dict, Optional


MAX_MTP_CACHE_LAYERS_SCANNED = 32
MAX_MTP_CACHE_METRIC_VALUE = (1 << 63) - 1
_HEAD_CACHE_FIELDS = (
    "layers",
    "layers_scanned",
    "introspectable_layers",
    "truncated",
    "offset",
    "offset_min",
    "offset_max",
    "length",
    "length_min",
    "length_max",
)


def _plain_nonnegative_int(value: Any) -> Optional[int]:
    """Return a bounded host integer without coercing tensor-like values."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    return min(max(0, int(value)), MAX_MTP_CACHE_METRIC_VALUE)


def _plain_cache_attrs(layer: Any) -> Dict[str, Any]:
    """Read instance storage directly, avoiding descriptors/properties."""

    try:
        attrs = vars(layer)
    except TypeError:
        return {}
    return attrs if isinstance(attrs, dict) else {}


def native_mtp_cache_snapshot(cache: Any) -> Dict[str, Any]:
    """Return a bounded, synchronization-free MTP-head cache snapshot.

    ``offset`` is the cumulative logical position when all safely inspected
    layers agree. ``length`` is the retained logical length (bounded by
    ``max_size`` for rotating caches) when all inspected layers agree.
    Min/max fields remain available when layers legitimately differ.
    """

    if not isinstance(cache, (list, tuple)):
        return {
            "layers": 0,
            "layers_scanned": 0,
            "introspectable_layers": 0,
            "truncated": False,
            "offset": None,
            "offset_min": None,
            "offset_max": None,
            "length": None,
            "length_min": None,
            "length_max": None,
        }

    layer_count = len(cache)
    scan_count = min(layer_count, MAX_MTP_CACHE_LAYERS_SCANNED)
    offsets = []
    lengths = []

    for layer in cache[:scan_count]:
        attrs = _plain_cache_attrs(layer)
        offset = _plain_nonnegative_int(attrs.get("offset"))
        if offset is None:
            # Batch caches often keep tensor-valued ``offset`` but a plain
            # host-side insertion index.  It is a safe retained-length signal,
            # not a substitute cumulative offset.
            idx = _plain_nonnegative_int(attrs.get("_idx"))
            if idx is not None:
                lengths.append(idx)
            continue

        offsets.append(offset)
        max_size = _plain_nonnegative_int(attrs.get("max_size"))
        lengths.append(min(offset, max_size) if max_size else offset)

    offset_min = min(offsets) if offsets else None
    offset_max = max(offsets) if offsets else None
    length_min = min(lengths) if lengths else None
    length_max = max(lengths) if lengths else None
    return {
        "layers": min(layer_count, MAX_MTP_CACHE_METRIC_VALUE),
        "layers_scanned": scan_count,
        "introspectable_layers": len(lengths),
        "truncated": layer_count > scan_count,
        "offset": offset_min if offset_min == offset_max else None,
        "offset_min": offset_min,
        "offset_max": offset_max,
        "length": length_min if length_min == length_max else None,
        "length_min": length_min,
        "length_max": length_max,
    }


def native_mtp_cache_lifecycle_snapshot(
    *,
    head_cache: Optional[Dict[str, Any]],
    recreated_on_rejects: int,
    retained_on_rejects: int,
) -> Dict[str, Any]:
    """Normalize lifecycle counters into a bounded stable health payload."""

    normalized_head = native_mtp_cache_snapshot(None)
    if isinstance(head_cache, dict):
        for key in _HEAD_CACHE_FIELDS:
            value = head_cache.get(key)
            if key == "truncated":
                if isinstance(value, bool):
                    normalized_head[key] = value
                continue
            if value is None and key in {
                "offset",
                "offset_min",
                "offset_max",
                "length",
                "length_min",
                "length_max",
            }:
                normalized_head[key] = None
                continue
            bounded = _plain_nonnegative_int(value)
            if bounded is not None:
                normalized_head[key] = bounded

    normalized_head["layers_scanned"] = min(
        normalized_head["layers_scanned"],
        MAX_MTP_CACHE_LAYERS_SCANNED,
        normalized_head["layers"],
    )
    normalized_head["introspectable_layers"] = min(
        normalized_head["introspectable_layers"],
        normalized_head["layers_scanned"],
    )
    normalized_head["truncated"] = bool(
        normalized_head["truncated"]
        or normalized_head["layers"] > normalized_head["layers_scanned"]
    )

    return {
        "head_cache": normalized_head,
        "recreated_on_rejects": (
            _plain_nonnegative_int(recreated_on_rejects) or 0
        ),
        "retained_on_rejects": _plain_nonnegative_int(retained_on_rejects) or 0,
    }
