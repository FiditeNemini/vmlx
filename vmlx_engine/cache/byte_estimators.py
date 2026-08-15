# SPDX-License-Identifier: Apache-2.0
"""Shared payload byte-walking for cache tiers.

The paged tier's residency accounting (``estimate_block_nbytes``) and the
block-disk tier's admission accounting (``_estimate_cache_payload_bytes``)
each grew their own recursive nbytes walker. Divergent walkers are how the
two tiers come to DISAGREE about the same payload's size — an admission
reservation smaller than residency reality is the byte-budget-hole class
(a dropped write leaves a hole and the restorable prefix ends there).

This module holds the ONE walker. Callers keep their historical counting
semantics via flags so consolidation changes no behavior:

- paged residency: every ``.nbytes`` leaf counts (quantized composites nest
  arrays under dict leaves), containers recurse, no depth cap, no raw-bytes
  or ``__dict__`` traversal.
- disk admission: array leaves need a ``shape`` (a stray int ``nbytes``
  attribute must not count), raw ``bytes``/``bytearray`` lengths count,
  object ``__dict__`` trees are walked, depth capped, and the tensor count
  feeds the safetensors header reservation.
"""

from __future__ import annotations

from typing import Any


def walk_payload_bytes(
    value: Any,
    *,
    require_shape_for_arrays: bool,
    dedupe_arrays: bool,
    count_raw_bytes: bool,
    walk_object_dicts: bool,
    max_depth: int | None,
) -> tuple[int, int]:
    """Return ``(total_bytes, tensor_count)`` for a nested cache payload.

    Never raises; malformed leaves contribute nothing. Containers are
    de-duplicated by identity so shared or self-referential subtrees count
    once. Array leaves de-duplicate only when ``dedupe_arrays`` — paged
    residency historically counts an aliased array at every occurrence and
    that behavior is preserved. ``tensor_count`` counts every qualifying
    array leaf (including zero-byte ones — a safetensors header entry
    exists regardless of payload size).
    """

    seen: set[int] = set()
    total = 0
    tensors = 0

    def walk(node: Any, depth: int) -> None:
        nonlocal total, tensors
        if node is None:
            return
        if max_depth is not None and depth > max_depth:
            return
        if isinstance(node, (str, int, float, bool)):
            return
        if isinstance(node, (bytes, bytearray)):
            if count_raw_bytes:
                total += len(node)
            return
        identity = id(node)
        nbytes = getattr(node, "nbytes", None)
        if nbytes is not None and (
            not require_shape_for_arrays or hasattr(node, "shape")
        ):
            if dedupe_arrays:
                if identity in seen:
                    return
                seen.add(identity)
            tensors += 1
            try:
                counted = int(nbytes)
            except (TypeError, ValueError):
                return
            if counted > 0:
                total += counted
            return
        if isinstance(node, dict):
            if identity in seen:
                return
            seen.add(identity)
            for item in node.values():
                walk(item, depth + 1)
            return
        if isinstance(node, (list, tuple)):
            if identity in seen:
                return
            seen.add(identity)
            for item in node:
                walk(item, depth + 1)
            return
        if walk_object_dicts:
            attributes = getattr(node, "__dict__", None)
            if isinstance(attributes, dict):
                if identity in seen:
                    return
                seen.add(identity)
                for item in attributes.values():
                    walk(item, depth + 1)

    try:
        walk(value, 0)
    except Exception:
        return 0, 0
    return total, tensors
