# SPDX-License-Identifier: Apache-2.0
"""
Cache type detection and classification for vmlx-engine.

Provides a unified system for identifying and working with all mlx-lm
cache types: KVCache, RotatingKVCache, QuantizedKVCache, MambaCache,
ArraysCache, and CacheList.
"""

import logging
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CacheType(Enum):
    """All cache types supported by mlx-lm."""

    KV_CACHE = "kv_cache"
    ROTATING_KV_CACHE = "rotating_kv_cache"
    QUANTIZED_KV_CACHE = "quantized_kv_cache"
    MAMBA_CACHE = "mamba_cache"
    ARRAYS_CACHE = "arrays_cache"
    CACHE_LIST = "cache_list"
    UNKNOWN = "unknown"


# Class name -> CacheType mapping for fast detection
_CLASS_NAME_MAP = {
    "KVCache": CacheType.KV_CACHE,
    "BatchKVCache": CacheType.KV_CACHE,
    "RotatingKVCache": CacheType.ROTATING_KV_CACHE,
    "BatchRotatingKVCache": CacheType.ROTATING_KV_CACHE,
    "QuantizedKVCache": CacheType.QUANTIZED_KV_CACHE,
    "MambaCache": CacheType.MAMBA_CACHE,
    "BatchMambaCache": CacheType.MAMBA_CACHE,
    "ArraysCache": CacheType.ARRAYS_CACHE,
    "CacheList": CacheType.CACHE_LIST,
    "TurboQuantKVCache": CacheType.KV_CACHE,  # TQ behaves like KVCache (positional, sliceable)
}

# Positional caches store per-token KV data that can be sliced by position
_POSITIONAL_TYPES = {
    CacheType.KV_CACHE,
    CacheType.ROTATING_KV_CACHE,
    CacheType.QUANTIZED_KV_CACHE,
}

# Cumulative caches store rolling state that represents all processed tokens
_CUMULATIVE_TYPES = {
    CacheType.MAMBA_CACHE,
    CacheType.ARRAYS_CACHE,
}


# Cache class names whose state accumulates over every token consumed rather
# than being indexed by position. These can never be sliced to a shorter token
# boundary, so they are stored in a typed companion or re-derived instead.
CUMULATIVE_CACHE_CLASS_NAMES = frozenset(
    {"MambaCache", "BatchMambaCache", "ArraysCache"}
)

# Cache class names that hold position-indexed attention KV.
ATTENTION_CACHE_CLASS_NAMES = frozenset(
    {"KVCache", "RotatingKVCache", "QuantizedKVCache", "TurboQuantKVCache"}
)

DSV4_CACHE_CLASS_NAMES = frozenset(
    {"DeepseekV4Cache", "PoolQuantizedV4Cache"}
)


def expand_cache_class_names(cache: Any) -> set:
    """Return the *effective* cache class names for a model cache, resolving
    ``CacheList`` wrappers to the classes they actually contain.

    Architecture detection that reads ``type(layer).__name__`` directly sees
    only ``CacheList`` and has to guess what is inside.  Both guesses are wrong
    for some family:

      - ``CacheList(KVCache(), KVCache())`` -- deepseek_v32, longcat_flash,
        longcat_flash_ngram.  Purely attention; treating it as hybrid would
        force an SSM companion that has nothing to hold.
      - ``CacheList(ArraysCache(...), KVCache())`` -- falcon_h1, baichuan_m1.
        Genuinely hybrid; treating it as plain KV skips the SSM companion and
        the layer never gets a cache lane at all.

    falcon_h1 was found running with the second case misclassified: every layer
    is a ``CacheList``, so the wrapper name was discarded, the model was called
    non-hybrid, and it served whole sessions with zero prefix reuse.

    Nested wrappers resolve recursively.  A wrapper with no readable
    sub-caches keeps its own name rather than vanishing, so an unreadable
    layout can never silently look like "no non-KV types present".
    """
    names = set()

    def _visit(layer: Any, depth: int = 0) -> None:
        cls_name = type(layer).__name__
        subs = getattr(layer, "caches", None)
        if (
            cls_name == "CacheList"
            and isinstance(subs, (list, tuple))
            and subs
            and depth < 8
        ):
            for sub in subs:
                _visit(sub, depth + 1)
            return
        names.add(cls_name)

    for layer in cache or []:
        _visit(layer)
    return names


def detect_dsv4_cache_contract(model: Any) -> Optional[bool]:
    """Observe whether ``model.make_cache()`` creates a DSV4 composite cache.

    ``False`` is returned only after the instantiated cache graph was inspected
    successfully. ``None`` means the runtime contract could not be observed and
    callers must retain conservative payload validation. This distinction is
    important for SSD-only prefix reuse: probing DSV4 terminal records on a
    runtime-proven ordinary KV/hybrid model reads every block once before the
    actual reconstruction reads it again.

    CacheList-style wrappers and subclasses are inspected recursively. Family
    names and bundle/registry metadata deliberately do not participate.
    """
    make_cache = getattr(model, "make_cache", None)
    if not callable(make_cache):
        return None
    try:
        cache = list(make_cache() or [])
    except Exception:
        return None

    def _is_dsv4_cache(slot: Any) -> bool:
        try:
            if any(
                cls.__name__ in DSV4_CACHE_CLASS_NAMES
                for cls in type(slot).__mro__
            ):
                return True
        except Exception:
            return False
        nested = getattr(slot, "caches", None)
        if isinstance(nested, (list, tuple)):
            return any(_is_dsv4_cache(child) for child in nested)
        return False

    return any(_is_dsv4_cache(slot) for slot in cache)


def describe_runtime_cache_layout(
    cache: Any,
    *,
    model: Any = None,
) -> Dict[str, Any]:
    """Describe the cache objects actually returned by ``model.make_cache``.

    Family names and registry metadata are useful launch hints, but neither is
    proof of the cache topology a loaded model instantiated.  Keep this helper
    deliberately observational: it records every top-level slot and nested
    ``CacheList`` leaf, then classifies only cache shapes whose semantics are
    known.  DSV4's explicit composite classes own both a local rotating SWA
    ring and cumulative compressor/indexer pools. Other unknown native classes
    stay visible as unknown instead of being guessed into a generic KV or SSM
    lane.

    The returned values contain class names and integer indices only.  No live
    tensors or cache objects are retained by telemetry.
    """

    slots = list(cache or [])
    slot_types = []
    slot_class_counts: Dict[str, int] = {}
    effective_class_counts: Dict[str, int] = {}
    attention_positions = []
    cumulative_positions = []
    parallel_positions = []
    rotating_positions = []
    quantized_positions = []
    unknown_positions = []

    def _leaf_names(slot: Any, depth: int = 0) -> list[str]:
        class_name = type(slot).__name__
        nested = getattr(slot, "caches", None)
        if (
            class_name == "CacheList"
            and isinstance(nested, (list, tuple))
            and nested
            and depth < 8
        ):
            names = []
            for child in nested:
                names.extend(_leaf_names(child, depth + 1))
            return names
        return [class_name]

    def _slot_label(slot: Any, depth: int = 0) -> str:
        class_name = type(slot).__name__
        nested = getattr(slot, "caches", None)
        if (
            class_name == "CacheList"
            and isinstance(nested, (list, tuple))
            and nested
            and depth < 8
        ):
            return f"CacheList({','.join(_slot_label(child, depth + 1) for child in nested)})"
        return class_name

    def _is_attention(class_name: str) -> bool:
        # mlx-lm has Batch* variants and model implementations can provide a
        # cache subclass with a family-specific prefix.  A KVCache suffix is a
        # semantic declaration; native composite caches without that suffix
        # remain unknown and are handled by their architecture-owned branch.
        return (
            class_name in ATTENTION_CACHE_CLASS_NAMES
            or class_name.endswith("KVCache")
            or class_name == "MiniMaxM3SparseCache"
        )

    for idx, slot in enumerate(slots):
        top_level_class = type(slot).__name__
        slot_class_counts[top_level_class] = (
            slot_class_counts.get(top_level_class, 0) + 1
        )
        slot_types.append(_slot_label(slot))

        leaf_names = _leaf_names(slot)
        for class_name in leaf_names:
            effective_class_counts[class_name] = (
                effective_class_counts.get(class_name, 0) + 1
            )

        has_dsv4_composite = any(
            name in DSV4_CACHE_CLASS_NAMES for name in leaf_names
        )
        has_attention = has_dsv4_composite or any(
            _is_attention(name) for name in leaf_names
        )
        has_cumulative = any(
            name in CUMULATIVE_CACHE_CLASS_NAMES for name in leaf_names
        ) or has_dsv4_composite
        has_rotating = has_dsv4_composite or any(
            "RotatingKVCache" in name for name in leaf_names
        )
        has_quantized = any(
            "QuantizedKVCache" in name or name == "TurboQuantKVCache"
            for name in leaf_names
        )

        if has_attention:
            attention_positions.append(idx)
        if has_cumulative:
            cumulative_positions.append(idx)
        if has_attention and has_cumulative:
            parallel_positions.append(idx)
        if has_rotating:
            rotating_positions.append(idx)
        if has_quantized:
            quantized_positions.append(idx)
        if not has_attention and not has_cumulative:
            unknown_positions.append(idx)

    result = {
        "layer_count": len(slots),
        "slot_types": slot_types,
        "slot_class_counts": slot_class_counts,
        "effective_class_counts": effective_class_counts,
        "attention_layer_indices": attention_positions,
        "cumulative_layer_indices": cumulative_positions,
        "parallel_layer_indices": parallel_positions,
        "rotating_layer_indices": rotating_positions,
        "quantized_layer_indices": quantized_positions,
        "unknown_layer_indices": unknown_positions,
    }

    # Cache classes prove storage semantics, but generic containers such as
    # ArraysCache do not identify the architecture component that owns them.
    # Observe the loaded model's matching layer objects too. For Qwen3.5/3.8,
    # this distinguishes GatedDeltaNet recurrent state from Mamba/SSM even
    # though both use cumulative cache containers. This is telemetry only; it
    # never drives routing.
    try:
        layers = list(getattr(model, "layers", []) or []) if model is not None else []
    except Exception:
        layers = []
    if len(layers) == len(slots) and layers:
        owner_types: list[str] = []
        owner_counts: Dict[str, int] = {}
        owner_attrs = (
            "linear_attn",
            "self_attn",
            "attention",
            "attn",
            "mixer",
            "mamba",
            "ssm",
            "conv",
        )
        for layer in layers:
            owner = None
            for attr in owner_attrs:
                candidate = getattr(layer, attr, None)
                if candidate is not None:
                    owner = type(candidate).__name__
                    break
            if owner is None:
                owner = type(layer).__name__
            owner_types.append(owner)
            owner_counts[owner] = owner_counts.get(owner, 0) + 1
        result["owner_component_types"] = owner_types
        result["owner_component_class_counts"] = owner_counts
        result["owner_component_source"] = "instantiated_model_layers"

    return result


def detect_cache_type(cache_obj: Any) -> CacheType:
    """
    Detect the cache type of a cache object by class name and structure.

    Args:
        cache_obj: A cache object from mlx-lm

    Returns:
        CacheType enum value
    """
    if cache_obj is None:
        return CacheType.UNKNOWN

    class_name = type(cache_obj).__name__

    # Fast path: exact class name match
    if class_name in _CLASS_NAME_MAP:
        return _CLASS_NAME_MAP[class_name]

    # Check inheritance chain
    for cls in type(cache_obj).__mro__:
        if cls.__name__ in _CLASS_NAME_MAP:
            return _CLASS_NAME_MAP[cls.__name__]

    # Structure-based detection (fallback)
    if hasattr(cache_obj, "caches") and hasattr(cache_obj, "__iter__"):
        return CacheType.CACHE_LIST

    if hasattr(cache_obj, "max_size") and hasattr(cache_obj, "keys"):
        return CacheType.ROTATING_KV_CACHE

    if hasattr(cache_obj, "keys_quantized") or hasattr(cache_obj, "quantized"):
        return CacheType.QUANTIZED_KV_CACHE

    if hasattr(cache_obj, "keys") and hasattr(cache_obj, "values"):
        return CacheType.KV_CACHE

    if hasattr(cache_obj, "cache") and isinstance(
        getattr(cache_obj, "cache", None), list
    ):
        # MambaCache/ArraysCache have a .cache list attribute
        return CacheType.MAMBA_CACHE

    return CacheType.UNKNOWN


def detect_cache_type_from_state(
    state: Any, class_name: str = ""
) -> CacheType:
    """
    Detect cache type from an extracted state and class name.

    Used when working with extracted cache states (dicts with 'state' key)
    rather than live cache objects.

    Args:
        state: The cache state (tuple of tensors, list of arrays, etc.)
        class_name: Original cache class name (most reliable)

    Returns:
        CacheType enum value
    """
    # Class name is the most reliable signal
    # Match longest names first to avoid "KVCache" matching before "RotatingKVCache"
    if class_name:
        for name in sorted(_CLASS_NAME_MAP, key=len, reverse=True):
            if name in class_name:
                return _CLASS_NAME_MAP[name]

    if not state:
        return CacheType.UNKNOWN

    # Structure-based detection
    if isinstance(state, (tuple, list)):
        if len(state) == 2:
            first = state[0]
            if hasattr(first, "shape"):
                if len(first.shape) == 4:
                    # 4D tensor: (batch, heads, seq, dim) -> KV cache
                    return CacheType.KV_CACHE
                elif len(first.shape) == 3:
                    # 3D tensor: could be cumulative state
                    return CacheType.MAMBA_CACHE

        if len(state) == 4:
            # Could be quantized: (keys_q, values_q, scales, zero_points)
            first = state[0]
            if hasattr(first, "shape") and len(first.shape) == 4:
                return CacheType.QUANTIZED_KV_CACHE

    return CacheType.UNKNOWN


def is_positional_cache(cache_type: CacheType) -> bool:
    """Check if cache type stores position-indexed KV data (sliceable)."""
    return cache_type in _POSITIONAL_TYPES


def is_cumulative_cache(cache_type: CacheType) -> bool:
    """Check if cache type stores cumulative state (not sliceable)."""
    return cache_type in _CUMULATIVE_TYPES


def get_cache_structure_info(cache_obj: Any) -> Dict[str, Any]:
    """
    Get structural information about a cache object for debugging.

    Args:
        cache_obj: A cache object from mlx-lm

    Returns:
        Dict with cache structure details
    """
    info: Dict[str, Any] = {
        "class_name": type(cache_obj).__name__,
        "cache_type": detect_cache_type(cache_obj).value,
    }

    if hasattr(cache_obj, "keys") and hasattr(cache_obj.keys, "shape"):
        info["keys_shape"] = list(cache_obj.keys.shape)
    if hasattr(cache_obj, "values") and hasattr(cache_obj.values, "shape"):
        info["values_shape"] = list(cache_obj.values.shape)
    if hasattr(cache_obj, "offset"):
        info["offset"] = cache_obj.offset
    if hasattr(cache_obj, "max_size"):
        info["max_size"] = cache_obj.max_size
    if hasattr(cache_obj, "keep"):
        info["keep"] = cache_obj.keep
    if hasattr(cache_obj, "cache") and isinstance(cache_obj.cache, list):
        info["num_arrays"] = len(cache_obj.cache)
        info["array_shapes"] = [
            list(a.shape) if hasattr(a, "shape") else None
            for a in cache_obj.cache
        ]
    if hasattr(cache_obj, "caches"):
        info["num_sub_caches"] = len(cache_obj.caches)
        info["sub_cache_types"] = [
            detect_cache_type(c).value for c in cache_obj.caches
        ]

    return info
