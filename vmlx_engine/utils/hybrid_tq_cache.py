# SPDX-License-Identifier: Apache-2.0
"""Hybrid-aware TurboQuant live KV cache helpers."""

from __future__ import annotations

from typing import Any, Callable


QWEN_HYBRID_MODEL_TYPES = frozenset(
    {
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
        "qwen3_next",
    }
)

# Backward-compatible export for existing imports/tests.  Qwen3.5 is the
# runtime model_type used by Qwen3.6 bundles, but Qwen3-Next is a distinct
# hybrid architecture and is classified explicitly below.
QWEN36_HYBRID_MODEL_TYPES = QWEN_HYBRID_MODEL_TYPES

NEMOTRON_OMNI_HYBRID_MODEL_TYPES = frozenset(
    {
        "nemotron_h",
        "nemotron_h_v2",
    }
)

TURBOQUANT_MAKE_CACHE_NAMES = frozenset(
    {
        "_tq_make_cache",
        "_turboquant_make_cache",
        "_hybrid_turboquant_make_cache",
    }
)


def _model_types(config: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    stack = [config]
    while stack:
        item = stack.pop()
        if not isinstance(item, dict):
            continue
        model_type = str(item.get("model_type") or "").lower()
        if model_type:
            values.add(model_type)
        for key in ("text_config", "language_config", "llm_config"):
            child = item.get(key)
            if isinstance(child, dict):
                stack.append(child)
    return values


def classify_qwen_cache_architecture(
    model_config: dict[str, Any],
    layer_types: list[str] | tuple[str, ...],
) -> str:
    """Classify Qwen cache topology from nested model type plus layer layout.

    Do not route from a name containing ``qwen``.  Qwen3/Qwen3-MoE are ordinary
    full-KV models, Qwen3.5/3.6 and Qwen3-Next are hybrid only when their actual
    layout contains both cumulative and attention slots, and Qwen-Mamba is
    cumulative-only.
    """
    model_types = _model_types(model_config)
    kinds = {str(item).lower() for item in layer_types}
    has_ssm = "ssm" in kinds
    has_attention = "attention" in kinds
    if model_types & {
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }:
        return "qwen3_5_hybrid_gated_delta" if has_ssm and has_attention else "unsupported"
    if "qwen3_next" in model_types:
        return "qwen3_next_hybrid_gated_delta" if has_ssm and has_attention else "unsupported"
    if model_types & {"qwen_mamba"}:
        return "qwen_cumulative_only"
    if model_types & {
        "qwen",
        "qwen2",
        "qwen2_moe",
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3",
        "qwen3_moe",
        "qwen3_vl",
        "qwen3_vl_text",
        "qwen3_vl_moe",
    }:
        return "qwen_full_kv" if has_attention and not has_ssm else "unsupported"
    return "not_qwen"


def is_qwen36_hybrid_tq_supported(
    model_config: dict[str, Any],
    layer_types: list[str] | tuple[str, ...],
) -> bool:
    """Return True only for Qwen hybrid layouts with standard KV slots."""
    return classify_qwen_cache_architecture(model_config, layer_types) in {
        "qwen3_5_hybrid_gated_delta",
        "qwen3_next_hybrid_gated_delta",
    }


def is_selective_hybrid_tq_supported(
    model_config: dict[str, Any],
    layer_types: list[str] | tuple[str, ...],
) -> bool:
    """Return True for allow-listed hybrid families with native KV companions."""
    if "ssm" not in layer_types or "attention" not in layer_types:
        return False
    if is_qwen36_hybrid_tq_supported(model_config, layer_types):
        return True
    return bool(_model_types(model_config) & NEMOTRON_OMNI_HYBRID_MODEL_TYPES)


def is_turboquant_make_cache(make_cache: Any) -> bool:
    return getattr(make_cache, "__name__", "") in TURBOQUANT_MAKE_CACHE_NAMES


def _config_layer_types(model_config: dict[str, Any]) -> list[str]:
    """Return the model's declared per-layer attention kinds, or []."""
    candidates = [model_config]
    text_cfg = model_config.get("text_config")
    if isinstance(text_cfg, dict):
        candidates.append(text_cfg)
    for candidate in candidates:
        layer_types = candidate.get("layer_types")
        if isinstance(layer_types, list) and layer_types:
            return [str(item).lower() for item in layer_types]
    return []


def is_mixed_swa_tq_supported(model_config: dict[str, Any], n_cache_slots: int) -> bool:
    """True for gemma-4-style mixed sliding/full attention with a usable layout.

    Requires the config to declare one layer_type per native cache slot, and to
    actually mix sliding with non-sliding layers. Without a 1:1 mapping we cannot
    tell which slot is the unbounded full-attention one, so TQ stays off.
    """
    layer_types = _config_layer_types(model_config)
    if len(layer_types) != n_cache_slots:
        return False
    has_sliding = any("sliding" in t for t in layer_types)
    has_full = any("sliding" not in t for t in layer_types)
    return has_sliding and has_full


def build_mixed_swa_layer_types(model_config: dict[str, Any]) -> list[str]:
    """Map declared layer kinds onto the hybrid builder's slot vocabulary.

    Full-attention slots become "attention" (TurboQuantKVCache): their KV grows
    with the whole context, so 3-bit encode is where the memory actually is.
    Sliding slots become "sliding", which the hybrid builder leaves as the model's
    native RotatingKVCache — TurboQuantKVCache is monotonic-growth (no max_size /
    keep / trim), so it cannot honour a sliding window, and a 512-token window
    saves almost nothing anyway. This keeps the mixed_swa_kv_v1 rotating-window
    metadata intact, which a flat TQ wrap would destroy.
    """
    return [
        "sliding" if "sliding" in t else "attention"
        for t in _config_layer_types(model_config)
    ]


def _is_kv_cache_slot(slot: Any) -> bool:
    return type(slot).__name__ in (
        "KVCache",
        "RotatingKVCache",
        "QuantizedKVCache",
        "TurboQuantKVCache",
    )


def build_hybrid_turboquant_make_cache(
    native_make_cache: Callable[[], list[Any]],
    tq_config: Any,
    key_dim: int,
    val_dim: int,
    layer_types: list[str],
):
    """Build a make_cache patch that TQs attention KV slots only.

    The native cache factory is still called on every invocation so all
    path-dependent hybrid companion state keeps its original class and
    initialization contract.
    """
    from .turboquant_config import install_turboquant_live_telemetry

    install_turboquant_live_telemetry()
    from jang_tools.turboquant.cache import TurboQuantKVCache

    n_layers = len(layer_types)
    attention_layers = tuple(
        i for i, layer_type in enumerate(layer_types) if layer_type == "attention"
    )
    companion_layers = tuple(
        i for i, layer_type in enumerate(layer_types) if layer_type != "attention"
    )

    def _hybrid_turboquant_make_cache(
        _native_make_cache=native_make_cache,
        _cfg=tq_config,
        _n=n_layers,
        _kd=key_dim,
        _vd=val_dim,
        _lt=tuple(layer_types),
    ):
        native_cache = list(_native_make_cache() or [])
        caches = []
        for i, slot in enumerate(native_cache):
            layer_type = _lt[i] if i < _n else "attention"
            if layer_type == "attention" and _is_kv_cache_slot(slot):
                caches.append(
                    TurboQuantKVCache(
                        key_dim=_kd,
                        value_dim=_vd,
                        key_bits=_cfg.key_bits_for_layer(i),
                        value_bits=_cfg.value_bits_for_layer(i),
                        seed=_cfg.seed + i,
                        compress_after=int(getattr(_cfg, "compress_after", 0) or 0),
                        sink_tokens=_cfg.sink_tokens,
                    )
                )
            else:
                caches.append(slot)
        return caches

    _hybrid_turboquant_make_cache._vmlx_hybrid_tq_policy = "attention_kv_only"
    _hybrid_turboquant_make_cache._vmlx_hybrid_tq_attention_layers = attention_layers
    _hybrid_turboquant_make_cache._vmlx_hybrid_tq_companion_layers = companion_layers
    _hybrid_turboquant_make_cache._vmlx_tq_compress_after = int(
        getattr(tq_config, "compress_after", 0) or 0
    )
    return _hybrid_turboquant_make_cache
