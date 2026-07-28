# SPDX-License-Identifier: Apache-2.0
"""Nanbeige looped-transformer runtime registration and cache contract.

Nanbeige 4.2 reuses ``num_hidden_layers`` module objects for ``num_loops``
forward passes.  The module tree therefore has 22 layers while the prompt
cache has 44 independent slots.  Treating ``len(model.layers)`` as the cache
length does not necessarily crash; it silently changes the model.  Keep the
registration and post-load invariant in one module so generic mlx-lm and JANG
loaders enforce the same architecture contract.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

NANBEIGE_MODEL_TYPE = "nanbeige"
NANBEIGE_CACHE_LAYOUT = "looped_kv_v1"


def _read_json(path: str | Path | None, name: str) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        data = json.loads((Path(path) / name).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _effective_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    text_config = config.get("text_config")
    if (
        isinstance(text_config, dict)
        and str(text_config.get("model_type") or "").lower()
        == NANBEIGE_MODEL_TYPE
    ):
        return text_config
    return config


def is_nanbeige_config(config: dict[str, Any] | None) -> bool:
    """Return whether config metadata names the Nanbeige architecture."""

    if not isinstance(config, dict):
        return False
    candidates = (
        config.get("model_type"),
        (config.get("text_config") or {}).get("model_type")
        if isinstance(config.get("text_config"), dict)
        else None,
    )
    return any(str(value or "").lower() == NANBEIGE_MODEL_TYPE for value in candidates)


def is_nanbeige_model_path(model_path: str | Path | None) -> bool:
    """Detect Nanbeige from a local bundle's authoritative config.json."""

    return is_nanbeige_config(_read_json(model_path, "config.json"))


def ensure_nanbeige_runtime_registered(
    model_path: str | Path | None = None,
    *,
    config: dict[str, Any] | None = None,
) -> bool:
    """Install ``mlx_lm.models.nanbeige`` before model-class resolution.

    Registration is attempted opportunistically for unresolved Hugging Face
    IDs, but an unavailable registration module only becomes an error when the
    local config is known to be Nanbeige.  That keeps unrelated model loading
    unchanged while making an actual Nanbeige load fail with an actionable
    error instead of a late generic ``model_type not supported`` exception.
    """

    resolved_config = config if isinstance(config, dict) else _read_json(
        model_path, "config.json"
    )
    required = is_nanbeige_config(resolved_config)
    if resolved_config and not required:
        return False
    try:
        importlib.import_module("jang_tools.nanbeige.mlx_register")
        importlib.import_module("mlx_lm.models.nanbeige")
        return True
    except Exception as exc:
        if required:
            raise RuntimeError(
                "Nanbeige bundle requires the loop-aware "
                "jang_tools.nanbeige runtime, but registration failed. "
                "Install or bundle the current jang-tools runtime; do not "
                "fall back to a stock 22-layer Llama cache."
            ) from exc
        return False


def _positive_int(source: str, value: Any) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"Nanbeige loop-cache contract: {source} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Nanbeige loop-cache contract: {source} must be a positive integer"
        ) from exc
    if result <= 0:
        raise RuntimeError(
            f"Nanbeige loop-cache contract: {source} must be positive (got {result})"
        )
    return result


def _model_arg(model: Any, name: str) -> Any:
    args = getattr(model, "args", None)
    if args is not None and hasattr(args, name):
        return getattr(args, name)
    inner = getattr(model, "model", None)
    args = getattr(inner, "args", None)
    if args is not None and hasattr(args, name):
        return getattr(args, name)
    if inner is not None and hasattr(inner, name):
        return getattr(inner, name)
    return None


def validate_nanbeige_loop_cache_contract(
    model: Any,
    model_path: str | Path | None = None,
    *,
    config: dict[str, Any] | None = None,
    jang_config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Fail closed unless every Nanbeige loop-cache source agrees.

    The current artifact has 22 shared layer modules and two passes, hence 44
    cache slots.  ``config.json`` is authoritative for layer/loop counts;
    JANG bundles additionally stamp the loop layout and cache-slot count in
    ``config.jang_runtime`` and ``jang_config.runtime``.  The loaded runtime
    must expose both ``cache_slots`` and ``make_cache()`` with the same derived
    count.  A missing or 22-entry cache is rejected before generation.
    """

    config_data = config if isinstance(config, dict) else _read_json(
        model_path, "config.json"
    )
    if not is_nanbeige_config(config_data):
        return None

    effective = _effective_config(config_data)
    jang_data = (
        jang_config
        if isinstance(jang_config, dict)
        else _read_json(model_path, "jang_config.json")
    )
    config_runtime = config_data.get("jang_runtime")
    config_runtime = config_runtime if isinstance(config_runtime, dict) else {}
    jang_runtime = jang_data.get("runtime")
    jang_runtime = jang_runtime if isinstance(jang_runtime, dict) else {}

    layer_sources = {
        "config.num_hidden_layers": _positive_int(
            "config.num_hidden_layers", effective.get("num_hidden_layers")
        ),
        "model.args.num_hidden_layers": _positive_int(
            "model.args.num_hidden_layers",
            _model_arg(model, "num_hidden_layers"),
        ),
    }
    if config_runtime.get("num_hidden_layers") is not None:
        layer_sources["config.jang_runtime.num_hidden_layers"] = _positive_int(
            "config.jang_runtime.num_hidden_layers",
            config_runtime.get("num_hidden_layers"),
        )
    model_layers = getattr(model, "layers", None)
    if model_layers is not None:
        layer_sources["len(model.layers)"] = _positive_int(
            "len(model.layers)", len(model_layers)
        )

    raw_model_loops = _model_arg(model, "total_loops")
    if raw_model_loops is None:
        raw_model_loops = _model_arg(model, "num_loops")
    loop_sources = {
        "config.num_loops": _positive_int(
            "config.num_loops", effective.get("num_loops")
        ),
        "model.args.total_loops": _positive_int(
            "model.args.total_loops", raw_model_loops
        ),
    }
    if config_runtime.get("num_loops") is not None:
        loop_sources["config.jang_runtime.num_loops"] = _positive_int(
            "config.jang_runtime.num_loops",
            config_runtime.get("num_loops"),
        )
    if jang_runtime:
        loop_sources["jang_config.runtime.num_loops"] = _positive_int(
            "jang_config.runtime.num_loops", jang_runtime.get("num_loops")
        )

    if len(set(layer_sources.values())) != 1:
        raise RuntimeError(
            f"Nanbeige loop-cache layer mismatch: {layer_sources}"
        )
    if len(set(loop_sources.values())) != 1:
        raise RuntimeError(
            f"Nanbeige loop-cache loop-count mismatch: {loop_sources}"
        )

    num_hidden_layers = next(iter(layer_sources.values()))
    num_loops = next(iter(loop_sources.values()))
    if num_loops <= 1:
        raise RuntimeError(
            "Nanbeige loop-cache contract requires a looped runtime "
            f"(num_loops={num_loops})"
        )
    expected_slots = num_hidden_layers * num_loops

    cache_layout_sources: dict[str, str] = {}
    if config_runtime:
        cache_layout_sources["config.jang_runtime.cache_layout"] = str(
            config_runtime.get("cache_layout") or ""
        )
    if jang_runtime.get("cache_layout") is not None:
        cache_layout_sources["jang_config.runtime.cache_layout"] = str(
            jang_runtime.get("cache_layout") or ""
        )
    for source, layout in cache_layout_sources.items():
        if layout != NANBEIGE_CACHE_LAYOUT:
            raise RuntimeError(
                f"Nanbeige loop-cache contract: {source}={layout!r}, "
                f"expected {NANBEIGE_CACHE_LAYOUT!r}"
            )

    make_cache = getattr(model, "make_cache", None)
    if not callable(make_cache):
        raise RuntimeError(
            "Nanbeige loop-cache contract: loaded model has no make_cache()"
        )
    try:
        native_cache = make_cache()
        made_slots = len(native_cache)
    except Exception as exc:
        raise RuntimeError(
            "Nanbeige loop-cache contract: model.make_cache() failed"
        ) from exc

    slot_sources = {
        "derived(num_hidden_layers*num_loops)": expected_slots,
        "model.cache_slots": _positive_int(
            "model.cache_slots", getattr(model, "cache_slots", None)
        ),
        "len(model.make_cache())": _positive_int(
            "len(model.make_cache())", made_slots
        ),
    }
    if config_runtime:
        slot_sources["config.jang_runtime.cache_slots"] = _positive_int(
            "config.jang_runtime.cache_slots",
            config_runtime.get("cache_slots"),
        )
    if jang_runtime:
        slot_sources["jang_config.runtime.cache_slots"] = _positive_int(
            "jang_config.runtime.cache_slots",
            jang_runtime.get("cache_slots"),
        )
    if len(set(slot_sources.values())) != 1:
        raise RuntimeError(
            "Nanbeige loop-cache slot mismatch; refusing silent shared-loop "
            f"cache reuse: {slot_sources}"
        )

    contract = {
        "cache_layout": NANBEIGE_CACHE_LAYOUT,
        "num_hidden_layers": num_hidden_layers,
        "num_loops": num_loops,
        "cache_slots": expected_slots,
    }
    model._vmlx_looped_cache_contract = contract
    return contract
