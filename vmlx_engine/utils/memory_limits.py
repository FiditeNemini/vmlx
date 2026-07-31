# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for Metal working-set and scheduler memory guard limits."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple

_MB = 1024**2
_GB = 1024**3

# jang-tools' native DSV4 pool codec keeps each compressed branch in BF16
# through 2 MiB, then stores uint8 codes plus FP16 affine min/scale metadata in
# groups of 32 features.  Keep these constants beside the admission estimator
# so the server's memory policy describes the cache that the runtime actually
# constructs rather than generic full-sequence K/V geometry.
_DSV4_POOL_BF16_MAX_BYTES = 2 * _MB
_DSV4_POOL_QUANT_GROUP_SIZE = 32
_DSV4_POOL_QUANT_CODE_BYTES = 1
_DSV4_POOL_QUANT_PARAM_BYTES = 2


@dataclass(frozen=True)
class DSV4CacheMemoryEstimate:
    """Conservative retained-memory admission estimate for native DSV4 state.

    DSV4 retains a bounded local SWA K/V ring on every layer.  Ratio-4 CSA
    layers additionally retain compressor and sparse-indexer pools; ratio-128
    HCA layers retain compressor pools.  ``tail_bytes`` covers the incomplete
    window state (including ratio-4's previous overlap window) that lives
    beside those pools.
    """

    token_count: int
    total_bytes: int
    local_swa_bytes: int
    csa_pool_bytes: int
    csa_indexer_bytes: int
    hca_pool_bytes: int
    tail_bytes: int
    ratio_zero_layers: int
    ratio_four_layers: int
    ratio_high_layers: int
    pool_quant_enabled: bool


def _parse_bool_env(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) != "0"


def _parse_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        value = float(raw)
        if value <= 0:
            return float(default)
        return value
    except (TypeError, ValueError):
        return float(default)


def _cfg_get(config, name: str, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _dtype_scalar_bytes(dtype) -> int:
    raw = str(dtype or "").lower()
    if any(marker in raw for marker in ("float32", "fp32", "f32")):
        return 4
    if any(marker in raw for marker in ("float64", "fp64", "f64")):
        return 8
    if any(marker in raw for marker in ("int8", "uint8", "fp8", "float8")):
        return 1
    # MLX KV cache for these local LLM paths is normally fp16/bf16.
    return 2


def _positive_int(value, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return parsed if parsed > 0 else int(default)


def _dsv4_config(config):
    """Return the text config when it owns a native DeepSeek-V4 topology."""

    text_config = _cfg_get(config, "text_config")
    candidates = [text_config, config] if text_config is not None else [config]
    for candidate in candidates:
        if candidate is None:
            continue
        model_type = str(_cfg_get(candidate, "model_type", "") or "").lower()
        normalized_type = model_type.replace("-", "_")
        architectures = _cfg_get(candidate, "architectures", ()) or ()
        if isinstance(architectures, str):
            architectures = (architectures,)
        architecture_text = " ".join(str(item) for item in architectures).lower()
        if normalized_type in {"deepseek_v4", "deepseekv4"} or (
            "deepseekv4" in architecture_text.replace("_", "")
        ):
            return candidate
    return None


def _dsv4_compress_ratios(config, num_layers: int) -> tuple[int, ...]:
    """Mirror the native DSV4 per-layer ratio resolver.

    Current bundles carry an explicit ``compress_ratios`` list.  The fallback
    below intentionally matches ``jang_tools.dsv4.mlx_model`` for legacy
    bundles instead of inventing a new layout.
    """

    raw_ratios = _cfg_get(config, "compress_ratios") or ()
    ratios: list[int] = []
    for layer_id in range(num_layers):
        raw = raw_ratios[layer_id] if layer_id < len(raw_ratios) else None
        try:
            ratio = int(raw) if raw is not None else None
        except (TypeError, ValueError):
            ratio = None
        if ratio is None:
            if layer_id == 0 or layer_id == num_layers - 1:
                ratio = 0
            else:
                ratio = 4 if (layer_id - 1) % 2 else 128
        ratios.append(max(0, ratio))
    return tuple(ratios)


def _dsv4_pool_storage_bytes(
    rows: int,
    feature_dim: int,
    scalar_bytes: int,
    *,
    pool_quant_enabled: bool,
) -> int:
    """Return the monotonic admission envelope for one retained DSV4 pool."""

    rows = max(0, int(rows))
    feature_dim = max(0, int(feature_dim))
    if rows == 0 or feature_dim == 0:
        return 0
    bf16_bytes = rows * feature_dim * scalar_bytes
    if not pool_quant_enabled or bf16_bytes <= _DSV4_POOL_BF16_MAX_BYTES:
        return bf16_bytes

    group_size = _DSV4_POOL_QUANT_GROUP_SIZE
    if feature_dim % group_size:
        group_size = feature_dim
    groups = max(1, feature_dim // group_size)
    encoded_bytes = rows * (
        feature_dim * _DSV4_POOL_QUANT_CODE_BYTES
        + 2 * groups * _DSV4_POOL_QUANT_PARAM_BYTES
    )
    # The q8 representation is the only retained long-pool tier. Native
    # attention scans bounded dequantized indexer tiles and gathers only the
    # selected compressor rows; it must never retain a full historical BF16
    # view beside these codes. Preserve a monotonic admission envelope across
    # the BF16-to-q8 promotion boundary because the capacity search relies on
    # monotonicity and promotion briefly owns the old small BF16 value.
    return max(_DSV4_POOL_BF16_MAX_BYTES, encoded_bytes)


def estimate_dsv4_cache_memory_from_config(
    config,
    token_count: int,
    *,
    pool_quant_enabled: Optional[bool] = None,
) -> Optional[DSV4CacheMemoryEstimate]:
    """Estimate native DSV4 SWA+CSA/HCA cache admission memory.

    The estimate is batch-one, matching vMLX's enforced DSV4 single-sequence
    runtime.  It includes the bounded local ring on *every* instantiated layer,
    compressed pools, ratio-4 indexer pools, and incomplete-window buffers.
    Returns ``None`` for non-DSV4 or incomplete configurations.
    """

    cfg = _dsv4_config(config)
    if cfg is None:
        return None
    tokens = max(0, int(token_count))
    num_layers = _positive_int(
        _cfg_get(cfg, "num_hidden_layers")
        or _cfg_get(cfg, "n_layers")
        or _cfg_get(cfg, "num_layers")
    )
    n_heads = _positive_int(
        _cfg_get(cfg, "num_attention_heads")
        or _cfg_get(cfg, "n_heads")
        or _cfg_get(cfg, "attention_heads")
    )
    n_kv_heads = _positive_int(
        _cfg_get(cfg, "num_key_value_heads")
        or _cfg_get(cfg, "n_kv_heads")
        or _cfg_get(cfg, "num_kv_heads")
        or n_heads
    )
    head_dim = _positive_int(_cfg_get(cfg, "head_dim"))
    if not head_dim:
        hidden = _positive_int(
            _cfg_get(cfg, "hidden_size") or _cfg_get(cfg, "d_model")
        )
        head_dim = hidden // n_heads if hidden and n_heads else 0
    sliding_window = _positive_int(_cfg_get(cfg, "sliding_window"), 128)
    index_head_dim = _positive_int(_cfg_get(cfg, "index_head_dim"), 128)
    if not num_layers or not n_kv_heads or not head_dim:
        return None

    dtype = (
        _cfg_get(cfg, "torch_dtype")
        or _cfg_get(cfg, "dtype")
        or _cfg_get(cfg, "mlx_dtype")
    )
    scalar_bytes = _dtype_scalar_bytes(dtype)
    if pool_quant_enabled is None:
        pool_quant_enabled = os.environ.get("DSV4_POOL_QUANT", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    ratios = _dsv4_compress_ratios(cfg, num_layers)
    ratio_zero_layers = sum(ratio == 0 for ratio in ratios)
    ratio_four_layers = sum(ratio == 4 for ratio in ratios)
    ratio_high_layers = sum(ratio not in (0, 4) for ratio in ratios)

    # Every layer, including CSA/HCA layers, wraps a bounded local SWA cache.
    local_rows = min(tokens, sliding_window)
    local_swa_bytes = (
        num_layers * local_rows * 2 * n_kv_heads * head_dim * scalar_bytes
    )

    csa_pool_bytes = 0
    csa_indexer_bytes = 0
    hca_pool_bytes = 0
    tail_bytes = 0
    for ratio in ratios:
        if ratio <= 0:
            continue
        rows = tokens // ratio
        pool_bytes = _dsv4_pool_storage_bytes(
            rows,
            head_dim,
            scalar_bytes,
            pool_quant_enabled=bool(pool_quant_enabled),
        )
        if ratio == 4:
            csa_pool_bytes += pool_bytes
            csa_indexer_bytes += _dsv4_pool_storage_bytes(
                rows,
                index_head_dim,
                scalar_bytes,
                pool_quant_enabled=bool(pool_quant_enabled),
            )
            # The overlap compressor/indexer retain the previous complete
            # ratio-4 window plus the current remainder.  Each branch keeps
            # both projected KV and gate tensors at twice its pooled width.
            # Admission tracks the peak buffer reached while consuming the
            # prompt, not only the smaller remainder left at its endpoint.
            # This keeps the capacity function monotonic across ratio
            # boundaries where the runtime rolls a completed window forward.
            buffered_rows = min(tokens, 2 * ratio - 1)
            tail_bytes += buffered_rows * (2 * head_dim) * 2 * scalar_bytes
            tail_bytes += buffered_rows * (2 * index_head_dim) * 2 * scalar_bytes
        else:
            hca_pool_bytes += pool_bytes
            # Non-overlap HCA retains only the incomplete current window, with
            # one KV projection and one gate projection.
            buffered_rows = min(tokens, ratio - 1)
            tail_bytes += buffered_rows * head_dim * 2 * scalar_bytes

    total_bytes = (
        local_swa_bytes
        + csa_pool_bytes
        + csa_indexer_bytes
        + hca_pool_bytes
        + tail_bytes
    )
    return DSV4CacheMemoryEstimate(
        token_count=tokens,
        total_bytes=total_bytes,
        local_swa_bytes=local_swa_bytes,
        csa_pool_bytes=csa_pool_bytes,
        csa_indexer_bytes=csa_indexer_bytes,
        hca_pool_bytes=hca_pool_bytes,
        tail_bytes=tail_bytes,
        ratio_zero_layers=ratio_zero_layers,
        ratio_four_layers=ratio_four_layers,
        ratio_high_layers=ratio_high_layers,
        pool_quant_enabled=bool(pool_quant_enabled),
    )


def estimate_cache_bytes_for_tokens_from_config(
    config,
    token_count: int,
    *,
    dsv4_pool_quant_enabled: Optional[bool] = None,
) -> int:
    """Estimate retained cache bytes for a batch-one prompt of ``token_count``."""

    dsv4 = estimate_dsv4_cache_memory_from_config(
        config,
        token_count,
        pool_quant_enabled=dsv4_pool_quant_enabled,
    )
    if dsv4 is not None:
        return dsv4.total_bytes
    return max(0, int(token_count)) * estimate_kv_bytes_per_token_from_config(config)


def estimate_cache_token_capacity_from_config(
    config,
    budget_bytes: int,
    *,
    max_tokens: int = 0,
    dsv4_pool_quant_enabled: Optional[bool] = None,
) -> int:
    """Return the largest cache length admitted by ``budget_bytes``.

    Native DSV4 memory is nonlinear because SWA stops growing at its window
    while CSA/HCA pools grow at their own ratios.  Its admission envelope is
    monotonic, so a bounded binary search replaces the incorrect generic
    bytes-per-token division.  Other families retain the prior linear path.
    """

    try:
        budget = int(budget_bytes)
        ceiling = int(max_tokens)
    except (TypeError, ValueError):
        return 0
    if budget <= 0:
        return 0

    if _dsv4_config(config) is None:
        bytes_per_token = estimate_kv_bytes_per_token_from_config(config)
        if bytes_per_token <= 0:
            return 0
        capacity = budget // bytes_per_token
        return min(capacity, ceiling) if ceiling > 0 else capacity

    if ceiling <= 0:
        ceiling = 1
        while (
            ceiling < 1_000_000_000
            and estimate_cache_bytes_for_tokens_from_config(
                config,
                ceiling,
                dsv4_pool_quant_enabled=dsv4_pool_quant_enabled,
            )
            <= budget
        ):
            ceiling *= 2
        ceiling = min(ceiling, 1_000_000_000)

    low, high = 0, ceiling
    while low < high:
        middle = (low + high + 1) // 2
        required = estimate_cache_bytes_for_tokens_from_config(
            config,
            middle,
            dsv4_pool_quant_enabled=dsv4_pool_quant_enabled,
        )
        if required <= budget:
            low = middle
        else:
            high = middle - 1
    return low


def estimate_kv_bytes_per_token_from_config(config) -> int:
    """Estimate live KV-cache bytes added per generated token.

    The estimate intentionally uses standard K+V cache geometry and leaves
    family-specific temporary/fragmentation safety to callers via their
    projected-budget multiplier. Dict and attr-style configs are both accepted,
    including multimodal wrappers that store text fields under ``text_config``.
    """
    dsv4_cfg = _dsv4_config(config)
    if dsv4_cfg is not None:
        # Worst-case one-token growth is the local ring growth on every layer
        # plus the average native pool growth.  This is used by the output
        # projection guard; prompt admission uses the nonlinear total above.
        num_layers = _positive_int(_cfg_get(dsv4_cfg, "num_hidden_layers"))
        n_heads = _positive_int(_cfg_get(dsv4_cfg, "num_attention_heads"))
        n_kv_heads = _positive_int(
            _cfg_get(dsv4_cfg, "num_key_value_heads") or n_heads
        )
        head_dim = _positive_int(_cfg_get(dsv4_cfg, "head_dim"))
        index_head_dim = _positive_int(
            _cfg_get(dsv4_cfg, "index_head_dim"), 128
        )
        dtype = (
            _cfg_get(dsv4_cfg, "torch_dtype")
            or _cfg_get(dsv4_cfg, "dtype")
            or _cfg_get(dsv4_cfg, "mlx_dtype")
        )
        if num_layers and n_kv_heads and head_dim:
            scalar_bytes = _dtype_scalar_bytes(dtype)
            local_growth = (
                num_layers * 2 * n_kv_heads * head_dim * scalar_bytes
            )
            pool_growth = 0.0
            for ratio in _dsv4_compress_ratios(dsv4_cfg, num_layers):
                if ratio <= 0:
                    continue
                # Without the current per-branch row count, the projected
                # output guard must budget the BF16 hot-tier growth.  Prompt
                # admission above knows the length and applies q8 exactly once
                # a branch crosses its 2 MiB promotion boundary.
                compressor_row_bytes = head_dim * scalar_bytes
                pool_growth += compressor_row_bytes / ratio
                if ratio == 4:
                    indexer_row_bytes = index_head_dim * scalar_bytes
                    pool_growth += indexer_row_bytes / ratio
            return local_growth + max(1, int(pool_growth + 0.999999))

    text_config = _cfg_get(config, "text_config")
    candidates = [text_config, config] if text_config is not None else [config]
    for cfg in candidates:
        n_layers = (
            _cfg_get(cfg, "num_hidden_layers")
            or _cfg_get(cfg, "n_layers")
            or _cfg_get(cfg, "num_layers")
            or 0
        )
        num_loops = (
            _cfg_get(cfg, "total_loops")
            or _cfg_get(cfg, "num_loops")
            or _cfg_get(config, "total_loops")
            or _cfg_get(config, "num_loops")
            or 1
        )
        n_heads = (
            _cfg_get(cfg, "num_attention_heads")
            or _cfg_get(cfg, "n_heads")
            or _cfg_get(cfg, "attention_heads")
            or 0
        )
        n_kv_heads = (
            _cfg_get(cfg, "num_key_value_heads")
            or _cfg_get(cfg, "n_kv_heads")
            or _cfg_get(cfg, "num_kv_heads")
            or n_heads
            or 0
        )
        head_dim = _cfg_get(cfg, "head_dim") or 0
        if not head_dim:
            hidden = _cfg_get(cfg, "hidden_size") or _cfg_get(cfg, "d_model") or 0
            head_dim = int(hidden) // int(n_heads) if hidden and n_heads else 0
        dtype = (
            _cfg_get(cfg, "torch_dtype")
            or _cfg_get(cfg, "dtype")
            or _cfg_get(cfg, "mlx_dtype")
        )
        try:
            n_layers = int(n_layers)
            num_loops = int(num_loops)
            n_kv_heads = int(n_kv_heads)
            head_dim = int(head_dim)
        except (TypeError, ValueError):
            continue
        if n_layers > 0 and n_kv_heads > 0 and head_dim > 0:
            effective_layers = n_layers * max(1, num_loops)
            return (
                effective_layers
                * 2
                * n_kv_heads
                * head_dim
                * _dtype_scalar_bytes(dtype)
            )
    return 0


def projected_output_token_cap(
    *,
    active_bytes: int,
    max_working_set_bytes: int,
    bytes_per_token: int,
    budget_fraction: float = 0.50,
    transient_multiplier: float = 4.0,
) -> int:
    """Return safe generated-token cap from current Metal headroom.

    ``budget_fraction`` reserves headroom for non-KV temporaries. The
    ``transient_multiplier`` accounts for attention workspaces, allocator
    fragmentation, paged/native cache metadata, and media tensors that are not
    represented by the steady-state KV bytes/token estimate.
    """
    try:
        active = int(active_bytes)
        max_ws = int(max_working_set_bytes)
        bpt = int(bytes_per_token)
        fraction = float(budget_fraction)
        multiplier = float(transient_multiplier)
    except (TypeError, ValueError):
        return 0
    if active < 0 or max_ws <= 0 or bpt <= 0:
        return 0
    if fraction <= 0 or fraction > 1:
        fraction = 0.50
    if multiplier < 1:
        multiplier = 1.0
    headroom = max(0, max_ws - active)
    effective_budget = int(headroom * fraction)
    effective_bpt = max(1, int(bpt * multiplier))
    return max(0, effective_budget // effective_bpt)


def _parse_working_set_bytes(raw: str) -> Optional[int]:
    raw = (raw or "").strip().replace(",", "")
    if not raw:
        return None
    match = re.match(r"^(\d+(?:\.\d+)?)\s*(gb|g|mb|m)?$", raw.lower())
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2) or "g"
    if unit in ("mb", "m"):
        return int(value * _MB)
    return int(value * _GB)


def resolve_working_set_override(base_bytes: int) -> int:
    """Resolve an explicit working-set ceiling override.

    Supported env vars:
    - VMLX_METAL_WS_MAX_BYTES: exact bytes.
    - VMLX_METAL_WS_MAX_GB: size in gigabytes.

    When override is provided and valid, it is clamped to ``base_bytes`` so the
    guard cannot exceed the MLX-reported device limit. This prevents callers from
    accidentally bypassing a safe OS/device guard on unsupported hardware.
    """
    override = os.environ.get("VMLX_METAL_WS_MAX_BYTES")
    parsed = None
    if override is not None:
        try:
            parsed = int((override or "").strip().replace(",", ""))
        except (TypeError, ValueError):
            parsed = _parse_working_set_bytes(override)
    else:
        override = os.environ.get("VMLX_METAL_WS_MAX_GB")
        parsed = _parse_working_set_bytes(override) if override is not None else None
    if parsed is None:
        return base_bytes
    if parsed <= 0:
        return base_bytes
    if base_bytes > 0:
        return min(base_bytes, parsed)
    return parsed


def get_metal_working_set_stats(mx_module=None) -> Tuple[int, int]:
    """Return ``(active_memory_bytes, max_working_set_bytes)`` from MLX."""
    if mx_module is None:
        import mlx.core as mx_module

    get_active = getattr(mx_module, "get_active_memory", None) or mx_module.metal.get_active_memory
    get_device_info = getattr(mx_module, "device_info", None) or mx_module.metal.device_info

    try:
        active = int(get_active() or 0)
    except Exception:
        active = 0

    max_ws = 0
    try:
        info = get_device_info() or {}
        max_ws = int(info.get("max_recommended_working_set_size", 0) or 0)
    except Exception:
        max_ws = 0

    return active, max_ws


def get_effective_metal_working_set_bytes(mx_module=None) -> Tuple[int, int]:
    """Return ``(active_memory_bytes, effective_max_working_set_bytes)``.

    Effective limit uses override env vars when valid, otherwise uses MLX's
    `max_recommended_working_set_size`.
    """
    active, base_ws = get_metal_working_set_stats(mx_module)
    return active, resolve_working_set_override(base_ws)


def get_metal_ws_guard_threshold(default: float = 98.0) -> float:
    """Current percent threshold for working-set rejection (e.g. 98 = 98%).

    Default raised from 85 to 98 (Eric directive 2026-05-11): users should be
    able to fill near-all of their unified memory before the guard fires.
    The 2% headroom still catches the genuine Metal command-buffer OOM edge
    case before MLX raises [METAL] Insufficient Memory and crashes the
    engine process. Override via VMLX_METAL_WS_REJECT_PCT.
    """
    return _parse_float_env("VMLX_METAL_WS_REJECT_PCT", default)


def is_metal_ws_guard_enabled() -> bool:
    """Whether the metal working-set guard is enabled."""
    return _parse_bool_env("VMLX_METAL_WS_GUARD", "1")
