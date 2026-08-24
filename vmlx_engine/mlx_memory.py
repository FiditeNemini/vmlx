"""MLX memory-cache cleanup helpers."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def clear_mlx_memory_cache(
    *,
    mx: Any | None = None,
    log: logging.Logger | None = None,
) -> str | None:
    """Clear MLX's allocator cache using the API available in this MLX build.

    MLX 0.31.x removed/does not expose ``mx.clear_memory_cache()``. Prefer the
    current top-level cache API and keep the deprecated Metal API only as an
    older-build fallback.
    """

    active_log = log or logger
    if mx is None:
        try:
            import mlx.core as mx  # type: ignore[no-redef]
        except Exception as exc:  # noqa: BLE001 - cleanup must not mask request flow
            active_log.warning("Unable to import MLX for memory cache cleanup: %s", exc)
            return None

    if callable(getattr(mx, "clear_cache", None)):
        try:
            mx.clear_cache()
            return "mx.clear_cache"
        except Exception as exc:  # noqa: BLE001 - log API drift/failure
            active_log.warning(
                "MLX memory cache cleanup via mx.clear_cache failed: %s",
                exc,
            )
            return None

    metal = getattr(mx, "metal", None)
    if metal is not None and callable(getattr(metal, "clear_cache", None)):
        try:
            metal.clear_cache()
            return "mx.metal.clear_cache"
        except Exception as exc:  # noqa: BLE001 - log API drift/failure
            active_log.warning(
                "MLX memory cache cleanup via mx.metal.clear_cache failed: %s",
                exc,
            )
            return None

    active_log.warning("No known MLX memory cache clearing API available")
    return None


# Serving-time bound on MLX's allocator cache.
#
# MLX keeps freed buffers in an allocator cache so it can hand them back without
# a round trip to Metal. Its DEFAULT limit is effectively the whole wired
# working set -- on this box the engine logs "MLX wired working-set limit
# defaulted to 115 GB from OS sysctl iogpu.wired_limit_mb". Nothing bounded it
# on the serving path: `set_cache_limit` was only ever called from the
# sleep/wake admin routes, while two docstrings in mllm_batch_generator.py
# claimed a 25%-of-working-set limit was applied at init. It was not.
#
# Consequence, measured 2026-08-23 on Qwen3.8-27B (17.5 GB of weights):
# vmmap showed IOAccelerator 26.2 GB resident / 26.2 GB DIRTY / 26.2 GB NONVOL
# with VOLATILE = 0K -- every retained buffer wired, so the OS could not
# reclaim any of it. Physical footprint peaked at 103.9 GB.
#
# The bound below is proportional to the model actually loaded (the allocator
# needs more scratch for a bigger model) but capped, so residency stays close
# to the model's own size. It ADVISES the allocator to release; it never
# refuses work and never fails a request.
_CACHE_LIMIT_FRACTION = 0.05
_CACHE_LIMIT_FLOOR_BYTES = 512 * 1024 * 1024
_CACHE_LIMIT_CEIL_BYTES = 4 * 1024 * 1024 * 1024


def resolve_serving_cache_limit_bytes(
    model_resident_bytes: int,
    *,
    fraction: float = _CACHE_LIMIT_FRACTION,
    floor_bytes: int = _CACHE_LIMIT_FLOOR_BYTES,
    ceil_bytes: int = _CACHE_LIMIT_CEIL_BYTES,
) -> int:
    """Return the allocator-cache bound for a model of this resident size.

    Pure so it can be tested without MLX: fraction of the model's own footprint,
    clamped to [floor, ceil]. A non-positive input falls back to the floor
    rather than to "unbounded" -- failed detection must never widen a limit.
    """
    try:
        resident = int(model_resident_bytes or 0)
    except (TypeError, ValueError):
        resident = 0
    if resident <= 0:
        return int(floor_bytes)
    scaled = int(resident * float(fraction))
    return int(max(floor_bytes, min(ceil_bytes, scaled)))


_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_FALSE_ENV_VALUES = {"0", "false", "no", "off"}


def _declared_model_dtype(model_path: str | Path | None) -> str | None:
    """Read the bundle's declared compute dtype without loading any weights."""
    if model_path is None:
        return None
    try:
        import json

        config = json.loads((Path(model_path) / "config.json").read_text())
    except (OSError, TypeError, ValueError):
        return None
    candidates = [config]
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        candidates.insert(0, text_config)
    for candidate in candidates:
        for key in ("dtype", "torch_dtype"):
            value = candidate.get(key)
            if value is not None:
                return str(value).strip().lower()
    return None


def harmonize_quant_metadata_dtypes(
    model: Any,
    *,
    model_path: str | Path | None = None,
    declared_dtype: str | None = None,
    mx: Any | None = None,
    log: logging.Logger | None = None,
) -> dict[str, int]:
    """Cast F16 packed-quant metadata to a bundle-declared BF16 compute dtype.

    Several JANG bundles declare BF16 compute but ship packed quantisation
    scales/biases as F16. MLX promotes
    ``f16 op bf16`` to **float32** (verified in isolation on-device:
    ``quantized_matmul(bf16_act, wq, f16_scales, f16_biases) -> float32``),
    so every quantised projection returns fp32 and that cascades into K, V and
    activations.

    The cost is exactly 2x on the KV cache. For a 16-full-attention-layer model
    with 4 KV heads at head_dim 256 that is 128 KB/token instead of 64, i.e.
    9.1 GB instead of 4.5 GB at 74k context. Measured on-device the same
    ~131-137 KB/token signature appears on other wide-KV bundles too, so this is
    not one model's quirk.

    Only ``*.scales`` and ``*.biases`` leaves whose sibling ``*.weight`` is a
    packed non-floating tensor are eligible. Real F16 weights stay untouched.
    That distinction matters on Qwen3.8-27B-JANG_4D-CRACK: its artifact contains
    1,160 eligible F16 quant-metadata tensors *and* 15 genuine F16 MTP tensors.

    The caller must invoke this on the model's loader/step worker. MLX streams
    are thread-local; post-load mutation from the server thread is invalid for
    JANGTQ and VLM runtimes. This function intentionally fails the opt-in load
    if materialisation or installation fails rather than serving a potentially
    half-mutated model.
    """
    active_log = log or logger
    summary = {
        "f16": 0,
        "bf16": 0,
        "f32": 0,
        "eligible": 0,
        "cast": 0,
        "preserved_f16": 0,
    }
    if mx is None:
        try:
            import mlx.core as mx  # type: ignore[no-redef]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("dtype harmonisation requires MLX") from exc

    try:
        from mlx.utils import tree_flatten, tree_unflatten
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("dtype harmonisation requires mlx.utils") from exc

    try:
        leaves = tree_flatten(model.parameters())
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("dtype harmonisation could not read parameters") from exc

    for _name, arr in leaves:
        dt = getattr(arr, "dtype", None)
        if dt == mx.float16:
            summary["f16"] += 1
        elif dt == mx.bfloat16:
            summary["bf16"] += 1
        elif dt == mx.float32:
            summary["f32"] += 1

    native_dtype = str(
        declared_dtype or _declared_model_dtype(model_path) or ""
    ).strip().lower()
    if native_dtype not in {"bfloat16", "bf16"}:
        active_log.info(
            "Quant metadata dtype harmonisation skipped: bundle dtype=%s "
            "(f16=%d bf16=%d f32=%d)",
            native_dtype or "undeclared",
            summary["f16"],
            summary["bf16"],
            summary["f32"],
        )
        summary["preserved_f16"] = summary["f16"]
        return summary

    leaves_by_name = {name: arr for name, arr in leaves}
    floating_dtypes = {mx.float16, mx.bfloat16, mx.float32}
    eligible_names: set[str] = set()
    for name, arr in leaves:
        if getattr(arr, "dtype", None) != mx.float16:
            continue
        stem, separator, leaf_name = name.rpartition(".")
        if not separator or leaf_name not in {"scales", "biases"}:
            continue
        packed_weight = leaves_by_name.get(f"{stem}.weight")
        packed_dtype = getattr(packed_weight, "dtype", None)
        if packed_weight is None or packed_dtype in floating_dtypes:
            continue
        eligible_names.add(name)

    summary["eligible"] = len(eligible_names)
    summary["preserved_f16"] = summary["f16"] - summary["eligible"]
    if not eligible_names:
        active_log.info(
            "Quant metadata dtypes need no harmonisation "
            "(f16=%d bf16=%d f32=%d; no packed F16 scales/biases)",
            summary["f16"],
            summary["bf16"],
            summary["f32"],
        )
        return summary

    updated = []
    materialized_casts = []
    for name, arr in leaves:
        if name in eligible_names:
            cast = arr.astype(mx.bfloat16)
            updated.append((name, cast))
            materialized_casts.append(cast)
        else:
            updated.append((name, arr))

    # Materialize before mutating the module so an MLX failure leaves the model
    # unchanged. mx.eval accepts nested containers, avoiding thousands of
    # positional arguments on large JANG bundles.
    mx.eval(materialized_casts)
    model.update(tree_unflatten(updated))
    summary["cast"] = len(eligible_names)
    active_log.info(
        "Harmonised %d packed quant metadata tensors F16 -> BF16 "
        "(bundle dtype=%s; total f16=%d, preserved real f16=%d, bf16=%d)",
        summary["cast"],
        native_dtype,
        summary["f16"],
        summary["preserved_f16"],
        summary["bf16"],
    )
    return summary


def maybe_harmonize_quant_metadata_dtypes(
    model: Any,
    *,
    model_path: str | Path | None = None,
    mx: Any | None = None,
    log: logging.Logger | None = None,
) -> dict[str, int] | None:
    """Apply the native-dtype correction unless explicitly disabled.

    The operation itself remains narrowly self-gating: it only changes packed
    F16 affine metadata in a bundle that declares BF16 compute.  Making that
    correction the default restores MLX's native BF16 quantized-linear
    contract; ``VMLX_HARMONIZE_PARAM_DTYPES=0`` remains an emergency A/B
    escape hatch.
    """
    setting = os.environ.get("VMLX_HARMONIZE_PARAM_DTYPES")
    if setting is not None and setting.strip().lower() in _FALSE_ENV_VALUES:
        try:
            setattr(
                model,
                "_vmlx_quant_metadata_dtype_harmonization",
                {"enabled": False, "reason": "explicitly_disabled"},
            )
        except Exception:
            pass
        return None
    summary = harmonize_quant_metadata_dtypes(
        model,
        model_path=model_path,
        mx=mx,
        log=log,
    )
    try:
        setattr(
            model,
            "_vmlx_quant_metadata_dtype_harmonization",
            {
                "enabled": True,
                "policy": "bundle_declared_bfloat16_packed_affine_only",
                "explicit": bool(
                    setting is not None
                    and setting.strip().lower() in _TRUE_ENV_VALUES
                ),
                **summary,
            },
        )
    except Exception:
        pass
    return summary


def apply_serving_cache_limit(
    model_resident_bytes: int,
    *,
    mx: Any | None = None,
    log: logging.Logger | None = None,
) -> int | None:
    """Bound MLX's allocator cache for the serving path. Advisory, never fatal.

    Returns the limit applied, or None when the MLX build exposes no setter.
    ``VMLX_MLX_CACHE_LIMIT_MB`` overrides the computed value (0 = leave MLX's
    default alone, for A/B measurement).
    """
    active_log = log or logger
    if mx is None:
        try:
            import mlx.core as mx  # type: ignore[no-redef]
        except Exception as exc:  # noqa: BLE001 - never break model load
            active_log.warning("Unable to import MLX to bound allocator cache: %s", exc)
            return None

    override = os.environ.get("VMLX_MLX_CACHE_LIMIT_MB")
    if override is not None:
        try:
            mb = int(override)
        except (TypeError, ValueError):
            mb = -1
        if mb == 0:
            active_log.info(
                "MLX allocator cache left at the MLX default "
                "(VMLX_MLX_CACHE_LIMIT_MB=0)"
            )
            return None
        if mb > 0:
            limit = mb * 1024 * 1024
        else:
            limit = resolve_serving_cache_limit_bytes(model_resident_bytes)
    else:
        limit = resolve_serving_cache_limit_bytes(model_resident_bytes)

    setter = getattr(mx, "set_cache_limit", None)
    if setter is None:
        metal = getattr(mx, "metal", None)
        setter = getattr(metal, "set_cache_limit", None) if metal else None
    if not callable(setter):
        active_log.warning(
            "This MLX build exposes no set_cache_limit; allocator cache stays "
            "at the MLX default and residency will exceed the model's size"
        )
        return None

    try:
        previous = setter(limit)
    except Exception as exc:  # noqa: BLE001 - advisory only
        active_log.warning("Could not bound MLX allocator cache: %s", exc)
        return None

    active_log.info(
        "MLX allocator cache bounded to %.2fGB for serving "
        "(model resident %.1fGB, previous limit %s)",
        limit / (1024**3),
        max(0, int(model_resident_bytes or 0)) / (1024**3),
        (
            "%.2fGB" % (previous / (1024**3))
            if isinstance(previous, (int, float)) and previous
            else previous
        ),
    )
    return int(limit)
