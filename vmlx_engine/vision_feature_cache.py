# SPDX-License-Identifier: Apache-2.0
"""Per-item SSD vision-feature cache for Qwen-family VLMs.

Why: both media-tail paths (Flash-Next conditioned tail, 27B DFlash2 media
plan) must encode the COMPLETE request so merged features and M-RoPE positions
stay exact. Without a feature cache, every new-image turn re-ran the vision
tower over every earlier image too (a 4,963-token screenshot costs ~3 s on the
M5 Max) even though the KV prefix itself was restored from SSD.

What: ``install(model, model_path)`` gives the model's vision tower a per-item
call. Every media item (one ``grid_thw`` row and its patch rows) is looked up by
sha256(pixels, grid row) in an SSD store inside the managed block-cache pool;
only missing items run the tower. Items are ALWAYS encoded one at a time --
cached or not -- so a cold request and a warm one produce byte-identical
features (joint vs per-item encodes may round differently; mixing them would
make cache hits change answers).

Policy: SSD only (no RAM LRU -- the shipping cache profile keeps RAM tiers off),
same pool and budget as the block cache, own schema namespace. Installed only
when the SSD tier is on and the model is a Qwen3.5/Qwen4Exp VLM whose callers
discard the tower's second return value. ``VMLX_VISION_FEATURE_SSD=0`` disables.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import weakref
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

SCHEMA = "vision_features_v1"
_FAMILIES = {"qwen4_exp", "qwen3_5", "qwen3_5_moe"}
_CONFIG: dict = {}
_STORES: weakref.WeakSet = weakref.WeakSet()
STATS = {"hits": 0, "misses": 0, "load_s": 0.0, "encode_s": 0.0, "bypass": 0}


def clear_active_stores() -> int:
    """Clear feature SSD for loaded towers without keeping models alive.

    The store serializes clear with reads/writes under the shared pool lock.
    In-flight features already returned to a request remain request-owned.
    """
    stores = list(_STORES)
    for store in stores:
        store.clear()
    return len(stores)


def configure(*, root: str, max_size_bytes: int) -> None:
    """Called by the CLI once the SSD tier is resolved (before model load)."""
    if os.environ.get("VMLX_VISION_FEATURE_SSD", "1").strip().lower() in ("0", "off", "false", "no"):
        logger.info("Vision feature SSD cache off (VMLX_VISION_FEATURE_SSD=0)")
        return
    _CONFIG.update(root=str(root), max_size_bytes=int(max_size_bytes))


def _model_identity(model: Any, model_path: str) -> str:
    """Bind features to the loaded bundle and runtime, once at installation.

    Config/index files alone do not identify weights replaced in place. Use
    the same shard-aware fingerprint as the native session SSD cache; this
    only stats bundle files and never reads weight contents or runs per turn.
    """
    from .model_bundle_integrity import _bundle_fingerprint
    from .prefix_cache import runtime_cache_fingerprint

    root = Path(model_path).expanduser().resolve()
    parts = [str(root), _bundle_fingerprint(root), runtime_cache_fingerprint()]
    config = getattr(model, "config", None)
    vision = config.get("vision_config") if isinstance(config, dict) else getattr(config, "vision_config", None)
    try:
        parts.append(json.dumps(vision if isinstance(vision, dict) else vars(vision), sort_keys=True, default=str))
    except Exception:
        parts.append(repr(vision))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:32]


def _grid_rows(grid) -> Optional[list]:
    try:
        import numpy as np

        rows = np.asarray(grid).reshape(-1, 3).tolist()
        rows = [tuple(int(v) for v in r) for r in rows]
        return rows if rows and all(all(v > 0 for v in r) for r in rows) else None
    except Exception:
        return None


def install(model: Any, model_path: str) -> bool:
    if not _CONFIG:
        return False
    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "") or (config.get("model_type", "") if isinstance(config, dict) else "")).lower()
    tower = getattr(model, "vision_tower", None)
    if model_type not in _FAMILIES or tower is None or getattr(tower, "_vmlx_feature_cache", None) is not None:
        return False
    from .utils.omni_session_disk_store import OmniSessionDiskStore

    store = OmniSessionDiskStore(
        root=_CONFIG["root"], model_key=_model_identity(model, model_path),
        max_size_bytes=_CONFIG["max_size_bytes"], schema=SCHEMA,
    )
    base = type(tower)

    class _PerItemCachedVisionTower(base):  # keeps the module tree / parameters intact
        def __call__(self, pixel_values, grid_thw=None, *args, **kwargs):
            return _cached_call(self, base, store, pixel_values, grid_thw, args, kwargs)

    _PerItemCachedVisionTower.__name__ = base.__name__
    _PerItemCachedVisionTower.__qualname__ = base.__qualname__
    tower.__class__ = _PerItemCachedVisionTower
    tower._vmlx_feature_cache = store
    _STORES.add(store)
    logger.info("Vision feature SSD cache installed for %s (%s, schema %s)", model_type, store.directory, SCHEMA)
    return True


def _cached_call(tower, base, store, pixel_values, grid_thw, args, kwargs):
    import time

    import mlx.core as mx
    import numpy as np

    rows = _grid_rows(grid_thw) if grid_thw is not None else None
    if args or kwargs or rows is None or pixel_values is None:
        STATS["bypass"] += 1
        return base.__call__(tower, pixel_values, grid_thw, *args, **kwargs)
    spans = [t * h * w for t, h, w in rows]
    if sum(spans) != int(pixel_values.shape[0]):
        STATS["bypass"] += 1
        return base.__call__(tower, pixel_values, grid_thw, *args, **kwargs)
    host = np.asarray(pixel_values.astype(mx.float32) if pixel_values.dtype == mx.bfloat16 else pixel_values)
    grid_dtype = getattr(grid_thw, "dtype", mx.int32)
    feats = []
    offset = 0
    for row, span in zip(rows, spans):
        pv = pixel_values[offset:offset + span]
        key = hashlib.sha256(host[offset:offset + span].tobytes() + np.asarray(row, dtype=np.int64).tobytes()
                             + str(pixel_values.dtype).encode()).hexdigest()
        offset += span
        t0 = time.perf_counter()
        loaded = None
        try:
            loaded = store.load(key, lambda p: mx.load(str(p)))
        except Exception:
            loaded = None
        if loaded is not None and "features" in loaded:
            STATS["hits"] += 1
            STATS["load_s"] += time.perf_counter() - t0
            feats.append(loaded["features"])
            continue
        STATS["misses"] += 1
        out = base.__call__(tower, pv, mx.array([list(row)], dtype=grid_dtype))
        feat, second = (out[0], out[1] if len(out) > 1 else None) if isinstance(out, tuple) else (out, None)
        mx.eval(feat)
        STATS["encode_s"] += time.perf_counter() - t0
        try:
            store.save(key, lambda tmp, f=feat: mx.save_safetensors(str(tmp), {"features": f}, {"schema": SCHEMA}))
        except Exception:
            logger.debug("vision feature SSD write failed", exc_info=True)
        feats.append(feat)
    features = feats[0] if len(feats) == 1 else mx.concatenate(feats, axis=0)
    # Callers in the admitted families discard the second value (Qwen3.5 has deepstack off).
    return features, None
