# SPDX-License-Identifier: Apache-2.0
"""vMLX-owned TurboQuant KV configuration and cache construction."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Optional

from jang_tools.turboquant.config import TurboQuantConfig as _JangTurboQuantConfig

logger = logging.getLogger(__name__)


def _layer_resident_bytes(layer) -> int:
    total = 0
    for name in (
        "keys",
        "values",
        "_decoded_k_buffer",
        "_decoded_v_buffer",
        "_joined_k",
        "_joined_v",
    ):
        value = getattr(layer, name, None)
        if value is not None and hasattr(value, "nbytes"):
            total += int(value.nbytes)
    total += int(getattr(layer, "compressed_nbytes", 0) or 0)
    return total


def install_turboquant_live_telemetry() -> None:
    """Instrument the real codec transition with authoritative byte counters."""
    from jang_tools.turboquant.cache import TurboQuantKVCache

    if getattr(TurboQuantKVCache, "_vmlx_compress_telemetry_installed", False):
        return
    original = TurboQuantKVCache.compress

    def compress(self, n_tokens=None):
        before = _layer_resident_bytes(self)
        before_tokens = int(getattr(self, "_compressed_tokens", 0) or 0)
        result = original(self, n_tokens)
        after = _layer_resident_bytes(self)
        record = {
            "calls": int(
                getattr(TurboQuantKVCache, "_vmlx_compress_calls", 0) or 0
            ) + 1,
            "requested_tokens": n_tokens,
            "compressed_tokens_before": before_tokens,
            "compressed_tokens_after": int(
                getattr(self, "_compressed_tokens", 0) or 0
            ),
            "resident_before_bytes": before,
            "resident_after_bytes": after,
            "resident_delta_bytes": after - before,
            "resident_memory_reduction_claimed": False,
        }
        TurboQuantKVCache._vmlx_compress_calls = record["calls"]
        TurboQuantKVCache._vmlx_last_compress = record
        logger.info("TurboQuant compress telemetry: %s", record)
        return result

    TurboQuantKVCache.compress = compress
    TurboQuantKVCache._vmlx_compress_telemetry_installed = True


def resolve_compress_after(tq_cfg: dict, model_config: dict | None = None) -> int:
    """Resolve the live-encode threshold without silently widening numerics.

    ``VMLX_TQ_COMPRESS_AFTER`` is the explicit live-gate control. A bundle may
    also own the setting in ``jang_config.turboquant``. There is intentionally
    no implicit family default until that family passes the live coherence and
    resident-memory gate documented in the issue ledger.
    """
    override = os.environ.get("VMLX_TQ_COMPRESS_AFTER")
    if override is not None:
        try:
            return max(0, int(override))
        except ValueError:
            raise ValueError("VMLINUX_TQ_COMPRESS_AFTER must be a non-negative integer")
    return max(0, int(tq_cfg.get("compress_after", 0) or 0))


@dataclass
class TurboQuantConfig(_JangTurboQuantConfig):
    """TurboQuant configuration including the live-encode threshold.

    Zero deliberately means inert. Numerics-changing live encoding is enabled
    only by model-owned configuration, an explicit runtime gate, or a family
    default backed by live coherence evidence.
    """

    compress_after: int = 0

    @classmethod
    def from_jang_config(
        cls, jang_cfg: dict, n_layers: int
    ) -> Optional["TurboQuantConfig"]:
        tq = jang_cfg.get("turboquant")
        if not tq or not tq.get("enabled", False):
            return None
        return cls(
            n_layers=n_layers,
            default_key_bits=tq.get("default_key_bits", 3),
            default_value_bits=tq.get("default_value_bits", 3),
            critical_key_bits=tq.get("critical_key_bits", 4),
            critical_value_bits=tq.get("critical_value_bits", 4),
            critical_layers=tq.get(
                "critical_layers", [0, 1, 2, -3, -2, -1]
            ),
            sink_tokens=tq.get("sink_tokens", 4),
            seed=tq.get("seed", 42),
            compress_after=max(0, int(tq.get("compress_after", 0) or 0)),
        )


def make_turboquant_cache(
    config: TurboQuantConfig,
    n_layers: int,
    key_dims: list[int],
    value_dims: list[int],
    layer_types: list[str],
) -> list:
    """Build native companion caches plus threshold-aware TQ attention caches."""
    install_turboquant_live_telemetry()
    from jang_tools.turboquant.cache import TurboQuantKVCache

    try:
        from mlx_lm.models.cache import ArraysCache
    except ImportError:
        ArraysCache = None
    try:
        from mlx_lm.models.cache import KVCache
    except ImportError:
        KVCache = None

    caches = []
    for i in range(n_layers):
        if layer_types[i] == "attention":
            caches.append(
                TurboQuantKVCache(
                    key_dim=key_dims[i],
                    value_dim=value_dims[i],
                    key_bits=config.key_bits_for_layer(i),
                    value_bits=config.value_bits_for_layer(i),
                    seed=config.seed + i,
                    compress_after=config.compress_after,
                    sink_tokens=config.sink_tokens,
                )
            )
        elif ArraysCache is not None:
            caches.append(ArraysCache(size=2))
        elif KVCache is not None:
            caches.append(KVCache())
        else:
            raise ImportError("Neither ArraysCache nor KVCache is available")
    return caches


def turboquant_cache_telemetry(cache: list) -> dict:
    """Measure actual TQ object/encode state and resident array bytes."""
    layers = []

    def visit(value):
        if type(value).__name__ == "TurboQuantKVCache":
            layers.append(value)
            return
        children = getattr(value, "caches", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                visit(child)

    for layer in cache or []:
        visit(layer)

    resident_bytes = 0
    float_view_bytes = 0
    packed_bytes = 0
    baseline_float_bytes = 0
    for layer in layers:
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        joined_k = getattr(layer, "_joined_k", None)
        joined_v = getattr(layer, "_joined_v", None)
        if keys is not None and values is not None:
            baseline_float_bytes += int(keys.nbytes + values.nbytes)
        elif joined_k is not None and joined_v is not None:
            baseline_float_bytes += int(joined_k.nbytes + joined_v.nbytes)
        for name in (
            "keys",
            "values",
            "_decoded_k_buffer",
            "_decoded_v_buffer",
            "_joined_k",
            "_joined_v",
        ):
            value = getattr(layer, name, None)
            if value is not None and hasattr(value, "nbytes"):
                nbytes = int(value.nbytes)
                resident_bytes += nbytes
                float_view_bytes += nbytes
        nbytes = int(getattr(layer, "compressed_nbytes", 0) or 0)
        resident_bytes += nbytes
        packed_bytes += nbytes

    compressed = [
        int(getattr(layer, "_compressed_tokens", 0) or 0) for layer in layers
    ]
    thresholds = [int(getattr(layer, "compress_after", 0) or 0) for layer in layers]
    return {
        "object_layers": len(layers),
        "encode_enabled_layers": sum(value > 0 for value in thresholds),
        "encoded_layers": sum(value > 0 for value in compressed),
        "compressed_tokens_total": sum(compressed),
        "compressed_tokens_max": max(compressed, default=0),
        "compress_after_values": sorted(set(thresholds)),
        "resident_bytes": resident_bytes,
        "float_view_bytes": float_view_bytes,
        "packed_bytes": packed_bytes,
        "baseline_float_bytes": baseline_float_bytes,
        "resident_delta_vs_float_bytes": resident_bytes - baseline_float_bytes,
        "resident_memory_reduction_claimed": False,
    }
