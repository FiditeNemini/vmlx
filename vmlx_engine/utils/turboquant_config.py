# SPDX-License-Identifier: Apache-2.0
"""vMLX-owned TurboQuant KV configuration and cache construction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import os
from typing import Optional

from jang_tools.turboquant.config import TurboQuantConfig as _JangTurboQuantConfig

logger = logging.getLogger(__name__)

QWEN_HYBRID_LIVE_TQ_COMPRESS_AFTER = 0
UNCALIBRATED_AUTO_TQ_BITS = 4
BONSAI_UNCALIBRATED_AUTO_TQ_BITS = 8
MINIMAX_M2_UNCALIBRATED_AUTO_TQ_BITS = 8
MIXED_SWA_AUTO_TQ_BITS = 4
HY3_FULL_KV_AUTO_TQ_BITS = 4
# Compatibility name retained for downstream tests/imports.  The safety rule is
# no longer Qwen-only: compatible bundles without model-owned TQ calibration use
# q4 stored attention KV. Full-KV Qwen keeps q4 for the bulk layers but protects
# the first/last six critical layers with q8; uniform q4 corrupted a live Qwen3
# restore while mixed q4/q8 and uniform q8 remained coherent in long-context
# cache probes. Bonsai retains its all-q8 exception established by its live
# agent-loop proof; native non-KV companion state is never quantized.
QWEN_HYBRID_UNCALIBRATED_TQ_BITS = UNCALIBRATED_AUTO_TQ_BITS
_QWEN_HYBRID_MODEL_TYPES = frozenset(
    {
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
        "qwen3_next",
    }
)


def _model_types(model_config: dict | None) -> set[str]:
    if not isinstance(model_config, dict):
        return set()
    candidates = [model_config]
    text_config = model_config.get("text_config")
    if isinstance(text_config, dict):
        candidates.append(text_config)
    return {
        str(candidate.get("model_type") or "").lower()
        for candidate in candidates
    }


def _is_bonsai_config(model_config: dict | None) -> bool:
    """Identify the Bonsai Qwen derivative that retains the q8 cache policy."""
    if not isinstance(model_config, dict):
        return False
    candidates = [model_config]
    text_config = model_config.get("text_config")
    if isinstance(text_config, dict):
        candidates.append(text_config)
    identity_fields = ("_name_or_path", "name_or_path", "model_name", "name")
    return any(
        "bonsai" in str(candidate.get(field) or "").lower()
        for candidate in candidates
        for field in identity_fields
    )


def _is_minimax_m2_config(model_config: dict | None) -> bool:
    """Identify MiniMax M2.x full-KV models requiring the q8 safety policy."""
    return bool(_model_types(model_config) & {"minimax", "minimax_m2"})


def _is_qwen_hybrid_attention_config(
    model_config: dict | None,
    detected_layer_types: list[str] | tuple[str, ...] | None = None,
) -> bool:
    if not isinstance(model_config, dict):
        return False
    candidates = [model_config]
    text_config = model_config.get("text_config")
    if isinstance(text_config, dict):
        candidates.append(text_config)
    layer_types: list[str] = []
    if detected_layer_types:
        layer_types = [str(value).lower() for value in detected_layer_types]
    else:
        for candidate in candidates:
            declared = candidate.get("layer_types")
            if isinstance(declared, list) and declared:
                layer_types = [str(value).lower() for value in declared]
                break
    has_companion = any(
        value in {
            "ssm",
            "linear_attention",
            "gated_delta",
            "gated_delta_net",
            "gated_linear_attention",
            "mamba",
            "mamba2",
        }
        for value in layer_types
    )
    has_kv = any(value in {"full_attention", "attention"} for value in layer_types)
    return bool(_model_types(model_config) & _QWEN_HYBRID_MODEL_TYPES) and (
        has_companion and has_kv
    )


def apply_uncalibrated_auto_tq_policy(
    tq_cfg: dict,
    model_config: dict | None,
    layer_types: list[str] | tuple[str, ...],
) -> dict:
    """Return a correctness-first Auto config for an uncalibrated bundle.

    An uncalibrated 3-bit storage round-trip is not a safe family default: it
    has caused Bonsai/Qwen tool loops and, on Laguna, a deterministic coherent
    cold -> incoherent paged+TQ warm transition.  Auto therefore keeps the
    mid-request live transition disabled and uses a correctness-gated storage
    codec on real attention KV slots. Qwen hybrid and other compatible families
    use q4 stored attention KV. Full-KV Qwen uses q4 bulk layers with q8 first/
    last-six critical layers because uniform q4 failed live semantic restore and
    the narrower first/last-three boundary still had a strict-format miss.
    Bonsai keeps q8 on every compatible KV layer because its repeated
    agent-loop proof established that boundary. MiniMax M2.x also uses q8
    throughout: independent long-prefix semantic A/B gates showed prompt-
    sensitive visible/tool-route drift after otherwise structurally exact q4
    page and disk round-trips, while bypassed requests remained coherent.
    HY3's plain full-KV runtime continues to use q4.
    Cumulative SSM/GatedDelta companions remain native and are never assigned
    fake TQ slots. A bundle-owned calibrated ``turboquant`` block or explicit
    operator settings remain authoritative.
    """
    resolved = dict(tq_cfg)
    from .hybrid_tq_cache import classify_qwen_cache_architecture

    architecture = classify_qwen_cache_architecture(
        model_config or {}, list(layer_types)
    )
    attention_layers = [
        index
        for index, layer_type in enumerate(layer_types)
        if str(layer_type).lower() == "attention"
    ]
    if not attention_layers:
        return resolved

    hy3_full_kv = bool(_model_types(model_config) & {"hy_v3", "hy3"}) and (
        len(attention_layers) == len(layer_types)
    )
    bonsai = _is_bonsai_config(model_config)
    minimax_m2 = _is_minimax_m2_config(model_config)
    storage_bits = (
        BONSAI_UNCALIBRATED_AUTO_TQ_BITS
        if bonsai
        else MINIMAX_M2_UNCALIBRATED_AUTO_TQ_BITS
        if minimax_m2
        else HY3_FULL_KV_AUTO_TQ_BITS
        if hy3_full_kv
        else UNCALIBRATED_AUTO_TQ_BITS
    )

    if architecture == "qwen_full_kv":
        auto_policy = (
            "bonsai_full_kv_storage_tq8"
            if bonsai
            else "qwen_full_kv_storage_tq4_critical_tq8"
        )
    elif architecture in {
        "qwen3_5_hybrid_gated_delta",
        "qwen3_next_hybrid_gated_delta",
    }:
        auto_policy = (
            "bonsai_hybrid_attention_kv_storage_tq8"
            if bonsai
            else "qwen_hybrid_attention_kv_storage_tq4"
        )
    elif hy3_full_kv:
        auto_policy = "hy3_full_kv_storage_tq4"
    elif minimax_m2 and len(attention_layers) == len(layer_types):
        auto_policy = "minimax_m2_full_kv_storage_tq8"
    elif len(attention_layers) == len(layer_types):
        auto_policy = "uncalibrated_full_kv_storage_tq4"
    else:
        auto_policy = "uncalibrated_selective_attention_kv_storage_tq4"

    critical_layers = (
        [0, 1, 2, 3, 4, 5, -6, -5, -4, -3, -2, -1]
        if architecture == "qwen_full_kv" and not bonsai
        else attention_layers
    )
    critical_bits = (
        8
        if architecture == "qwen_full_kv" and not bonsai
        else storage_bits
    )
    resolved.update(
        {
            "default_key_bits": storage_bits,
            "default_value_bits": storage_bits,
            "critical_key_bits": critical_bits,
            "critical_value_bits": critical_bits,
            # Hybrid lists name every real KV position. Full-KV Qwen instead
            # uses first/last transformer positions because every slot is KV.
            "critical_layers": critical_layers,
            "sink_tokens": 0,
            "compress_after": QWEN_HYBRID_LIVE_TQ_COMPRESS_AFTER,
            "auto_policy": auto_policy,
        }
    )
    return resolved


MIXED_SWA_STORAGE_TQ_ENV = "VMLX_MIXED_SWA_STORAGE_TQ"


def mixed_swa_storage_tq_opted_in() -> bool:
    """Has the operator asked for lossy STORAGE-ONLY KV on mixed-SWA models?

    Off by default: storage-only quantization with live encode disabled makes a
    warm cache hit disagree with a cold recompute at temperature 0. See the
    measurements in :func:`apply_mixed_swa_auto_tq_policy`.
    """
    return os.environ.get(MIXED_SWA_STORAGE_TQ_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def apply_mixed_swa_auto_tq_policy(
    tq_cfg: dict,
    layer_types: list[str] | tuple[str, ...],
) -> dict:
    """Use q4 TQ only on full-attention slots of a mixed rotating-SWA cache.

    Sliding slots retain their native RotatingKVCache and window metadata. The
    returned critical-layer list therefore names only the unbounded KV slots
    that the mixed-cache builder replaces with TurboQuantKVCache.
    """
    normalized = [str(value).lower() for value in layer_types]
    full_attention_layers = [
        index for index, value in enumerate(normalized) if value == "attention"
    ]
    has_sliding = any(value == "sliding" for value in normalized)
    if not has_sliding or not full_attention_layers:
        return dict(tq_cfg)

    if not mixed_swa_storage_tq_opted_in():
        # This policy sets compress_after=0 — live encode OFF — while asking for
        # q4 on the full-attention slots. That combination is ASYMMETRIC: those
        # layers keep exact KV in RAM, but come back through the q4 codec when a
        # block is restored, so a cache hit does not equal recomputation.
        #
        # MEASURED on Muse Glimmer 30B, temperature 0, ~7.27k prompt, warm reuse
        # confirmed in every arm:
        #   storage-tq4 (this policy): cold 157 tok, warm 241 tok, reasoning DIFFERS
        #   tq disabled              : cold 154 tok, warm 154 tok, byte-identical
        #   live q8                  : cold 201 tok, warm 201 tok, byte-identical
        # Both consistent arms apply the SAME precision to both paths; only the
        # storage-only arm diverges, and it is the one that shipped by default.
        #
        # This config is engine-invented (it only runs when the bundle shipped no
        # calibrated turboquant block), so nothing here is the model author's
        # intent. Reuse works identically without it — the compression buys L2
        # footprint and costs answer stability, which is the wrong trade to make
        # silently. Opt back in with VMLX_MIXED_SWA_STORAGE_TQ=1.
        resolved = dict(tq_cfg)
        resolved["enabled"] = False
        resolved["auto_policy"] = "mixed_swa_exact_kv_storage"
        return resolved

    resolved = dict(tq_cfg)
    resolved.update(
        {
            "default_key_bits": MIXED_SWA_AUTO_TQ_BITS,
            "default_value_bits": MIXED_SWA_AUTO_TQ_BITS,
            "critical_key_bits": MIXED_SWA_AUTO_TQ_BITS,
            "critical_value_bits": MIXED_SWA_AUTO_TQ_BITS,
            "critical_layers": full_attention_layers,
            "sink_tokens": 0,
            "compress_after": 0,
            "auto_policy": "mixed_swa_full_attention_kv_storage_tq4",
        }
    )
    return resolved


def turboquant_storage_signature(
    config: "TurboQuantConfig", auto_policy: str | None = None
) -> str:
    """Stable identity for every field that changes a persisted TQ payload."""
    payload = {
        "schema": "tq_codec_config_v2",
        "n_layers": int(config.n_layers),
        "default_key_bits": int(config.default_key_bits),
        "default_value_bits": int(config.default_value_bits),
        "critical_key_bits": int(config.critical_key_bits),
        "critical_value_bits": int(config.critical_value_bits),
        "critical_layers": [int(value) for value in config.critical_layers],
        "sink_tokens": int(config.sink_tokens),
        "seed": int(config.seed),
        "compress_after": int(getattr(config, "compress_after", 0) or 0),
        "auto_policy": auto_policy or "bundle_calibrated_or_explicit",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]


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


def install_turboquant_batch_deepcopy() -> None:
    """Make TQ caches independently copyable by mlx-lm BatchGenerator.

    ``PromptBatch.split()`` deep-copies every cache before filtering the old and
    new batches. MLX arrays already implement an independent deepcopy, but
    ``mlx.core.Dtype`` deliberately does not implement Python pickling. The TQ
    cache records the original attention dtypes after its first update, so the
    default object deepcopy fails after prefill and the scheduler retries the
    same request forever. Preserve immutable dtype handles by identity and
    deepcopy every other field, including live arrays and packed state.
    """
    import copy

    from jang_tools.turboquant.cache import TurboQuantKVCache

    if getattr(TurboQuantKVCache, "_vmlx_batch_deepcopy_installed", False):
        return

    def __deepcopy__(self, memo):
        cls = type(self)
        clone = cls.__new__(cls)
        memo[id(self)] = clone
        for name, value in self.__dict__.items():
            if (
                type(value).__name__ == "Dtype"
                and type(value).__module__ == "mlx.core"
            ):
                copied = value
            else:
                copied = copy.deepcopy(value, memo)
            setattr(clone, name, copied)
        return clone

    TurboQuantKVCache.__deepcopy__ = __deepcopy__
    TurboQuantKVCache._vmlx_batch_deepcopy_installed = True


def resolve_compress_after(tq_cfg: dict, model_config: dict | None = None) -> int:
    """Resolve the live-encode threshold without silently widening numerics.

    ``VMLX_TQ_COMPRESS_AFTER`` is the explicit live-gate control. A bundle may
    also own the setting in ``jang_config.turboquant`` (including an explicit
    zero to disable live encode). Uncalibrated hybrid Qwen Auto mode keeps this
    zero and performs TQ at the paged/L2 storage boundary; a mid-request lossy
    transition must be bundle-calibrated or explicitly requested.
    """
    override = os.environ.get("VMLX_TQ_COMPRESS_AFTER")
    if override is not None:
        try:
            return max(0, int(override))
        except ValueError:
            raise ValueError("VMLINUX_TQ_COMPRESS_AFTER must be a non-negative integer")
    if "compress_after" in tq_cfg:
        return max(0, int(tq_cfg.get("compress_after", 0) or 0))
    if _is_qwen_hybrid_attention_config(model_config):
        return QWEN_HYBRID_LIVE_TQ_COMPRESS_AFTER
    return 0


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
    install_turboquant_batch_deepcopy()
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
