"""Decide whether a checkpoint's RMSNorm weights still need the +1 shift.

Qwen3.5-family (and Step3.7) checkpoints store *zero-centered* norm weights:
the effective scale is ``1 + w``, so the loader adds 1.0 during sanitize.
That is correct for a checkpoint in the original layout — and WRONG for a
bundle that a conversion pipeline saved AFTER running sanitize, because
those weights are already shifted. Applying the shift twice roughly doubles
the norm scale, which presents as fluent-looking multilingual garbage with
empty visible content and mid-generation 502s (vmlx#259).

Key naming cannot separate the two cases: JANG bundles and mlx-community
conversions both ship MLX-side names (``language_model.model.*``). The value
distribution can, and by a wide margin — measured on real checkpoints:

    dealignai/Qwen3.8-27B-JANG_4D-CRACK  layer0 input_layernorm mean = -0.04
    jangq-ai/Qwen3.8-27B-JANG_2D          layer0 input_layernorm mean = -0.04
    mlx-community/Qwen3.8-27B-4bit        layer0 input_layernorm mean = +0.96

This is the detector the native-MTP Qwen adapter has used since it shipped
(``_qwen_norm_shard_looks_unshifted``); it lives here so the base family
loaders — which had an UNCONDITIONAL shift and therefore garbled every
already-converted bundle — share one implementation with it rather than
growing a second, subtly different one.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Sequence

logger = logging.getLogger(__name__)


def qwen_norm_shard_looks_unshifted(
    weights: Dict[str, Any], norm_keys: Sequence[str]
) -> bool:
    """True when this shard's norm weights are still zero-centered.

    Evidence-first: the first 1-D norm tensor that reads clearly below the
    shifted population decides it. The final ``model.norm.weight`` carries a
    larger scale than per-layer norms, so it gets its own threshold. No
    readable evidence returns False (leave already-working loads alone).
    """
    import mlx.core as mx

    for key, value in weights.items():
        if not any(key.endswith(sfx) for sfx in norm_keys):
            continue
        if getattr(value, "ndim", 0) != 1:
            continue
        try:
            sample = value[: min(int(value.shape[0]), 1024)].astype(mx.float32)
            mean = float(mx.mean(sample).item())
        except Exception:
            continue
        if key.endswith("model.norm.weight") and mean < 1.5:
            return True
        if mean < 0.5:
            return True
    return False


def zero_centered_norm_shift_needed(
    weights: Dict[str, Any],
    norm_key_suffixes: Sequence[str],
    *,
    model_label: str = "checkpoint",
) -> bool:
    """``qwen_norm_shard_looks_unshifted`` plus a loud log when skipping.

    The skip changes how weights are interpreted, so it must never be
    silent: a user debugging output quality needs to see which branch ran.
    """
    needed = qwen_norm_shard_looks_unshifted(weights, norm_key_suffixes)
    if not needed:
        logger.warning(
            "%s: RMSNorm weights are already shifted; skipping the +1 "
            "zero-centered shift. This checkpoint was saved after "
            "conversion — shifting again would roughly double the norm "
            "scale and produce garbled output (vmlx#259).",
            model_label,
        )
    return needed
