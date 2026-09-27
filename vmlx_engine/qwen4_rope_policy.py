"""Process-lifetime identity for opt-in Qwen4 attention rotary math."""

import os

QWEN4_EXACT_ROPE_ATTN_MATH_ABI = "elementwise-mrope-v1"
# This numerical option is a startup setting, not a supported hot toggle.
# One frozen value must own forward math, persisted state, and attestation.
_EXACT_ROPE_ATTN_ENABLED = os.environ.get(
    "VMLX_QWEN4_EXACT_ROPE_ATTN", "0"
).strip().lower() not in {"", "0", "false", "no", "off"}


def exact_rope_attn_enabled() -> bool:
    return _EXACT_ROPE_ATTN_ENABLED


def exact_rope_attn_cache_identity() -> str:
    # Preserve the established off-mode identity in installed releases.
    if not _EXACT_ROPE_ATTN_ENABLED:
        return ""
    return "qwen4_exact_rope_attn=" + QWEN4_EXACT_ROPE_ATTN_MATH_ABI


def exact_rope_attn_status() -> dict:
    return {
        "enabled": _EXACT_ROPE_ATTN_ENABLED,
        "math_abi": QWEN4_EXACT_ROPE_ATTN_MATH_ABI,
        "configuration_scope": "process_startup",
    }
