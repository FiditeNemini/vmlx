"""Persisted-state identity for the canonical JANGH numerical runtime."""

import hashlib
import os
from pathlib import Path

# Freeze diagnostic routing controls alongside persisted-state identity.
# Preserve producer flag semantics: only the exact string "1" enables these
# routes. Empty, whitespace, aliases and other values remain disabled. Store
# effective values so equivalent disabled spellings share one cache identity.
def _producer_binary_flag(name: str, default: str) -> str:
    return "1" if os.environ.get(name, default) == "1" else "0"


QWEN4_PREFILL_FUSED = _producer_binary_flag("JANGH_QWEN4_PREFILL_FUSED", "1")
WEIGHTED_UNSORT = _producer_binary_flag("JANGH_WEIGHTED_UNSORT", "0")
TAIL_SPLIT = _producer_binary_flag("JANGH_TAIL_SPLIT", "1")
# Decode-vs-sorted routed path by TOKEN count (see switch.TQSwitchGLU._use_sorted). Empty = the
# family loader's measured default; an integer pins it. Part of the identity because a short
# prefill chunk below the limit runs the gather kernels (fp32 weighted accumulate) instead of the
# sorted NAX path, which is a different (both exact-gated) rounding path for stored KV.
DECODE_MAX_TOKENS = os.environ.get("JANGH_DECODE_MAX_TOKENS", "").strip()
# Measured 2026-10-04 on M5 Max / MLX 0.32.3 for qwen4_exp (D=2560, I=640, E=512, k=10, JANGH 4/6-bit),
# jangh_crossover_bench.py, interleaved A/B/A/B, medians of 14 rounds: the gather decode path wins
# through 96 tokens and the sorted NAX path from 128.
QWEN4_DECODE_MAX_TOKENS = 96


def qwen4_decode_max_tokens() -> int:
    """Effective qwen4_exp decode-vs-sorted token limit: the env pin when it is an integer, else the measured default."""
    return int(DECODE_MAX_TOKENS) if DECODE_MAX_TOKENS.isdigit() else QWEN4_DECODE_MAX_TOKENS

DECODE_ROT = os.environ.get("JANGTQ2_DECODE_ROT", "host").strip().lower()
PREFILL = os.environ.get("JANGTQ2_PREFILL", "").strip().lower()
EXPERT_TILES = os.environ.get("JANGH_EXPERT_TILES", "0").strip()
PREFILL_REDUCE = os.environ.get("JANGH_PREFILL_REDUCE", "0").strip()
if PREFILL_REDUCE not in {"0", "1"}:
    raise ValueError("JANGH_PREFILL_REDUCE must be 0 or 1")
H32_ROWS = os.environ.get("JANGH_H32_ROWS", "0").strip()
if H32_ROWS not in {"0", "1"}:
    raise ValueError("JANGH_H32_ROWS must be 0 or 1")
GATEUP_H32 = os.environ.get("JANGH_GATEUP_H32", "0").strip()
if GATEUP_H32 not in {"0", "1"}:
    raise ValueError("JANGH_GATEUP_H32 must be 0 or 1")
if EXPERT_TILES not in {"0", "1"}:
    raise ValueError("JANGH_EXPERT_TILES must be 0 or 1")
if DECODE_ROT not in {"host", "kernel"}:
    raise ValueError("JANGTQ2_DECODE_ROT must be host or kernel")
if PREFILL not in {"", "steel", "nax"}:
    raise ValueError("JANGTQ2_PREFILL must be steel, nax, or unset")


def runtime_identity() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return (
        "jangh=" + digest.hexdigest()
        + ";decode_rot=" + DECODE_ROT + ";prefill=" + (PREFILL or "auto")
        + ";expert_tiles=" + EXPERT_TILES
        + ";gateup_h32=" + GATEUP_H32
        + ";h32_rows=" + H32_ROWS
        + ";prefill_reduce=" + PREFILL_REDUCE
        + ";qwen4_prefill_fused=" + QWEN4_PREFILL_FUSED
        + ";weighted_unsort=" + WEIGHTED_UNSORT
        + ";tail_split=" + TAIL_SPLIT
        + ";decode_max_tokens=" + DECODE_MAX_TOKENS
    )
