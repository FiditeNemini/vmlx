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
    )
