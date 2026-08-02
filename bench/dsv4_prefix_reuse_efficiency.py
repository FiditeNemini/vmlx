#!/usr/bin/env python3
"""How much of the *available* prefix does DSV4 actually reuse?

Each turn of a growing conversation strictly extends the previous one, so the
tokens the engine already processed last turn (prompt + completion) are all
present at the front of this turn's prompt. That is the reuse that is
physically available. This probe compares it against the `cached_tokens` the
engine reports, turn over turn.

  available_i = prompt_tokens_{i-1} + completion_tokens_{i-1}
  efficiency  = cached_tokens_i / available_i

A block-aligned cache floors reuse to a multiple of the block size, so the
shortfall should never exceed one block. A larger shortfall means the longest
matching prefix is not being found.

Usage:
    python3 bench/dsv4_prefix_reuse_efficiency.py [base_url]
"""
from __future__ import annotations

import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
MODEL = "DeepSeek-V4-Flash-0731-JANG"

# A long, stable system prompt so the conversation spans many blocks.
SYSTEM = (
    "You are a meticulous release engineer for an on-device inference runtime. "
    "You answer briefly and precisely. "
) + " ".join(
    f"Rule {i}: prefer the smallest change that is provably correct and never "
    f"claim a result you have not observed."
    for i in range(1, 46)
)

TURNS = [
    "Name one benefit of a prefix cache. One sentence.",
    "Name one risk of a prefix cache. One sentence.",
    "Name one way to measure prefix cache health. One sentence.",
    "Summarise your three previous answers in one short sentence.",
]


def block_size():
    with urllib.request.urlopen(f"{BASE}/health", timeout=15) as r:
        d = json.load(r)
    return d["cache"]["scheduler_cache"].get("block_size", 256)


def chat(messages):
    body = {"model": MODEL, "messages": messages, "stream": False,
            "max_tokens": 120, "enable_thinking": False}
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    usage = d.get("usage") or {}
    msg = d["choices"][0]["message"]
    return {
        "prompt": usage.get("prompt_tokens", 0),
        "completion": usage.get("completion_tokens", 0),
        "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0,
        "detail": (usage.get("prompt_tokens_details") or {}).get("cache_detail"),
        "content": msg.get("content") or "",
    }


def main():
    bs = block_size()
    print(f"base={BASE}  block_size={bs}")
    print(f"system prompt chars={len(SYSTEM)}")
    print()

    messages = [{"role": "system", "content": SYSTEM}]
    prev = None
    rows = []
    for i, turn in enumerate(TURNS):
        messages.append({"role": "user", "content": turn})
        r = chat(messages)
        messages.append({"role": "assistant", "content": r["content"]})

        available = (prev["prompt"] + prev["completion"]) if prev else 0
        shortfall = available - r["cached"] if available else 0
        eff = (100.0 * r["cached"] / available) if available else float("nan")
        rows.append((i, r, available, shortfall, eff))
        print(f"turn {i}: prompt={r['prompt']:<5} completion={r['completion']:<4} "
              f"cached={r['cached']:<5} available={available:<5} "
              f"shortfall={shortfall:<5} efficiency={eff:5.1f}%  detail={r['detail']}")
        prev = r

    print()
    print("=== verdict ===")
    bad = []
    for i, r, available, shortfall, eff in rows:
        if not available:
            continue
        if shortfall > bs:
            bad.append(
                f"turn {i}: shortfall {shortfall} tokens exceeds one block ({bs}) — "
                f"longest matching prefix not found")
    if bad:
        for b in bad:
            print(f"  FAIL  {b}")
        print("\nReuse is worse than block alignment explains.")
        return 1
    print(f"  every turn reused all but at most one block ({bs} tokens)")
    print("  shortfall is block-alignment only, which is the expected floor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
