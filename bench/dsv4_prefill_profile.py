#!/usr/bin/env python3
"""Characterise DSV4 cold-prefill throughput before optimising anything.

Forces a cold prefill per sample (unique ``cache_salt``) and measures
time-to-first-token against prompt length, so prefill cost can be separated
into a fixed per-request overhead and a per-token slope:

    ttft ≈ fixed_overhead + prompt_tokens / prefill_rate

Optimising the MoE prefill kernel only helps the slope. If the fixed term
dominates at realistic prompt sizes, kernel work is the wrong target.

Usage:
    python3 bench/dsv4_prefill_profile.py [base_url]
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
import uuid

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
MODEL = "DeepSeek-V4-Flash-0731-JANG"

FILLER = (
    "Rule {i}: when validating a release candidate, prefer the smallest change "
    "that is provably correct, record the exact command that produced each "
    "receipt, and never claim a result that was not observed directly. "
)


def make_prompt(target_rules: int) -> str:
    return "".join(FILLER.format(i=i) for i in range(1, target_rules + 1))


def ttft_sample(prompt: str) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt + "\n\nReply with the single word OK."}],
        "stream": True,
        "max_tokens": 8,
        "enable_thinking": False,
        "cache_salt": str(uuid.uuid4()),  # force a cold prefill
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first = None
    usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            evt = json.loads(payload)
            if evt.get("usage"):
                usage = evt["usage"]
            for ch in evt.get("choices") or []:
                delta = ch.get("delta") or {}
                if first is None and (delta.get("content") or delta.get("reasoning_content")):
                    first = time.perf_counter() - start
    total = time.perf_counter() - start
    pt = (usage or {}).get("prompt_tokens", 0)
    cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    return {"prompt_tokens": pt, "cached": cached, "ttft": first or total, "total": total}


def main():
    print(f"base={BASE} model={MODEL}")
    print("Cold prefill forced per sample via unique cache_salt.\n")
    print(f"{'rules':>6} {'prompt_tok':>11} {'cached':>7} {'ttft_s':>8} {'tok/s':>9}")
    print("-" * 48)

    rows = []
    for rules in (20, 80, 200, 400, 800):
        sample = ttft_sample(make_prompt(rules))
        rate = sample["prompt_tokens"] / sample["ttft"] if sample["ttft"] else 0.0
        rows.append(sample)
        print(f"{rules:>6} {sample['prompt_tokens']:>11} {sample['cached']:>7} "
              f"{sample['ttft']:>8.3f} {rate:>9.1f}")

    usable = [r for r in rows if r["prompt_tokens"] and r["cached"] == 0]
    print()
    if len(usable) < 2:
        print("Not enough cold samples to fit a slope "
              f"({len(usable)} of {len(rows)} were cold).")
        return 1

    lo, hi = usable[0], usable[-1]
    dt = hi["ttft"] - lo["ttft"]
    dn = hi["prompt_tokens"] - lo["prompt_tokens"]
    if dn <= 0:
        print("Prompt lengths did not increase; cannot fit.")
        return 1
    slope_rate = dn / dt if dt > 0 else float("inf")
    fixed = lo["ttft"] - lo["prompt_tokens"] / slope_rate if slope_rate else lo["ttft"]

    print("=== two-point fit over the cold samples ===")
    print(f"  marginal prefill rate : {slope_rate:8.1f} tok/s")
    print(f"  fixed per-request cost: {fixed:8.3f} s")
    print()
    print("  interpretation at each measured size:")
    for r in usable:
        marginal = r["prompt_tokens"] / slope_rate if slope_rate else 0.0
        share = 100.0 * marginal / r["ttft"] if r["ttft"] else 0.0
        print(f"    {r['prompt_tokens']:>6} tok: ttft={r['ttft']:.3f}s, "
              f"token-proportional part ~{marginal:.3f}s ({share:.0f}%)")
    print()
    print("  A MoE prefill kernel can only improve the token-proportional part.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
