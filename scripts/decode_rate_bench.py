#!/usr/bin/env python3
"""Measure sustained decode rate honestly.

Three rules this encodes, each of which has produced a wrong number before:

  * Token counts come from ``usage.completion_tokens``. SSE chunks are NOT
    tokens -- deltas coalesce, so counting chunks understates the true rate
    (measured 67.8 vs a true 94.6 on one model).
  * Report p50-of-inter-token-gaps AND the mean AND the max gap. The mean alone
    hides a stall; p50 alone hides that a stall happened at all.
  * Reasoning models emit ``reasoning_content`` deltas that are real decoded
    tokens. Ignoring them undercounts the work done.

Usage: decode_rate_bench.py <port> <served-model-name> <tag> [max_tokens] [runs]

Prints per-run and aggregate figures. Run more than once: a single sample of a
decode rate is not a measurement, and a ~9% swing between identical runs has
been observed on this hardware.
"""

import json
import statistics
import sys
import time
import urllib.request

PROMPT = (
    "Explain, in detail and without lists, how a write-ahead log keeps a "
    "database consistent across an unclean shutdown. Cover the ordering "
    "guarantees, what fsync actually promises, and where the recovery scan "
    "starts."
)


def _run(port, model, max_tokens):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:%s/v1/chat/completions" % port,
        data=body, headers={"Content-Type": "application/json"},
    )

    start = time.time()
    ttft = None
    gaps = []
    last = None
    usage = {}
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except ValueError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            # reasoning_content deltas are decoded tokens too.
            if not (delta.get("content") or delta.get("reasoning_content")):
                continue
            now = time.time()
            if ttft is None:
                ttft = now - start
            else:
                gaps.append(now - last)
            last = now

    total = time.time() - start
    completion = int(usage.get("completion_tokens") or 0)
    return {
        "ttft": ttft or 0.0,
        "total": total,
        "completion_tokens": completion,
        "gaps": gaps,
        # Authoritative rate: real token count over the decode window.
        "rate": (completion - 1) / (total - (ttft or 0.0))
        if completion > 1 and total > (ttft or 0.0) else 0.0,
    }


def main():
    port, model, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 400
    runs = int(sys.argv[5]) if len(sys.argv) > 5 else 3

    rates = []
    for i in range(runs):
        r = _run(port, model, max_tokens)
        if r["completion_tokens"] <= 1:
            print("[%s] run %d: NO USAGE -- server did not return "
                  "completion_tokens; rate cannot be trusted" % (tag, i + 1))
            continue
        g = r["gaps"]
        rates.append(r["rate"])
        p50 = statistics.median(g) if g else 0.0
        # Where the worst stall lands matters more than its size. One big gap
        # early is warmup; evenly spaced ones are periodic (cache growth,
        # eviction); scattered ones are jitter. max_gap alone cannot tell them
        # apart.
        #
        # @d<n> is a DELTA index, NOT a token index. SSE deltas coalesce -- that
        # is the whole reason this script takes counts from usage rather than
        # from the stream -- so the nth delta is at or beyond the nth token.
        # Use it to see spacing and periodicity, never to claim an exact token
        # boundary.
        worst_i = max(range(len(g)), key=lambda k: g[k]) if g else -1
        stalls = sum(1 for x in g if p50 > 0 and x > 4 * p50)
        print("[%s] run %d: %.1f t/s  tokens=%d  ttft=%.3fs  "
              "p50_gap=%.4fs  mean_gap=%.4fs  max_gap=%.4fs @d%d  "
              "gaps>4xp50=%d"
              % (tag, i + 1, r["rate"], r["completion_tokens"], r["ttft"],
                 p50, statistics.fmean(g) if g else 0.0,
                 max(g) if g else 0.0, worst_i + 2, stalls))
        time.sleep(2.0)

    if rates:
        print("[%s] AGGREGATE: median %.1f t/s  min %.1f  max %.1f  (n=%d)"
              % (tag, statistics.median(rates), min(rates), max(rates),
                 len(rates)))
    return 0 if rates else 1


if __name__ == "__main__":
    sys.exit(main())
