#!/usr/bin/env python3
"""Growing-multiturn prefix-cache scaling row for one family.

Each turn appends a large block of fresh reference text plus a short question,
so the conversation grows by a controlled amount while the prefix stays reusable.
That isolates the thing being measured: whether prefix reuse HOLDS as context
grows, and how sustained prefill/decode behave once it does.

Emits one JSON object per turn (schema matches
docs/internal/benchmarks/dsv4-growing-multiturn-308k-2026-08-10.jsonl) so rows
from different families are directly comparable.

    python3 growing_multiturn_bench.py --port 8036 --model <served-name> \
        --turns 8 --grow-tokens 12000 --out row.jsonl

Report decode as BOTH p50-of-gaps and mean: stalls live in the tail, and a
MEAN that collapses while p50 stays flat means per-response fixed cost, not
slower decode. Counts come from usage.completion_tokens — never from counting
SSE chunks.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.request


def _post(url, body, timeout=3600):
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def _health(base):
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=30) as fh:
            h = json.load(fh)
    except Exception:
        return {}
    # These live under cache.totals / cache.scheduler_cache, NOT at the top of
    # cache. Reading them one level too high returned None for every field and
    # every row this script produced silently reported "no cache activity" —
    # measured against a run whose real numbers were cache_hits 304,
    # cache_hit_rate 0.66, tokens_saved 9703, l2_tokens_on_disk 66,206.
    cache = h.get("cache") or {}
    totals = cache.get("totals") or {}
    sched = cache.get("scheduler_cache") or {}
    return {
        "ram_tokens": totals.get("ram_tokens_cached"),
        "l1_resident_mb": totals.get("l1_resident_bytes_mb"),
        "l1_evictions": totals.get("l1_evictions"),
        "l1_indexed_tokens": totals.get("l1_indexed_tokens"),
        "l2_block_tokens": totals.get("l2_block_tokens_on_disk"),
        "l2_ssm_tokens": totals.get("l2_ssm_tokens_on_disk"),
        "l2_total_tokens": totals.get("l2_tokens_on_disk"),
        "cache_hits": sched.get("cache_hits"),
        "cache_hit_rate": sched.get("cache_hit_rate"),
        "tokens_saved": sched.get("tokens_saved"),
        "disk_hits": sched.get("disk_hits"),
    }


def _filler(words):
    # Deterministic and low-entropy so the model has nothing to latch onto, and
    # so the row is reproducible across runs.
    return " ".join(
        f"Reference {i}: topic {i % 97} weight {(i * 31) % 89}." for i in range(words)
    )


def run_turn(base, model, messages, max_tokens):
    """Stream one turn so per-token gaps are measurable."""
    body = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": messages,
    }
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    ttft = None
    gaps = []
    last = start
    usage = {}
    content = []
    stream_error: str | None = None
    with urllib.request.urlopen(req, timeout=3600) as fh:
        for raw in fh:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            # An SSE `error` chunk is how the server reports a failed stream —
            # it still arrives under a 200, because the response headers were
            # sent before generation began. Ignoring it made a killed turn
            # indistinguishable from an empty success: turn 4 of the 400k run
            # recorded ptok=0 ctok=0 ttft=None and read as "the model said
            # nothing", when the engine had actually raised
            # `Streaming exceeded 900.0s timeout` after ~15 minutes of prefill.
            if chunk.get("error"):
                stream_error = chunk["error"].get("message") or str(chunk["error"])
                break
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                # Reasoning models stream on reasoning_content — watching only
                # `content` reports TTFT=None and hides the whole decode span.
                piece = delta.get("content") or delta.get("reasoning_content")
                if not piece:
                    continue
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - start
                else:
                    gaps.append(now - last)
                last = now
                if delta.get("content"):
                    content.append(delta["content"])
    wall = time.perf_counter() - start
    ptd = usage.get("prompt_tokens_details") or {}
    ctok = int(usage.get("completion_tokens") or 0)
    decode_span = max(1e-9, wall - (ttft or 0))
    return {
        "error": stream_error,
        "ptok": int(usage.get("prompt_tokens") or 0),
        "ctok": ctok,
        "cached": int(ptd.get("cached_tokens") or 0),
        "detail": ptd.get("cache_detail"),
        "ttft": round(ttft, 2) if ttft else None,
        "decode_p50": round(1.0 / statistics.median(gaps), 1) if gaps else None,
        "decode_mean": round(ctok / decode_span, 1) if ctok else None,
        "max_gap_s": round(max(gaps), 2) if gaps else None,
        "wall": round(wall, 1),
        "content": "".join(content).strip(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--grow-words", type=int, default=1500)
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--out")
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    messages = []
    rows = []
    for turn in range(1, args.turns + 1):
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Reference block {turn}: {_filler(args.grow_words)}\n\n"
                    f"Turn {turn}: name one Nordic city not yet mentioned, one line."
                ),
            }
        )
        row = run_turn(base, args.model, messages, args.max_tokens)
        answer = row.pop("content")
        messages.append({"role": "assistant", "content": answer})
        fresh = max(0, row["ptok"] - row["cached"])
        row.update(
            turn=turn,
            fresh_tokens=fresh,
            prefill_pps=(
                round(fresh / row["ttft"], 1) if row["ttft"] and fresh else None
            ),
            health=_health(base),
        )
        rows.append(row)
        print(
            "T%-2d ptok=%-7s cached=%-7s fresh=%-6s ttft=%-6s pp/s=%-7s "
            "p50=%-6s mean=%-6s maxgap=%-6s wall=%s"
            % (
                turn, row["ptok"], row["cached"], fresh, row["ttft"],
                row["prefill_pps"], row["decode_p50"], row["decode_mean"],
                row["max_gap_s"], row["wall"],
            ),
            flush=True,
        )
        if row["error"]:
            # Loud, and on its own line: a turn that failed is not a data point,
            # and averaging it in as "0 tokens" understates the family.
            print(f"    !! TURN {turn} FAILED: {row['error']}", flush=True)
        if args.out:
            with open(args.out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")

    if rows:
        failed = [r for r in rows if r["error"]]
        # Summarise only the turns that actually produced a response. Folding a
        # failed turn's zeros into the trend reports a slowdown that never
        # happened, and drops the failure itself out of sight.
        ok = [r for r in rows if not r["error"]] or rows
        first, last = ok[0], ok[-1]
        if failed:
            print(f"\n{len(failed)} of {len(rows)} TURNS FAILED — excluded below:")
            for r in failed:
                print(f"  turn {r['turn']}: {r['error']} (after {r['wall']}s)")
        print(
            f"\ncontext {first['ptok']} -> {last['ptok']} tokens over {len(ok)} turns"
        )
        print(f"  reuse held      : {all(r['cached'] > 0 for r in ok[1:])}")
        print(f"  decode p50      : {first['decode_p50']} -> {last['decode_p50']} t/s")
        print(f"  decode mean     : {first['decode_mean']} -> {last['decode_mean']} t/s")
        print(f"  evictions (last): {last['health'].get('l1_evictions')}")
        print(f"  cache hits      : {last['health'].get('cache_hits')} "
              f"(rate {last['health'].get('cache_hit_rate')})")
        print(f"  tokens saved    : {last['health'].get('tokens_saved')}")


if __name__ == "__main__":
    sys.exit(main())
