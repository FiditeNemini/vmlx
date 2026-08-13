#!/usr/bin/env python3
"""Prove a prefix-cache HIT reproduces a cold prefill, byte for byte.

The contract this checks: asking the same question twice must give the same
answer. A cache hit that returns different text than the cold prefill is a
correctness bug, not a performance trade-off — the second user sees a different
model.

Usage:
    python3 scripts/cache_fidelity_check.py <port> <served-model-name> [tag]

Exit status: 0 when cold == hit with real reuse, 1 when they diverge, and 2 when
no reuse happened at all — that last case compared two cold prefills and proves
nothing, so it must not read as a pass.

WHY THE CALLER MUST SET UP THE SERVER CAREFULLY
-----------------------------------------------
This script cannot tell a cold prefill from a restore on its own, so the
caller has to guarantee turn 1 is genuinely cold:

  * `--no-paged-cache` does NOT disable caching. It drops only the paged RAM
    tier; the SSD/L2 prefix backend keeps serving restores
    ("block disk-only prefix backend ... payloads restore transiently from
    SSD"). A run with it still hits cache.
  * A fresh `--block-disk-cache-dir` / `--disk-cache-dir` is necessary but NOT
    sufficient — the server process also holds RAM state, so a second
    measurement against the SAME server has a warm turn 1.

So: start a FRESH server with FRESH cache dirs for every measurement, e.g.

    D=$(mktemp -d)
    vmlx-engine serve <bundle> --port 8099 \
        --block-disk-cache-dir "$D" --disk-cache-dir "$D"

The script prints both TTFTs. **Sanity-check the cold one**: a "cold" TTFT that
matches the warm TTFT means turn 1 was a cache hit and the comparison is void.
That mistake inverted a diagnosis twice before this check existed.

READING THE RESULT
------------------
Reasoning models emit `reasoning_content`, so both streams are concatenated;
a content-only comparison silently passes when the divergence is in the
reasoning. Equal `completion_tokens` with different text means differing
logits, not truncation. The common-prefix length says where they split.
"""

import hashlib
import json
import sys
import time
import urllib.request

PROMPT_BODY = "Consensus protocols must tolerate partial failure. " * 200
PROMPT_TAIL = " List exactly five numbered facts about quorums."


def _turn(port: str, model: str, max_tokens: int):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT_BODY + PROMPT_TAIL}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:%s/v1/chat/completions" % port,
        body,
        {"Content-Type": "application/json"},
    )
    text = ""
    usage = None
    started = time.time()
    ttft = None
    for raw in urllib.request.urlopen(req, timeout=900):
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        chunk = json.loads(payload)
        if chunk.get("usage"):
            usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        # Reasoning families put the divergence here, not in `content`.
        piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
        if piece:
            text += piece
            if ttft is None:
                ttft = time.time() - started
    return text, usage or {}, ttft


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    port, model = sys.argv[1], sys.argv[2]
    tag = sys.argv[3] if len(sys.argv) > 3 else model.split("/")[-1]
    max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 260

    cold, cold_usage, cold_ttft = _turn(port, model, max_tokens)
    time.sleep(3)  # let the cold turn's cache write-back settle
    hit, hit_usage, hit_ttft = _turn(port, model, max_tokens)

    shared = 0
    while shared < min(len(cold), len(hit)) and cold[shared] == hit[shared]:
        shared += 1

    def _sha(s: str) -> str:
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    print("[%s] COLD ttft=%.3fs len=%d prompt=%s tokens=%s sha=%s"
          % (tag, cold_ttft or -1, len(cold), cold_usage.get("prompt_tokens"),
             cold_usage.get("completion_tokens"), _sha(cold)))
    print("[%s] HIT  ttft=%.3fs len=%d prompt=%s tokens=%s sha=%s cached=%s"
          % (tag, hit_ttft or -1, len(hit), hit_usage.get("prompt_tokens"),
             hit_usage.get("completion_tokens"), _sha(hit),
             (hit_usage.get("prompt_tokens_details") or {}).get("cached_tokens")))

    cached = (hit_usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    reused = int(cached or 0)

    # The re-fed tail: how many trailing positions the warm turn recomputes on
    # top of the restored prefix. This is the quantity the divergence is
    # suspected to track -- the warm turn computes these as a small forward on a
    # restored base while the cold turn computes them inside one large prefill,
    # and FP reductions are not shape-invariant. Reporting it on every row is
    # what makes that hypothesis checkable instead of anecdotal.
    hit_prompt = hit_usage.get("prompt_tokens")
    if isinstance(hit_prompt, int) and reused > 0:
        print("[%s] re-fed tail = %d token(s) (prompt %d - cached %d)"
              % (tag, hit_prompt - reused, hit_prompt, reused))

    if reused <= 0:
        # Two cold prefills prove determinism, not restore fidelity. Reporting
        # PASS here would be a vacuous pass: Falcon-H1R produced identical text
        # on both turns while its log showed ZERO cache hits, because a
        # hybrid/SSM family can decline reuse entirely.
        print("[%s] INCONCLUSIVE — no prefix reuse occurred (cached_tokens=%r), "
              "so this compared two cold prefills and proves nothing about the "
              "cache. Check the engine log for a hit line." % (tag, cached))
        return 2

    if cold == hit:
        print("[%s] PASS — cache hit (%d tokens reused) reproduces the cold "
              "prefill" % (tag, reused))
        return 0

    print("[%s] FAIL — cache hit ANSWERS DIFFERENTLY (diverges at char %d)"
          % (tag, shared))
    print("  cold: %r" % cold[shared:shared + 90])
    print("  hit : %r" % hit[shared:shared + 90])
    if cold_usage.get("completion_tokens") == hit_usage.get("completion_tokens"):
        print("  equal completion_tokens -> differing logits, not truncation")
    if cold_ttft and hit_ttft and cold_ttft < hit_ttft * 2:
        print("  WARNING: cold ttft is not much slower than the hit — turn 1 may "
              "have been a cache hit; re-run with a FRESH server and FRESH "
              "--block-disk-cache-dir/--disk-cache-dir")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
