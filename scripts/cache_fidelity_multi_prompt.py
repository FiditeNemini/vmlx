#!/usr/bin/env python3
"""Is a family's cold-vs-hit fidelity a stable property, or an artifact of ONE prompt?

The family map classifies each family EXACT or DIVERGE from a single prompt.
But margin probing showed that families classified EXACT (Nanbeige, Laguna) also
emit exact top1/top2 ties, i.e. positions where an FP delta decides the token.
If so, "EXACT" may only mean the tail-recompute perturbation did not happen to
land on a near-tie in that particular generation -- and the whole map would be
misleading rather than a property of the architecture.

This runs N distinct prompts through the same server. Each prompt is cold on
first use (distinct leading text, so no shared prefix to hit) and warm on the
immediate repeat. Any single divergence proves the family is NOT reliably exact.

Usage: multi_prompt_fidelity.py <port> <served-model-name> <tag> [n_prompts]
"""

import hashlib
import json
import sys
import time
import urllib.request

TOPICS = [
    ("quorum", "Consensus protocols must tolerate partial failure. ",
     " List exactly five numbered facts about quorums."),
    ("vacuum", "Thermal transport in porous media resists steady analysis. ",
     " List exactly five numbered facts about insulation."),
    ("harbour", "Tidal harbours silt predictably under seasonal flow. ",
     " List exactly five numbered facts about dredging."),
    ("lattice", "Crystal lattices deform under sustained shear stress. ",
     " List exactly five numbered facts about dislocations."),
    ("plumage", "Migratory plumage changes track daylight more than heat. ",
     " List exactly five numbered facts about moulting."),
    ("ledger", "Double-entry ledgers survive because errors cancel visibly. ",
     " List exactly five numbered facts about reconciliation."),
]


def _turn(port, model, prompt):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:%s/v1/chat/completions" % port,
        data=body, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        data = json.loads(resp.read())
    choice = data["choices"][0]["message"]
    text = (choice.get("reasoning_content") or "") + (choice.get("content") or "")
    usage = data.get("usage") or {}
    return text, usage


def main():
    port, model, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else len(TOPICS)

    def sha(s):
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    diverged, checked, skipped = [], 0, 0
    for name, body, tail in TOPICS[:n]:
        prompt = (body * 200) + tail
        cold, cold_u = _turn(port, model, prompt)
        time.sleep(2.5)  # settle: turn N's write-back must not collide with N+1
        hit, hit_u = _turn(port, model, prompt)
        cached = (hit_u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        if int(cached) <= 0:
            print("  [%s] SKIP — no reuse (cached=%s), proves nothing" % (name, cached))
            skipped += 1
            continue
        checked += 1
        same = cold == hit
        if not same:
            i = 0
            while i < min(len(cold), len(hit)) and cold[i] == hit[i]:
                i += 1
            diverged.append((name, i))
        print("  [%s] %s cold=%s hit=%s cached=%s%s"
              % (name, "OK  " if same else "DIVERGE", sha(cold), sha(hit), cached,
                 "" if same else "  (splits at char %d)" % i))

    print("[%s] %d/%d prompts diverged (%d checked, %d skipped)"
          % (tag, len(diverged), checked, checked, skipped))
    if diverged:
        print("[%s] NOT reliably exact — divergence is prompt-dependent" % tag)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
