#!/usr/bin/env python3
"""Control: with the prefix cache OFF, does the same prompt twice give the same answer?

This separates two very different claims:
  - the engine is nondeterministic run-to-run (then cold-vs-hit divergence says
    nothing about the cache), versus
  - the engine is deterministic and RESTORING A PREFIX is what changes answers.

Run against a server started with --disable-prefix-cache --disable-block-disk-cache.
Both turns must report cached_tokens = 0; any reuse voids the control.

Usage: determinism_control.py <port> <served-model-name> <tag>
"""

import hashlib
import json
import sys
import time
import urllib.request

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache_fidelity_multi_prompt import TOPICS, _turn  # noqa: E402


def main():
    port, model, tag = sys.argv[1], sys.argv[2], sys.argv[3]

    def sha(s):
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    differing, checked, voided = 0, 0, 0
    for name, body, tail in TOPICS:
        prompt = (body * 200) + tail
        a, ua = _turn(port, model, prompt)
        time.sleep(2.0)
        b, ub = _turn(port, model, prompt)
        ca = (ua.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        cb = (ub.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        if int(ca) or int(cb):
            print("  [%s] VOID — cache was consulted (%s/%s)" % (name, ca, cb))
            voided += 1
            continue
        checked += 1
        same = a == b
        if not same:
            differing += 1
        print("  [%s] %s  run1=%s run2=%s (both cold)"
              % (name, "SAME    " if same else "DIFFERENT", sha(a), sha(b)))

    print("[%s] %d/%d cold-vs-cold pairs DIFFER (%d voided)"
          % (tag, differing, checked, voided))
    if checked and differing == 0:
        print("[%s] engine is DETERMINISTIC with the cache off -> "
              "restoring a prefix is what changes the answer" % tag)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
