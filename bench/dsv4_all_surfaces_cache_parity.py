#!/usr/bin/env python3
"""DSV4 all-API-path sweep with cross-surface prefix-cache parity.

Sends the *same logical conversation* (identical system, tools and user turn)
through every gateway surface in sequence and records, per surface:

  * reasoning separated from content
  * tool call name and arguments
  * terminal event / finish reason
  * scheduler cache delta and any reported cached_tokens

The first surface populates the prefix cache. If the later surfaces render the
same conversation identically, they must hit those blocks. A surface that takes
only misses is rendering a different prefix, which means blocks written by one
API path are unusable by another -- the cross-surface parity risk.

Usage:
    python3 bench/dsv4_all_surfaces_cache_parity.py [base_url]
"""
from __future__ import annotations

import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
MODEL = "DeepSeek-V4-Flash-0731-JANG"

SYSTEM = "You are a precise assistant. Use tools when they help."
USER = ("Look up the build status for project atlas-7 and report it. "
        "Use the status tool.")

OPENAI_TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_build_status",
        "description": "Return the build status for a project.",
        "parameters": {
            "type": "object",
            "properties": {"project": {"type": "string"}},
            "required": ["project"],
        },
    },
}]

ANTHROPIC_TOOLS = [{
    "name": "get_build_status",
    "description": "Return the build status for a project.",
    "input_schema": {
        "type": "object",
        "properties": {"project": {"type": "string"}},
        "required": ["project"],
    },
}]


def post(path, body, timeout=900):
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def cache_stats():
    with urllib.request.urlopen(f"{BASE}/health", timeout=15) as r:
        d = json.load(r)
    c = d["cache"]["scheduler_cache"]
    return {"hits": c.get("cache_hits", 0), "misses": c.get("cache_misses", 0),
            "saved": c.get("tokens_saved", 0)}


def delta(before, after):
    return {k: after[k] - before[k] for k in before}


def summarize(name, *, reasoning, content, tool_name, tool_args, finish,
              cached_tokens, cache_delta):
    print(f"\n--- {name} ---")
    print(f"  reasoning_chars = {len(reasoning or '')}")
    print(f"  content_chars   = {len(content or '')}  {(content or '')[:70]!r}")
    print(f"  tool_call       = {tool_name} {str(tool_args)[:90]}")
    print(f"  finish          = {finish}")
    print(f"  cached_tokens   = {cached_tokens}")
    print(f"  cache delta     = hits +{cache_delta['hits']} "
          f"misses +{cache_delta['misses']} saved +{cache_delta['saved']}")
    return {"name": name, "reasoning": len(reasoning or ""),
            "content": len(content or ""), "tool": tool_name,
            "finish": finish, "cached_tokens": cached_tokens,
            "delta": cache_delta}


def chat_completions():
    before = cache_stats()
    d = post("/v1/chat/completions", {
        "model": MODEL, "stream": False, "max_tokens": 400,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": USER}],
        "tools": OPENAI_TOOLS, "tool_choice": "auto",
    })
    after = cache_stats()
    ch = d["choices"][0]
    m = ch["message"]
    tcs = m.get("tool_calls") or []
    usage = d.get("usage") or {}
    return summarize(
        "chat.completions (/v1/chat/completions)",
        reasoning=m.get("reasoning_content"), content=m.get("content"),
        tool_name=(tcs[0]["function"]["name"] if tcs else None),
        tool_args=(tcs[0]["function"]["arguments"] if tcs else None),
        finish=ch.get("finish_reason"),
        cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        cache_delta=delta(before, after))


def responses():
    before = cache_stats()
    d = post("/v1/responses", {
        "model": MODEL, "stream": False, "max_output_tokens": 400,
        "instructions": SYSTEM,
        "input": [{"role": "user", "content": USER}],
        "tools": [{"type": "function", "name": t["function"]["name"],
                   "description": t["function"]["description"],
                   "parameters": t["function"]["parameters"]}
                  for t in OPENAI_TOOLS],
        "tool_choice": "auto",
    })
    after = cache_stats()
    reasoning, content, tool_name, tool_args = "", "", None, None
    for item in d.get("output") or []:
        itype = item.get("type")
        if itype == "reasoning":
            for c in item.get("summary") or item.get("content") or []:
                reasoning += (c.get("text") or "")
        elif itype == "message":
            for c in item.get("content") or []:
                content += (c.get("text") or "")
        elif itype == "function_call":
            tool_name = item.get("name")
            tool_args = item.get("arguments")
    usage = d.get("usage") or {}
    return summarize(
        "responses (/v1/responses)",
        reasoning=reasoning, content=content, tool_name=tool_name,
        tool_args=tool_args, finish=d.get("status"),
        cached_tokens=(usage.get("input_tokens_details") or {}).get("cached_tokens"),
        cache_delta=delta(before, after))


def anthropic():
    before = cache_stats()
    d = post("/v1/messages", {
        "model": MODEL, "max_tokens": 400, "system": SYSTEM,
        "messages": [{"role": "user", "content": USER}],
        "tools": ANTHROPIC_TOOLS,
    })
    after = cache_stats()
    reasoning, content, tool_name, tool_args = "", "", None, None
    for block in d.get("content") or []:
        btype = block.get("type")
        if btype == "thinking":
            reasoning += block.get("thinking") or ""
        elif btype == "text":
            content += block.get("text") or ""
        elif btype == "tool_use":
            tool_name = block.get("name")
            tool_args = block.get("input")
    usage = d.get("usage") or {}
    return summarize(
        "anthropic (/v1/messages)",
        reasoning=reasoning, content=content, tool_name=tool_name,
        tool_args=tool_args, finish=d.get("stop_reason"),
        cached_tokens=usage.get("cache_read_input_tokens"),
        cache_delta=delta(before, after))


def ollama():
    before = cache_stats()
    d = post("/api/chat", {
        "model": MODEL, "stream": False,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": USER}],
        "tools": OPENAI_TOOLS,
    })
    after = cache_stats()
    m = d.get("message") or {}
    tcs = m.get("tool_calls") or []
    tool_name = tool_args = None
    if tcs:
        fn = tcs[0].get("function") or {}
        tool_name, tool_args = fn.get("name"), fn.get("arguments")
    return summarize(
        "ollama (/api/chat)",
        reasoning=m.get("thinking") or m.get("reasoning_content"),
        content=m.get("content"), tool_name=tool_name, tool_args=tool_args,
        finish=d.get("done_reason") or d.get("done"),
        cached_tokens=d.get("prompt_eval_count"),
        cache_delta=delta(before, after))


def main():
    print(f"base={BASE} model={MODEL}")
    print("Same system + tools + user turn through every surface, in order.")
    results = []
    for fn in (chat_completions, responses, anthropic, ollama):
        try:
            results.append(fn())
        except Exception as exc:  # noqa: BLE001
            print(f"\n--- {fn.__name__} FAILED: {type(exc).__name__}: {exc}")
            results.append({"name": fn.__name__, "delta": None})

    print("\n" + "=" * 70)
    print("CROSS-SURFACE PREFIX PARITY")
    print("=" * 70)
    first = results[0]
    if not first.get("delta"):
        print("  first surface failed; parity undetermined")
        return 1
    for r in results[1:]:
        d = r.get("delta")
        if d is None:
            print(f"  {r['name']}: FAILED, parity undetermined")
            continue
        verdict = ("REUSED earlier blocks" if d["hits"] > 0
                   else "NO reuse -- renders a different prefix")
        print(f"  {r['name']}: hits +{d['hits']} misses +{d['misses']} -> {verdict}")

    print("\nTool-call agreement across surfaces:")
    for r in results:
        print(f"  {r.get('name')}: tool={r.get('tool')} finish={r.get('finish')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
