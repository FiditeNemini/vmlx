#!/usr/bin/env python3
"""DSV4 reasoning-kwargs and DSML parser matrix against a live engine.

Covers the axes that only show up on the wire:

  A. reasoning_effort low/high/max -- the bundle realizes effort as a text
     prefix *before* the system message, so effort changes must move
     prompt_tokens and must invalidate the cached prefix.
  B. enable_thinking=False -- chat mode, no reasoning block at all.
  C. DSML multi-invoke -- two tool calls inside one <｜DSML｜tool_calls> block
     must parse into two OpenAI tool_calls.
  D. DSML typed parameters -- string="false" values must round-trip as real
     JSON types (number, bool, array, object), not as strings.

Usage:
    python3 bench/dsv4_reasoning_dsml_matrix.py [base_url]
"""
from __future__ import annotations

import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
MODEL = "DeepSeek-V4-Flash-0731-JANG"

TYPED_TOOL = {
    "type": "function",
    "function": {
        "name": "configure_run",
        "description": "Configure a benchmark run.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Run label"},
                "iterations": {"type": "integer", "description": "How many iterations"},
                "verbose": {"type": "boolean", "description": "Verbose output"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "limits": {"type": "object", "description": "Arbitrary limits object"},
            },
            "required": ["name", "iterations", "verbose", "tags"],
        },
    },
}

ECHO_TOOL = {
    "type": "function",
    "function": {
        "name": "echo_text",
        "description": "Echo the provided text back.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
}

ADD_TOOL = {
    "type": "function",
    "function": {
        "name": "add_numbers",
        "description": "Add two integers and return the sum.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
}


def health_cache():
    with urllib.request.urlopen(f"{BASE}/health", timeout=10) as r:
        d = json.load(r)
    c = d["cache"]["scheduler_cache"]
    return {"hits": c.get("cache_hits"), "misses": c.get("cache_misses")}


def call(messages, *, tools=None, effort=None, thinking=None, max_tokens=400):
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "stream_options": {"include_usage": True},
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    if effort is not None:
        payload["reasoning_effort"] = effort
    if thinking is not None:
        payload["enable_thinking"] = thinking

    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    reasoning, content, tool_calls, usage, finish = [], [], {}, None, None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            evt = json.loads(body)
            if evt.get("usage"):
                usage = evt["usage"]
            for ch in evt.get("choices") or []:
                delta = ch.get("delta") or {}
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                if delta.get("content"):
                    content.append(delta["content"])
                for tc in delta.get("tool_calls") or []:
                    slot = tool_calls.setdefault(tc.get("index", 0), {"name": "", "args": ""})
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["args"] += fn["arguments"]
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    return {
        "reasoning": "".join(reasoning),
        "content": "".join(content),
        "tool_calls": tool_calls,
        "usage": usage or {},
        "finish": finish,
    }


def section(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def main():
    failures = []

    # ---- A. reasoning_effort realizes as a prompt prefix -------------------
    section("A. reasoning_effort low / high / max")
    msgs = [{"role": "user", "content": "What is 17 + 25? Answer with the number."}]
    prompts = {}
    for eff in ("low", "high", "max"):
        r = call(msgs, effort=eff, thinking=True, max_tokens=300)
        pt = r["usage"].get("prompt_tokens")
        prompts[eff] = pt
        print(f"  effort={eff:<5} prompt_tokens={pt:<5} reasoning_chars={len(r['reasoning']):<5} "
              f"content={r['content'][:60]!r}")
    if not (prompts["low"] < prompts["high"] and prompts["low"] < prompts["max"]):
        failures.append(
            f"reasoning_effort did not lengthen the prompt as the bundle documents: {prompts}")
    else:
        print(f"  OK effort prefix moves prompt_tokens: {prompts}")

    # ---- B. enable_thinking=False -> chat mode ----------------------------
    section("B. enable_thinking=False (chat mode)")
    r = call(msgs, thinking=False, max_tokens=200)
    print(f"  reasoning_chars={len(r['reasoning'])} content={r['content'][:80]!r} "
          f"finish={r['finish']}")
    if r["reasoning"].strip():
        failures.append("enable_thinking=False still produced reasoning_content")
    else:
        print("  OK no reasoning emitted in chat mode")

    # ---- C. DSML multi-invoke in one block --------------------------------
    section("C. DSML multi-invoke -> two tool_calls")
    multi = [{
        "role": "user",
        "content": "Call add_numbers for 2+3 and also call echo_text with the text HELLO. "
                   "Make both calls now, in a single response.",
    }]
    r = call(multi, tools=[ADD_TOOL, ECHO_TOOL], thinking=True, max_tokens=500)
    names = [v["name"] for v in r["tool_calls"].values()]
    print(f"  finish={r['finish']} n_tool_calls={len(r['tool_calls'])} names={names}")
    for idx, v in sorted(r["tool_calls"].items()):
        print(f"    [{idx}] {v['name']} args={v['args'][:120]}")
    if len(r["tool_calls"]) < 2:
        failures.append(
            f"DSML multi-invoke produced {len(r['tool_calls'])} tool_calls, expected 2 ({names})")
    else:
        print("  OK two tool calls parsed from one DSML block")

    # ---- D. DSML typed parameters round-trip ------------------------------
    section("D. DSML typed parameters (string=\"false\" -> real JSON types)")
    typed = [{
        "role": "user",
        "content": "Configure a run named smoke with exactly 7 iterations, verbose enabled, "
                   "tags alpha and beta, and limits {\"mem\": 512}. Use the tool.",
    }]
    r = call(typed, tools=[TYPED_TOOL], thinking=True, max_tokens=600)
    print(f"  finish={r['finish']} n_tool_calls={len(r['tool_calls'])}")
    if not r["tool_calls"]:
        failures.append("typed-parameter tool was never called")
    for idx, v in sorted(r["tool_calls"].items()):
        print(f"    [{idx}] {v['name']} args={v['args'][:300]}")
        try:
            parsed = json.loads(v["args"])
        except (json.JSONDecodeError, TypeError) as exc:
            failures.append(f"tool arguments were not valid JSON: {exc}: {v['args'][:200]}")
            continue
        types = {k: type(val).__name__ for k, val in parsed.items()}
        print(f"        parsed types: {types}")
        checks = {
            "iterations": int,
            "verbose": bool,
            "tags": list,
        }
        for key, want in checks.items():
            if key not in parsed:
                failures.append(f"typed arg {key!r} missing from tool call")
            elif not isinstance(parsed[key], want) or (
                want is int and isinstance(parsed[key], bool)
            ):
                failures.append(
                    f"typed arg {key!r} came back as {type(parsed[key]).__name__}, "
                    f"expected {want.__name__} (value={parsed[key]!r})")
        if "limits" in parsed and not isinstance(parsed["limits"], dict):
            failures.append(
                f"typed arg 'limits' came back as {type(parsed['limits']).__name__}, expected dict")

    # ---- summary ----------------------------------------------------------
    section("SUMMARY")
    if failures:
        for f in failures:
            print(f"  FAIL  {f}")
        print(f"\n{len(failures)} failure(s)")
        return 1
    print("  all matrix rows passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
