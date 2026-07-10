#!/usr/bin/env python3
"""Hy3-JANG_2L live exhaustive gate against the vmlx server."""
import json, sys, time, urllib.request, threading, re

BASE = "http://127.0.0.1:8130/v1"
MODEL = "Hy3-JANG_2L"
FAIL = []


def post(path, body, stream=False, timeout=1800):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=timeout)
    if not stream:
        return json.loads(r.read())
    return r


def chat(messages, **kw):
    b = {"model": MODEL, "messages": messages, "temperature": 0.0,
         "max_tokens": 128, "stream": False}
    b.update(kw)
    t = time.time()
    r = post("/chat/completions", b)
    ch = r["choices"][0]
    msg = ch["message"]
    return {
        "text": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
        "tools": msg.get("tool_calls") or [],
        "finish": ch.get("finish_reason"),
        "usage": r.get("usage", {}),
        "dt": time.time() - t,
    }


def chat_stream(messages, **kw):
    b = {"model": MODEL, "messages": messages, "temperature": 0.0,
         "max_tokens": 128, "stream": True}
    b.update(kw)
    r = post("/chat/completions", b, stream=True)
    content, reasoning, n_c, n_r, finish = "", "", 0, 0, None
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        choices = json.loads(data).get("choices") or []
        if not choices:          # usage-only / keepalive chunk
            continue
        d = choices[0]
        delta = d.get("delta", {})
        if delta.get("content"):
            content += delta["content"]; n_c += 1
        if delta.get("reasoning_content"):
            reasoning += delta["reasoning_content"]; n_r += 1
        if d.get("finish_reason"):
            finish = d["finish_reason"]
    return {"text": content, "reasoning": reasoning,
            "n_content_deltas": n_c, "n_reasoning_deltas": n_r, "finish": finish}


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAIL.append(f"{name}: {detail}")


U = lambda s: [{"role": "user", "content": s}]

print("=" * 78)
print("A. GREEDY DETERMINISM (3x same request, byte-identical)")
print("=" * 78)
PQ = "List the first 8 prime numbers and explain why 1 is not prime."
cold0 = chat(U(PQ), max_tokens=200)          # cold miss: seeds the prefix cache
outs = [chat(U(PQ), max_tokens=200) for _ in range(3)]
check("3x greedy byte-identical (warm)",
      outs[0]["text"] == outs[1]["text"] == outs[2]["text"],
      f"lens={[len(o['text']) for o in outs]}")
# Cold-vs-warm on an MoE is NOT expected to be byte-equal: a prefix hit
# re-runs only the tail of the prompt, which lands on a different MoE kernel
# (M=1 vs M=N). The ~1e-3 hidden-state delta tips near-ties in the top-8
# sigmoid router (measured: up to 12/79 layers swap an expert), which is an
# O(1) logit change. Both answers are correct; only bytes differ.
same_cw = cold0["text"] == outs[0]["text"]
print(f"    cold-vs-warm byte-equal: {same_cw}  (expected False on 2-bit MoE; "
      f"cold={len(cold0['text'])}ch warm={len(outs[0]['text'])}ch)")
check("cold output is coherent + correct", "2, 3, 5, 7, 11, 13, 17, 19" in cold0["text"],
      cold0["text"][:80])
for i, o in enumerate(outs):
    print(f"    run{i}: {o['usage'].get('completion_tokens')} tok in {o['dt']:.2f}s "
          f"= {o['usage'].get('completion_tokens',0)/o['dt']:.2f} tok/s")
print("  --- FULL OUTPUT run0 ---")
print(outs[0]["text"])
print("  --- END ---")

print()
print("=" * 78)
print("B. PREFIX CACHE: cold == warm (identical prompt re-issued)")
print("=" * 78)
long_prefix = ("You are a precise assistant. Here is a document.\n\n" +
               "\n".join(f"Fact {i}: the code for item {i} is {i*37 % 1000:03d}."
                         for i in range(60)))
q = long_prefix + "\n\nWhat is the code for item 47? Answer with just the number."
cold = chat(U(q), max_tokens=40)
warm = chat(U(q), max_tokens=40)
check("cold == warm byte-identical", cold["text"] == warm["text"],
      f"cold={cold['text']!r} warm={warm['text']!r}")
check("needle recall (item 47 -> 739)", "739" in cold["text"], repr(cold["text"]))
print(f"    cold {cold['dt']:.2f}s / warm {warm['dt']:.2f}s "
      f"(speedup {cold['dt']/max(warm['dt'],1e-6):.2f}x)")

print()
print("=" * 78)
print("C. EXHAUSTIVE MULTITURN (full read, 5 turns, continuation + recall)")
print("=" * 78)
convo = []
turns = [
    "My name is Eric and my favorite number is 47. Write a Python function `fib(n)` returning the nth Fibonacci number iteratively.",
    "Now make it handle negative n by raising ValueError. Show the full function.",
    "What is 17 * 23? Show the arithmetic.",
    "What was my favorite number, and what is it times the answer to the previous question?",
    "Summarize everything we discussed in exactly 3 bullet points.",
]
for i, t in enumerate(turns):
    convo.append({"role": "user", "content": t})
    o = chat(convo, max_tokens=400)
    convo.append({"role": "assistant", "content": o["text"]})
    print(f"\n  ---- TURN {i+1} ({o['usage'].get('completion_tokens')} tok, "
          f"{o['finish']}) ----\n  Q: {t}\n  A: {o['text']}")
    check(f"turn{i+1} non-empty + clean finish",
          len(o["text"].strip()) > 0 and o["finish"] in ("stop", "length"), o["finish"])
    check(f"turn{i+1} no role-marker leak",
          "hy_User" not in o["text"] and "hy_Assistant" not in o["text"])
last = convo[-1]["content"]
t3 = convo[-3]["content"]
check("turn4 recalls 47", "47" in t3)
check("turn4 computes 47*391=18377", "18377" in t3 or "18,377" in t3, t3[-200:])

print()
print("=" * 78)
print("D. REASONING RAILS (no_think | low | high) + STREAMING DELTAS")
print("=" * 78)
for effort in [None, "low", "high"]:
    kw = {} if effort is None else {"reasoning_effort": effort}
    s = chat_stream(U("A farmer has 17 sheep. All but 9 die. How many are left?"),
                    max_tokens=2048, **kw)
    label = effort or "default(no_think)"
    print(f"\n  ---- reasoning_effort={label} ----")
    print(f"  content deltas={s['n_content_deltas']} reasoning deltas={s['n_reasoning_deltas']} finish={s['finish']}")
    if s["reasoning"]:
        print(f"  REASONING ({len(s['reasoning'])} ch): {s['reasoning'][:600]}")
    print(f"  CONTENT: {s['text']}")
    check(f"{label}: content deltas streamed", s["n_content_deltas"] > 1)
    check(f"{label}: no <think> leak in content", "<think>" not in s["text"] and "</think>" not in s["text"])
    check(f"{label}: answer is 9", "9" in s["text"])
    if effort == "high":
        check("high: reasoning deltas present", s["n_reasoning_deltas"] > 0,
              f"n={s['n_reasoning_deltas']}")
    if effort is None:
        check("no_think: no reasoning content", s["n_reasoning_deltas"] == 0,
              f"n={s['n_reasoning_deltas']}")

print()
print("=" * 78)
print("E. TOOL CALLS (hunyuan XML parser)")
print("=" * 78)
tools = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "enum": ["c", "f"]}},
        "required": ["city"]}}}]
o = chat(U("What's the weather in Tokyo in celsius? Use the tool."),
         tools=tools, max_tokens=256)
print(f"  finish={o['finish']} content={o['text']!r}")
print(f"  tool_calls={json.dumps(o['tools'], indent=2)}")
check("tool_calls parsed", len(o["tools"]) == 1, f"n={len(o['tools'])}")
if o["tools"]:
    fn = o["tools"][0]["function"]
    check("tool name == get_weather", fn["name"] == "get_weather", fn["name"])
    args = json.loads(fn["arguments"])
    check("tool arg city == Tokyo", args.get("city", "").lower() == "tokyo", str(args))
    check("finish_reason == tool_calls", o["finish"] == "tool_calls", str(o["finish"]))
    # tool response round trip
    convo2 = U("What's the weather in Tokyo in celsius? Use the tool.")
    convo2.append({"role": "assistant", "content": o["text"] or None, "tool_calls": o["tools"]})
    convo2.append({"role": "tool", "tool_call_id": o["tools"][0].get("id", "0"),
                   "content": '{"temp": 18, "unit": "c", "sky": "clear"}'})
    o2 = chat(convo2, tools=tools, max_tokens=200)
    print(f"  --- after tool response ---\n  {o2['text']}")
    check("tool round-trip mentions 18", "18" in o2["text"], o2["text"][:200])

print()
print("=" * 78)
print("F. CONTINUOUS BATCHING (mns=2): concurrent == serial, byte-equal")
print("=" * 78)
prompts = ["Explain what a hash table is in 3 sentences.",
           "Name the 4 inner planets in order from the sun."]
# Warm the prefix cache FIRST so serial and concurrent runs are on the same
# footing — otherwise this test would just re-measure the cold/warm MoE
# routing split from section A and blame continuous batching for it.
for p in prompts:
    chat(U(p), max_tokens=150)
serial = [chat(U(p), max_tokens=150) for p in prompts]
res = {}
def worker(i, p):
    res[i] = chat(U(p), max_tokens=150)
th = [threading.Thread(target=worker, args=(i, p)) for i, p in enumerate(prompts)]
t0 = time.time()
[t.start() for t in th]; [t.join() for t in th]
dt = time.time() - t0
for i in range(2):
    ok = serial[i]["text"] == res[i]["text"]
    check(f"cobatch req{i} byte-equal to serial", ok)
    if not ok:
        print(f"    serial: {serial[i]['text'][:300]!r}")
        print(f"    concur: {res[i]['text'][:300]!r}")
print(f"    serial total {sum(o['dt'] for o in serial):.2f}s vs concurrent {dt:.2f}s")
print(f"  --- concurrent req0 ---\n{res[0]['text']}")
print(f"  --- concurrent req1 ---\n{res[1]['text']}")
# MTP re-engage at batch 1 (should stay blocked, but must not crash)
after = chat(U(prompts[0]), max_tokens=150)
check("batch-1 after cobatch still byte-equal", after["text"] == serial[0]["text"])

print()
print("=" * 78)
print("G. LONG STOCHASTIC GENERATION (temp 0.9, >2.5K tokens, loop check)")
print("=" * 78)
o = chat(U("Write a detailed technical essay about the history and design of "
           "memory hierarchies in computer architecture, from magnetic core "
           "memory through modern HBM. Be thorough and specific."),
         temperature=0.9, top_p=0.95, max_tokens=3000, seed=1234)
txt = o["text"]
ntok = o["usage"].get("completion_tokens", 0)
print(f"  {ntok} tokens in {o['dt']:.1f}s = {ntok/o['dt']:.2f} tok/s, finish={o['finish']}")
# Only prose lines: markdown separators ("---") and short headers legitimately
# repeat and are not evidence of a degeneration loop.
lines = [l.strip() for l in txt.split("\n") if len(l.strip()) >= 25]
dupes = len(lines) - len(set(lines))
# n-gram loop detector: any 12-word window repeated >3x
words = re.findall(r"\w+", txt.lower())
grams = {}
for i in range(len(words) - 12):
    g = " ".join(words[i:i+12])
    grams[g] = grams.get(g, 0) + 1
worst = max(grams.values()) if grams else 0
check("generated >2500 tokens", ntok > 2500, f"n={ntok}")
check("no 12-gram repeated >3x (loop)", worst <= 3, f"max_repeat={worst}")
check("duplicate-line ratio < 5%", dupes <= max(2, len(lines) * 0.05),
      f"{dupes}/{len(lines)}")
check("no role-marker leak", "hy_User" not in txt and "hy_Assistant" not in txt)
print(f"  --- first 900 chars ---\n{txt[:900]}")
print(f"  --- last 900 chars ---\n{txt[-900:]}")

print()
print("=" * 78)
print(f"GATE RESULT: {'ALL PASS' if not FAIL else f'{len(FAIL)} FAILURES'}")
for f in FAIL:
    print(f"  FAIL: {f}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
