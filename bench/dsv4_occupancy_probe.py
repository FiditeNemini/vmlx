"""Is allocated_blocks 'pinned right now' or 'holding cached content'?

Polls /health while a long generation is in flight. If allocated_blocks rises
during the request and falls back to ~1 afterwards while total_tokens_cached
stays high, then 'allocated' means actively pinned and the idle reading is
semantics, not drift.
"""
import json
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "DeepSeek-V4-Flash-0731-JANG"

SYSTEM = "You are a careful assistant. " + " ".join(
    f"Fact {i}: item {i} is catalogued under shelf {i % 17}." for i in range(1, 300)
)

samples = []
stop = threading.Event()


def poll():
    while not stop.is_set():
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=5) as r:
                sc = json.load(r)["cache"]["scheduler_cache"]
            samples.append({
                "alloc": sc.get("allocated_blocks"),
                "free": sc.get("free_blocks"),
                "util": sc.get("utilization"),
                "tokens": sc.get("total_tokens_cached"),
                "active": sc.get("active_requests"),
            })
        except Exception:
            pass
        time.sleep(0.4)


def snapshot(label):
    with urllib.request.urlopen(f"{BASE}/health", timeout=10) as r:
        sc = json.load(r)["cache"]["scheduler_cache"]
    print("%-8s alloc=%-5s free=%-6s util=%-8s tokens=%-6s" % (
        label, sc.get("allocated_blocks"), sc.get("free_blocks"),
        sc.get("utilization"), sc.get("total_tokens_cached")))


snapshot("IDLE")

t = threading.Thread(target=poll, daemon=True)
t.start()

body = {
    "model": MODEL, "stream": False, "max_tokens": 220,
    "enable_thinking": True,
    "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "List shelf numbers for items 5, 40 and 120."},
    ],
}
req = urllib.request.Request(f"{BASE}/v1/chat/completions",
                             data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=900) as r:
    d = json.load(r)
stop.set()
time.sleep(0.6)

usage = d.get("usage") or {}
print("request prompt_tokens=%s completion_tokens=%s" % (
    usage.get("prompt_tokens"), usage.get("completion_tokens")))

if samples:
    peak = max(samples, key=lambda s: (s["alloc"] or 0))
    peak_tokens = max(s["tokens"] or 0 for s in samples)
    print("PEAK during request: alloc=%s free=%s util=%s tokens=%s" % (
        peak["alloc"], peak["free"], peak["util"], peak_tokens))
    print("samples collected: %d" % len(samples))

time.sleep(1.0)
snapshot("AFTER")
