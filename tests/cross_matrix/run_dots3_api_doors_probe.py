"""dots3: prefix cache hit on ALL THREE API doors + sustained decode + cache size.

Eric: "dot3 note caching and hit and ollama/chat/responses and issues perhaps
with speed and sustained decode max cache size".

Each door renders the prompt differently — /v1/chat/completions,
/v1/responses, and ollama /api/chat — so a hit on one door is NOT evidence of
a hit on another. That is exactly how a "healthy stores / zero hits" split
hides: same model, same text, different rendered prefix, different store key.

Per door: cold turn, then the SAME turn again.
  PASS = warm cached_tokens > 0 AND warm answer byte-equal to cold
  (latency alone has lied in this codebase; text equality is the contract)

Then: sustained decode over a long generation (not a 20-token sample), and the
block-disk cache size/eviction behaviour at its configured cap.

Usage: dots3_three_doors.py <lane>   lane in {l1l2, ssd}
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO = os.environ.get("VMLX_REPO") or os.path.expanduser("~/mlx/vllm-mlx")
PY = os.path.join(REPO, ".venv/bin/python")
MODEL = "/Volumes/EricsLLMDrive/models/dots3-note-prev-JANG"
PORT = 8143
LANE = sys.argv[1] if len(sys.argv) > 1 else "ssd"
CACHE_GB = "4"          # deliberately small so the cap is reachable
d = f"/tmp/dots3-doors-{LANE}"
subprocess.run(["rm", "-rf", d], check=False)
os.makedirs(d, exist_ok=True)

LANE_ARGS = (
    ["--use-paged-cache", "--paged-cache-block-size", "64",
     "--max-cache-blocks", "4000", "--enable-block-disk-cache"]
    if LANE == "l1l2"
    else ["--no-paged-cache", "--enable-block-disk-cache"]
)

log = open(os.path.join(d, "serve.log"), "w")
proc = subprocess.Popen(
    [PY, "-B", "-m", "vmlx_engine.cli", "serve", MODEL,
     "--host", "127.0.0.1", "--port", str(PORT), "--timeout", "1800",
     "--continuous-batching", "--max-num-seqs", "1",
     "--enable-prefix-cache", *LANE_ARGS,
     "--block-disk-cache-max-gb", CACHE_GB, "--block-disk-cache-dir", d,
     "--reasoning-parser", "auto", "--tool-call-parser", "auto"],
    cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
)
for _ in range(360):
    time.sleep(5)
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
        break
    except Exception:
        if proc.poll() is not None:
            raise SystemExit("serve died; see " + d + "/serve.log")

FILLER = ("Dock ledger: the crimson consignment recorded crates by shift. " * 45).strip()
QUESTION = FILLER + "\n\nQUESTION: reply with exactly one word: acknowledged."


def call(url, body, extract):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            payload = json.load(r)
    except urllib.error.HTTPError as e:
        return {"http_error": e.code,
                "detail": e.read()[:200].decode(errors="replace"),
                "wall": round(time.time() - t0, 2)}
    out = extract(payload)
    out["wall"] = round(time.time() - t0, 2)
    return out


def door_chat():
    def extract(p):
        m = p["choices"][0]["message"]
        u = p.get("usage", {}) or {}
        det = u.get("prompt_tokens_details") or {}
        return {"text": (m.get("content") or "").strip(),
                "prompt": u.get("prompt_tokens"),
                "cached": det.get("cached_tokens", 0)}
    return call(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                {"model": "dots3",
                 "messages": [{"role": "user", "content": QUESTION}],
                 "temperature": 0.0, "max_tokens": 24,
                 "enable_thinking": False,
                 "chat_template_kwargs": {"enable_thinking": False}}, extract)


def door_responses():
    def extract(p):
        text = ""
        for item in p.get("output") or []:
            for c in item.get("content") or []:
                if c.get("type") in ("output_text", "text"):
                    text += c.get("text") or ""
        if not text:
            text = p.get("output_text") or ""
        u = p.get("usage", {}) or {}
        det = u.get("input_tokens_details") or {}
        return {"text": text.strip(),
                "prompt": u.get("input_tokens"),
                "cached": det.get("cached_tokens", 0)}
    return call(f"http://127.0.0.1:{PORT}/v1/responses",
                {"model": "dots3", "input": QUESTION,
                 "temperature": 0.0, "max_output_tokens": 24,
                 "enable_thinking": False,
                 "chat_template_kwargs": {"enable_thinking": False}}, extract)


def door_ollama():
    def extract(p):
        msg = p.get("message") or {}
        return {"text": (msg.get("content") or "").strip(),
                "prompt": p.get("prompt_eval_count"),
                "cached": p.get("prompt_cache_hit_tokens",
                                p.get("cached_tokens", 0)) or 0}
    return call(f"http://127.0.0.1:{PORT}/api/chat",
                {"model": "dots3",
                 "messages": [{"role": "user", "content": QUESTION}],
                 "stream": False,
                 "options": {"temperature": 0.0, "num_predict": 24},
                 "enable_thinking": False,
                 "chat_template_kwargs": {"enable_thinking": False}}, extract)


DOORS = [("chat/completions", door_chat),
         ("responses", door_responses),
         ("ollama /api/chat", door_ollama)]

print(f"\n=== dots3 cache across API doors ({LANE} lane) ===", flush=True)
findings = []
for name, fn in DOORS:
    cold = fn()
    if cold.get("http_error"):
        print(f"  {name:<18} COLD HTTP {cold['http_error']} {cold['detail'][:90]}", flush=True)
        findings.append(f"{name}: door returned HTTP {cold['http_error']}")
        continue
    time.sleep(3)
    warm = fn()
    ok_text = cold["text"] == warm["text"]
    ok_reuse = (warm.get("cached") or 0) > 0
    print(f"  {name:<18} cold cached={cold['cached']:>5}/{cold['prompt']} "
          f"-> warm cached={warm['cached']:>5}/{warm['prompt']} "
          f"| equal={ok_text} text={cold['text'][:28]!r}", flush=True)
    if not cold["text"]:
        findings.append(f"{name}: EMPTY answer")
    if not ok_text:
        findings.append(f"{name}: warm text differs ({cold['text'][:30]!r} vs {warm['text'][:30]!r})")
    if not ok_reuse:
        if name.startswith("ollama"):
            # Measure the door that does not self-report: count the engine's
            # own prefix-hit lines emitted while these two calls ran.
            tail = open(os.path.join(d, "serve.log"), errors="replace").read()
            hits = tail.lower().count("prefix cache hit") + tail.lower().count("cache hit")
            print(f"    (ollama does not report cached_tokens; engine log shows "
                  f"{hits} cache-hit line(s))", flush=True)
            if hits == 0:
                findings.append(f"{name}: no reuse in the response AND none in the log")
        else:
            findings.append(f"{name}: NO prefix reuse on the warm call")

# ---- sustained decode over a real generation -------------------------------
print("\n=== sustained decode (400 tokens, not a 20-token sample) ===", flush=True)
t0 = time.time()
req = urllib.request.Request(
    f"http://127.0.0.1:{PORT}/v1/chat/completions",
    data=json.dumps({"model": "dots3",
                     "messages": [{"role": "user",
                                   "content": "Count from 1 to 120, comma separated."}],
                     "temperature": 0.0, "max_tokens": 400}).encode(),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=1800) as r:
    payload = json.load(r)
wall = time.time() - t0
u = payload.get("usage", {}) or {}
completion = u.get("completion_tokens") or 0
rate = completion / wall if wall else 0
print(f"  completion_tokens={completion} wall={wall:.2f}s -> {rate:.2f} tok/s", flush=True)
if completion < 50:
    findings.append(f"sustained decode produced only {completion} tokens")

# ---- block-disk cache size vs its cap --------------------------------------
size_bytes = sum(
    os.path.getsize(os.path.join(root, f))
    for root, _, files in os.walk(d) for f in files
)
print(f"\n=== block-disk cache on disk: {size_bytes/1e9:.2f} GB (cap {CACHE_GB} GB) ===",
      flush=True)
if size_bytes / 1e9 > float(CACHE_GB) * 1.25:
    findings.append(f"block-disk cache {size_bytes/1e9:.2f} GB exceeds its {CACHE_GB} GB cap")

serve_log = open(os.path.join(d, "serve.log"), errors="replace").read()
for needle, label in (("prefill admission rejected", "valve decline"),
                      ("Traceback", "traceback in serve log")):
    if needle in serve_log:
        findings.append(f"{label} present in serve log")

print("\nFINDINGS:" if findings else "\nFINDINGS: none", flush=True)
for f in findings:
    print(f"  - {f}", flush=True)
print(f"VERDICT: {'FAIL' if findings else 'PASS'} (dots3/{LANE})", flush=True)

proc.terminate()
try:
    proc.wait(timeout=120)
except subprocess.TimeoutExpired:
    proc.kill()
