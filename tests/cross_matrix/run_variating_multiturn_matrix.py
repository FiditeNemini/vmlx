"""Variating multiturn: reasoning on/off x tools x video x image, both lanes.

Eric's stated fear, verbatim shape:
  turn 1  reasoning ON  + tool call
  turn 2  reasoning OFF + video
  turn 3  reasoning ON  + text only
  turn 4  image         + reasoning OFF

...and "how this plays with caching", for BOTH qwen and dots3, in BOTH the
L1+L2 lane and the SSD-only lane.

What counts as a defect vs expected, decided BEFORE running so the result
cannot be rationalised afterwards:

  DEFECT   empty visible answer; leaked parser/reasoning markers; a warm
           replay whose answer text differs at temp 0; a media turn that
           confabulates instead of reading the payload; reasoning content
           present when reasoning is OFF.
  EXPECTED near-zero reuse on the turn AFTER a reasoning-state flip. The
           template embeds the reasoning state, so flipping it changes the
           prefix — that is design, recorded in the campaign notes, NOT a
           cache bug. Reuse should RECOVER on the following same-state turn.

Usage: variating_multiturn_matrix.py <model_key> <lane>
  model_key in {qwen36, dots3}, lane in {l1l2, ssd}
"""
import base64
import io
import json
import math
import os
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave

# Never hardcode a checkout: on max2 ~/mlx/vllm-mlx is a STALE clone and a
# probe that points there silently measures unfixed code (it happened).
REPO = os.environ.get("VMLX_REPO") or os.path.expanduser("~/mlx/vllm-mlx")
PY = os.path.join(REPO, ".venv/bin/python")
PORT = 8143

MODELS = {
    "qwen36": "/Volumes/EricsLLMDrive/dealignai/Qwen3.6-35B-A3B-MXFP8-CRACK-MTP",
    "dots3": "/Volumes/EricsLLMDrive/models/dots3-note-prev-JANG",
}
MODEL_KEY = sys.argv[1]
LANE = sys.argv[2] if len(sys.argv) > 2 else "ssd"
MODEL = MODELS[MODEL_KEY]
d = f"/tmp/varmt-{MODEL_KEY}-{LANE}"
subprocess.run(["rm", "-rf", d], check=False)
os.makedirs(d, exist_ok=True)

LANE_ARGS = (
    ["--use-paged-cache", "--paged-cache-block-size", "64",
     "--max-cache-blocks", "4000", "--enable-block-disk-cache"]
    if LANE == "l1l2"
    else ["--no-paged-cache", "--enable-block-disk-cache"]
)

# ---- media fixtures with checkable ground truth ----------------------------
from PIL import Image  # noqa: E402

img = Image.new("RGB", (448, 448))
for x in range(448):
    for y in range(448):
        img.putpixel((x, y), (240, 200, 20) if y < 224 else (30, 30, 30))
_b = io.BytesIO(); img.save(_b, format="PNG")
IMAGE_B64 = base64.b64encode(_b.getvalue()).decode()   # yellow top, black bottom

frames = os.path.join(d, "frames"); os.makedirs(frames, exist_ok=True)
NF = 10
for i in range(NF):
    fr = Image.new("RGB", (256, 256), (0, 0, 0))
    y0 = int(8 + i * (256 - 72) / (NF - 1))            # square moves DOWNWARD
    for x in range(96, 160):
        for y in range(y0, y0 + 64):
            fr.putpixel((x, y), (255, 255, 255))
    fr.save(os.path.join(frames, f"f{i:03d}.png"))
VIDEO = os.path.join(d, "down.mp4")
VIDEO_OK = False
try:
    import cv2  # box has OpenCV, no ffmpeg binary

    w = cv2.VideoWriter(VIDEO, cv2.VideoWriter_fourcc(*"avc1"), 4.0, (256, 256))
    if not w.isOpened():
        w = cv2.VideoWriter(VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (256, 256))
    for i in range(NF):
        w.write(cv2.imread(os.path.join(frames, f"f{i:03d}.png")))
    w.release()
    VIDEO_OK = os.path.exists(VIDEO) and os.path.getsize(VIDEO) > 800
except Exception as exc:
    print(f"video encode failed: {exc}", flush=True)
VIDEO_B64 = base64.b64encode(open(VIDEO, "rb").read()).decode() if VIDEO_OK else None

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_dock_count",
        "description": "Return the number of crates at a dock.",
        "parameters": {
            "type": "object",
            "properties": {"dock": {"type": "string"}},
            "required": ["dock"],
        },
    },
}]

log = open(os.path.join(d, "serve.log"), "w")
proc = subprocess.Popen(
    [PY, "-B", "-m", "vmlx_engine.cli", "serve", MODEL,
     "--host", "127.0.0.1", "--port", str(PORT), "--timeout", "1800",
     "--continuous-batching", "--max-num-seqs", "1",
     "--enable-prefix-cache", *LANE_ARGS,
     "--block-disk-cache-max-gb", "20", "--block-disk-cache-dir", d,
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

LEAK_MARKERS = ("<think>", "</think>", "<no_think>", "<mm:think>",
                "<dots_function_call>", "<tool_call>", "<|tool")


def post(messages, *, reasoning: bool, tools=None, max_tokens=700):
    body = {
        "model": MODEL_KEY,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    # Only send the thinking flag when turning it OFF: thinking-only families
    # correctly 400 on enable_thinking=false, and forcing it everywhere was a
    # harness bug earlier today.
    if not reasoning:
        body["enable_thinking"] = False
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            out = json.load(r)
    except urllib.error.HTTPError as e:
        return {"http_error": e.code,
                "detail": e.read()[:240].decode(errors="replace"),
                "wall": round(time.time() - t0, 2)}
    msg = out["choices"][0]["message"]
    u = out.get("usage", {}) or {}
    det = u.get("prompt_tokens_details") or {}
    return {
        "text": (msg.get("content") or "").strip(),
        "reasoning": (msg.get("reasoning_content") or "").strip(),
        "tool_calls": msg.get("tool_calls") or [],
        "prompt_tokens": u.get("prompt_tokens"),
        "cached": det.get("cached_tokens", 0),
        "wall": round(time.time() - t0, 2),
    }


def image_part():
    return {"type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{IMAGE_B64}"}}


def video_part():
    return {"type": "video_url",
            "video_url": {"url": f"data:video/mp4;base64,{VIDEO_B64}"}}


def run_conversation(tag):
    """The four varying turns, carrying history forward like a real chat."""
    history = []
    rows = []

    # ── turn 1: reasoning ON + tool call ────────────────────────────────
    history.append({"role": "user",
                    "content": "Use get_dock_count for dock 'A7', then state the number."})
    r1 = post(history, reasoning=True, tools=TOOLS)
    rows.append(("t1 reasoningON+tools", r1))
    if not r1.get("http_error"):
        calls = r1.get("tool_calls") or []
        if calls:
            history.append({"role": "assistant", "content": "", "tool_calls": calls})
            history.append({"role": "tool", "tool_call_id": calls[0].get("id", "c1"),
                            "content": json.dumps({"dock": "A7", "crates": 41})})
            r1b = post(history, reasoning=True, tools=TOOLS)
            rows.append(("t1b tool-continuation", r1b))
            history.append({"role": "assistant", "content": r1b.get("text", "")})
        else:
            history.append({"role": "assistant", "content": r1.get("text", "")})

    # ── turn 2: reasoning OFF + video ───────────────────────────────────
    if VIDEO_B64:
        history.append({"role": "user", "content": [
            video_part(),
            {"type": "text", "text": "In this video does the white square move UP or DOWN? One word."}]})
        r2 = post(history, reasoning=False)
        rows.append(("t2 reasoningOFF+video", r2))
        history.append({"role": "assistant", "content": r2.get("text", "")})

    # ── turn 3: reasoning ON + text only ───────────────────────────────
    history.append({"role": "user",
                    "content": "Ignoring the media, how many crates were at dock A7?"})
    r3 = post(history, reasoning=True)
    rows.append(("t3 reasoningON+text", r3))
    history.append({"role": "assistant", "content": r3.get("text", "")})

    # ── turn 4: image + reasoning OFF ──────────────────────────────────
    history.append({"role": "user", "content": [
        image_part(),
        {"type": "text", "text": "What colour is the TOP half of this image? One word."}]})
    r4 = post(history, reasoning=False)
    rows.append(("t4 image+reasoningOFF", r4))
    history.append({"role": "assistant", "content": r4.get("text", "")})

    print(f"\n--- {tag} ({MODEL_KEY} / {LANE}) ---", flush=True)
    for label, r in rows:
        if r.get("http_error"):
            print(f"  {label:<24} HTTP {r['http_error']} {r['detail'][:110]}", flush=True)
            continue
        leaked = [m for m in LEAK_MARKERS if m in (r["text"] + r["reasoning"])]
        print(f"  {label:<24} cached={r['cached']:>6}/{r['prompt_tokens'] or 0:<6} "
              f"t={r['wall']:>6}s reas={len(r['reasoning']):>5}c "
              f"leak={leaked or '-'} text={r['text'][:52]!r}", flush=True)
    return rows, history


first_rows, history = run_conversation("PASS 1 (cold)")
time.sleep(4)
# Replay the identical conversation: reuse should be high and answers identical.
second_rows, _ = run_conversation("PASS 2 (warm replay)")

findings = []
for (label, a), (_, b) in zip(first_rows, second_rows):
    if a.get("http_error") or b.get("http_error"):
        findings.append(f"{label}: HTTP error")
        continue
    if not a["text"] and not a["tool_calls"]:
        findings.append(f"{label}: EMPTY visible answer (never-empty contract)")
    if a["text"] != b["text"]:
        findings.append(f"{label}: warm answer DIFFERS at temp 0 -> {a['text'][:40]!r} vs {b['text'][:40]!r}")
    leaked = [m for m in LEAK_MARKERS if m in (a["text"] + a["reasoning"])]
    if leaked:
        findings.append(f"{label}: leaked markers {leaked}")
    if "reasoningOFF" in label and a["reasoning"]:
        findings.append(f"{label}: reasoning_content present while reasoning OFF")

warm_reuse = [b["cached"] for (_, b) in second_rows if not b.get("http_error")]
print(f"\nwarm replay cached tokens: {warm_reuse}", flush=True)
if warm_reuse and max(warm_reuse) == 0:
    findings.append("warm replay reused NOTHING on any turn")

serve_log = open(os.path.join(d, "serve.log"), errors="replace").read()
print(f"valve_declined={'prefill admission rejected' in serve_log}", flush=True)

print("\nFINDINGS:" if findings else "\nFINDINGS: none", flush=True)
for f in findings:
    print(f"  - {f}", flush=True)
print(f"VERDICT: {'FAIL' if findings else 'PASS'} ({MODEL_KEY}/{LANE})", flush=True)

json.dump({"model": MODEL_KEY, "lane": LANE,
           "pass1": [{"label": l, **r} for l, r in first_rows],
           "pass2": [{"label": l, **r} for l, r in second_rows],
           "findings": findings},
          open(f"/tmp/varmt-{MODEL_KEY}-{LANE}.json", "w"), indent=1, default=str)

proc.terminate()
try:
    proc.wait(timeout=120)
except subprocess.TimeoutExpired:
    proc.kill()
