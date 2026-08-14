#!/usr/bin/env bash
# Native-MTP A/B for a JANG bundle.
#
# Serves the bundle twice on the same port -- once with native MTP as-is, once
# with VMLX_NATIVE_MTP=0 -- sends an identical temperature-0 prompt to each, and
# reports decode rate for both arms plus whether the outputs match byte for byte.
#
# MTP only engages at temperature 0, so the probe pins temperature 0 explicitly.
# At temp 0 the two arms MUST produce identical text: MTP is a speculative
# decode, not a different model. Divergence means the MTP path is wrong, and no
# speedup number from that run is worth quoting.
#
# Usage: scripts/mtp-ab.sh /Volumes/EricsLLMDrive/jangq-ai/Qwen3.8-27B-JANG_4D [port]
set -uo pipefail

BUNDLE="${1:?usage: mtp-ab.sh <bundle-path> [port]}"
PORT="${2:-8014}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python3"
PROMPT='Explain in exactly four sentences why speculative decoding can be faster than ordinary decoding without changing the output.'

[ -d "$BUNDLE" ] || { echo "no such bundle: $BUNDLE" >&2; exit 1; }
[ -x "$PY" ] || { echo "no venv python at $PY" >&2; exit 1; }

serve() {  # serve <env-assignment-or-empty>
  pkill -f "port $PORT" 2>/dev/null
  sleep 5
  local extra_args=()
  case "$BUNDLE" in *Qwen3.[5678]*|*qwen3.[5678]*) extra_args=(--is-mllm --reasoning-parser qwen3);; esac
  env $1 nohup "$PY" -B -s -m vmlx_engine.cli serve "$BUNDLE" \
      --host 127.0.0.1 --port "$PORT" --timeout 900 --max-num-seqs 1 \
      --continuous-batching --use-paged-cache --stream-interval 1 --no-jit \
      "${extra_args[@]}" > "/tmp/mtpab_$PORT.log" 2>&1 &
  until curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models" 2>/dev/null; do sleep 5; done
}

probe() {  # probe <arm-label>
  "$PY" - "$PORT" "$1" "$PROMPT" <<'PYEOF'
import json, sys, time, urllib.request
port, arm, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
body = {"model": "m", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400, "temperature": 0}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                             data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.time()
d = json.load(urllib.request.urlopen(req, timeout=900))
wall = time.time() - t0
# Count from usage, not from streamed chunks -- chunks are not tokens.
out = d["usage"]["completion_tokens"]
text = d["choices"][0]["message"].get("content") or ""
print("%-8s %5.1f tok/s  (%d tokens in %.1fs)" % (arm, out / wall if wall else 0, out, wall))
open(f"/tmp/mtpab_{arm}.txt", "w").write(text)
PYEOF
}

echo "=== bundle: $(basename "$BUNDLE") ==="
echo "loading arm ON (native MTP as shipped)..."
serve ""
probe on
echo "loading arm OFF (VMLX_NATIVE_MTP=0)..."
serve "VMLX_NATIVE_MTP=0"
probe off

pkill -f "port $PORT" 2>/dev/null
if cmp -s /tmp/mtpab_on.txt /tmp/mtpab_off.txt; then
  echo "outputs: BYTE-IDENTICAL (expected at temperature 0)"
else
  echo "outputs: *** DIFFER *** — the MTP path is wrong; ignore the speedup above"
  diff <(head -c 400 /tmp/mtpab_on.txt) <(head -c 400 /tmp/mtpab_off.txt) | head -10
fi
echo "engine log: /tmp/mtpab_$PORT.log   texts: /tmp/mtpab_{on,off}.txt"
