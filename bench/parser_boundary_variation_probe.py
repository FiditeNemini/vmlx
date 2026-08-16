#!/usr/bin/env python3
"""Parser-boundary + per-turn-variation cache equivalence probe (campaign
#181 arms A2/A3/A7).

Proves (or refutes) that a served bundle produces byte-equivalent answer TEXT
at temperature 0 when a multiturn conversation with EXTREME per-turn
variation is replayed warm in the same process — exactly the states where
parser-owned tokens (think segments, tool-call markers, tool schemas in the
system render, preserved prior-turn reasoning) shift token boundaries inside
cache blocks:

- T1 plain turn (family-default thinking).
- T2 history replays T1's assistant WITH reasoning_content (PRESERVED
  THINKING, arm A7) — think blocks ride the rendered prefix.
- T3 tools APPEAR mid-conversation (one schema) — the tool render changes
  the prefix before the shared history (arm A3, changed provided tools).
- T4 the tool SET changes (second schema added).
- T5 tools removed again + an in-set reasoning-effort change (arm A3).
- T6 same settings as T5 — the re-establishment turn that must reuse.

Rules inherited from the proven sameproc probe (ledger discipline):
- Latency is never the proof; the comparison is answer TEXT at temp 0
  (exact bytes first, whitespace-normalized reported separately).
- Fresh server process per run + run-unique ``--nonce`` salted into every
  user turn (novel-seed rule). ``--no-paged-cache`` is NOT a no-cache
  control and is not used.
- The WARM arm replays the COLD arm's recorded answers/reasoning as history
  so every prompt is byte-identical to the prompt that primed the cache.
- >=2s settle before every request in both arms.
- Reuse must be PROVEN (cached_tokens > 0 or a cache-stats hit delta on at
  least one warm turn) or the verdict is "inconclusive_no_reuse".
- Serve flags are caller-supplied (``--serve-arg`` repeatable) so the SAME
  probe runs the L1+L2 and SSD-only tier configs.

Writes one JSON artifact (``--out``); prints a one-line summary.
Exit codes: 0 = pass, 2 = inconclusive_no_reuse, 1 = fail/error.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_PYTHON_ENV = Path(os.environ.get("VMLINUX_BENCH_PYTHON", sys.executable)).expanduser()
PYTHON = _PYTHON_ENV if _PYTHON_ENV.is_absolute() else (ROOT / _PYTHON_ENV).resolve()

SERVED_MODEL_NAME = "parser-boundary-probe"
REQUEST_MAX_TOKENS = 384
REQUEST_TIMEOUT_S_DEFAULT = 420.0
SETTLE_GAP_S = 2.5
STATS_LAG_S = 2.0
LOG_TAIL_CHARS = 4000

TOOL_SCHEMA_A = {
    "type": "function",
    "function": {
        "name": "lookup_constant",
        "description": "Look up a named physical constant",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
}
TOOL_SCHEMA_B = {
    "type": "function",
    "function": {
        "name": "convert_units",
        "description": "Convert a value between units",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "from_unit": {"type": "string"},
                "to_unit": {"type": "string"},
            },
            "required": ["value", "from_unit", "to_unit"],
        },
    },
}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def request_json(
    method: str,
    url: str,
    body: Any | None = None,
    *,
    timeout: float = 180.0,
) -> tuple[int, Any, float]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            elapsed = time.perf_counter() - t0
            if not raw:
                return response.status, None, elapsed
            try:
                return response.status, json.loads(raw.decode("utf-8")), elapsed
            except Exception:
                return response.status, raw.decode("utf-8", "replace"), elapsed
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        elapsed = time.perf_counter() - t0
        text = raw.decode("utf-8", "replace") if raw else ""
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = {"error": f"HTTPError: {exc}", "body": text}
        if isinstance(parsed, dict):
            parsed.setdefault("error", f"HTTPError: {exc}")
        return exc.code, parsed, elapsed
    except Exception as exc:
        return 0, {"error": f"{type(exc).__name__}: {exc}"}, time.perf_counter() - t0


def log_tail(log_path: Path) -> str:
    try:
        return log_path.read_text(errors="replace")[-LOG_TAIL_CHARS:]
    except OSError:
        return ""


def wait_ready(
    base_url: str,
    proc: subprocess.Popen[Any],
    load_timeout_s: float,
    log_path: Path,
) -> dict[str, Any]:
    deadline = time.monotonic() + load_timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"server exited during load rc={proc.returncode}: {log_tail(log_path)}"
            )
        code, body, _ = request_json("GET", f"{base_url}/health", timeout=5.0)
        if code == 200 and isinstance(body, dict) and body.get("model_loaded") is True:
            return body
        time.sleep(3.0)
    raise RuntimeError(f"server not ready in {load_timeout_s}s: {log_tail(log_path)}")


def terminate_process(proc: subprocess.Popen[Any]) -> int | None:
    if proc.poll() is not None:
        return proc.returncode
    proc.send_signal(signal.SIGTERM)
    try:
        return proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        return proc.wait(timeout=10)


def extract_message(resp: Any) -> dict[str, Any]:
    if not isinstance(resp, dict):
        return {}
    choices = resp.get("choices") or []
    if not choices:
        return {}
    return choices[0].get("message") or {}


def cached_tokens_of(resp: Any) -> int:
    try:
        details = (resp.get("usage") or {}).get("prompt_tokens_details") or {}
        return int(details.get("cached_tokens") or 0)
    except Exception:
        return 0


def normalize_text(text: str) -> str:
    return " ".join((text or "").split())


def conversation_script(nonce: str) -> list[dict[str, Any]]:
    """Six turns of per-turn variation. Each entry: user text + request
    variation (tools list / reasoning_effort) + whether the prior assistant
    turn is replayed WITH its reasoning_content (preserved thinking)."""
    return [
        {
            "key": "t1_plain",
            "user": f"[{nonce}] In two sentences, why does ice float on water?",
            "tools": None,
            "reasoning_effort": None,
            "preserve_prior_reasoning": False,
        },
        {
            "key": "t2_preserved_thinking",
            "user": f"[{nonce}] Relate that to why lakes freeze top-down, briefly.",
            "tools": None,
            "reasoning_effort": None,
            "preserve_prior_reasoning": True,
        },
        {
            "key": "t3_tools_appear",
            "user": f"[{nonce}] If you needed the exact density values, which tool call would you make? Answer in one sentence without calling it.",
            "tools": [TOOL_SCHEMA_A],
            "reasoning_effort": None,
            "preserve_prior_reasoning": True,
        },
        {
            "key": "t4_toolset_changes",
            "user": f"[{nonce}] And to express that density in imperial units, which tool would you use? One sentence, no call.",
            "tools": [TOOL_SCHEMA_A, TOOL_SCHEMA_B],
            "reasoning_effort": None,
            "preserve_prior_reasoning": False,
        },
        {
            "key": "t5_tools_off_effort_change",
            "user": f"[{nonce}] Summarize our discussion so far in one sentence.",
            "tools": None,
            "reasoning_effort": "medium",
            "preserve_prior_reasoning": False,
        },
        {
            "key": "t6_reestablish",
            "user": f"[{nonce}] Now add one caveat to that summary, in one sentence.",
            "tools": None,
            "reasoning_effort": "medium",
            "preserve_prior_reasoning": False,
        },
    ]


def chat_payload(
    messages: list[dict[str, Any]],
    turn: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": SERVED_MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        "max_tokens": REQUEST_MAX_TOKENS,
        "stream": False,
    }
    if turn["tools"]:
        payload["tools"] = turn["tools"]
    if turn["reasoning_effort"]:
        payload["reasoning_effort"] = turn["reasoning_effort"]
    return payload


def run_conversation(
    base_url: str,
    script: list[dict[str, Any]],
    *,
    request_timeout_s: float,
    replay_answers: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run the six-turn script. With ``replay_answers`` (warm arm) the
    history is rebuilt from the COLD arm's recorded assistant messages so
    every prompt is byte-identical to the one that primed the cache."""
    records: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    for index, turn in enumerate(script):
        messages = [dict(m) for m in history] + [
            {"role": "user", "content": turn["user"]}
        ]
        time.sleep(SETTLE_GAP_S)
        code, resp, elapsed = request_json(
            "POST",
            f"{base_url}/v1/chat/completions",
            chat_payload(messages, turn),
            timeout=request_timeout_s,
        )
        message = extract_message(resp)
        record = {
            "key": turn["key"],
            "status_code": code,
            "elapsed_s": round(elapsed, 3),
            "cached_tokens": cached_tokens_of(resp),
            "prompt_tokens": int(
                ((resp or {}).get("usage") or {}).get("prompt_tokens") or 0
            )
            if isinstance(resp, dict)
            else 0,
            "content": str(message.get("content") or ""),
            "reasoning_content": str(message.get("reasoning_content") or ""),
            "error": None if code == 200 else resp,
        }
        records.append(record)
        if code != 200:
            break
        # Build the next turn's history entry. The WARM arm replays the COLD
        # answers verbatim so the rendered prompt matches the primed prefix.
        source = (
            replay_answers[index]
            if replay_answers is not None and index < len(replay_answers)
            else record
        )
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": source["content"],
        }
        next_turn = script[index + 1] if index + 1 < len(script) else None
        if (
            next_turn is not None
            and next_turn["preserve_prior_reasoning"]
            and source["reasoning_content"]
        ):
            assistant_msg["reasoning_content"] = source["reasoning_content"]
        history = messages + [assistant_msg]
    return records


def compute_verdict(
    cold: list[dict[str, Any]],
    warm: list[dict[str, Any]],
    *,
    table_hit_evidence: bool,
) -> tuple[str, list[dict[str, Any]]]:
    turns: list[dict[str, Any]] = []
    if len(cold) != len(warm) or any(r["status_code"] != 200 for r in cold + warm):
        return "fail", turns
    byte_equal = True
    for c, w in zip(cold, warm):
        exact = c["content"] == w["content"]
        normalized = normalize_text(c["content"]) == normalize_text(w["content"])
        byte_equal = byte_equal and exact
        turns.append(
            {
                "key": c["key"],
                "byte_equal": exact,
                "normalized_equal": normalized,
                "cold_cached_tokens": c["cached_tokens"],
                "warm_cached_tokens": w["cached_tokens"],
            }
        )
    reuse_proven = table_hit_evidence or any(
        t["warm_cached_tokens"] > 0 for t in turns
    )
    if not reuse_proven:
        return "inconclusive_no_reuse", turns
    return ("pass" if byte_equal else "fail"), turns


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parser-boundary + per-turn-variation cache equivalence probe "
            "(campaign #181 arms A2/A3/A7). Private/internal gate."
        )
    )
    parser.add_argument("--model", required=True, help="Bundle path to serve.")
    parser.add_argument(
        "--nonce",
        required=True,
        help="Run-unique nonce salted into every user turn (novel-seed rule).",
    )
    parser.add_argument("--out", required=True, help="JSON artifact path.")
    parser.add_argument("--port", type=int, default=8172)
    parser.add_argument(
        "--serve-arg",
        action="append",
        default=[],
        help="Extra server CLI arg (repeatable) — e.g. --serve-arg=--use-paged-cache "
        "--serve-arg=--enable-block-disk-cache for the L1+L2 tier config.",
    )
    parser.add_argument("--load-timeout-s", type=float, default=900.0)
    parser.add_argument(
        "--request-timeout-s", type=float, default=REQUEST_TIMEOUT_S_DEFAULT
    )
    parser.add_argument("--workdir", default=None)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="pbv_"))
    workdir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / "server.log"
    base_url = f"http://127.0.0.1:{args.port}"
    out_path = Path(args.out)

    serve_cmd = [
        str(PYTHON),
        "-m",
        "vmlx_engine.server",
        "--model",
        args.model,
        "--port",
        str(args.port),
        *args.serve_arg,
    ]
    artifact: dict[str, Any] = {
        "schema": "vmlx-parser-boundary-variation-probe-v1",
        "model": args.model,
        "nonce": args.nonce,
        "serve_cmd": serve_cmd,
        "status": "error",
    }
    proc: subprocess.Popen[Any] | None = None
    try:
        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(
                serve_cmd, cwd=str(ROOT), stdout=log_file, stderr=subprocess.STDOUT
            )
        artifact["health_at_ready"] = wait_ready(
            base_url, proc, args.load_timeout_s, log_path
        )
        script = conversation_script(args.nonce)
        _, stats_before, _ = request_json("GET", f"{base_url}/v1/cache/stats", timeout=10)
        cold = run_conversation(
            base_url, script, request_timeout_s=args.request_timeout_s
        )
        warm = run_conversation(
            base_url,
            script,
            request_timeout_s=args.request_timeout_s,
            replay_answers=cold,
        )
        time.sleep(STATS_LAG_S)
        _, stats_after, _ = request_json("GET", f"{base_url}/v1/cache/stats", timeout=10)

        def _hits(stats: Any) -> int:
            if not isinstance(stats, dict):
                return 0

            def walk(node: Any) -> int:
                found = 0
                if isinstance(node, dict):
                    for k, v in node.items():
                        if isinstance(v, (int, float)) and "hit" in str(k).lower():
                            found += int(v)
                        else:
                            found += walk(v)
                elif isinstance(node, list):
                    for item in node:
                        found += walk(item)
                return found
            return walk(stats)

        hits_delta = max(0, _hits(stats_after) - _hits(stats_before))
        status, turns = compute_verdict(
            cold, warm, table_hit_evidence=hits_delta > 0
        )
        artifact.update(
            {
                "status": status,
                "turns": turns,
                "cold_records": cold,
                "warm_records": warm,
                "cache_stats_before": stats_before,
                "cache_stats_after": stats_after,
                "cache_stats_hits_delta": hits_delta,
                "server_log_tail": log_tail(log_path),
            }
        )
    except Exception as exc:
        artifact["error"] = f"{type(exc).__name__}: {exc}"
        artifact["server_log_tail"] = log_tail(log_path)
    finally:
        if proc is not None:
            artifact["server_exit_code"] = terminate_process(proc)
    write_json(out_path, artifact)
    turns_summary = [
        (t["key"], t["byte_equal"], t["warm_cached_tokens"])
        for t in artifact.get("turns", [])
    ]
    print(
        f"[parser-boundary-variation] status={artifact['status']} "
        f"model={Path(args.model).name} nonce={args.nonce} "
        f"turns={turns_summary} out={out_path}"
    )
    return {"pass": 0, "inconclusive_no_reuse": 2}.get(artifact["status"], 1)


if __name__ == "__main__":
    sys.exit(main())
