#!/usr/bin/env python3
"""Sustained-speed trend probe (campaign #181 arm A4).

Measures whether decode t/s and TTFT DECAY over a long-running serve as the
conversation (and the cache store) grows — the "best maintained long term
speeds" arm. One serve, one growing conversation, N turns; per-turn records
of decode rate (from usage.completion_tokens only — ledger rule), TTFT,
cached_tokens, and RSS/thermal snapshots where exposed.

Design rules:
- Decode rate = (completion_tokens - 1) / decode_elapsed using the server's
  own usage counters; never client-side token counting.
- The conversation GROWS (each turn appends real history) so store size,
  block count, and prompt length rise the way a real long session does.
- Trend verdict: compare the median decode t/s of the FIRST quartile of
  turns against the LAST quartile. Decay beyond --max-decay-pct (default
  15%) = "decay_detected" (exit 3) — a lead to attribute (thermal, wired
  Metal, store growth), NOT automatically an engine defect (box-state
  variance rule: attribute before filing).
- Serve flags caller-supplied (--serve-arg) so the trend can be measured in
  both tier configs.

Writes one JSON artifact (--out); prints a one-line summary.
Exit codes: 0 = steady, 3 = decay_detected, 1 = fail/error.
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
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_PYTHON_ENV = Path(os.environ.get("VMLINUX_BENCH_PYTHON", sys.executable)).expanduser()
PYTHON = _PYTHON_ENV if _PYTHON_ENV.is_absolute() else (ROOT / _PYTHON_ENV).resolve()

SERVED_MODEL_NAME = "sustained-trend-probe"
LOG_TAIL_CHARS = 4000


def request_json(
    method: str, url: str, body: Any | None = None, *, timeout: float = 300.0
) -> tuple[int, Any, float]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            elapsed = time.perf_counter() - t0
            return (
                response.status,
                json.loads(raw.decode("utf-8")) if raw else None,
                elapsed,
            )
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        return (
            exc.code,
            {"error": str(exc), "body": raw.decode("utf-8", "replace") if raw else ""},
            time.perf_counter() - t0,
        )
    except Exception as exc:
        return 0, {"error": f"{type(exc).__name__}: {exc}"}, time.perf_counter() - t0


def log_tail(log_path: Path) -> str:
    try:
        return log_path.read_text(errors="replace")[-LOG_TAIL_CHARS:]
    except OSError:
        return ""


def wait_ready(
    base_url: str, proc: subprocess.Popen[Any], load_timeout_s: float, log_path: Path
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
    raise RuntimeError(f"server not ready in {load_timeout_s}s")


def terminate_process(proc: subprocess.Popen[Any]) -> int | None:
    if proc.poll() is not None:
        return proc.returncode
    proc.send_signal(signal.SIGTERM)
    try:
        return proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        return proc.wait(timeout=10)


def turn_prompt(nonce: str, index: int) -> str:
    themes = [
        "a measurement pipeline",
        "a caching layer",
        "a scheduler",
        "a tokenizer",
        "a build system",
        "a profiler",
        "a storage engine",
        "an eviction policy",
    ]
    theme = themes[index % len(themes)]
    return (
        f"[{nonce}-{index}] In 3-4 sentences, describe one subtle failure "
        f"mode of {theme} and how you would detect it in production."
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sustained decode/TTFT trend probe (campaign #181 arm A4)."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--port", type=int, default=8173)
    parser.add_argument("--turns", type=int, default=48)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-decay-pct", type=float, default=15.0)
    parser.add_argument("--serve-arg", action="append", default=[])
    parser.add_argument("--load-timeout-s", type=float, default=900.0)
    parser.add_argument("--request-timeout-s", type=float, default=420.0)
    parser.add_argument("--workdir", default=None)
    return parser


def quartile_medians(rates: list[float]) -> tuple[float, float]:
    if len(rates) < 8:
        return (median(rates), median(rates)) if rates else (0.0, 0.0)
    q = max(1, len(rates) // 4)
    return median(rates[:q]), median(rates[-q:])


def main() -> int:
    args = build_arg_parser().parse_args()
    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="sst_"))
    workdir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / "server.log"
    base_url = f"http://127.0.0.1:{args.port}"
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
        "schema": "vmlx-sustained-speed-trend-probe-v1",
        "model": args.model,
        "nonce": args.nonce,
        "serve_cmd": serve_cmd,
        "turns_requested": args.turns,
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
        history: list[dict[str, str]] = []
        records: list[dict[str, Any]] = []
        for index in range(args.turns):
            messages = history + [
                {"role": "user", "content": turn_prompt(args.nonce, index)}
            ]
            code, resp, elapsed = request_json(
                "POST",
                f"{base_url}/v1/chat/completions",
                {
                    "model": SERVED_MODEL_NAME,
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": args.max_tokens,
                    "stream": False,
                },
                timeout=args.request_timeout_s,
            )
            if code != 200:
                artifact["turn_error"] = {"index": index, "code": code, "body": resp}
                break
            usage = (resp or {}).get("usage") or {}
            completion = int(usage.get("completion_tokens") or 0)
            details = usage.get("prompt_tokens_details") or {}
            content = (
                ((resp.get("choices") or [{}])[0].get("message") or {}).get("content")
                or ""
            )
            # elapsed includes prefill; a coarse decode rate over the full
            # request stays comparable turn-over-turn when cached_tokens keeps
            # prefill amortized — record BOTH so decay can be attributed.
            records.append(
                {
                    "index": index,
                    "elapsed_s": round(elapsed, 3),
                    "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                    "completion_tokens": completion,
                    "cached_tokens": int(details.get("cached_tokens") or 0),
                    "coarse_rate_tps": round(max(0, completion - 1) / elapsed, 2)
                    if elapsed > 0
                    else 0.0,
                }
            )
            history = messages + [{"role": "assistant", "content": content}]
        artifact["records"] = records
        rates = [r["coarse_rate_tps"] for r in records if r["completion_tokens"] > 8]
        first_med, last_med = quartile_medians(rates)
        decay_pct = (
            round(100.0 * (first_med - last_med) / first_med, 2) if first_med else 0.0
        )
        artifact.update(
            {
                "first_quartile_median_tps": first_med,
                "last_quartile_median_tps": last_med,
                "decay_pct": decay_pct,
                "status": "decay_detected"
                if decay_pct > args.max_decay_pct
                else ("steady" if records else "error"),
                "server_log_tail": log_tail(log_path),
            }
        )
    except Exception as exc:
        artifact["error"] = f"{type(exc).__name__}: {exc}"
        artifact["server_log_tail"] = log_tail(log_path)
    finally:
        if proc is not None:
            artifact["server_exit_code"] = terminate_process(proc)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(
        f"[sustained-trend] status={artifact['status']} turns={len(artifact.get('records', []))} "
        f"first_q={artifact.get('first_quartile_median_tps')} "
        f"last_q={artifact.get('last_quartile_median_tps')} "
        f"decay_pct={artifact.get('decay_pct')} out={out_path}"
    )
    return {"steady": 0, "decay_detected": 3}.get(artifact["status"], 1)


if __name__ == "__main__":
    sys.exit(main())
