#!/usr/bin/env python3
"""API-path kwargs parity probe (campaign #181 arm A5).

Sends the SAME logical request through all four dialects — chat completions,
Responses, Ollama, Anthropic-shaped — and diffs the engine's OWN
"Resolved sampling kwargs" log lines. The engine log is the ground truth for
what actually reached generation; client-side echoes are never trusted.

Method:
- The probe owns the serve (fresh process, caller-supplied extra flags), so
  it owns the log file. Requests are sent SEQUENTIALLY; each route logs
  exactly one Resolved line, so pairing is by order + the route= field
  (header-identity plumbing differences cannot break the pairing).
- Requested values are chosen to be expressible in every dialect:
  temperature, top_p, an output cap, and an IN-SET reasoning effort for the
  served family (out-of-set would engage the substitution policy and change
  what the A/B measures).
- Comparison keys: temperature, top_p, top_k, max_tokens, reasoning_effort,
  enable_thinking, and the logged chat_template_kwargs subset. A mismatch on
  any compared key across dialects = "divergent" with the offending keys.

Exit codes: 0 = parity, 3 = divergent, 1 = fail/error.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
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

RESOLVED_RE = re.compile(
    r"Resolved sampling kwargs route=(\S+) model=\S+.*? kwargs=(\{.*\})\s*$"
)
COMPARED_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "reasoning_effort",
    "enable_thinking",
    "chat_template_kwargs",
)
LOG_TAIL_CHARS = 4000


def request_json(
    method: str, url: str, body: Any | None = None, *, timeout: float = 300.0
) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            return response.status, json.loads(raw.decode("utf-8")) if raw else None
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        return exc.code, {
            "error": str(exc),
            "body": raw.decode("utf-8", "replace") if raw else "",
        }
    except Exception as exc:
        return 0, {"error": f"{type(exc).__name__}: {exc}"}


def wait_ready(
    base_url: str, proc: subprocess.Popen[Any], load_timeout_s: float, log_path: Path
) -> None:
    deadline = time.monotonic() + load_timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited during load rc={proc.returncode}")
        code, body = request_json("GET", f"{base_url}/health", timeout=5.0)
        if code == 200 and isinstance(body, dict) and body.get("model_loaded") is True:
            return
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


def dialect_requests(nonce: str, effort: str, max_out: int) -> list[dict[str, Any]]:
    """The same logical request expressed in each dialect's native shape."""
    user_text = f"[{nonce}] Reply with one short sentence about tides."
    return [
        {
            "dialect": "chat",
            "path": "/v1/chat/completions",
            "body": {
                "model": "parity-probe",
                "messages": [{"role": "user", "content": user_text}],
                "temperature": 0.4,
                "top_p": 0.9,
                "max_tokens": max_out,
                "reasoning_effort": effort,
                "stream": False,
            },
        },
        {
            "dialect": "responses",
            "path": "/v1/responses",
            "body": {
                "model": "parity-probe",
                "input": user_text,
                "temperature": 0.4,
                "top_p": 0.9,
                "max_output_tokens": max_out,
                "reasoning_effort": effort,
                "stream": False,
            },
        },
        {
            "dialect": "ollama",
            "path": "/api/chat",
            "body": {
                "model": "parity-probe",
                "messages": [{"role": "user", "content": user_text}],
                "options": {
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "num_predict": max_out,
                },
                "reasoning_effort": effort,
                "stream": False,
            },
        },
        {
            "dialect": "anthropic",
            "path": "/v1/messages",
            "body": {
                "model": "parity-probe",
                "messages": [{"role": "user", "content": user_text}],
                "temperature": 0.4,
                "top_p": 0.9,
                "max_tokens": max_out,
                "reasoning_effort": effort,
                "stream": False,
            },
        },
    ]


def parse_resolved_lines(log_text: str) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for raw in log_text.splitlines():
        match = RESOLVED_RE.search(raw)
        if not match:
            continue
        route, kwargs_repr = match.groups()
        try:
            kwargs = ast.literal_eval(kwargs_repr)
        except Exception:
            kwargs = {"_unparseable": kwargs_repr}
        lines.append({"route": route, "kwargs": kwargs})
    return lines


def compare_dialects(
    resolved: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Diff the compared keys across the captured resolved lines."""
    if len(resolved) < 2:
        return "fail", {"reason": "fewer than 2 resolved lines captured"}
    baseline = resolved[0]
    divergences: dict[str, Any] = {}
    for other in resolved[1:]:
        for key in COMPARED_KEYS:
            base_value = baseline["kwargs"].get(key)
            other_value = other["kwargs"].get(key)
            if base_value != other_value:
                divergences.setdefault(key, {})[
                    f"{baseline['route']} vs {other['route']}"
                ] = [base_value, other_value]
    return ("parity" if not divergences else "divergent"), divergences


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-dialect resolved-kwargs parity probe (campaign #181 A5)."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--port", type=int, default=8174)
    parser.add_argument(
        "--effort",
        default="medium",
        help="Must be IN the served bundle's stamped effort set.",
    )
    parser.add_argument("--max-out", type=int, default=77)
    parser.add_argument("--serve-arg", action="append", default=[])
    parser.add_argument("--load-timeout-s", type=float, default=900.0)
    parser.add_argument("--request-timeout-s", type=float, default=420.0)
    parser.add_argument("--workdir", default=None)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="akp_"))
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
        "schema": "vmlx-api-kwargs-parity-probe-v1",
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
        wait_ready(base_url, proc, args.load_timeout_s, log_path)
        log_offset = log_path.stat().st_size
        results = []
        for req in dialect_requests(args.nonce, args.effort, args.max_out):
            code, body = request_json(
                "POST",
                f"{base_url}{req['path']}",
                req["body"],
                timeout=args.request_timeout_s,
            )
            results.append(
                {"dialect": req["dialect"], "status_code": code}
                | ({"error": body} if code != 200 else {})
            )
            time.sleep(1.0)
        new_log = log_path.read_text(errors="replace")[log_offset:]
        resolved = parse_resolved_lines(new_log)
        status, divergences = compare_dialects(resolved)
        request_failures = [r for r in results if r["status_code"] != 200]
        if request_failures:
            status = "fail"
        artifact.update(
            {
                "status": status,
                "request_results": results,
                "resolved_lines": resolved,
                "divergences": divergences,
                "server_log_tail": new_log[-LOG_TAIL_CHARS:],
            }
        )
    except Exception as exc:
        artifact["error"] = f"{type(exc).__name__}: {exc}"
        try:
            artifact["server_log_tail"] = log_path.read_text(errors="replace")[
                -LOG_TAIL_CHARS:
            ]
        except OSError:
            pass
    finally:
        if proc is not None:
            artifact["server_exit_code"] = terminate_process(proc)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(
        f"[kwargs-parity] status={artifact['status']} "
        f"routes={[r['route'] for r in artifact.get('resolved_lines', [])]} "
        f"divergences={list(artifact.get('divergences', {}))} out={out_path}"
    )
    return {"parity": 0, "divergent": 3}.get(artifact["status"], 1)


if __name__ == "__main__":
    sys.exit(main())
