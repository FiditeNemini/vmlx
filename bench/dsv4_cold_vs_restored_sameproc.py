#!/usr/bin/env python3
"""DSV4 cold-vs-restored equivalence probe, same-process arm (task #149).

Proves (or refutes) that a served DeepSeek-V4 bundle produces byte-equivalent
answer TEXT at temperature 0 when the paged prefix cache is restored in the
SAME server process, across a 3-turn conversation replay.

Probe design (each point maps to a proven-fact ledger rule):
- Latency is never the proof: the comparison is answer TEXT at temperature 0,
  exact bytes first, whitespace-normalized second (reported separately).
- The COLD arm is a fresh server process (no block-disk cache flags at all)
  plus a run-supplied ``--nonce`` salted into every user turn, so no prior
  cache of any kind can hit. ``--no-paged-cache`` is NOT a no-cache control
  and is not used here.
- The RESTORED arm reruns the SAME 3-turn conversation in the same process,
  replaying the COLD arm's recorded assistant answers as history so every
  turn's prompt is byte-identical to the prompt that primed the cache.
- A >=2s settle gap precedes every request in both arms (prefix-cache A/B
  settle rule; symmetric pacing keeps cache state the only variable).
- Reuse must be PROVEN: ``usage.prompt_tokens_details.cached_tokens`` > 0 on
  at least one restored turn, else status is "inconclusive_no_reuse", never
  "pass". /v1/cache/stats is snapshotted before and after.
- Decode counting comes from ``usage.completion_tokens`` only.
- DSV4 contract: reasoning resolves via the bundle's chat.reasoning default,
  so ``enable_thinking`` is never sent; temperature 0, max_tokens 512; the
  two-pass answer boundary makes long "stalls" normal, hence the 420s
  request timeout.

Writes one JSON artifact (``--out``) and prints a one-line summary.
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

SERVED_MODEL_NAME = "dsv4-eq-probe"
REQUEST_MAX_TOKENS = 512
REQUEST_TIMEOUT_S_DEFAULT = 420.0  # DSV4 two-pass answer boundary: stalls are normal.
SETTLE_GAP_S = 2.5  # >=2s settle between a priming request and its reuse request.
STATS_LAG_S = 2.0  # /v1/cache/stats counters can lag the request that moved them.
LOG_TAIL_CHARS = 4000


# ---------------------------------------------------------------------------
# Small helpers (copied from bench/all_local_model_smoke.py conventions rather
# than imported: bench/ is not a package and sibling probes duplicate these).
# ---------------------------------------------------------------------------


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
    last_error = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"server exited during load rc={proc.returncode}: {log_tail(log_path)}"
            )
        code, body, _ = request_json("GET", f"{base_url}/health", timeout=5.0)
        if code == 200 and isinstance(body, dict) and body.get("model_loaded") is True:
            return body
        if isinstance(body, dict) and body.get("error"):
            last_error = str(body["error"])
        time.sleep(1.0)
    raise TimeoutError(f"server not healthy after {load_timeout_s}s: {last_error}: {log_tail(log_path)}")


def terminate_process(proc: subprocess.Popen[Any]) -> int | None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
    return proc.returncode


def extract_text(resp: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(resp, dict):
        return "", {}
    usage = resp.get("usage") if isinstance(resp.get("usage"), dict) else {}
    choices = resp.get("choices") or []
    if not choices:
        return "", usage
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content") if isinstance(message, dict) else ""
    return content if isinstance(content, str) else "", usage


def cached_tokens_of(resp: Any) -> int:
    usage = resp.get("usage") if isinstance(resp, dict) else None
    details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
    if not isinstance(details, dict):
        return 0
    try:
        return int(details.get("cached_tokens") or 0)
    except (TypeError, ValueError):
        return 0


def cache_detail_of(resp: Any) -> str:
    usage = resp.get("usage") if isinstance(resp, dict) else None
    details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
    return str(details.get("cache_detail") or "") if isinstance(details, dict) else ""


def completion_tokens_of(usage: Any) -> int:
    if not isinstance(usage, dict):
        return 0
    try:
        return int(usage.get("completion_tokens") or 0)
    except (TypeError, ValueError):
        return 0


def normalize_text(text: str) -> str:
    """Whitespace-normalized view: strip + collapse runs of whitespace."""
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Probe contract
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "DSV4 cold-vs-restored answer equivalence probe (same-process "
            "paged-cache arm). Private/internal gate."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--model", required=True, help="Path to the DSV4 bundle to serve.")
    parser.add_argument(
        "--nonce",
        required=True,
        help=(
            "Run-unique nonce salted into every user turn (novel-seed rule: "
            "reruns must never collide with stale cache entries). Required, "
            "no default."
        ),
    )
    parser.add_argument("--out", required=True, help="Path of the JSON artifact to write.")
    parser.add_argument("--port", type=int, default=8866)
    parser.add_argument("--load-timeout-s", type=float, default=900.0)
    parser.add_argument("--request-timeout-s", type=float, default=REQUEST_TIMEOUT_S_DEFAULT)
    parser.add_argument(
        "--workdir",
        default=None,
        help="Directory for server logs (default: fresh temp dir per run).",
    )
    return parser


def build_serve_cmd(model_path: str, port: int) -> list[str]:
    """Serve invocation mirroring bench/all_local_model_smoke.py.

    Deliberately NO block-disk-cache flags: the cold arm's freshness contract
    is a fresh process with no disk tier at all (plus the prompt nonce).
    """
    return [
        str(PYTHON),
        "-B",
        "-s",
        "-m",
        "vmlx_engine.cli",
        "serve",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--timeout",
        "240",
        "--max-num-seqs",
        "1",
        "--prefill-batch-size",
        "512",
        "--prefill-step-size",
        "1024",
        "--completion-batch-size",
        "256",
        "--continuous-batching",
        "--use-paged-cache",
        "--paged-cache-block-size",
        "64",
        "--max-cache-blocks",
        "1000",
        "--max-tokens",
        "1024",
        "--log-level",
        "INFO",
    ]


def conversation_turns(nonce: str) -> list[str]:
    """Fixed turn texts, nonce-salted so reruns never collide with stale cache.

    Turn one carries a deterministic filler paragraph so the FIRST prompt
    already exceeds the DSV4 composite's 256-token block floor — shorter
    prompts are skipped by the store ("prompt below snapshot/store
    threshold", proven live 2026-08-15), which made every earlier run
    structurally unable to store, hit, or prove reuse.
    """
    filler = " ".join(
        f"Context sentence {i} for the equivalence ledger: the probe "
        f"records deterministic filler text so token {i * 7} exceeds the "
        "composite block floor without changing the questions."
        for i in range(1, 19)
    )
    return [
        (
            f"Equivalence probe nonce {nonce}, turn one. Background: {filler} "
            "Now list the first five prime numbers in ascending order, then "
            "state their sum on its own line as 'SUM: <n>'."
        ),
        (
            f"Probe {nonce}, turn two. Multiply that sum by seven. Show the "
            "multiplication as '<a> x 7 = <b>' and finish with a line "
            "'PRODUCT: <b>'."
        ),
        (
            f"Probe {nonce}, turn three. Restate the prime list, the sum, and "
            "the product from this conversation in one short paragraph."
        ),
    ]


def chat_payload(messages: list[dict[str, str]]) -> dict[str, Any]:
    # DSV4 resolves reasoning from the bundle's chat.reasoning default:
    # never send enable_thinking. Temperature 0 / max_tokens 512 per contract.
    return {
        "model": SERVED_MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        "max_tokens": REQUEST_MAX_TOKENS,
    }


def run_conversation(
    base_url: str,
    turns: list[str],
    *,
    request_timeout_s: float,
    settle_gap_s: float = SETTLE_GAP_S,
    replay_answers: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run the 3-turn conversation; turns 2 and 3 replay the full history.

    With ``replay_answers`` (the restored arm), the assistant history uses the
    COLD arm's recorded answers so every turn's prompt is byte-identical to
    the prompt that primed the cache — otherwise a turn-1 divergence would
    silently change the later prompts and void the per-turn comparison.
    """
    results: list[dict[str, Any]] = []
    messages: list[dict[str, str]] = []
    for index, turn_text in enumerate(turns):
        messages.append({"role": "user", "content": turn_text})
        if settle_gap_s > 0:
            time.sleep(settle_gap_s)
        code, resp, elapsed = request_json(
            "POST",
            f"{base_url}/v1/chat/completions",
            chat_payload(messages),
            timeout=request_timeout_s,
        )
        text, usage = extract_text(resp)
        results.append(
            {
                "turn": index + 1,
                "http_code": code,
                "text": text,
                "usage": usage,
                "completion_tokens": completion_tokens_of(usage),
                "cached_tokens": cached_tokens_of(resp),
                "cache_detail": cache_detail_of(resp),
                "elapsed_s": round(elapsed, 3),
                "error": (resp or {}).get("error") if isinstance(resp, dict) else None,
            }
        )
        history_answer = text if replay_answers is None else replay_answers[index]
        messages.append({"role": "assistant", "content": history_answer})
    return results


def build_turn_records(
    cold: list[dict[str, Any]],
    restored: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for cold_turn, restored_turn in zip(cold, restored):
        cold_text = cold_turn["text"]
        restored_text = restored_turn["text"]
        records.append(
            {
                "turn": cold_turn["turn"],
                "cold_text": cold_text,
                "restored_text": restored_text,
                "byte_equal": cold_text.encode("utf-8") == restored_text.encode("utf-8"),
                "normalized_equal": normalize_text(cold_text) == normalize_text(restored_text),
                "cold_usage": cold_turn["usage"],
                "restored_usage": restored_turn["usage"],
                "cached_tokens": restored_turn["cached_tokens"],
                "cold_cached_tokens": cold_turn["cached_tokens"],
                "restored_cache_detail": restored_turn["cache_detail"],
                "cold_completion_tokens": cold_turn["completion_tokens"],
                "restored_completion_tokens": restored_turn["completion_tokens"],
                "cold_http_code": cold_turn["http_code"],
                "restored_http_code": restored_turn["http_code"],
            }
        )
    return records


def compute_verdict(
    turn_records: list[dict[str, Any]],
    table_hit_evidence: bool = False,
) -> str:
    """Map per-turn records to pass | fail | inconclusive_no_reuse.

    Ledger rules encoded here:
    - "pass" requires EVERY turn byte_equal AND proven reuse. The DSV4
      composite performs a clean N-1 reprefill over restored anchors, so
      ``usage.cached_tokens`` can stay 0 BY DESIGN on a real in-memory
      reuse; the composite's own attribution is the prefix-table hit
      counter. Reuse is therefore proven by ``table_hit_evidence`` (the
      scheduler_cache hits delta between the arms) OR cached_tokens > 0.
    - Without proven reuse the comparison proves nothing:
      "inconclusive_no_reuse", never "pass".
    - Any non-200 turn means the A/B did not complete: "fail".
    """
    if not turn_records:
        return "fail"
    for record in turn_records:
        if record.get("cold_http_code", 200) != 200:
            return "fail"
        if record.get("restored_http_code", 200) != 200:
            return "fail"
    reuse_proven = bool(table_hit_evidence) or any(
        int(record.get("cached_tokens") or 0) > 0 for record in turn_records
    )
    if not reuse_proven:
        return "inconclusive_no_reuse"
    if not all(record.get("byte_equal") is True for record in turn_records):
        return "fail"
    return "pass"


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if not args.nonce.strip():
        parser.error("--nonce must be non-empty")

    out_path = Path(args.out).resolve()
    if args.workdir:
        workdir = Path(args.workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        workdir = Path(tempfile.mkdtemp(prefix="dsv4_eq_sameproc_"))
    log_path = workdir / "server.log"
    base_url = f"http://127.0.0.1:{args.port}"
    turns = conversation_turns(args.nonce)
    serve_cmd = build_serve_cmd(args.model, args.port)

    artifact: dict[str, Any] = {
        "probe": "dsv4_cold_vs_restored_sameproc",
        "status": "fail",
        "model": args.model,
        "nonce": args.nonce,
        "turns": [],
        "cache_stats_before": None,
        "cache_stats_after": None,
        "server_log_tail": {},
        "port": args.port,
        "workdir": str(workdir),
        "serve_command": serve_cmd,
        "settle_gap_s": SETTLE_GAP_S,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(ROOT)
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            serve_cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True
        )
    try:
        artifact["health"] = wait_ready(base_url, proc, args.load_timeout_s, log_path)
        code, stats_initial, _ = request_json("GET", f"{base_url}/v1/cache/stats", timeout=30)
        artifact["cache_stats_initial"] = {"code": code, "body": stats_initial}

        cold = run_conversation(base_url, turns, request_timeout_s=args.request_timeout_s)
        artifact["cold_turns_raw"] = cold

        code, stats_before, _ = request_json("GET", f"{base_url}/v1/cache/stats", timeout=30)
        artifact["cache_stats_before"] = {"code": code, "body": stats_before}

        restored = run_conversation(
            base_url,
            turns,
            request_timeout_s=args.request_timeout_s,
            replay_answers=[turn["text"] for turn in cold],
        )
        artifact["restored_turns_raw"] = restored

        time.sleep(STATS_LAG_S)
        code, stats_after, _ = request_json("GET", f"{base_url}/v1/cache/stats", timeout=30)
        artifact["cache_stats_after"] = {"code": code, "body": stats_after}

        records = build_turn_records(cold, restored)
        artifact["turns"] = records

        def _table_hits(stats: Any) -> int:
            body = stats.get("body") if isinstance(stats, dict) else None
            body = body if isinstance(body, dict) else (stats or {})
            table = body.get("scheduler_cache") or {}
            try:
                return int(table.get("hits") or 0)
            except (TypeError, ValueError):
                return 0

        hits_delta = _table_hits(artifact.get("cache_stats_after") or {}) - _table_hits(
            artifact.get("cache_stats_before") or {}
        )
        artifact["table_hits_delta"] = hits_delta
        artifact["status"] = compute_verdict(records, table_hit_evidence=hits_delta > 0)
    except Exception as exc:  # noqa: BLE001 - artifact must always be written
        artifact["error"] = repr(exc)
        artifact["status"] = "fail"
    finally:
        artifact["server_returncode"] = terminate_process(proc)
        artifact["server_log_tail"] = {"server": log_tail(log_path)}

    write_json(out_path, artifact)
    records = artifact.get("turns") or []
    print(
        f"[dsv4-eq-sameproc] status={artifact['status']} model={Path(args.model).name} "
        f"nonce={args.nonce} byte_equal="
        f"{sum(1 for r in records if r.get('byte_equal'))}/{len(records)} "
        f"cached_tokens={[r.get('cached_tokens') for r in records]} out={out_path}",
        flush=True,
    )
    return {"pass": 0, "inconclusive_no_reuse": 2}.get(artifact["status"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
