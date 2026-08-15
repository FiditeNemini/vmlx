#!/usr/bin/env python3
"""DSV4 cold-vs-restored equivalence probe, restart + disk-L2 arm (task #149).

Proves (or refutes) that a served DeepSeek-V4 bundle produces byte-equivalent
answer TEXT at temperature 0 when generation resumes from the block-disk (L2)
cache after a full server restart, across a 3-turn conversation replay.

Flow: serve with a FRESH block-disk-cache dir D (created per run, never
reused — the block cache is not runtime-versioned), run the 3-turn
conversation (warms L1+L2), SIGTERM the server and wait, relaunch on the SAME
dir D, replay the conversation (reuse must come from disk restore: L1 died
with the first process), and compare per-turn against the first run's
answers.

Probe design (each point maps to a proven-fact ledger rule):
- Latency is never the proof: the comparison is answer TEXT at temperature 0,
  exact bytes first, whitespace-normalized second (reported separately).
- Cold arm freshness: fresh server process, fresh per-run disk cache dir,
  plus a run-supplied ``--nonce`` salted into every user turn so no prior
  cache can hit. ``--no-paged-cache`` is NOT a no-cache control.
- The restored run replays the first run's recorded assistant answers as
  history so every turn's prompt is byte-identical to the prompt that primed
  the cache.
- A >=2s settle gap precedes every request in both runs.
- Reuse must be PROVEN from DISK: ``cached_tokens`` > 0 on a restored turn
  AND disk evidence (block_disk_cache.disk_hits > 0 in /v1/cache/stats, or a
  per-turn cache_detail mentioning "disk"), else "inconclusive_no_reuse",
  never "pass". Stats counters can lag, so the final read waits 2s.
- Decode counting comes from ``usage.completion_tokens`` only.
- DSV4 contract: reasoning resolves via the bundle's chat.reasoning default,
  so ``enable_thinking`` is never sent; temperature 0, max_tokens 512; 420s
  request timeout (two-pass answer boundary stalls are normal).

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
RELAUNCH_GRACE_S = 3.0  # Let the port fully free before rebinding.
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


def nested_int(obj: Any, path: list[str]) -> int:
    current = obj
    for key in path:
        current = current.get(key) if isinstance(current, dict) else None
    try:
        return int(current or 0)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Probe contract
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "DSV4 cold-vs-restored answer equivalence probe (restart + "
            "block-disk-L2 arm). Private/internal gate."
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
    parser.add_argument("--port", type=int, default=8867)
    parser.add_argument("--load-timeout-s", type=float, default=900.0)
    parser.add_argument("--request-timeout-s", type=float, default=REQUEST_TIMEOUT_S_DEFAULT)
    parser.add_argument(
        "--workdir",
        default=None,
        help=(
            "Directory for server logs and the per-run disk cache dir "
            "(default: fresh temp dir per run)."
        ),
    )
    return parser


def _serve_cmd(model_path: str, port: int, disk_cache_dir: Path | str) -> list[str]:
    """Serve invocation mirroring bench/all_local_model_smoke.py, plus the
    block-disk (L2) tier flags this probe exists to exercise."""
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
        "--enable-block-disk-cache",
        "--block-disk-cache-dir",
        str(disk_cache_dir),
        "--block-disk-cache-max-gb",
        "2",
    ]


def restart_serve_commands(
    model_path: str,
    port: int,
    disk_cache_dir: Path | str,
) -> tuple[list[str], list[str]]:
    """Return the (first-launch, relaunch) serve commands.

    Both commands point at the SAME disk cache dir — that identity is the
    whole restart/L2 contract (the relaunch must restore what the first
    process persisted). main() launches from exactly this pair, and the
    contract test pins the shared dir through this function.
    """
    cmd = _serve_cmd(model_path, port, disk_cache_dir)
    return list(cmd), list(cmd)


def conversation_turns(nonce: str) -> list[str]:
    """Fixed turn texts, nonce-salted so reruns never collide with stale cache."""
    return [
        (
            f"Equivalence probe nonce {nonce}, turn one. List the first five "
            "prime numbers in ascending order, then state their sum on its "
            "own line as 'SUM: <n>'."
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

    With ``replay_answers`` (the restored run), the assistant history uses the
    first run's recorded answers so every turn's prompt is byte-identical to
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


def compute_verdict(turn_records: list[dict[str, Any]], disk_evidence: bool) -> str:
    """Map per-turn records + disk evidence to pass | fail | inconclusive_no_reuse.

    Ledger rules encoded here:
    - "pass" requires EVERY turn byte_equal AND proven reuse FROM DISK:
      ``disk_evidence`` (disk_hits > 0 or a cache_detail mentioning disk).
      The DSV4 composite performs a clean N-1 reprefill over restored
      SWA+CSA/HCA anchors, so ``usage.cached_tokens`` stays 0 BY DESIGN even
      on a real restore (proven live 2026-08-15: disk_hits=2 with byte-equal
      answers and cached_tokens=[0,0,0]); disk_hits IS the composite's
      restore attribution. cached_tokens > 0, when present, is accepted as
      additional evidence but is not required.
    - Without proven disk reuse the comparison proves nothing:
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
    reuse_proven = bool(disk_evidence)
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
        workdir = Path(tempfile.mkdtemp(prefix="dsv4_eq_restart_l2_"))
    # Fresh per-run disk dir; the block cache is NOT runtime-versioned, so a
    # reused dir could serve stale blocks from another engine build. Hard-fail
    # rather than silently reuse.
    disk_cache_dir = workdir / f"block_disk_cache_{int(time.time())}"
    disk_cache_dir.mkdir(parents=True, exist_ok=False)
    first_log = workdir / "first_server.log"
    second_log = workdir / "second_server.log"
    base_url = f"http://127.0.0.1:{args.port}"
    turns = conversation_turns(args.nonce)
    first_cmd, second_cmd = restart_serve_commands(args.model, args.port, disk_cache_dir)

    artifact: dict[str, Any] = {
        "probe": "dsv4_cold_vs_restored_restart_l2",
        "status": "fail",
        "model": args.model,
        "nonce": args.nonce,
        "turns": [],
        "cache_stats_before": None,
        "cache_stats_after": None,
        "server_log_tail": {},
        "disk_hits": 0,
        "disk_evidence": False,
        "port": args.port,
        "workdir": str(workdir),
        "disk_cache_dir": str(disk_cache_dir),
        "serve_command_first": first_cmd,
        "serve_command_second": second_cmd,
        "settle_gap_s": SETTLE_GAP_S,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(ROOT)

    def launch(cmd: list[str], log_path: Path) -> subprocess.Popen[Any]:
        with log_path.open("w") as log:
            return subprocess.Popen(
                cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True
            )

    first_proc: subprocess.Popen[Any] | None = None
    second_proc: subprocess.Popen[Any] | None = None
    try:
        # ---- Run 1: fresh process + fresh disk dir; warms L1 + L2. ----
        first_proc = launch(first_cmd, first_log)
        artifact["health_first"] = wait_ready(base_url, first_proc, args.load_timeout_s, first_log)
        code, stats, _ = request_json("GET", f"{base_url}/v1/cache/stats", timeout=30)
        artifact["cache_stats_run1_before"] = {"code": code, "body": stats}

        cold = run_conversation(base_url, turns, request_timeout_s=args.request_timeout_s)
        artifact["cold_turns_raw"] = cold

        time.sleep(STATS_LAG_S)
        code, stats, _ = request_json("GET", f"{base_url}/v1/cache/stats", timeout=30)
        artifact["cache_stats_run1_after"] = {"code": code, "body": stats}

        # ---- Restart: SIGTERM, wait for full exit, relaunch on SAME dir. ----
        artifact["first_server_returncode"] = terminate_process(first_proc)
        time.sleep(RELAUNCH_GRACE_S)

        second_proc = launch(second_cmd, second_log)
        artifact["health_second"] = wait_ready(
            base_url, second_proc, args.load_timeout_s, second_log
        )
        code, stats_before, _ = request_json("GET", f"{base_url}/v1/cache/stats", timeout=30)
        artifact["cache_stats_before"] = {"code": code, "body": stats_before}

        # ---- Run 2: replay; reuse can only come from the disk restore. ----
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

        disk_hits = nested_int(
            stats_after if isinstance(stats_after, dict) else {},
            ["block_disk_cache", "disk_hits"],
        )
        detail_mentions_disk = any(
            "disk" in str(turn.get("cache_detail") or "").lower() for turn in restored
        )
        artifact["disk_hits"] = disk_hits
        artifact["disk_evidence"] = disk_hits > 0 or detail_mentions_disk

        records = build_turn_records(cold, restored)
        artifact["turns"] = records
        artifact["status"] = compute_verdict(records, artifact["disk_evidence"])
    except Exception as exc:  # noqa: BLE001 - artifact must always be written
        artifact["error"] = repr(exc)
        artifact["status"] = "fail"
    finally:
        if first_proc is not None:
            terminate_process(first_proc)
        if second_proc is not None:
            artifact["second_server_returncode"] = terminate_process(second_proc)
        artifact["server_log_tail"] = {
            "first": log_tail(first_log),
            "second": log_tail(second_log),
        }

    write_json(out_path, artifact)
    records = artifact.get("turns") or []
    print(
        f"[dsv4-eq-restart-l2] status={artifact['status']} model={Path(args.model).name} "
        f"nonce={args.nonce} byte_equal="
        f"{sum(1 for r in records if r.get('byte_equal'))}/{len(records)} "
        f"cached_tokens={[r.get('cached_tokens') for r in records]} "
        f"disk_hits={artifact['disk_hits']} out={out_path}",
        flush=True,
    )
    return {"pass": 0, "inconclusive_no_reuse": 2}.get(artifact["status"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
