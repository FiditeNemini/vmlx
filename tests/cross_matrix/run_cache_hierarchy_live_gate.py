#!/usr/bin/env python3
"""Exercise prefix-cache hierarchy behavior against an already-running engine.

The gate deliberately uses one long, deterministic prompt prefix with two
different suffixes.  This makes exact and partial-prefix reuse distinguishable
without depending on model-specific answer quality.  Every request retains its
request JSON, raw Responses SSE, parsed summary, and post-request health state.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


def _json_get(url: str, timeout: int) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def _parse_sse(raw: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for block in raw.replace("\r\n", "\n").split("\n\n"):
        event_type = "message"
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_type = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].lstrip())
        if not data_lines:
            continue
        data_text = "\n".join(data_lines)
        if data_text == "[DONE]":
            events.append({"event": event_type, "data": data_text})
            continue
        try:
            data: Any = json.loads(data_text)
        except json.JSONDecodeError:
            data = {"_raw": data_text}
        events.append({"event": event_type, "data": data})
    return events


def _usage_from_event(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    usage = data.get("usage")
    if isinstance(usage, dict):
        return usage
    response = data.get("response")
    if isinstance(response, dict) and isinstance(response.get("usage"), dict):
        return response["usage"]
    return None


def _summarize(raw: str, elapsed_s: float, status_code: int) -> dict[str, Any]:
    events = _parse_sse(raw)
    output_text = "".join(
        str(item["data"].get("delta", ""))
        for item in events
        if item["event"] == "response.output_text.delta"
        and isinstance(item["data"], dict)
    )
    reasoning_text = "".join(
        str(item["data"].get("delta", ""))
        for item in events
        if item["event"] in {
            "response.reasoning.delta",
            "response.reasoning_text.delta",
            "response.reasoning_summary_text.delta",
        }
        and isinstance(item["data"], dict)
    )
    usages = [
        usage
        for item in events
        if (usage := _usage_from_event(item["data"])) is not None
    ]
    usage = usages[-1] if usages else {}
    input_details = usage.get("input_tokens_details")
    if not isinstance(input_details, dict):
        input_details = usage.get("prompt_tokens_details")
    if not isinstance(input_details, dict):
        input_details = {}
    terminals = [
        item["event"]
        for item in events
        if item["event"] in {"response.completed", "response.incomplete", "response.failed"}
    ]
    return {
        "status_code": status_code,
        "elapsed_s": round(elapsed_s, 3),
        "event_counts": dict(Counter(item["event"] for item in events)),
        "terminal_events": terminals,
        "output_text": output_text,
        "reasoning_text": reasoning_text,
        "usage": usage,
        "cached_tokens": int(input_details.get("cached_tokens") or 0),
        "cache_detail": input_details.get("cache_detail"),
    }


def _post_sse(url: str, payload: dict[str, Any], timeout: int) -> tuple[int, str, float]:
    started = time.monotonic()
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"accept": "text/event-stream", "content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
            return response.status, raw, time.monotonic() - started
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        return exc.code, raw, time.monotonic() - started


def _common_prefix(nonce: str, records: int) -> str:
    rows = [
        (
            f"CACHE-CONTRACT {nonce} record {index:04d}: preserve alpha-{index:04d}, "
            f"beta-{index * 7:05d}, gamma-{index * 13:05d}; do not summarize this record."
        )
        for index in range(records)
    ]
    return "\n".join(
        [
            "You are executing a cache-transport contract. Read the records, then obey only the final line.",
            *rows,
            "The records above are an immutable shared prefix.",
        ]
    )


def _payload(model: str, prompt: str) -> dict[str, Any]:
    return {
        "model": model,
        "input": prompt,
        "stream": True,
        "store": False,
        "max_output_tokens": 32,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 20,
        "enable_thinking": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("store", "probe"), required=True)
    parser.add_argument("--records", type=int, default=320)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    prefix = _common_prefix(args.nonce, args.records)
    prompt_a = f"{prefix}\nSUFFIX-A: Reply exactly CACHE-HIERARCHY-A-{args.nonce}."
    prompt_b = f"{prefix}\nSUFFIX-B: Reply exactly CACHE-HIERARCHY-B-{args.nonce}."
    requests = (
        [("cold_a", prompt_a), ("warm_a", prompt_a), ("partial_b", prompt_b)]
        if args.phase == "store"
        else [("restart_a", prompt_a), ("restart_partial_b", prompt_b)]
    )

    health_before = _json_get(f"{args.base_url}/health", args.timeout)
    (args.artifact_dir / "health_before.json").write_text(
        json.dumps(health_before, indent=2, sort_keys=True) + "\n"
    )
    rows: list[dict[str, Any]] = []
    for tag, prompt in requests:
        payload = _payload(args.model, prompt)
        request_path = args.artifact_dir / f"{tag}.request.json"
        raw_path = args.artifact_dir / f"{tag}.sse"
        request_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        code, raw, elapsed = _post_sse(
            f"{args.base_url}/v1/responses", payload, args.timeout
        )
        raw_path.write_text(raw)
        summary = _summarize(raw, elapsed, code)
        health = _json_get(f"{args.base_url}/health", args.timeout)
        health_path = args.artifact_dir / f"{tag}.health.json"
        health_path.write_text(json.dumps(health, indent=2, sort_keys=True) + "\n")
        suffix = "A" if tag.endswith("_a") else "B"
        summary.update(
            {
                "tag": tag,
                "expected_marker": f"CACHE-HIERARCHY-{suffix}-{args.nonce}",
                "request_path": str(request_path),
                "raw_path": str(raw_path),
                "health_path": str(health_path),
                "last_cache_execution": (health.get("scheduler") or {}).get(
                    "last_cache_execution"
                ),
                "scheduler_cache": ((health.get("cache") or {}).get("scheduler_cache") or {}),
                "block_disk_cache": ((health.get("cache") or {}).get("block_disk_cache") or {}),
            }
        )
        summary["marker_ok"] = summary["expected_marker"] in summary["output_text"]
        summary["terminal_ok"] = summary["terminal_events"] == ["response.completed"]
        rows.append(summary)
        print(
            json.dumps(
                {
                    "tag": tag,
                    "status": code,
                    "marker_ok": summary["marker_ok"],
                    "cached_tokens": summary["cached_tokens"],
                    "cache_detail": summary["cache_detail"],
                    "last_cache_execution": summary["last_cache_execution"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    result = {
        "phase": args.phase,
        "nonce": args.nonce,
        "base_url": args.base_url,
        "model": args.model,
        "health_before": health_before,
        "requests": rows,
    }
    (args.artifact_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    required_ok = all(
        row["status_code"] == 200 and row["marker_ok"] and row["terminal_ok"]
        for row in rows
    )
    return 0 if required_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
