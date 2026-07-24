#!/usr/bin/env python3
"""Exercise prefix-cache hierarchy behavior against an already-running engine.

The gate deliberately uses one long, deterministic prompt prefix with distinct
suffixes.  This makes exact and partial-prefix reuse distinguishable without
depending on model-specific answer quality.  Every request retains its request
JSON, raw Responses SSE, parsed summary, and pre/post-request health state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def _json_get(url: str, timeout: int) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def _json_post(
    url: str,
    payload: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    body = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        parsed = json.loads(response.read())
    if not isinstance(parsed, dict):
        raise ValueError("JSON response is not an object")
    return parsed


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
        if item["event"]
        in {
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
        if item["event"]
        in {"response.completed", "response.incomplete", "response.failed"}
    ]
    response_ids: list[str] = []
    for item in events:
        if item["event"] not in {
            "response.created",
            "response.completed",
            "response.incomplete",
            "response.failed",
        }:
            continue
        data = item["data"]
        if not isinstance(data, dict):
            continue
        response = data.get("response")
        if not isinstance(response, dict):
            continue
        response_id = str(response.get("id") or "").strip()
        if response_id:
            response_ids.append(response_id)
    unique_response_ids = list(dict.fromkeys(response_ids))
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
        "response_id": (
            unique_response_ids[0] if len(unique_response_ids) == 1 else None
        ),
        "response_ids": unique_response_ids,
        "response_id_consistent": len(unique_response_ids) == 1,
    }


def _integer(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _health_cache_counters(health: dict[str, Any]) -> dict[str, int]:
    """Return monotonic cache counters needed to prove request-local deltas."""
    scheduler = health.get("scheduler")
    if not isinstance(scheduler, dict):
        scheduler = {}
    cache = health.get("cache")
    if not isinstance(cache, dict):
        cache = {}
    scheduler_cache = cache.get("scheduler_cache")
    if not isinstance(scheduler_cache, dict):
        scheduler_cache = {}
    block_disk_cache = cache.get("block_disk_cache")
    if not isinstance(block_disk_cache, dict):
        block_disk_cache = {}
    return {
        "scheduler.cache_hit_requests": _integer(scheduler.get("cache_hit_requests")),
        "scheduler.cache_hit_tokens": _integer(scheduler.get("cache_hit_tokens")),
        "scheduler_cache.cache_hits": _integer(scheduler_cache.get("cache_hits")),
        "scheduler_cache.cache_misses": _integer(scheduler_cache.get("cache_misses")),
        "scheduler_cache.disk_hits": _integer(scheduler_cache.get("disk_hits")),
        "scheduler_cache.disk_misses": _integer(scheduler_cache.get("disk_misses")),
        "scheduler_cache.disk_promotion_hits": _integer(
            scheduler_cache.get("disk_promotion_hits")
        ),
        "scheduler_cache.evictions": _integer(scheduler_cache.get("evictions")),
        "scheduler_cache.tokens_saved": _integer(scheduler_cache.get("tokens_saved")),
        "block_disk_cache.disk_hits": _integer(block_disk_cache.get("disk_hits")),
        "block_disk_cache.disk_misses": _integer(block_disk_cache.get("disk_misses")),
        "block_disk_cache.disk_writes": _integer(block_disk_cache.get("disk_writes")),
        "block_disk_cache.disk_evictions": _integer(
            block_disk_cache.get("disk_evictions")
        ),
        "block_disk_cache.blocks_on_disk": _integer(
            block_disk_cache.get("blocks_on_disk")
        ),
        "block_disk_cache.total_tokens_on_disk": _integer(
            block_disk_cache.get("total_tokens_on_disk")
        ),
    }


def _counter_deltas(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {
        key: _integer(after.get(key)) - _integer(before.get(key))
        for key in sorted(set(before) | set(after))
    }


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()


def _token_contract_request(
    model: str,
    prompts: dict[str, str],
) -> dict[str, Any]:
    """Describe the exact Responses prompt shape without performing generation."""
    return {
        "contract_version": 1,
        "surface": "responses",
        "model": model,
        "inputs": prompts,
        "request_controls": {
            "enable_thinking": False,
            "instructions": None,
            "tools": [],
        },
    }


def _validate_tokenizer_lcp_contract(
    contract: dict[str, Any],
    *,
    request_payload: dict[str, Any],
    health_attestation: dict[str, Any],
) -> list[str]:
    """Validate a no-cache, final-render tokenizer contract.

    The contract must be produced before generation by a dry-render endpoint
    that runs the same Responses conversion/template/tokenizer path but never
    invokes cache lookup. It exposes only counts and hashes, not token IDs.
    """
    failures: list[str] = []
    if contract.get("contract_version") != 1:
        failures.append("token contract: contract_version must be 1")
    if contract.get("method") != "final-render-tokenize-no-cache":
        failures.append(
            "token contract: method must be final-render-tokenize-no-cache"
        )
    if contract.get("surface") != "responses":
        failures.append("token contract: surface must be responses")
    if contract.get("cache_lookup_bypassed") is not True:
        failures.append("token contract: cache_lookup_bypassed is not true")

    expected_request_sha = _canonical_sha256(request_payload)
    if contract.get("request_sha256") != expected_request_sha:
        failures.append(
            "token contract: request_sha256 does not bind the exact dry-render "
            "request"
        )
    for field, contract_field in (
        ("model_bundle_provenance", "model_bundle_fingerprint_sha256"),
        ("cache_topology_provenance", "cache_topology_fingerprint_sha256"),
    ):
        attestation = health_attestation.get(field)
        expected_fingerprint = (
            str(attestation.get("fingerprint_sha256") or "")
            if isinstance(attestation, dict)
            else ""
        )
        if not expected_fingerprint:
            failures.append(
                f"token contract: /health {field} fingerprint is unavailable"
            )
        elif contract.get(contract_field) != expected_fingerprint:
            failures.append(
                f"token contract: {contract_field} does not match /health"
            )

    inputs = request_payload.get("inputs")
    rows = contract.get("prompts")
    if not isinstance(inputs, dict):
        return failures + ["token contract: request inputs are missing"]
    if not isinstance(rows, dict):
        return failures + ["token contract: prompts are missing"]
    expected_labels = set(inputs)
    if set(rows) != expected_labels:
        failures.append(
            f"token contract: prompt labels {sorted(rows)} do not equal "
            f"{sorted(expected_labels)}"
        )
    for label, prompt in inputs.items():
        row = rows.get(label)
        if not isinstance(row, dict):
            failures.append(f"token contract: prompt {label} metadata is missing")
            continue
        expected_input_sha = hashlib.sha256(str(prompt).encode()).hexdigest()
        if row.get("input_sha256") != expected_input_sha:
            failures.append(
                f"token contract: prompt {label} input_sha256 does not match"
            )
        token_count = _integer(row.get("cache_prompt_token_count"))
        if token_count <= 1:
            failures.append(
                f"token contract: prompt {label} cache token count is not usable"
            )
        token_sha = str(row.get("cache_prompt_token_ids_sha256") or "")
        if len(token_sha) != 64 or any(
            character not in "0123456789abcdef"
            for character in token_sha.lower()
        ):
            failures.append(
                f"token contract: prompt {label} token-ID digest is invalid"
            )

    lcp = contract.get("longest_common_prefix_tokens")
    if not isinstance(lcp, dict):
        return failures + [
            "token contract: longest_common_prefix_tokens are missing"
        ]
    a_row = rows.get("A") if isinstance(rows, dict) else None
    a_count = (
        _integer(a_row.get("cache_prompt_token_count"))
        if isinstance(a_row, dict)
        else 0
    )
    if _integer(lcp.get("A:A")) != a_count:
        failures.append(
            "token contract: A:A LCP does not equal the independently tokenized "
            "A prompt length"
        )
    for label in sorted(expected_labels - {"A"}):
        pair = f"A:{label}"
        value = _integer(lcp.get(pair))
        other = rows.get(label)
        other_count = (
            _integer(other.get("cache_prompt_token_count"))
            if isinstance(other, dict)
            else 0
        )
        if value <= 1:
            failures.append(
                f"token contract: {pair} does not prove a multi-token prefix"
            )
        if a_count > 0 and other_count > 0 and value >= min(a_count, other_count):
            failures.append(
                f"token contract: {pair} must leave a tokenizer-visible "
                "differing tail"
            )
    return failures


def _fetch_tokenizer_lcp_contract(
    *,
    base_url: str,
    model: str,
    prompts: dict[str, str],
    timeout: int,
    health_attestation: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Fetch an independent final-render token/LCP contract, failing closed."""
    request_payload = _token_contract_request(model, prompts)
    try:
        contract = _json_post(
            f"{base_url}/v1/cache/token-contract",
            request_payload,
            timeout,
        )
    except (OSError, ValueError, urllib.error.URLError) as exc:
        return {}, [
            "token contract: independent tokenizer endpoint is unavailable; "
            "required interface is POST /v1/cache/token-contract over the "
            "loaded model's final Responses render path, returning path-free "
            "per-prompt token counts/digests and pairwise LCP counts without "
            f"invoking cache lookup ({exc})"
        ]
    failures = _validate_tokenizer_lcp_contract(
        contract,
        request_payload=request_payload,
        health_attestation=health_attestation,
    )
    return contract, failures


def _run_text(command: list[str], *, cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(cwd) if cwd is not None else None,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"could not run {command[0]}: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"{command[0]} exited {completed.returncode}: {detail or 'no output'}"
        )
    return completed.stdout.strip()


def _listener_pids(port: int) -> set[int]:
    output = _run_text(
        [
            "/usr/sbin/lsof",
            "-nP",
            f"-iTCP:{port}",
            "-sTCP:LISTEN",
            "-Fp",
        ]
    )
    return {
        int(line[1:])
        for line in output.splitlines()
        if line.startswith("p") and line[1:].isdigit()
    }


def _listener_cwd(pid: int) -> Path:
    output = _run_text(
        [
            "/usr/sbin/lsof",
            "-nP",
            "-a",
            "-p",
            str(pid),
            "-d",
            "cwd",
            "-Fn",
        ]
    )
    paths = [
        Path(line[1:]).resolve()
        for line in output.splitlines()
        if line.startswith("n") and line[1:]
    ]
    if len(paths) != 1:
        raise RuntimeError(
            f"expected one cwd path for listener PID {pid}, found {paths}"
        )
    return paths[0]


def _listener_launch(command: str, cwd: Path) -> tuple[Path, str]:
    """Validate a supported vmlx-engine launch and return its interpreter."""
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        raise RuntimeError(f"could not parse listener command: {exc}") from exc
    if not argv:
        raise RuntimeError("listener command is empty")
    launch_path = Path(argv[0]).expanduser()
    if not launch_path.is_absolute():
        launch_path = cwd / launch_path
    launch_path = launch_path.absolute()
    launch_name = launch_path.name.lower()
    if launch_name.startswith("python"):
        try:
            module_index = argv.index("-m")
        except ValueError as exc:
            raise RuntimeError(
                "Python listener was not launched with -m vmlx_engine.cli"
            ) from exc
        if (
            module_index + 2 >= len(argv)
            or argv[module_index + 1] != "vmlx_engine.cli"
            or argv[module_index + 2] != "serve"
        ):
            raise RuntimeError(
                "Python listener launch is not `-m vmlx_engine.cli serve`"
            )
        return launch_path, "python-module-vmlx-engine-cli"
    if launch_name != "vmlx-engine":
        raise RuntimeError(
            f"listener launcher {launch_path} is not the vmlx-engine console script"
        )
    if len(argv) < 2 or argv[1] != "serve":
        raise RuntimeError("vmlx-engine listener command does not use `serve`")
    try:
        launcher_text = launch_path.read_text(errors="replace")
        shebang = launcher_text.splitlines()[0]
    except (OSError, IndexError) as exc:
        raise RuntimeError(
            f"could not read listener launcher {launch_path}: {exc}"
        ) from exc
    if not shebang.startswith("#!"):
        raise RuntimeError(f"listener launcher {launch_path} has no Python shebang")
    if (
        "from vmlx_engine.cli import main" not in launcher_text
        or "main()" not in launcher_text
    ):
        raise RuntimeError(
            f"listener launcher {launch_path} does not invoke vmlx_engine.cli:main"
        )
    interpreter_argv = shlex.split(shebang[2:].strip())
    if len(interpreter_argv) != 1:
        raise RuntimeError(f"listener launcher {launch_path} uses an ambiguous shebang")
    interpreter = Path(interpreter_argv[0]).expanduser()
    if not interpreter.is_absolute():
        raise RuntimeError(
            f"listener launcher {launch_path} does not name an absolute interpreter"
        )
    if not interpreter.name.lower().startswith("python"):
        raise RuntimeError(
            f"listener launcher {launch_path} does not name Python directly"
        )
    return interpreter, "console-script-vmlx-engine"


def _observe_local_listener_identity(base_url: str) -> dict[str, Any]:
    """Observe the actual macOS process listening for the tested local engine."""
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise RuntimeError(
            f"base URL host {host!r} is not a supported localhost listener"
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise RuntimeError(f"base URL has an invalid port: {exc}") from exc
    if port is None:
        port = 443 if parsed.scheme == "https" else 80

    pids_before = _listener_pids(port)
    if len(pids_before) != 1:
        raise RuntimeError(
            f"expected one LISTEN PID on localhost:{port}, found {sorted(pids_before)}"
        )
    pid = next(iter(pids_before))
    started_at = _run_text(["/bin/ps", "-p", str(pid), "-o", "lstart="])
    command = _run_text(["/bin/ps", "-p", str(pid), "-o", "command="])
    cwd = _listener_cwd(pid)
    python_executable, launch_shape = _listener_launch(command, cwd)
    pids_after = _listener_pids(port)
    if pids_after != {pid}:
        raise RuntimeError(
            f"listener changed while observing localhost:{port}: "
            f"{sorted(pids_before)} -> {sorted(pids_after)}"
        )
    if not started_at or not command:
        raise RuntimeError(f"could not capture start time and command for PID {pid}")

    fingerprint_payload = {
        "host": host,
        "port": port,
        "pid": pid,
        "started_at": started_at,
        "command": command,
        "cwd": str(cwd),
        "python_executable": str(python_executable),
        "launch_shape": launch_shape,
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "method": "macos-lsof-ps",
        **fingerprint_payload,
        "fingerprint_sha256": fingerprint,
    }


def _observe_source_checkout() -> dict[str, Any]:
    """Record the checkout containing this harness and its current Git HEAD."""
    expected_root = Path(__file__).resolve().parents[2]
    git_root = Path(
        _run_text(
            [
                "/usr/bin/git",
                "-C",
                str(expected_root),
                "rev-parse",
                "--show-toplevel",
            ]
        )
    ).resolve()
    if git_root != expected_root:
        raise RuntimeError(
            f"harness resolved to Git root {git_root}, expected {expected_root}"
        )
    head = _run_text(
        [
            "/usr/bin/git",
            "-C",
            str(git_root),
            "rev-parse",
            "HEAD",
        ]
    )
    if len(head) != 40:
        raise RuntimeError(f"Git HEAD is not a full commit SHA: {head!r}")
    status_text = _run_text(
        [
            "/usr/bin/git",
            "-C",
            str(git_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ]
    )
    status_lines = status_text.splitlines() if status_text else []
    return {
        "git_root": str(git_root),
        "head": head,
        "dirty": bool(status_lines),
        "status_porcelain": status_lines,
        "status_sha256": hashlib.sha256(status_text.encode()).hexdigest(),
    }


def _compare_source_checkout_observations(
    before: dict[str, Any],
    after: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    for field in ("git_root", "head", "dirty", "status_sha256"):
        if before.get(field) != after.get(field):
            failures.append(
                f"provenance: observed source {field} changed during the live gate"
            )
    if after.get("dirty") is not False:
        failures.append(
            "provenance: observed source checkout is dirty at final attestation"
        )
    return failures


def _validate_runtime_source_provenance(
    observed_engine: dict[str, Any],
    observed_source: dict[str, Any],
) -> list[str]:
    """Bind the listener's import resolution to one clean source checkout."""
    failures: list[str] = []
    try:
        git_root = Path(str(observed_source["git_root"])).resolve()
    except (KeyError, OSError):
        return ["provenance: observed source Git root is missing or invalid"]
    try:
        listener_cwd = Path(str(observed_engine["cwd"])).resolve()
    except (KeyError, OSError):
        failures.append("provenance: listener cwd is missing or invalid")
    else:
        if listener_cwd != git_root and git_root not in listener_cwd.parents:
            failures.append(
                "provenance: listener cwd is outside the observed source checkout"
            )
    if not str(observed_engine.get("python_executable") or ""):
        failures.append("provenance: listener Python executable is empty")
    if observed_engine.get("launch_shape") not in {
        "console-script-vmlx-engine",
        "python-module-vmlx-engine-cli",
    }:
        failures.append("provenance: listener launch shape is unsupported")
    if observed_source.get("dirty") is not False:
        failures.append(
            "provenance: observed source checkout is dirty; HEAD does not identify "
            "the running source exactly"
        )
    return failures


def _python_source_tree_digest(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    count = 0
    read_errors = 0
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root.parent).as_posix().encode()
        try:
            content = path.read_bytes()
        except OSError:
            digest.update(relative)
            digest.update(b"\0UNREADABLE\0")
            read_errors += 1
            continue
        digest.update(relative)
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
        count += 1
    return digest.hexdigest(), count, read_errors


def _validate_health_runtime_provenance(
    health: dict[str, Any],
    observed_engine: dict[str, Any],
    observed_source: dict[str, Any],
) -> list[str]:
    """Verify the listener PID attests to exact source files without path leaks."""
    failures: list[str] = []
    attestation = health.get("runtime_provenance")
    if not isinstance(attestation, dict):
        return ["provenance: /health runtime_provenance attestation is missing"]
    git_root = Path(str(observed_source.get("git_root") or "")).resolve()
    source_tree_sha256, source_file_count, source_read_error_count = (
        _python_source_tree_digest(git_root / "vmlx_engine")
    )
    expected = {
        "pid": _integer(observed_engine.get("pid")),
        "server_module_relpath": "vmlx_engine/server.py",
        "server_module_sha256": hashlib.sha256(
            (git_root / "vmlx_engine" / "server.py").read_bytes()
        ).hexdigest(),
        "package_init_relpath": "vmlx_engine/__init__.py",
        "package_init_sha256": hashlib.sha256(
            (git_root / "vmlx_engine" / "__init__.py").read_bytes()
        ).hexdigest(),
        "python_source_tree_sha256": source_tree_sha256,
        "python_source_file_count": source_file_count,
        "python_source_read_error_count": source_read_error_count,
        "python_executable_fingerprint_sha256": hashlib.sha256(
            str(observed_engine.get("python_executable") or "").encode()
        ).hexdigest(),
    }
    for field, expected_value in expected.items():
        if attestation.get(field) != expected_value:
            failures.append(
                f"provenance: /health {field} does not match the observed "
                "listener/source"
            )
    if source_read_error_count:
        failures.append(
            "provenance: source-tree attestation could not read every Python file"
        )
    return failures


def _valid_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text.lower()
    )


def _health_attestation_snapshot(
    health: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Capture the immutable runtime, bundle, and cache topology attestations."""
    failures: list[str] = []
    snapshot: dict[str, Any] = {}
    for field in (
        "runtime_provenance",
        "model_bundle_provenance",
        "cache_topology_provenance",
    ):
        value = health.get(field)
        if not isinstance(value, dict) or not value:
            failures.append(f"provenance: /health {field} attestation is missing")
            continue
        snapshot[field] = value

    for field in ("model_bundle_provenance", "cache_topology_provenance"):
        value = snapshot.get(field)
        if isinstance(value, dict) and not _valid_sha256(
            value.get("fingerprint_sha256")
        ):
            failures.append(
                f"provenance: /health {field}.fingerprint_sha256 is invalid"
            )

    def _contains_private_absolute_path(value: Any) -> bool:
        if isinstance(value, dict):
            return any(
                _contains_private_absolute_path(item) for item in value.values()
            )
        if isinstance(value, list):
            return any(_contains_private_absolute_path(item) for item in value)
        if not isinstance(value, str):
            return False
        return value.startswith(("/Users/", "/Volumes/", "/private/", "/tmp/"))

    for field, value in snapshot.items():
        if _contains_private_absolute_path(value):
            failures.append(
                f"provenance: /health {field} leaks an absolute local path"
            )

    if snapshot:
        snapshot["combined_sha256"] = _canonical_sha256(snapshot)
    return snapshot, failures


def _compare_attestation_snapshots(
    before: dict[str, Any],
    after: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    for field in (
        "runtime_provenance",
        "model_bundle_provenance",
        "cache_topology_provenance",
        "combined_sha256",
    ):
        if before.get(field) != after.get(field):
            failures.append(
                f"provenance: /health {field} changed during the live gate"
            )
    return failures


def _nearest_existing_directory(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            return candidate
        candidate = parent
    return candidate if candidate.is_dir() else candidate.parent


def _is_within_directory(path: Path, parent: Path) -> bool:
    """Use filesystem identity so symlinks and APFS case aliases cannot escape."""
    try:
        guarded_parent = parent.resolve(strict=True)
    except OSError:
        guarded_parent = parent.resolve()
    existing = _nearest_existing_directory(path)
    for candidate in (existing, *existing.parents):
        try:
            if candidate.samefile(guarded_parent):
                return True
        except OSError:
            continue
    return False


def _is_in_git_context(path: Path) -> bool:
    """Return whether an existing ancestor is in any worktree or Git metadata."""
    directory = _nearest_existing_directory(path)
    environment = dict(os.environ)
    for name in ("GIT_CEILING_DIRECTORIES", "GIT_DIR", "GIT_WORK_TREE"):
        environment.pop(name, None)
    environment["LC_ALL"] = "C"
    try:
        proc = subprocess.run(
            [
                "/usr/bin/git",
                "-C",
                str(directory),
                "rev-parse",
                "--is-inside-work-tree",
                "--is-inside-git-dir",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            "cannot safely inspect artifact directory for Git containment"
        ) from exc
    if proc.returncode != 0:
        if "not a git repository" in proc.stderr.lower():
            return False
        raise RuntimeError(
            "cannot safely inspect artifact directory for Git containment"
        )
    return any(line.strip().lower() == "true" for line in proc.stdout.splitlines())


def _external_artifact_dir(path: Path) -> Path:
    """Resolve private evidence storage outside every Git worktree/repository."""
    guarded_worktree = Path(__file__).resolve().parents[2]
    resolved = path.expanduser().resolve()
    if _is_within_directory(resolved, guarded_worktree) or _is_in_git_context(
        resolved
    ):
        raise RuntimeError(
            f"artifact directory {resolved} must resolve outside every Git "
            "worktree, repository, and Git metadata directory"
        )
    return resolved


def _wait_for_store_durability(
    *,
    base_url: str,
    request_timeout: int,
    request_id: str,
    baseline_counters: dict[str, int],
    timeout_s: float,
    poll_interval_s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Wait for one request's post-eviction write fence to settle twice."""
    started = time.monotonic()
    deadline = started + max(0.0, timeout_s)
    polls = 0
    last_health: dict[str, Any] = {}
    final_counters = dict(baseline_counters)
    deltas = _counter_deltas(baseline_counters, final_counters)
    last_error: str | None = None
    matching_fence: dict[str, Any] | None = None
    pipeline_snapshot: dict[str, Any] = {}
    stable_signature: str | None = None
    stable_observations = 0
    contract_failures: list[str] = []

    while True:
        polls += 1
        try:
            last_health = _json_get(f"{base_url}/health", request_timeout)
            final_counters = _health_cache_counters(last_health)
            deltas = _counter_deltas(baseline_counters, final_counters)
            last_error = None
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = str(exc)

        cache = last_health.get("cache")
        if not isinstance(cache, dict):
            cache = {}
        block_disk = cache.get("block_disk_cache")
        if not isinstance(block_disk, dict):
            block_disk = {}
        pipeline = block_disk.get("write_pipeline")
        pipeline_snapshot = pipeline if isinstance(pipeline, dict) else {}
        recent_fences = pipeline_snapshot.get("recent_fences")
        if not isinstance(recent_fences, list):
            recent_fences = []
        matching_fence = next(
            (
                dict(fence)
                for fence in reversed(recent_fences)
                if isinstance(fence, dict)
                and str(fence.get("request_id") or "") == request_id
            ),
            None,
        )
        scheduler = last_health.get("scheduler")
        if not isinstance(scheduler, dict):
            scheduler = {}

        contract_failures = []
        if matching_fence is None:
            contract_failures.append(
                f"no block-disk write fence matches request_id={request_id}"
            )
        else:
            expected = _integer(matching_fence.get("expected"))
            queued = _integer(matching_fence.get("queued"))
            completed = _integer(matching_fence.get("completed"))
            retained = _integer(matching_fence.get("retained"))
            failed = _integer(matching_fence.get("failed"))
            dropped = _integer(matching_fence.get("dropped"))
            if expected <= 0:
                contract_failures.append("request write fence expected no blocks")
            if queued != expected:
                contract_failures.append(
                    f"request write fence queued={queued} != expected={expected}"
                )
            if completed != expected:
                contract_failures.append(
                    f"request write fence completed={completed} != expected={expected}"
                )
            if failed != 0:
                contract_failures.append(
                    f"request write fence failed={failed} is not zero"
                )
            if dropped != 0:
                contract_failures.append(
                    f"request write fence dropped={dropped} is not zero"
                )
            if retained != expected:
                contract_failures.append(
                    f"request write fence retained={retained} != expected={expected}"
                )
            if matching_fence.get("sealed") is not True:
                contract_failures.append("request write fence is not sealed")
            if matching_fence.get("seal_enqueued") is not True:
                contract_failures.append("request write fence sentinel was not queued")
            if matching_fence.get("seal_failed") is not False:
                contract_failures.append("request write fence sentinel failed")
            if matching_fence.get("post_eviction_complete") is not True:
                contract_failures.append(
                    "request write fence is not complete after eviction"
                )
            if _integer(matching_fence.get("completion_generation")) <= 0:
                contract_failures.append(
                    "request write fence has no completion generation"
                )
        if _integer(pipeline_snapshot.get("queue_depth")) != 0:
            contract_failures.append("block-disk write queue is not empty")
        if _integer(pipeline_snapshot.get("inflight")) != 0:
            contract_failures.append("block-disk writes are still in flight")
        if pipeline_snapshot.get("writer_alive") is not True:
            contract_failures.append("block-disk writer is not alive")
        if _integer(scheduler.get("num_waiting")) != 0:
            contract_failures.append("scheduler still has waiting requests")
        if _integer(scheduler.get("num_running")) != 0:
            contract_failures.append("scheduler still has running requests")
        if _integer(deltas.get("block_disk_cache.disk_writes")) <= 0:
            contract_failures.append(
                "block-disk committed-write counter did not increase"
            )

        candidate_signature = json.dumps(
            {
                "fence": matching_fence,
                "completion_generation": pipeline_snapshot.get("completion_generation"),
                "queue_depth": pipeline_snapshot.get("queue_depth"),
                "inflight": pipeline_snapshot.get("inflight"),
                "disk_writes": final_counters.get("block_disk_cache.disk_writes"),
                "disk_evictions": final_counters.get("block_disk_cache.disk_evictions"),
                "scheduler_num_waiting": scheduler.get("num_waiting"),
                "scheduler_num_running": scheduler.get("num_running"),
            },
            sort_keys=True,
        )
        if not contract_failures:
            if candidate_signature == stable_signature:
                stable_observations += 1
            else:
                stable_signature = candidate_signature
                stable_observations = 1
        else:
            stable_signature = None
            stable_observations = 0

        durable = not contract_failures and stable_observations >= 2
        now = time.monotonic()
        if durable or now >= deadline:
            return (
                {
                    "ok": durable,
                    "polls": polls,
                    "elapsed_s": round(now - started, 3),
                    "timeout_s": timeout_s,
                    "poll_interval_s": poll_interval_s,
                    "request_id": request_id,
                    "exact_request_identity_proven": matching_fence is not None,
                    "post_eviction_fence_required": True,
                    "stable_observations": stable_observations,
                    "matching_fence": matching_fence,
                    "write_pipeline": pipeline_snapshot,
                    "contract_failures": contract_failures,
                    "baseline_counters": baseline_counters,
                    "final_counters": final_counters,
                    "counter_deltas": deltas,
                    "last_error": last_error,
                },
                last_health,
            )
        time.sleep(min(poll_interval_s, max(0.0, deadline - now)))


def _validate_hit_row(
    row: dict[str, Any],
    *,
    require_partial: bool,
    require_disk_origin: bool,
    minimum_cached_tokens: int,
    maximum_cached_tokens: int,
    expected_prompt_tokens: int,
    allow_direct_reuse: bool,
) -> list[str]:
    """Validate that a nominal hit actually reused a continuous prompt prefix."""
    tag = str(row.get("tag") or "<missing-tag>")
    failures: list[str] = []
    execution = row.get("last_cache_execution")
    if not isinstance(execution, dict):
        return [f"{tag}: missing scheduler.last_cache_execution"]

    response_id = str(row.get("response_id") or "")
    execution_request_id = str(execution.get("request_id") or "")
    if row.get("response_id_consistent") is not True or not response_id:
        failures.append(f"{tag}: Responses SSE has no single consistent response ID")
    if not execution_request_id:
        failures.append(f"{tag}: execution request_id is empty")
    elif response_id and execution_request_id != response_id:
        failures.append(
            f"{tag}: Responses SSE id={response_id} does not match "
            f"execution request_id={execution_request_id}"
        )

    cached_tokens = _integer(row.get("cached_tokens"))
    execution_cached_tokens = _integer(execution.get("cached_tokens"))
    attempted_cached_tokens = _integer(execution.get("attempted_cached_tokens"))
    prompt_tokens = _integer(execution.get("prompt_tokens"))
    uncached_prompt_tokens = _integer(execution.get("uncached_prompt_tokens"))
    expected_uncached = max(prompt_tokens - cached_tokens, 0)
    has_generation_suffix_telemetry = "generation_prompt_suffix_tokens" in execution
    generation_suffix_tokens = (
        _integer(execution.get("generation_prompt_suffix_tokens"))
        if has_generation_suffix_telemetry
        else 0
    )
    prefill_tokens = _integer(execution.get("prefill_tokens"))

    if cached_tokens <= 0:
        failures.append(f"{tag}: cached_tokens must be positive")
    if cached_tokens < minimum_cached_tokens:
        failures.append(
            f"{tag}: cached_tokens={cached_tokens} is below expected shared-prefix "
            f"floor {minimum_cached_tokens}"
        )
    if maximum_cached_tokens > 0 and cached_tokens > maximum_cached_tokens:
        failures.append(
            f"{tag}: cached_tokens={cached_tokens} exceeds independent tokenizer "
            f"LCP {maximum_cached_tokens}"
        )
    if "attempted_cached_tokens" not in execution:
        failures.append(f"{tag}: execution attempted_cached_tokens is missing")
    if cached_tokens > attempted_cached_tokens:
        failures.append(
            f"{tag}: cached_tokens={cached_tokens} exceeds "
            f"attempted_cached_tokens={attempted_cached_tokens}"
        )
    if execution_cached_tokens != cached_tokens:
        failures.append(
            f"{tag}: usage cached_tokens={cached_tokens} does not match "
            f"execution cached_tokens={execution_cached_tokens}"
        )
    if prompt_tokens <= 0:
        failures.append(f"{tag}: execution prompt_tokens must be positive")
    elif expected_prompt_tokens > 0 and prompt_tokens != expected_prompt_tokens:
        failures.append(
            f"{tag}: execution prompt_tokens={prompt_tokens} does not match "
            f"independent tokenizer count={expected_prompt_tokens}"
        )
    if require_partial and prompt_tokens > 0 and cached_tokens >= prompt_tokens:
        failures.append(
            f"{tag}: cached_tokens must be smaller than prompt_tokens for "
            "partial-prefix reuse"
        )
    if require_partial and uncached_prompt_tokens <= 0:
        failures.append(f"{tag}: partial-prefix reuse must leave an uncached tail")
    if prompt_tokens > 0 and uncached_prompt_tokens != expected_uncached:
        failures.append(
            f"{tag}: uncached_prompt_tokens={uncached_prompt_tokens} does not "
            f"equal prompt_tokens-cached_tokens={expected_uncached}"
        )
    expected_prefill = uncached_prompt_tokens + generation_suffix_tokens
    if cached_tokens == prompt_tokens and expected_prefill == 0:
        # A truly exact full-cache hit must re-feed one kickoff token to obtain
        # logits. This is the only accepted deviation from uncached tail +
        # template-owned generation suffix.
        expected_prefill = 1
    if "prefill_tokens" not in execution:
        failures.append(f"{tag}: execution prefill_tokens is missing")
    elif prefill_tokens != expected_prefill:
        failures.append(
            f"{tag}: prefill_tokens={prefill_tokens} does not equal actual "
            f"uncached tail plus optional generation suffix={expected_prefill}"
        )
    if execution.get("cache_reuse_applied") is not True:
        failures.append(f"{tag}: cache_reuse_applied is not true")
    if execution.get("cache_outcome") != "hit":
        failures.append(f"{tag}: cache_outcome is not hit")

    usage_detail = str(row.get("cache_detail") or "")
    execution_detail = str(execution.get("cache_detail") or "")
    if not usage_detail:
        failures.append(f"{tag}: streaming usage cache_detail is empty")
    if not execution_detail:
        failures.append(f"{tag}: execution cache_detail is empty")
    if usage_detail and execution_detail and usage_detail != execution_detail:
        failures.append(
            f"{tag}: usage cache_detail={usage_detail!r} does not match "
            f"execution cache_detail={execution_detail!r}"
        )

    selection = str(execution.get("selection") or "").lower()
    detail_lower = execution_detail.lower()
    reconstructed_path_named = any(
        marker in detail_lower for marker in ("paged", "disk", "block")
    )
    direct_memory_or_prefix = (
        allow_direct_reuse
        and not reconstructed_path_named
        and (
            selection in {"memory", "prefix"}
            or "memory" in detail_lower
            or "prefix" in detail_lower
        )
    )
    if require_disk_origin or not direct_memory_or_prefix:
        if execution.get("reconstruction_ok") is not True:
            failures.append(f"{tag}: reconstruction_ok is not true")
        if execution.get("reconstructed") is not True:
            failures.append(f"{tag}: reconstructed is not true")

    if require_disk_origin:
        if "disk" not in usage_detail or "disk" not in execution_detail:
            failures.append(f"{tag}: restart hit cache_detail does not identify disk")
        disk_blocks = _integer(execution.get("disk_blocks"))
        if disk_blocks <= 0:
            failures.append(f"{tag}: execution disk_blocks must be positive")
        deltas = row.get("health_counter_deltas")
        if not isinstance(deltas, dict):
            failures.append(f"{tag}: missing pre/post health counter deltas")
        else:
            disk_hit_delta = _integer(deltas.get("block_disk_cache.disk_hits"))
            if disk_hit_delta <= 0:
                failures.append(f"{tag}: block-disk /health disk_hits did not increase")
            elif disk_blocks > 0 and disk_hit_delta < disk_blocks:
                failures.append(
                    f"{tag}: block-disk /health disk-hit delta={disk_hit_delta} "
                    f"is smaller than reconstructed disk_blocks={disk_blocks}"
                )

    return failures


def _tokenizer_prefix_floor(
    row: dict[str, Any],
    *,
    selector: str,
    require_partial: bool,
    token_contract: dict[str, Any],
) -> tuple[int, int, int, list[str]]:
    """Derive the cache floor from independent final-render tokenization."""
    tag = str(row.get("tag") or "<missing-tag>")
    failures: list[str] = []
    prompts = token_contract.get("prompts")
    lcp_rows = token_contract.get("longest_common_prefix_tokens")
    if not isinstance(prompts, dict) or not isinstance(lcp_rows, dict):
        return 0, 0, 0, [
            f"{tag}: independent tokenizer LCP contract is unavailable"
        ]
    prompt_row = prompts.get(selector)
    if not isinstance(prompt_row, dict):
        return 0, 0, 0, [
            f"{tag}: independent tokenizer metadata for selector {selector} is missing"
        ]
    expected_prompt_tokens = _integer(prompt_row.get("cache_prompt_token_count"))
    pair = f"A:{selector}"
    independent_lcp = _integer(lcp_rows.get(pair))
    if expected_prompt_tokens <= 1:
        failures.append(
            f"{tag}: independent tokenizer prompt count is not usable"
        )
    if independent_lcp <= 1:
        failures.append(
            f"{tag}: independent tokenizer LCP={independent_lcp} does not prove "
            "a multi-token prefix candidate"
        )

    if require_partial:
        cache = row.get("scheduler_cache")
        if not isinstance(cache, dict):
            cache = {}
        block_size = _integer(cache.get("block_size"))
        if block_size <= 0:
            failures.append(
                f"{tag}: scheduler cache block_size is missing or invalid"
            )
            minimum = 0
        else:
            # A cache may preserve the exact terminal partial block. At minimum
            # it must recover every complete block wholly inside the independent
            # token LCP; this floor does not trust cache-selection telemetry.
            minimum = (independent_lcp // block_size) * block_size
    else:
        # Exact prompt reuse intentionally re-feeds the final prompt token to
        # obtain logits, so all preceding independently tokenized tokens must
        # be reusable.
        minimum = max(independent_lcp - 1, 0)
    if minimum <= 1:
        failures.append(
            f"{tag}: tokenizer-derived reusable prefix floor={minimum} is not "
            "a meaningful cache proof"
        )
    return minimum, independent_lcp, expected_prompt_tokens, failures


def validate_cache_rows(
    phase: str,
    rows: list[dict[str, Any]],
    *,
    store_summary: dict[str, Any] | None = None,
    token_contract: dict[str, Any] | None = None,
) -> list[str]:
    """Validate the cache-specific contract for one store or restart phase.

    The prompts share one byte-identical prefix and differ only at the final
    suffix.  Validation therefore accepts only a nonzero continuous-prefix hit
    that leaves the differing suffix uncached; arbitrary overlap or
    HTTP/output-only success cannot satisfy the gate.
    """
    by_tag = {
        str(row.get("tag")): row
        for row in rows
        if isinstance(row, dict) and row.get("tag")
    }
    requirements = (
        {
            "cold_a": "cold",
            "warm_a": "hit",
            "partial_b": "partial",
        }
        if phase == "store"
        else {
            "restart_partial_c": "disk_partial",
            "restart_a": "hit",
        }
    )
    failures: list[str] = []
    if not isinstance(token_contract, dict):
        failures.append(
            f"{phase}: independent tokenizer-derived LCP contract is required"
        )
        token_contract = {}
    observed_order = [
        str(row.get("tag"))
        for row in rows
        if isinstance(row, dict) and row.get("tag") in requirements
    ]
    expected_order = list(requirements)
    if observed_order != expected_order:
        failures.append(
            f"{phase}: required row order is {expected_order}, got {observed_order}"
        )

    if phase == "probe" and not isinstance(store_summary, dict):
        failures.append("probe: linked passing store summary is required")
    for tag, requirement in requirements.items():
        row = by_tag.get(tag)
        if row is None:
            failures.append(f"{phase}: missing required row {tag}")
            continue
        row_failures: list[str]
        if requirement == "cold":
            row_failures = []
            execution = row.get("last_cache_execution")
            if not isinstance(execution, dict):
                row_failures.append(f"{tag}: missing scheduler.last_cache_execution")
            else:
                response_id = str(row.get("response_id") or "")
                execution_request_id = str(execution.get("request_id") or "")
                if row.get("response_id_consistent") is not True or not response_id:
                    row_failures.append(
                        f"{tag}: Responses SSE has no single consistent response ID"
                    )
                if not execution_request_id:
                    row_failures.append(f"{tag}: execution request_id is empty")
                elif response_id and execution_request_id != response_id:
                    row_failures.append(
                        f"{tag}: Responses SSE id={response_id} does not match "
                        f"execution request_id={execution_request_id}"
                    )
                if _integer(row.get("cached_tokens")) != 0:
                    row_failures.append(
                        f"{tag}: cold row unexpectedly used cached tokens"
                    )
                if execution.get("cache_reuse_applied") is not False:
                    row_failures.append(
                        f"{tag}: cold row cache_reuse_applied is not false"
                    )
                if execution.get("cache_outcome") != "miss":
                    row_failures.append(f"{tag}: cold row cache_outcome is not miss")
                prompt_row = (token_contract.get("prompts") or {}).get("A")
                expected_prompt_tokens = (
                    _integer(prompt_row.get("cache_prompt_token_count"))
                    if isinstance(prompt_row, dict)
                    else 0
                )
                if expected_prompt_tokens <= 1:
                    row_failures.append(
                        f"{tag}: independent tokenizer prompt count is missing"
                    )
                elif (
                    _integer(execution.get("prompt_tokens"))
                    != expected_prompt_tokens
                ):
                    row_failures.append(
                        f"{tag}: execution prompt_tokens="
                        f"{_integer(execution.get('prompt_tokens'))} does not "
                        f"match independent tokenizer count={expected_prompt_tokens}"
                    )
        else:
            selector = "A"
            if requirement == "partial":
                selector = "B"
            elif requirement == "disk_partial":
                selector = "C"
            (
                minimum_cached_tokens,
                independent_lcp_tokens,
                expected_prompt_tokens,
                floor_failures,
            ) = _tokenizer_prefix_floor(
                row,
                selector=selector,
                require_partial=requirement in {"partial", "disk_partial"},
                token_contract=token_contract,
            )
            row_failures = _validate_hit_row(
                row,
                require_partial=requirement in {"partial", "disk_partial"},
                require_disk_origin=requirement == "disk_partial",
                minimum_cached_tokens=minimum_cached_tokens,
                maximum_cached_tokens=independent_lcp_tokens,
                expected_prompt_tokens=expected_prompt_tokens,
                # Standard scheduler/TQ hits may be direct memory/prefix reuse.
                # Only restart-C must prove worker reconstruction from disk.
                allow_direct_reuse=requirement != "disk_partial",
            )
            row_failures.extend(floor_failures)
            row["expected_shared_prefix_floor_tokens"] = minimum_cached_tokens
            row["independent_longest_common_prefix_tokens"] = (
                independent_lcp_tokens
            )
            row["independent_prompt_tokens"] = expected_prompt_tokens
        row["cache_contract_required"] = True
        row["cache_contract_ok"] = not row_failures
        row["cache_contract_failures"] = row_failures
        failures.extend(row_failures)

    # A different suffix cannot legitimately produce a longer continuous match
    # than the exact-prompt reference row from the same phase.
    reference_tag = "warm_a" if phase == "store" else "restart_a"
    partial_tag = "partial_b" if phase == "store" else "restart_partial_c"
    reference = by_tag.get(reference_tag)
    partial = by_tag.get(partial_tag)
    if reference is not None and partial is not None:
        reference_cached = _integer(reference.get("cached_tokens"))
        partial_cached = _integer(partial.get("cached_tokens"))
        if partial_cached > reference_cached:
            failure = (
                f"{partial_tag}: cached_tokens={partial_cached} exceeds "
                f"{reference_tag} cached_tokens={reference_cached}; not a "
                "longest continuous shared-prefix result"
            )
            failures.append(failure)
            partial.setdefault("cache_contract_failures", []).append(failure)
            partial["cache_contract_ok"] = False
    return failures


def _prompt_contract(prefix: str, records: int) -> dict[str, Any]:
    return {
        "records": records,
        "common_prefix_characters": len(prefix),
        "common_prefix_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
    }


def validate_probe_linkage(
    probe_metadata: dict[str, Any],
    store_summary: dict[str, Any],
) -> list[str]:
    """Require a passing store from the same source/config but an old listener."""
    failures: list[str] = []
    if store_summary.get("phase") != "store":
        failures.append("probe linkage: --store-summary phase is not store")
    if store_summary.get("cache_contract_ok") is not True:
        failures.append("probe linkage: store summary cache contract did not pass")
    if store_summary.get("gate_ok") is not True:
        failures.append("probe linkage: store summary full gate did not pass")
    durability = store_summary.get("store_durability")
    if not isinstance(durability, dict) or durability.get("ok") is not True:
        failures.append("probe linkage: store durability barrier did not pass")
    for field in ("nonce", "model"):
        if probe_metadata.get(field) != store_summary.get(field):
            failures.append(f"probe linkage: {field} does not match store summary")

    probe_identity = probe_metadata.get("identity")
    store_identity = store_summary.get("identity")
    if not isinstance(probe_identity, dict):
        probe_identity = {}
    if not isinstance(store_identity, dict):
        store_identity = {}
    for field in ("declared_source", "declared_config"):
        if not str(probe_identity.get(field) or ""):
            failures.append(f"probe linkage: probe {field} identity is empty")
        if not str(store_identity.get(field) or ""):
            failures.append(f"probe linkage: store {field} identity is empty")

    probe_engine = probe_identity.get("observed_engine")
    store_engine = store_identity.get("observed_engine")
    if not isinstance(probe_engine, dict):
        probe_engine = {}
    if not isinstance(store_engine, dict):
        store_engine = {}
    probe_engine_fingerprint = str(probe_engine.get("fingerprint_sha256") or "")
    store_engine_fingerprint = str(store_engine.get("fingerprint_sha256") or "")
    if not probe_engine_fingerprint:
        failures.append("probe linkage: probe observed engine fingerprint is empty")
    if not store_engine_fingerprint:
        failures.append("probe linkage: store observed engine fingerprint is empty")
    for field in (
        "host",
        "port",
        "command",
        "cwd",
        "python_executable",
        "launch_shape",
    ):
        probe_value = probe_engine.get(field)
        store_value = store_engine.get(field)
        if probe_value in (None, ""):
            failures.append(f"probe linkage: probe observed listener {field} is empty")
        if store_value in (None, ""):
            failures.append(f"probe linkage: store observed listener {field} is empty")
        if (
            probe_value not in (None, "")
            and store_value not in (None, "")
            and probe_value != store_value
        ):
            failures.append(
                f"probe linkage: observed listener {field} does not match store"
            )

    probe_source_observed = probe_identity.get("observed_source")
    store_source_observed = store_identity.get("observed_source")
    if not isinstance(probe_source_observed, dict):
        probe_source_observed = {}
    if not isinstance(store_source_observed, dict):
        store_source_observed = {}
    for label, identity, observed in (
        ("probe", probe_identity, probe_source_observed),
        ("store", store_identity, store_source_observed),
    ):
        observed_head = str(observed.get("head") or "")
        if not observed_head:
            failures.append(f"probe linkage: {label} observed source HEAD is empty")
        elif identity.get("declared_source") != observed_head:
            failures.append(
                f"probe linkage: {label} declared source does not match "
                "its observed source HEAD"
            )
        if observed.get("dirty") is not False:
            failures.append(f"probe linkage: {label} observed source checkout is dirty")
    if probe_identity.get("declared_source") and probe_identity.get(
        "declared_source"
    ) != store_identity.get("declared_source"):
        failures.append("probe linkage: declared source identity does not match store")
    if probe_source_observed.get("head") and probe_source_observed.get(
        "head"
    ) != store_source_observed.get("head"):
        failures.append("probe linkage: observed source HEAD does not match store")
    if probe_identity.get("declared_config") and probe_identity.get(
        "declared_config"
    ) != store_identity.get("declared_config"):
        failures.append("probe linkage: declared config identity does not match store")
    for field in (
        "model_bundle_provenance",
        "cache_topology_provenance",
    ):
        probe_attestation = probe_identity.get(field)
        store_attestation = store_identity.get(field)
        if not isinstance(probe_attestation, dict):
            failures.append(f"probe linkage: probe {field} is missing")
        if not isinstance(store_attestation, dict):
            failures.append(f"probe linkage: store {field} is missing")
        if (
            isinstance(probe_attestation, dict)
            and isinstance(store_attestation, dict)
            and probe_attestation != store_attestation
        ):
            failures.append(f"probe linkage: {field} does not match store")
    probe_runtime = probe_identity.get("runtime_provenance")
    store_runtime = store_identity.get("runtime_provenance")
    if not isinstance(probe_runtime, dict):
        failures.append("probe linkage: probe runtime provenance is missing")
        probe_runtime = {}
    if not isinstance(store_runtime, dict):
        failures.append("probe linkage: store runtime provenance is missing")
        store_runtime = {}
    for field in (
        "server_module_relpath",
        "server_module_sha256",
        "package_init_relpath",
        "package_init_sha256",
        "python_source_tree_sha256",
        "python_source_file_count",
        "python_executable_fingerprint_sha256",
    ):
        if not str(probe_runtime.get(field) or ""):
            failures.append(f"probe linkage: probe runtime {field} is empty")
        if not str(store_runtime.get(field) or ""):
            failures.append(f"probe linkage: store runtime {field} is empty")
        if (
            probe_runtime.get(field)
            and store_runtime.get(field)
            and probe_runtime.get(field) != store_runtime.get(field)
        ):
            failures.append(f"probe linkage: runtime {field} does not match store")
    for label, runtime in (
        ("probe", probe_runtime),
        ("store", store_runtime),
    ):
        if "python_source_read_error_count" not in runtime:
            failures.append(
                f"probe linkage: {label} runtime source read-error count is missing"
            )
        elif _integer(runtime.get("python_source_read_error_count")) != 0:
            failures.append(
                f"probe linkage: {label} runtime could not read every Python source file"
            )
    if probe_runtime.get("python_source_read_error_count") != store_runtime.get(
        "python_source_read_error_count"
    ):
        failures.append(
            "probe linkage: runtime Python source read-error count does not match store"
        )
    if (
        probe_engine_fingerprint
        and probe_engine_fingerprint == store_engine_fingerprint
    ):
        failures.append(
            "probe linkage: observed listener identity must differ after process restart"
        )

    if probe_metadata.get("prompt_contract") != store_summary.get("prompt_contract"):
        failures.append("probe linkage: prompt contract does not match store")
    if probe_metadata.get("tokenizer_lcp_contract") != store_summary.get(
        "tokenizer_lcp_contract"
    ):
        failures.append(
            "probe linkage: independent tokenizer LCP contract does not match store"
        )
    return failures


def _post_sse(
    url: str, payload: dict[str, Any], timeout: int
) -> tuple[int, str, float]:
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


def _cache_prompts(prefix: str, nonce: str) -> dict[str, str]:
    """Build probes that differ at exactly one final ASCII selector character."""
    stem = f"{prefix}\nReply exactly CACHE-HIERARCHY-{nonce}-"
    return {selector: f"{stem}{selector}" for selector in ("A", "B", "C")}


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
    parser.add_argument(
        "--source-identity",
        required=True,
        help="Declared exact source revision; must match this checkout's Git HEAD.",
    )
    parser.add_argument(
        "--config-identity",
        required=True,
        help="Declared stable digest of the effective model/server configuration.",
    )
    parser.add_argument(
        "--store-summary",
        type=Path,
        help="Passing store-phase summary.json; required for probe.",
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("store", "probe"), required=True)
    parser.add_argument("--records", type=int, default=320)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--durability-timeout", type=float, default=30.0)
    parser.add_argument("--durability-poll-interval", type=float, default=0.25)
    args = parser.parse_args()
    for flag, value in (
        ("--source-identity", args.source_identity),
        ("--config-identity", args.config_identity),
    ):
        if not str(value or "").strip():
            parser.error(f"{flag} must not be empty")
    if args.durability_timeout < 0:
        parser.error("--durability-timeout must be non-negative")
    if args.durability_poll_interval <= 0:
        parser.error("--durability-poll-interval must be positive")

    try:
        args.artifact_dir = _external_artifact_dir(args.artifact_dir)
    except RuntimeError as exc:
        parser.error(str(exc))
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    provenance_failures: list[str] = []
    observed_engine: dict[str, Any] = {}
    observed_source: dict[str, Any] = {}
    try:
        observed_engine = _observe_local_listener_identity(args.base_url)
    except RuntimeError as exc:
        provenance_failures.append(f"provenance: {exc}")
    try:
        observed_source = _observe_source_checkout()
    except RuntimeError as exc:
        provenance_failures.append(f"provenance: {exc}")
    if observed_engine and observed_source:
        provenance_failures.extend(
            _validate_runtime_source_provenance(
                observed_engine,
                observed_source,
            )
        )
    if observed_source.get("head") and args.source_identity != observed_source["head"]:
        provenance_failures.append(
            "provenance: declared source identity does not match observed Git HEAD "
            f"({args.source_identity} != {observed_source['head']})"
        )

    health_before: dict[str, Any] = {}
    health_attestation_before: dict[str, Any] = {}
    if not provenance_failures:
        try:
            health_before = _json_get(f"{args.base_url}/health", args.timeout)
        except (OSError, ValueError, urllib.error.URLError) as exc:
            provenance_failures.append(
                f"provenance: could not read initial /health attestation: {exc}"
            )
        if health_before:
            try:
                provenance_failures.extend(
                    _validate_health_runtime_provenance(
                        health_before,
                        observed_engine,
                        observed_source,
                    )
                )
                (
                    health_attestation_before,
                    health_attestation_failures,
                ) = _health_attestation_snapshot(health_before)
                provenance_failures.extend(health_attestation_failures)
            except (OSError, ValueError) as exc:
                provenance_failures.append(
                    f"provenance: could not validate /health attestation: {exc}"
                )
    bundle_provenance = health_attestation_before.get(
        "model_bundle_provenance"
    )
    if isinstance(bundle_provenance, dict):
        observed_config_identity = str(
            bundle_provenance.get("fingerprint_sha256") or ""
        )
        if (
            observed_config_identity
            and args.config_identity != observed_config_identity
        ):
            provenance_failures.append(
                "provenance: declared config identity does not match the loaded "
                "model bundle attestation"
            )
    identity = {
        "observed_engine": observed_engine,
        "declared_source": args.source_identity,
        "observed_source": observed_source,
        "declared_config": args.config_identity,
        "runtime_provenance": health_before.get("runtime_provenance"),
        "model_bundle_provenance": health_before.get(
            "model_bundle_provenance"
        ),
        "cache_topology_provenance": health_before.get(
            "cache_topology_provenance"
        ),
        "health_attestation_sha256": health_attestation_before.get(
            "combined_sha256"
        ),
    }
    prefix = _common_prefix(args.nonce, args.records)
    prompts = _cache_prompts(prefix, args.nonce)
    prompt_a = prompts["A"]
    prompt_b = prompts["B"]
    prompt_c = prompts["C"]
    requests = (
        [
            ("cold_a", prompt_a, "A"),
            ("warm_a", prompt_a, "A"),
            ("partial_b", prompt_b, "B"),
        ]
        if args.phase == "store"
        else [
            ("restart_partial_c", prompt_c, "C"),
            ("restart_a", prompt_a, "A"),
        ]
    )
    prompt_contract = _prompt_contract(prefix, args.records)
    tokenizer_lcp_contract: dict[str, Any] = {}
    token_contract_failures: list[str] = []
    if not provenance_failures:
        (
            tokenizer_lcp_contract,
            token_contract_failures,
        ) = _fetch_tokenizer_lcp_contract(
            base_url=args.base_url,
            model=args.model,
            prompts=prompts,
            timeout=args.timeout,
            health_attestation=health_attestation_before,
        )
    metadata = {
        "phase": args.phase,
        "nonce": args.nonce,
        "base_url": args.base_url,
        "model": args.model,
        "identity": identity,
        "prompt_contract": prompt_contract,
        "tokenizer_lcp_contract": tokenizer_lcp_contract,
    }
    if provenance_failures or token_contract_failures:
        prerequisite_failures = provenance_failures + token_contract_failures
        result = {
            **metadata,
            "provenance_ok": not provenance_failures,
            "provenance_failures": provenance_failures,
            "token_contract_ok": not token_contract_failures,
            "token_contract_failures": token_contract_failures,
            "probe_linkage_ok": None,
            "probe_linkage_failures": [],
            "cache_contract_ok": False,
            "cache_contract_failures": prerequisite_failures,
            "gate_ok": False,
            "requests": [],
        }
        (args.artifact_dir / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        return 1

    (args.artifact_dir / "health_before.json").write_text(
        json.dumps(health_before, indent=2, sort_keys=True) + "\n"
    )
    store_summary: dict[str, Any] | None = None
    linkage_failures: list[str] = []
    if args.phase == "probe":
        if args.store_summary is None:
            parser.error("--store-summary is required for probe phase")
        try:
            loaded_store_summary = json.loads(args.store_summary.read_text())
            if not isinstance(loaded_store_summary, dict):
                raise ValueError("top-level JSON is not an object")
            store_summary = loaded_store_summary
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            linkage_failures.append(
                f"probe linkage: failed to read store summary: {exc}"
            )
        if store_summary is not None:
            linkage_failures.extend(validate_probe_linkage(metadata, store_summary))
        if linkage_failures:
            result = {
                **metadata,
                "store_summary_path": str(args.store_summary),
                "probe_linkage_ok": False,
                "probe_linkage_failures": linkage_failures,
                "cache_contract_ok": False,
                "cache_contract_failures": linkage_failures,
                "gate_ok": False,
                "requests": [],
            }
            (args.artifact_dir / "summary.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )
            return 1

    rows: list[dict[str, Any]] = []
    health_after = health_before
    for tag, prompt, suffix in requests:
        request_health_before = _json_get(f"{args.base_url}/health", args.timeout)
        request_health_before_path = args.artifact_dir / f"{tag}.health-before.json"
        request_health_before_path.write_text(
            json.dumps(request_health_before, indent=2, sort_keys=True) + "\n"
        )
        health_counters_before = _health_cache_counters(request_health_before)
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
        health_after = health
        health_path = args.artifact_dir / f"{tag}.health.json"
        health_path.write_text(json.dumps(health, indent=2, sort_keys=True) + "\n")
        health_counters_after = _health_cache_counters(health)
        health_counter_deltas = _counter_deltas(
            health_counters_before, health_counters_after
        )
        summary.update(
            {
                "tag": tag,
                "expected_marker": f"CACHE-HIERARCHY-{args.nonce}-{suffix}",
                "request_path": str(request_path),
                "raw_path": str(raw_path),
                "health_before_path": str(request_health_before_path),
                "health_path": str(health_path),
                "health_counters_before": health_counters_before,
                "health_counters_after": health_counters_after,
                "health_counter_deltas": health_counter_deltas,
                "last_cache_execution": (health.get("scheduler") or {}).get(
                    "last_cache_execution"
                ),
                "scheduler_cache": (
                    (health.get("cache") or {}).get("scheduler_cache") or {}
                ),
                "block_disk_cache": (
                    (health.get("cache") or {}).get("block_disk_cache") or {}
                ),
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

    validation_failures = validate_cache_rows(
        args.phase,
        rows,
        store_summary=store_summary,
        token_contract=tokenizer_lcp_contract,
    )
    request_contract_ok = all(
        row["status_code"] == 200 and row["marker_ok"] and row["terminal_ok"]
        for row in rows
    )
    store_durability: dict[str, Any] | None = None
    if args.phase == "store":
        if request_contract_ok and not validation_failures:
            store_durability, durability_health = _wait_for_store_durability(
                base_url=args.base_url,
                request_timeout=args.timeout,
                request_id=str(rows[0].get("response_id") or ""),
                baseline_counters=_health_cache_counters(health_before),
                timeout_s=args.durability_timeout,
                poll_interval_s=args.durability_poll_interval,
            )
            if durability_health:
                health_after = durability_health
                (args.artifact_dir / "health_after_durability.json").write_text(
                    json.dumps(
                        durability_health,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )
            if store_durability.get("ok") is not True:
                validation_failures.append(
                    "store: request-correlated post-eviction durability fence "
                    "did not settle; "
                    f"failures={store_durability.get('contract_failures')}"
                )
        else:
            store_durability = {
                "ok": False,
                "skipped": True,
                "reason": (
                    "request or cache contract failed before durability barrier"
                ),
            }
    observed_engine_after: dict[str, Any] = {}
    observed_source_after: dict[str, Any] = {}
    health_final: dict[str, Any] = {}
    health_attestation_after: dict[str, Any] = {}
    try:
        observed_engine_after = _observe_local_listener_identity(args.base_url)
    except RuntimeError as exc:
        provenance_failures.append(f"provenance: final listener observation: {exc}")
    if observed_engine_after and observed_engine_after.get(
        "fingerprint_sha256"
    ) != observed_engine.get("fingerprint_sha256"):
        provenance_failures.append(
            "provenance: listener identity changed during the live gate"
        )
    try:
        observed_source_after = _observe_source_checkout()
    except RuntimeError as exc:
        provenance_failures.append(
            f"provenance: final source observation: {exc}"
        )
    if observed_source_after:
        provenance_failures.extend(
            _compare_source_checkout_observations(
                observed_source,
                observed_source_after,
            )
        )
    if observed_engine_after and observed_source_after:
        provenance_failures.extend(
            _validate_runtime_source_provenance(
                observed_engine_after,
                observed_source_after,
            )
        )
    try:
        health_final = _json_get(f"{args.base_url}/health", args.timeout)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        provenance_failures.append(
            f"provenance: could not read final /health attestation: {exc}"
        )
    if health_final and observed_engine_after and observed_source_after:
        try:
            provenance_failures.extend(
                _validate_health_runtime_provenance(
                    health_final,
                    observed_engine_after,
                    observed_source_after,
                )
            )
            (
                health_attestation_after,
                final_attestation_failures,
            ) = _health_attestation_snapshot(health_final)
            provenance_failures.extend(final_attestation_failures)
            provenance_failures.extend(
                _compare_attestation_snapshots(
                    health_attestation_before,
                    health_attestation_after,
                )
            )
        except (OSError, ValueError) as exc:
            provenance_failures.append(
                f"provenance: could not validate final /health attestation: {exc}"
            )
    if health_final:
        (args.artifact_dir / "health_final.json").write_text(
            json.dumps(health_final, indent=2, sort_keys=True) + "\n"
        )
    validation_failures.extend(provenance_failures)
    gate_ok = request_contract_ok and not validation_failures
    result = {
        **metadata,
        "provenance_ok": not provenance_failures,
        "provenance_failures": provenance_failures,
        "observed_engine_after": observed_engine_after,
        "observed_source_after": observed_source_after,
        "health_attestation_before": health_attestation_before,
        "health_attestation_after": health_attestation_after,
        "store_summary_path": (
            str(args.store_summary) if args.store_summary is not None else None
        ),
        "store_durability": store_durability,
        "probe_linkage_ok": (not linkage_failures if args.phase == "probe" else None),
        "probe_linkage_failures": linkage_failures,
        "health_before": health_before,
        "health_after": health_after,
        "health_final": health_final,
        "token_contract_ok": not token_contract_failures,
        "token_contract_failures": token_contract_failures,
        "cache_contract_ok": not validation_failures,
        "cache_contract_failures": validation_failures,
        "request_contract_ok": request_contract_ok,
        "gate_ok": gate_ok,
        "requests": rows,
    }
    (args.artifact_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
