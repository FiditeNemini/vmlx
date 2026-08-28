#!/usr/bin/env python3
"""Attach-only, fail-closed API sustained-context acceptance probe.

This instrument never launches, stops, swaps, or loads a model. The operator
must first load the exact bundle through the real Electron UI and visually
inspect the target row's progress. The probe then binds the running endpoint to
the local source tree and bundle configuration fingerprints exposed by
``/health`` before it sends one streaming Chat Completions request.

One invocation is one fixed-context sample. Run the first large-prefill sample
as warmup, discard it, then retain at least three identical graded invocations
before making a speed claim. Cache lifecycle arms belong in separate invocations
with their requested cache state recorded; this script does not silently clear,
restart, or mutate cache namespaces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SCHEMA = "vmlx-api-context-acceptance-v1"
BUNDLE_FILES = (
    "config.json",
    "generation_config.json",
    "jang_config.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)


class GateError(RuntimeError):
    """A fail-closed acceptance error, never a passing measurement."""


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return "<absolute-path-redacted>" if os.path.isabs(value) else value
    if isinstance(value, dict):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_json_value(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
    return {"unsupported_type": type(value).__name__}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _json_value(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def bundle_attestation(bundle: Path) -> dict[str, Any]:
    root = bundle.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise GateError(f"bundle is not a directory: {root}")
    files: dict[str, dict[str, Any]] = {
        name: {"state": "missing"} for name in BUNDLE_FILES
    }
    for name in BUNDLE_FILES:
        path = root / name
        if not path.is_file():
            continue
        data = path.read_bytes()
        files[name] = {
            "state": "present",
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    observed = {
        "schema": "vmlx-bundle-config-v1",
        "directory_state": "available",
        "files": files,
    }
    fingerprint = canonical_sha256(observed)
    return {
        **observed,
        "aggregate_sha256": fingerprint,
        "fingerprint_sha256": fingerprint,
    }


def python_source_tree_digest(package_root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    count = 0
    read_errors = 0
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root.parent).as_posix().encode()
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


def local_runtime_provenance(source_root: Path) -> dict[str, Any]:
    package_root = source_root.resolve(strict=True) / "vmlx_engine"
    server = package_root / "server.py"
    package_init = package_root / "__init__.py"
    if not server.is_file() or not package_init.is_file():
        raise GateError(f"not a vMLX source root: {source_root}")
    tree_sha, file_count, read_errors = python_source_tree_digest(package_root)
    return {
        "server_module_sha256": hashlib.sha256(server.read_bytes()).hexdigest(),
        "package_init_sha256": hashlib.sha256(package_init.read_bytes()).hexdigest(),
        "python_source_tree_sha256": tree_sha,
        "python_source_file_count": file_count,
        "python_source_read_error_count": read_errors,
    }


def _headers(api_key: str | None, extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra:
        headers.update(extra)
    return headers


def request_json(
    method: str,
    url: str,
    *,
    api_key: str | None,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> tuple[int, Any, float]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        method=method,
        data=data,
        headers=_headers(api_key),
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
            body = json.loads(raw) if raw else None
            return response.status, body, time.perf_counter() - started
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"raw": raw}
        return exc.code, body, time.perf_counter() - started


def verify_provenance(
    health: dict[str, Any],
    models: dict[str, Any],
    *,
    source_root: Path,
    bundle: Path,
    served_model: str,
) -> list[str]:
    errors: list[str] = []
    if health.get("status") != "healthy" or health.get("model_loaded") is not True:
        errors.append("health does not report a loaded healthy model")
    ids = {
        str(item.get("id"))
        for item in (models.get("data") or [])
        if isinstance(item, dict) and item.get("id")
    }
    if served_model not in ids:
        errors.append(f"served model {served_model!r} absent from /v1/models: {sorted(ids)}")

    expected_bundle = bundle_attestation(bundle)
    observed_bundle = health.get("model_bundle_provenance") or {}
    if observed_bundle.get("fingerprint_sha256") != expected_bundle["fingerprint_sha256"]:
        errors.append("loaded bundle configuration fingerprint differs from --bundle")

    expected_runtime = local_runtime_provenance(source_root)
    observed_runtime = health.get("runtime_provenance") or {}
    for key, value in expected_runtime.items():
        if observed_runtime.get(key) != value:
            errors.append(
                f"runtime provenance mismatch for {key}: "
                f"expected={value!r} observed={observed_runtime.get(key)!r}"
            )
    return errors


def verify_cache_contract(health: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    topology = (health.get("cache_topology_provenance") or {}).get("configuration") or {}
    instantiated = topology.get("instantiated") or {}
    tq = topology.get("turboquant_kv_cache") or health.get("turboquant_kv_cache") or {}
    kvq = topology.get("kv_cache_quantization") or health.get("kv_cache_quantization") or {}
    native = topology.get("native_cache") or health.get("native_cache") or {}
    cache = health.get("cache") or {}
    totals = cache.get("totals") or {}
    vision = cache.get("vision_memory_cache") or {}

    if instantiated.get("paged_ram_enabled") is not False:
        errors.append("paged/retained RAM cache is not proven off")
    if instantiated.get("block_disk_l2") is not True:
        errors.append("SSD block L2 is not instantiated")
    if instantiated.get("block_disk_only") is not True:
        errors.append("block cache is not proven SSD-only")
    # ``use_memory_aware_cache`` is a legacy configured-policy input and can
    # remain true while --no-paged-cache makes every instantiated RAM tier
    # inactive.  Gate the effective runtime plus live retention telemetry;
    # rejecting the inert input produces a false negative for SSD-only mode.
    if instantiated.get("memory_aware_prefix") is not False:
        errors.append("memory-aware retained prefix cache is instantiated")
    if totals.get("retained_cache_ram_enabled") is not False:
        errors.append("retained cache RAM is not proven disabled")
    if int(totals.get("retained_cache_bytes") or 0) != 0:
        errors.append("retained cache RAM contains live bytes")
    if vision.get("enabled") is not False:
        errors.append("media preprocess RAM cache is not proven disabled")
    if int(vision.get("retained_bytes") or 0) != 0:
        errors.append("media preprocess RAM cache contains live bytes")
    if tq.get("enabled") is True or tq.get("storage_encode_enabled") is True:
        errors.append("TurboQuant/stored-KV encoding is enabled")
    if int(tq.get("objects_active") or 0) != 0:
        errors.append("TurboQuant cache objects are active")
    if kvq.get("enabled") is True or str(kvq.get("mode") or "none").lower() not in {
        "",
        "none",
        "disabled",
    }:
        errors.append("generic KV storage quantization is enabled")
    storage = json.dumps(native.get("storage_quantization"), sort_keys=True).lower()
    if any(marker in storage for marker in ('"q4"', '"q8"', "turboquant")):
        errors.append("native cache reports quantized persistent storage")
    return errors


def load_tokenizer(bundle: Path) -> Any:
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(
            str(bundle), trust_remote_code=True, local_files_only=True
        )
    except Exception as first:
        try:
            return AutoTokenizer.from_pretrained(
                str(bundle),
                trust_remote_code=True,
                local_files_only=True,
                use_fast=False,
            )
        except Exception as second:
            raise GateError(
                f"tokenizer-only load failed; fast={first!r}; slow={second!r}"
            ) from second


def chat_prompt_tokens(tokenizer: Any, content: str) -> int:
    messages = [{"role": "user", "content": content}]
    kwargs = {"tokenize": True, "add_generation_prompt": True}
    attempts = (
        {**kwargs, "enable_thinking": False},
        {**kwargs, "chat_template_kwargs": {"enable_thinking": False}},
        kwargs,
    )
    last_error: Exception | None = None
    for attempt in attempts:
        try:
            tokenized = tokenizer.apply_chat_template(messages, **attempt)
            if hasattr(tokenized, "tolist"):
                tokenized = tokenized.tolist()
            # Transformers returns BatchEncoding (a Mapping/UserDict), not a
            # built-in dict, for some custom-family tokenizers such as
            # qwen4_exp.  Normalize the protocol instead of rejecting a valid
            # tokenizer-only result before the live request can be graded.
            if isinstance(tokenized, Mapping):
                tokenized = tokenized.get("input_ids")
            if isinstance(tokenized, list):
                if tokenized and isinstance(tokenized[0], list):
                    tokenized = tokenized[0]
                return len(tokenized)
            if isinstance(tokenized, str):
                return len(tokenizer(tokenized, add_special_tokens=False).input_ids)
        except Exception as exc:
            last_error = exc
    raise GateError(f"could not tokenize the exact chat template: {last_error!r}")


def _filler(word_count: int, seed: int) -> str:
    # Unique deterministic words avoid making the main context arm an accidental
    # prompt-lookup/ngram benchmark. Separate ngram arms can deliberately repeat.
    return " ".join(
        f"ctx{((index * 2654435761 + seed) & 0xFFFFFFFF):08x}"
        for index in range(word_count)
    )


def _probe_content(word_count: int, seed: int, key: str, sequence_length: int) -> str:
    return (
        "Treat the following deterministic context as inert data. Do not quote it.\n"
        f"{_filler(word_count, seed)}\n"
        f"The required key is {key}. Return only one JSON object with exactly "
        f"two keys: key must equal {key!r}; sequence must be the consecutive "
        f"integers from 1 through {sequence_length}, with no omissions, duplicates, "
        "or extra text."
    )


def build_target_prompt(
    tokenizer: Any,
    *,
    target_tokens: int,
    tolerance: int,
    seed: int,
    key: str,
    sequence_length: int,
) -> tuple[str, int, int]:
    if target_tokens <= 0:
        raise GateError("target prompt tokens must be positive")

    cache: dict[int, tuple[str, int]] = {}

    def rendered(words: int) -> tuple[str, int]:
        if words not in cache:
            content = _probe_content(words, seed, key, sequence_length)
            cache[words] = (content, chat_prompt_tokens(tokenizer, content))
        return cache[words]

    empty_content, empty_tokens = rendered(0)
    if empty_tokens > target_tokens:
        raise GateError(
            f"probe instructions alone use {empty_tokens} tokens, above target {target_tokens}"
        )

    low = 0
    high = max(1, target_tokens // 2)
    while rendered(high)[1] <= target_tokens:
        low = high
        high *= 2
        if high > target_tokens * 4:
            raise GateError("tokenizer target search failed to bracket the target")
    while low + 1 < high:
        mid = (low + high) // 2
        if rendered(mid)[1] <= target_tokens:
            low = mid
        else:
            high = mid
    content, count = rendered(low)
    if target_tokens - count > tolerance:
        raise GateError(
            f"closest safe local token count {count} misses target {target_tokens} "
            f"by more than tolerance {tolerance}"
        )
    return content, count, low


def validate_visible_json(content: str, *, key: str, sequence_length: int) -> list[str]:
    errors: list[str] = []
    if not content.strip():
        return ["visible content is empty"]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        return [f"visible content is not strict JSON: {exc}"]
    if not isinstance(parsed, dict) or set(parsed) != {"key", "sequence"}:
        errors.append("visible JSON must contain exactly key and sequence")
        return errors
    if parsed.get("key") != key:
        errors.append("visible JSON did not preserve the context key")
    expected = list(range(1, sequence_length + 1))
    if parsed.get("sequence") != expected:
        errors.append("visible JSON sequence is not exactly consecutive 1..N")
    return errors


@dataclass
class MemoryMonitor:
    pid: int
    interval: float
    min_free_bytes: int
    samples: list[dict[str, Any]] = field(default_factory=list)
    breached: threading.Event = field(default_factory=threading.Event)
    stopped: threading.Event = field(default_factory=threading.Event)

    def run(self) -> None:
        import psutil

        process = psutil.Process(self.pid)
        started = time.perf_counter()
        while not self.stopped.wait(self.interval):
            try:
                vm = psutil.virtual_memory()
                rss = int(process.memory_info().rss)
                sample = {
                    "elapsed_s": round(time.perf_counter() - started, 3),
                    "rss_bytes": rss,
                    "system_available_bytes": int(vm.available),
                    "system_used_percent": float(vm.percent),
                }
                self.samples.append(sample)
                if self.min_free_bytes and vm.available < self.min_free_bytes:
                    self.breached.set()
            except Exception as exc:
                self.samples.append({"monitor_error": repr(exc)})
                self.breached.set()
                return


def stream_chat(
    url: str,
    *,
    payload: dict[str, Any],
    api_key: str | None,
    proof_id: str,
    timeout: float,
    monitor: MemoryMonitor,
) -> dict[str, Any]:
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers=_headers(
            api_key,
            {
                "x-vmlx-proof-request-id": proof_id,
                "x-vmlx-message-id": f"{proof_id}-message",
            },
        ),
    )
    started = time.perf_counter()
    thread = threading.Thread(target=monitor.run, name="vmlx-memory-monitor", daemon=True)
    thread.start()
    raw_events: list[dict[str, Any]] = []
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage_events: list[dict[str, Any]] = []
    finish_reasons: list[str] = []
    response_id: str | None = None
    done_count = 0
    decoded_times: list[float] = []
    cancel_result: dict[str, Any] | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                elapsed = time.perf_counter() - started
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                body = line[6:]
                if body == "[DONE]":
                    done_count += 1
                    raw_events.append({"elapsed_s": elapsed, "done": True})
                    continue
                try:
                    event = json.loads(body)
                except json.JSONDecodeError as exc:
                    raw_events.append(
                        {"elapsed_s": elapsed, "invalid_json": body, "error": str(exc)}
                    )
                    continue
                raw_events.append({"elapsed_s": elapsed, "event": event})
                if event.get("id") and response_id is None:
                    response_id = str(event["id"])
                if event.get("usage") is not None:
                    usage_events.append(event["usage"])
                choices = event.get("choices") or []
                for choice in choices:
                    finish = choice.get("finish_reason")
                    if finish is not None:
                        finish_reasons.append(str(finish))
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if content:
                        content_parts.append(str(content))
                    if reasoning:
                        reasoning_parts.append(str(reasoning))
                    if content or reasoning:
                        decoded_times.append(elapsed)
                if monitor.breached.is_set() and response_id and cancel_result is None:
                    code, body, cancel_elapsed = request_json(
                        "POST",
                        f"{url}/{urllib.parse.quote(response_id, safe='')}/cancel",
                        api_key=api_key,
                        timeout=10.0,
                    )
                    cancel_result = {
                        "code": code,
                        "body": body,
                        "elapsed_s": cancel_elapsed,
                    }
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        raise GateError(f"chat HTTP {exc.code}: {raw}") from exc
    finally:
        monitor.stopped.set()
        thread.join(timeout=max(2.0, monitor.interval * 4))

    total_elapsed = time.perf_counter() - started
    usage = usage_events[-1] if usage_events else {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    decode_elapsed = (
        decoded_times[-1] - decoded_times[0] if len(decoded_times) > 1 else 0.0
    )
    decode_rate = (
        (completion_tokens - 1) / decode_elapsed
        if completion_tokens > 1 and decode_elapsed > 0
        else 0.0
    )
    gaps = [b - a for a, b in zip(decoded_times, decoded_times[1:])]
    return {
        "response_id": response_id,
        "raw_events": raw_events,
        "done_count": done_count,
        "usage_events": usage_events,
        "usage": usage,
        "finish_reasons": finish_reasons,
        "content": "".join(content_parts),
        "reasoning_content": "".join(reasoning_parts),
        "timing": {
            "total_elapsed_s": total_elapsed,
            "ttft_s": decoded_times[0] if decoded_times else None,
            "decode_elapsed_s": decode_elapsed,
            "decode_tokens_per_second": decode_rate,
            "delta_gap_median_s": statistics.median(gaps) if gaps else None,
            "delta_gap_max_s": max(gaps) if gaps else None,
            "delta_count": len(decoded_times),
            "note": "delta gaps are transport deltas, not token boundaries",
        },
        "memory_samples": monitor.samples,
        "memory_breached": monitor.breached.is_set(),
        "cancel_result": cancel_result,
    }


def grade_stream(
    result: dict[str, Any],
    *,
    key: str,
    sequence_length: int,
    expected_prompt_tokens: int,
    token_tolerance: int,
) -> list[str]:
    errors: list[str] = []
    usage_events = result.get("usage_events") or []
    if len(usage_events) != 1:
        errors.append(f"expected exactly one usage event, observed {len(usage_events)}")
    if result.get("done_count") != 1:
        errors.append(f"expected exactly one [DONE], observed {result.get('done_count')}")
    finishes = result.get("finish_reasons") or []
    if len(finishes) != 1:
        errors.append(f"expected exactly one terminal finish reason, observed {finishes}")
    usage = result.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    if prompt_tokens <= 0 or completion_tokens <= 0:
        errors.append("usage prompt/completion token counts must both be positive")
    if abs(prompt_tokens - expected_prompt_tokens) > token_tolerance:
        errors.append(
            f"server prompt_tokens={prompt_tokens} differs from local template "
            f"count={expected_prompt_tokens} beyond tolerance={token_tolerance}"
        )
    if not result.get("reasoning_content") and not result.get("content"):
        errors.append("stream emitted neither reasoning nor visible content")
    errors.extend(
        validate_visible_json(
            str(result.get("content") or ""),
            key=key,
            sequence_length=sequence_length,
        )
    )
    if result.get("memory_breached"):
        errors.append("system free-memory floor was breached")
    if float((result.get("timing") or {}).get("decode_tokens_per_second") or 0) <= 0:
        errors.append("sustained decode rate is unavailable or zero")
    return errors


def read_declared_context(bundle: Path) -> int:
    config = json.loads((bundle / "config.json").read_text())
    text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    values = (
        config.get("max_position_embeddings"),
        text_config.get("max_position_embeddings"),
        config.get("model_max_length"),
    )
    return next((int(value) for value in values if isinstance(value, (int, float)) and value > 0), 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--source-root", default=str(Path(__file__).resolve().parents[1]), type=Path)
    parser.add_argument("--target-prompt-tokens", required=True, type=int)
    parser.add_argument("--max-tokens", required=True, type=int)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--token-tolerance", type=int, default=32)
    parser.add_argument("--seed", type=int, default=270827)
    parser.add_argument("--key", default="VMLINUX-CONTEXT-OK-270827")
    parser.add_argument("--api-key", default=os.environ.get("VMLINUX_API_KEY"))
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--memory-sample-s", type=float, default=0.25)
    parser.add_argument("--min-free-gb", type=float, default=16.0)
    parser.add_argument("--cache-salt", default=None)
    parser.add_argument("--skip-prefix-cache", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    base = args.base_url.rstrip("/")
    bundle = args.bundle.expanduser().resolve(strict=True)
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "INVALID",
        "started_at_unix": time.time(),
        "inputs": {
            "base_url": base,
            "served_model": args.served_model,
            "bundle_name": bundle.name,
            "target_prompt_tokens": args.target_prompt_tokens,
            "max_tokens": args.max_tokens,
            "sequence_length": args.sequence_length,
            "token_tolerance": args.token_tolerance,
            "seed": args.seed,
            "key": args.key,
            "skip_prefix_cache": args.skip_prefix_cache,
            "cache_salt_present": bool(args.cache_salt),
        },
        "errors": [],
    }
    exit_code = 2
    try:
        health_code, health, _ = request_json(
            "GET", f"{base}/health", api_key=args.api_key, timeout=30.0
        )
        models_code, models, _ = request_json(
            "GET", f"{base}/v1/models", api_key=args.api_key, timeout=30.0
        )
        if health_code != 200 or not isinstance(health, dict):
            raise GateError(f"/health failed: code={health_code} body={health!r}")
        if models_code != 200 or not isinstance(models, dict):
            raise GateError(f"/v1/models failed: code={models_code} body={models!r}")
        artifact["health_before"] = health
        artifact["models"] = models

        errors = verify_provenance(
            health,
            models,
            source_root=args.source_root,
            bundle=bundle,
            served_model=args.served_model,
        )
        errors.extend(verify_cache_contract(health))
        if errors:
            raise GateError("; ".join(errors))

        declared_context = read_declared_context(bundle)
        session_cap = int(health.get("max_prompt_tokens") or 0)
        total_requested = args.target_prompt_tokens + args.max_tokens
        if declared_context and total_requested > declared_context:
            raise GateError(
                f"prompt+output={total_requested} exceeds declared context {declared_context}"
            )
        if session_cap and args.target_prompt_tokens > session_cap:
            raise GateError(
                f"target prompt {args.target_prompt_tokens} exceeds session cap {session_cap}"
            )

        import psutil

        before_vm = psutil.virtual_memory()
        min_free_bytes = int(args.min_free_gb * (1024**3))
        if before_vm.available < min_free_bytes:
            raise GateError(
                f"preflight available memory {before_vm.available} below floor {min_free_bytes}"
            )
        tokenizer = load_tokenizer(bundle)
        content, local_tokens, word_count = build_target_prompt(
            tokenizer,
            target_tokens=args.target_prompt_tokens,
            tolerance=args.token_tolerance,
            seed=args.seed,
            key=args.key,
            sequence_length=args.sequence_length,
        )
        artifact["prompt"] = {
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
            "local_chat_template_tokens": local_tokens,
            "word_count": word_count,
            "utf8_bytes": len(content.encode()),
            "head": content[:160],
            "tail": content[-320:],
        }

        proof_id = f"api-context-{int(time.time())}-{args.seed}"
        payload: dict[str, Any] = {
            "model": args.served_model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "enable_thinking": False,
            "tool_choice": "none",
            "max_prompt_tokens": args.target_prompt_tokens + args.token_tolerance,
        }
        if args.cache_salt:
            payload["cache_salt"] = args.cache_salt
        if args.skip_prefix_cache:
            payload["skip_prefix_cache"] = True

        pid = int((health.get("runtime_provenance") or {}).get("pid") or 0)
        if pid <= 0:
            raise GateError("health runtime provenance has no valid PID")
        monitor = MemoryMonitor(
            pid=pid,
            interval=max(0.05, args.memory_sample_s),
            min_free_bytes=min_free_bytes,
        )
        result = stream_chat(
            f"{base}/v1/chat/completions",
            payload=payload,
            api_key=args.api_key,
            proof_id=proof_id,
            timeout=args.timeout_s,
            monitor=monitor,
        )
        artifact["stream"] = result
        artifact["errors"] = grade_stream(
            result,
            key=args.key,
            sequence_length=args.sequence_length,
            expected_prompt_tokens=local_tokens,
            token_tolerance=args.token_tolerance,
        )

        settle_deadline = time.monotonic() + 30.0
        health_after: dict[str, Any] | None = None
        while time.monotonic() < settle_deadline:
            code, body, _ = request_json(
                "GET", f"{base}/health", api_key=args.api_key, timeout=10.0
            )
            if code == 200 and isinstance(body, dict):
                health_after = body
                execution = ((body.get("scheduler") or {}).get("last_cache_execution") or {})
                if not result.get("response_id") or execution.get("request_id") == result.get("response_id"):
                    break
            time.sleep(0.5)
        artifact["health_after"] = health_after
        cache_code, cache_stats, _ = request_json(
            "GET", f"{base}/v1/cache/stats", api_key=args.api_key, timeout=30.0
        )
        artifact["cache_stats_after"] = {"code": cache_code, "body": cache_stats}
        if health_after is None:
            artifact["errors"].append("post-request health did not settle")
        artifact["status"] = "PASS" if not artifact["errors"] else "FAIL"
        exit_code = 0 if artifact["status"] == "PASS" else 1
    except Exception as exc:
        artifact["errors"].append(f"{type(exc).__name__}: {exc}")
        artifact["status"] = "INVALID"
        exit_code = 2
    finally:
        artifact["finished_at_unix"] = time.time()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": artifact["status"],
                    "errors": artifact["errors"],
                    "out": str(args.out),
                    "decode_tokens_per_second": (
                        ((artifact.get("stream") or {}).get("timing") or {}).get(
                            "decode_tokens_per_second"
                        )
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
