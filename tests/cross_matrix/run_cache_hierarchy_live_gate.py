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
import re
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ARTIFACT_SCHEMA = "vmlx-cache-hierarchy-live-gate-v2"
PREFIX_ATTESTATION_METHOD = (
    "final-render-tokenize-cache-prefix-identity-readonly"
)
PRIVATE_ATTESTATION_TOKEN_ENV = "VMLINUX_PRIVATE_CACHE_ATTESTATION_TOKEN"
PRIVATE_ATTESTATION_PROOF_HEADER = "vmlx-cache-prefix-attestation-v1"
L2_SIZE_EVICTION_SCHEMA = "vmlx-cache-l2-size-eviction-observation-v1"
L2_RESTART_RESTORE_SCHEMA = "vmlx-cache-l2-restart-restore-observation-v1"
CACHE_SCENARIOS = (
    "standard",
    "store-evict-refault",
    "restart-restore",
)
CACHE_SCENARIO_INSTRUCTIONS = (
    "This is a cache transport measurement. Do not call tools. "
    "Follow the final user reply instruction exactly."
)
CACHE_SCENARIO_TOOLS = (
    {
        "type": "function",
        "name": "cache_contract_unused",
        "description": (
            "Stable schema rendered only to prove cache identity includes tools."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "value": {"type": "string"},
            },
            "required": ["value"],
            "additionalProperties": False,
        },
    },
)
L2_EVICTION_INSTRUCTIONS = "Reply with the requested exact marker."


def _cache_scenario_request_controls() -> dict[str, Any]:
    return {
        "enable_thinking": False,
        "instructions": CACHE_SCENARIO_INSTRUCTIONS,
        "tools": [dict(tool) for tool in CACHE_SCENARIO_TOOLS],
        # Keep the stable tool schema in the rendered/cache-key contract while
        # making the transport-only marker response deterministic.  A natural-
        # language "do not call tools" instruction is not an API constraint:
        # models may legally return content and a redundant tool call when
        # tool_choice is left on auto, which does not indicate cache corruption.
        "tool_choice": "none",
    }


def _l2_scenario_request_controls() -> dict[str, Any]:
    """Keep L2 eviction producers cold on hybrid cache architectures.

    The regular cache contract deliberately includes a stable tool schema so
    it can prove that tools participate in cache identity.  Reusing that large
    stable prefix for every L2 eviction producer creates a paged-cache hit
    before the request-specific identity.  Hybrid SSM runtimes intentionally
    do not recursively promote reconstructed state, so such a request cannot
    own a new disk-write fence.  The eviction scenario has a different job:
    create independently owned disk entries and prove their LRU lifecycle.
    Use a concise, tool-free render shape and put the unique identity at the
    start of each user prompt so every producer is cold before block 0.
    """

    return {
        "enable_thinking": False,
        "instructions": L2_EVICTION_INSTRUCTIONS,
    }


def _json_get(url: str, timeout: int) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def _json_post(
    url: str,
    payload: dict[str, Any],
    timeout: int,
    *,
    private_attestation: bool = False,
) -> dict[str, Any]:
    # Preserve the exact insertion order used by the live Responses request.
    # Tool schemas are rendered into the chat template, so recursively sorting
    # their JSON keys here can change the tokenized prompt being attested.
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if private_attestation:
        token = os.environ.get(PRIVATE_ATTESTATION_TOKEN_ENV, "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{32,512}", token):
            raise ValueError(
                f"{PRIVATE_ATTESTATION_TOKEN_ENV} is required for private attestation"
            )
        headers.update(
            {
                "Authorization": f"Bearer {token}",
                "X-vMLX-Private-Proof": PRIVATE_ATTESTATION_PROOF_HEADER,
            }
        )
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
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
    function_calls: list[dict[str, Any]] = []
    for item in events:
        if item["event"] != "response.output_item.done":
            continue
        data = item["data"]
        if not isinstance(data, dict):
            continue
        output_item = data.get("item")
        if (
            not isinstance(output_item, dict)
            or output_item.get("type") != "function_call"
        ):
            continue
        function_calls.append(
            {
                "name": output_item.get("name"),
                "arguments": output_item.get("arguments"),
                "status": output_item.get("status"),
            }
        )
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
        "function_calls": function_calls,
    }


def _exact_cache_marker_observed(
    summary: dict[str, Any],
    expected_marker: str,
) -> bool:
    """Accept exactly one clean marker shape with no reasoning leakage."""
    if str(summary.get("reasoning_text") or "").strip():
        return False
    output_text = str(summary.get("output_text") or "").strip()
    calls = summary.get("function_calls", [])
    if not isinstance(calls, list):
        return False
    # Cold and warm generations are semantically equivalent, NOT
    # byte-identical (decided policy): at temp 0 the same model emits the
    # marker bare on one arm and markdown-bolded on the other (observed
    # live on qwen3.8 — cold/partial rows bolded, the full hit did not).
    # Tolerate pure markdown emphasis around the exact marker; anything
    # else (extra words, truncation, mutation) still fails.
    if output_text.startswith(("**", "*", "`", "_")) and output_text.endswith(
        ("**", "*", "`", "_")
    ):
        _stripped = output_text.strip("*`_")
        if _stripped == expected_marker:
            output_text = _stripped
    if output_text == expected_marker:
        return not calls
    if output_text:
        return False
    if len(calls) != 1:
        return False
    call = calls[0]
    if (
        not isinstance(call, dict)
        or call.get("name") != "cache_contract_unused"
        or call.get("status") != "completed"
        or not isinstance(call.get("arguments"), str)
    ):
        return False
    try:
        arguments = json.loads(call["arguments"])
    except json.JSONDecodeError:
        return False
    return arguments == {"value": expected_marker}


def _integer(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _exact_nonnegative_integer(value: Any) -> int | None:
    """Accept only an attested JSON integer, never a bool or coercible string."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _token_contract_prompt_counts(
    prompt: dict[str, Any],
    *,
    tag: str,
) -> tuple[int, int, int, list[str]]:
    """Return cache-key, full-cache, and generation-suffix token counts.

    The private token contract exposes two deliberately different domains:
    ``cache_prompt_token_count`` is the production lookup/store key after any
    N-1 boundary and generation-prompt stripping, while
    ``full_cache_prompt_token_count`` is the rendered prompt before the
    generation suffix is appended back for execution.
    """
    failures: list[str] = []
    cache_value = _exact_nonnegative_integer(
        prompt.get("cache_prompt_token_count")
    )
    full_cache_value = _exact_nonnegative_integer(
        prompt.get("full_cache_prompt_token_count")
    )
    suffix_value = _exact_nonnegative_integer(
        prompt.get("generation_prompt_suffix_tokens")
    )
    removed_value = _exact_nonnegative_integer(
        prompt.get("cache_key_boundary_removed_tokens")
    )
    cache_tokens = cache_value if cache_value is not None else 0
    full_cache_tokens = (
        full_cache_value if full_cache_value is not None else 0
    )
    suffix_tokens = suffix_value if suffix_value is not None else 0
    removed_tokens = removed_value if removed_value is not None else 0

    if cache_value is None or cache_tokens <= 1:
        failures.append(f"{tag}: cache-prompt token count is invalid")
    if full_cache_value is None or full_cache_tokens <= 1:
        failures.append(f"{tag}: full cache-prompt token count is invalid")
    if suffix_value is None:
        failures.append(f"{tag}: generation-prompt suffix token count is invalid")
    if removed_value is None:
        failures.append(f"{tag}: cache-key boundary removal count is invalid")

    boundary = prompt.get("cache_key_boundary")
    expected_removed = {
        "full_cache_prompt": 0,
        "mllm_re_feed_n_minus_one": 1,
    }.get(boundary)
    if expected_removed is None:
        failures.append(f"{tag}: cache-key boundary is invalid")
    elif removed_value is not None and removed_tokens != expected_removed:
        failures.append(
            f"{tag}: cache-key boundary={boundary} requires "
            f"removed_tokens={expected_removed}, got {removed_tokens}"
        )

    if (
        cache_value is not None
        and full_cache_value is not None
        and full_cache_tokens < cache_tokens
    ):
        failures.append(
            f"{tag}: full cache-prompt count={full_cache_tokens} is smaller than "
            f"cache-key count={cache_tokens}"
        )
    elif (
        cache_value is not None
        and full_cache_value is not None
        and removed_value is not None
        and full_cache_tokens - cache_tokens != removed_tokens
    ):
        failures.append(
            f"{tag}: full/cache-key count difference="
            f"{full_cache_tokens - cache_tokens} does not match attested "
            f"boundary removal={removed_tokens}"
        )
    return cache_tokens, full_cache_tokens, suffix_tokens, failures


def _expected_execution_prompt_tokens(
    prompt: dict[str, Any],
    execution: dict[str, Any],
    *,
    tag: str,
) -> tuple[int, list[str]]:
    """Translate the token contract into the scheduler telemetry domain."""
    _, full_cache_tokens, suffix_tokens, failures = (
        _token_contract_prompt_counts(prompt, tag=tag)
    )
    if "generation_prompt_suffix_tokens" in execution:
        # MLLM telemetry counts the cache-prompt tokens and reports the
        # template-owned suffix separately.
        execution_suffix_value = _exact_nonnegative_integer(
            execution.get("generation_prompt_suffix_tokens")
        )
        if execution_suffix_value is None:
            failures.append(
                f"{tag}: execution generation-prompt suffix token count is invalid"
            )
            return full_cache_tokens + suffix_tokens, failures
        execution_suffix = execution_suffix_value
        if execution_suffix != suffix_tokens:
            failures.append(
                f"{tag}: execution generation-prompt suffix={execution_suffix} "
                f"does not match independent tokenizer suffix={suffix_tokens}"
            )
        return full_cache_tokens, failures
    # The standard Scheduler reports Request.num_prompt_tokens, which includes
    # the same independently attested generation suffix.
    return full_cache_tokens + suffix_tokens, failures


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
    ssm_companion = cache.get("ssm_companion")
    if not isinstance(ssm_companion, dict):
        ssm_companion = {}
    ssm_disk = ssm_companion.get("disk")
    if not isinstance(ssm_disk, dict):
        ssm_disk = {}
    global_budget = block_disk_cache.get("global_budget")
    if not isinstance(global_budget, dict):
        global_budget = {}
    return {
        "scheduler.cache_hit_requests": _integer(scheduler.get("cache_hit_requests")),
        "scheduler.cache_hit_tokens": _integer(scheduler.get("cache_hit_tokens")),
        "scheduler.hybrid_kv_without_ssm_hits": _integer(
            scheduler.get("hybrid_kv_without_ssm_hits")
        ),
        "scheduler.hybrid_kv_without_ssm_tokens": _integer(
            scheduler.get("hybrid_kv_without_ssm_tokens")
        ),
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
        "block_disk_cache.tq_native_hits": _integer(
            block_disk_cache.get("tq_native_hits")
        ),
        "block_disk_cache.tq_native_writes": _integer(
            block_disk_cache.get("tq_native_writes")
        ),
        "block_disk_cache.blocks_on_disk": _integer(
            block_disk_cache.get("blocks_on_disk")
        ),
        "block_disk_cache.total_tokens_on_disk": _integer(
            block_disk_cache.get("total_tokens_on_disk")
        ),
        "block_disk_cache.global_reconciliation_generation": _integer(
            global_budget.get("reconciliation_generation")
        ),
        "ssm_companion.disk.hits": _integer(ssm_disk.get("hits")),
        "ssm_companion.disk.misses": _integer(ssm_disk.get("misses")),
        "ssm_companion.disk.stores": _integer(ssm_disk.get("stores")),
    }


def _path_free_native_cache(native_cache: Any) -> dict[str, Any]:
    """Return only typed-cache contract fields; never retain paths or argv."""

    if not isinstance(native_cache, dict):
        return {}
    storage_quant = native_cache.get("attention_kv_storage_quantization")
    if not isinstance(storage_quant, dict):
        storage_quant = {}
    generic_tq = native_cache.get("generic_turboquant_kv")
    if not isinstance(generic_tq, dict):
        generic_tq = {}
    components = native_cache.get("components")
    if not isinstance(components, list):
        components = []
    return {
        key: native_cache.get(key)
        for key in (
            "family",
            "cache_type",
            "schema",
            "paged",
            "block_disk_l2",
        )
        if key in native_cache
    } | {
        "components": list(components),
        "attention_kv_storage_quantization": {
            key: storage_quant.get(key)
            for key in (
                "enabled",
                "mode",
                "codec",
                "bits",
                "value_bits",
                "applies_to",
                "ssm_policy",
                "rederive",
            )
            if key in storage_quant
        },
        "generic_turboquant_kv": {
            key: generic_tq.get(key)
            for key in ("enabled", "reason")
            if key in generic_tq
        },
    }


def _path_free_prefix_lookup(last_prefix_lookup: Any) -> dict[str, Any]:
    """Retain only request-correlated SSM lookup contract fields."""

    if not isinstance(last_prefix_lookup, dict):
        return {}

    def _bounded_token_lengths(value: Any, *, max_items: int) -> list[int]:
        if (
            not isinstance(value, list)
            or len(value) > max_items
            or any(type(item) is not int or item <= 0 for item in value)
        ):
            return []
        return list(value)

    candidate_lengths = _bounded_token_lengths(
        last_prefix_lookup.get("candidate_lengths"),
        max_items=20,
    )
    attempted_candidate_lengths = _bounded_token_lengths(
        last_prefix_lookup.get("attempted_candidate_lengths"),
        max_items=21,
    )
    candidate_count = last_prefix_lookup.get("candidate_count")
    attempted_candidate_count = last_prefix_lookup.get(
        "attempted_candidate_count"
    )
    candidate_lengths_truncated = last_prefix_lookup.get(
        "candidate_lengths_truncated"
    )
    attempted_candidate_lengths_truncated = last_prefix_lookup.get(
        "attempted_candidate_lengths_truncated"
    )
    return {
        key: last_prefix_lookup.get(key)
        for key in (
            "request_id",
            "max_len",
            "matched",
            "checkpoint_tokens",
            "is_complete",
            "source",
            "reason",
            "store_size",
        )
        if key in last_prefix_lookup
    } | {
        "candidate_lengths": candidate_lengths,
        "attempted_candidate_lengths": attempted_candidate_lengths,
        "candidate_count": (
            candidate_count
            if type(candidate_count) is int and candidate_count >= 0
            else None
        ),
        "attempted_candidate_count": (
            attempted_candidate_count
            if type(attempted_candidate_count) is int
            and attempted_candidate_count >= 0
            else None
        ),
        "candidate_lengths_truncated": (
            candidate_lengths_truncated
            if type(candidate_lengths_truncated) is bool
            else None
        ),
        "attempted_candidate_lengths_truncated": (
            attempted_candidate_lengths_truncated
            if type(attempted_candidate_lengths_truncated) is bool
            else None
        ),
    }


def _health_cache_contract_evidence(health: dict[str, Any]) -> dict[str, Any]:
    """Retain the explicit path-free typed-cache facts needed by the gate."""

    native_cache = health.get("native_cache")
    cache = health.get("cache")
    if not isinstance(cache, dict):
        cache = {}
    ssm_companion = cache.get("ssm_companion")
    if not isinstance(ssm_companion, dict):
        ssm_companion = {}
    ssm_disk = ssm_companion.get("disk")
    if not isinstance(ssm_disk, dict):
        ssm_disk = {}
    last_prefix_lookup = ssm_companion.get("last_prefix_lookup")
    if not isinstance(last_prefix_lookup, dict):
        last_prefix_lookup = {}
    return {
        "native_cache": _path_free_native_cache(native_cache),
        "ssm_companion": {
            "last_prefix_lookup": _path_free_prefix_lookup(last_prefix_lookup),
            "disk": {
                key: ssm_disk.get(key)
                for key in ("hits", "misses", "stores")
                if key in ssm_disk
            },
        },
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
    *,
    request_controls: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the exact Responses prompt shape without performing generation."""
    return {
        "contract_version": 1,
        "surface": "responses",
        "model": model,
        "inputs": prompts,
        "request_controls": (
            _cache_scenario_request_controls()
            if request_controls is None
            else request_controls
        ),
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
        _, _, _, count_failures = _token_contract_prompt_counts(
            row,
            tag=f"token contract: prompt {label}",
        )
        failures.extend(count_failures)
        token_sha = str(row.get("cache_prompt_token_ids_sha256") or "")
        if len(token_sha) != 64 or any(
            character not in "0123456789abcdef"
            for character in token_sha.lower()
        ):
            failures.append(
                f"token contract: prompt {label} token-ID digest is invalid"
            )
        full_token_sha = str(
            row.get("full_cache_prompt_token_ids_sha256") or ""
        )
        if len(full_token_sha) != 64 or any(
            character not in "0123456789abcdef"
            for character in full_token_sha.lower()
        ):
            failures.append(
                f"token contract: prompt {label} full token-ID digest is invalid"
            )
        discriminator_present = row.get(
            "generation_prompt_discriminator_present"
        )
        discriminator_sha = row.get(
            "generation_prompt_discriminator_sha256"
        )
        if not isinstance(discriminator_present, bool):
            failures.append(
                f"token contract: prompt {label} generation-prompt "
                "discriminator presence is missing"
            )
        elif discriminator_present and not _valid_sha256(discriminator_sha):
            failures.append(
                f"token contract: prompt {label} generation-prompt "
                "discriminator digest is invalid"
            )
        elif not discriminator_present and discriminator_sha is not None:
            failures.append(
                f"token contract: prompt {label} has a discriminator digest "
                "without a discriminator"
            )

    lcp = contract.get("longest_common_prefix_tokens")
    if not isinstance(lcp, dict):
        return failures + [
            "token contract: longest_common_prefix_tokens are missing"
        ]
    a_row = rows.get("A") if isinstance(rows, dict) else None
    a_count_value = (
        _exact_nonnegative_integer(a_row.get("cache_prompt_token_count"))
        if isinstance(a_row, dict)
        else None
    )
    a_count = a_count_value if a_count_value is not None else 0
    exact_a_value = _exact_nonnegative_integer(lcp.get("A:A"))
    if exact_a_value is None:
        failures.append("token contract: A:A LCP count is invalid")
    elif exact_a_value != a_count:
        failures.append(
            "token contract: A:A LCP does not equal the independently tokenized "
            "A prompt length"
        )
    for label in sorted(expected_labels - {"A"}):
        pair = f"A:{label}"
        value_attested = _exact_nonnegative_integer(lcp.get(pair))
        value = value_attested if value_attested is not None else 0
        other = rows.get(label)
        other_count_value = (
            _exact_nonnegative_integer(other.get("cache_prompt_token_count"))
            if isinstance(other, dict)
            else None
        )
        other_count = (
            other_count_value if other_count_value is not None else 0
        )
        if value_attested is None:
            failures.append(f"token contract: {pair} LCP count is invalid")
        elif value <= 1:
            failures.append(
                f"token contract: {pair} does not prove a multi-token prefix"
            )
        if (
            value_attested is not None
            and a_count > 0
            and other_count > 0
            and value >= min(a_count, other_count)
        ):
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
            private_attestation=True,
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


def _prefix_attestation_request(
    model: str,
    prompts: dict[str, str],
    pairs: dict[str, tuple[str, str] | list[str]],
    *,
    request_controls: dict[str, Any] | None = None,
) -> dict[str, Any]:
    referenced_labels: set[str] = set()
    normalized_pairs: dict[str, list[str]] = {}
    for pair_name, labels in sorted(pairs.items()):
        if (
            not isinstance(pair_name, str)
            or not pair_name
            or not isinstance(labels, (tuple, list))
            or len(labels) != 2
            or any(not isinstance(label, str) or not label for label in labels)
        ):
            raise ValueError("prefix pair labels must name exactly two prompts")
        missing = [label for label in labels if label not in prompts]
        if missing:
            raise ValueError(
                f"prefix pair {pair_name!r} references missing prompts {missing!r}"
            )
        normalized = list(labels)
        normalized_pairs[pair_name] = normalized
        referenced_labels.update(normalized)
    return {
        "contract_version": 1,
        "surface": "responses",
        "model": model,
        "inputs": {
            label: prompts[label] for label in sorted(referenced_labels)
        },
        "prefix_pairs": normalized_pairs,
        "request_controls": (
            _cache_scenario_request_controls()
            if request_controls is None
            else request_controls
        ),
        "touch": False,
    }


def _valid_path_free_sha256(value: Any) -> bool:
    return _valid_sha256(value)


def _prefix_attestation_forbidden_exposure(
    value: Any,
    *,
    prompt_values: tuple[str, ...],
) -> list[str]:
    """Reject raw prompt/token/path/host material from source attestations."""
    failures: list[str] = []
    forbidden_keys = {
        "prompt",
        "prompt_text",
        "token_ids",
        "cache_prompt_token_ids",
        "block_hash",
        "block_hashes",
        "file_name",
        "file_path",
        "path",
        "host",
        "hostname",
        "cache_dir",
    }

    def _walk(item: Any, location: str) -> None:
        if isinstance(item, dict):
            for key, nested in item.items():
                normalized = str(key).lower()
                if normalized in forbidden_keys:
                    failures.append(
                        f"prefix attestation: forbidden field {location}.{key}"
                    )
                _walk(nested, f"{location}.{key}")
            return
        if isinstance(item, list):
            for index, nested in enumerate(item):
                _walk(nested, f"{location}[{index}]")
            return
        if not isinstance(item, str):
            return
        if item.startswith(("/Users/", "/Volumes/", "/private/", "/tmp/")):
            failures.append(
                f"prefix attestation: absolute local path leaked at {location}"
            )
        if any(prompt and prompt in item for prompt in prompt_values):
            failures.append(
                f"prefix attestation: raw prompt text leaked at {location}"
            )

    _walk(value, "$")
    return failures


def _validate_prefix_attestation_snapshot(
    snapshot: Any,
    *,
    layer: str,
    expected_blocks: int,
) -> list[str]:
    failures: list[str] = []
    if not isinstance(snapshot, dict):
        return [f"prefix attestation: {layer} snapshot is missing"]
    expected_schema = (
        "vmlx-cache-prefix-l1-snapshot-v1"
        if layer == "l1"
        else "vmlx-cache-prefix-l2-snapshot-v1"
    )
    if snapshot.get("schema") != expected_schema:
        failures.append(
            f"prefix attestation: {layer} schema is not {expected_schema}"
        )
    if snapshot.get("access_metadata_mutated") is not False:
        failures.append(
            f"prefix attestation: {layer} read probe mutated access metadata"
        )
    if _integer(snapshot.get("expected_blocks")) != expected_blocks:
        failures.append(
            f"prefix attestation: {layer} expected_blocks mismatch"
        )
    if layer == "l1":
        disk_only = snapshot.get("disk_only")
        paged_ram_enabled = snapshot.get("paged_ram_enabled")
        expected_mode = "block_disk_only" if disk_only is True else "paged"
        if disk_only not in (True, False):
            failures.append("prefix attestation: L1 disk_only truth is missing")
        if paged_ram_enabled is not (disk_only is False):
            failures.append(
                "prefix attestation: L1 paged-RAM/disk-only truth is inconsistent"
            )
        if snapshot.get("backend_mode") != expected_mode:
            failures.append(
                "prefix attestation: L1 backend mode is inconsistent"
            )
        metadata = _integer(snapshot.get("metadata_blocks_present"))
        resident = _integer(snapshot.get("resident_payload_blocks_present"))
        contiguous_metadata = _integer(
            snapshot.get("contiguous_metadata_blocks")
        )
        contiguous_resident = _integer(
            snapshot.get("contiguous_resident_payload_blocks")
        )
        if not 0 <= resident <= metadata <= expected_blocks:
            failures.append(
                "prefix attestation: L1 resident/metadata counts are invalid"
            )
        if not 0 <= contiguous_resident <= contiguous_metadata <= expected_blocks:
            failures.append(
                "prefix attestation: L1 contiguous counts are invalid"
            )
        if snapshot.get("terminal_resident_payload_present") is True and (
            snapshot.get("terminal_metadata_present") is not True
        ):
            failures.append(
                "prefix attestation: L1 terminal payload exists without metadata"
            )
    else:
        indexed = _integer(snapshot.get("indexed_blocks"))
        readable = _integer(snapshot.get("readable_blocks"))
        contiguous_indexed = _integer(
            snapshot.get("contiguous_indexed_blocks")
        )
        contiguous_readable = _integer(
            snapshot.get("contiguous_readable_blocks")
        )
        if not 0 <= readable <= indexed <= expected_blocks:
            failures.append(
                "prefix attestation: L2 readable/indexed counts are invalid"
            )
        if not 0 <= contiguous_readable <= contiguous_indexed <= expected_blocks:
            failures.append(
                "prefix attestation: L2 contiguous counts are invalid"
            )
        if _integer(snapshot.get("stale_index_blocks")) != indexed - readable:
            failures.append(
                "prefix attestation: L2 stale-index count is inconsistent"
            )
        if snapshot.get("terminal_readable") is True and (
            snapshot.get("terminal_indexed") is not True
        ):
            failures.append(
                "prefix attestation: L2 terminal payload exists without index"
            )
        max_size = _integer(snapshot.get("store_max_size_bytes"))
        total_size = _integer(snapshot.get("store_total_size_bytes"))
        if max_size > 0 and total_size > max_size:
            failures.append(
                "prefix attestation: L2 store exceeds configured maximum"
            )
    return failures


def _validate_prefix_attestation_contract(
    contract: dict[str, Any],
    *,
    request_payload: dict[str, Any],
    health_attestation: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    if contract.get("contract_version") != 1:
        failures.append("prefix attestation: contract_version must be 1")
    if contract.get("method") != PREFIX_ATTESTATION_METHOD:
        failures.append("prefix attestation: method is not source-owned")
    if contract.get("surface") != "responses":
        failures.append("prefix attestation: surface must be responses")
    if (
        contract.get("cache_extra_keys_contract")
        != "generation-prompt-only-text-render-v1"
    ):
        failures.append(
            "prefix attestation: generation-prompt-only text side-key "
            "contract is missing"
        )
    if contract.get("caller_cache_or_media_side_keys") != "rejected":
        failures.append(
            "prefix attestation: caller cache/media side keys are not rejected"
        )
    if contract.get("cache_lookup_bypassed") is not True:
        failures.append("prefix attestation: cache lookup was not bypassed")
    if contract.get("access_metadata_mutated") is not False:
        failures.append("prefix attestation: read probe mutated access metadata")
    if contract.get("request_sha256") != _canonical_sha256(request_payload):
        failures.append(
            "prefix attestation: request_sha256 does not bind the exact request"
        )
    for field, contract_field in (
        ("model_bundle_provenance", "model_bundle_fingerprint_sha256"),
        ("cache_topology_provenance", "cache_topology_fingerprint_sha256"),
    ):
        attestation = health_attestation.get(field)
        expected = (
            str(attestation.get("fingerprint_sha256") or "")
            if isinstance(attestation, dict)
            else ""
        )
        if not _valid_path_free_sha256(expected):
            failures.append(
                f"prefix attestation: /health {field} fingerprint is unavailable"
            )
        elif contract.get(contract_field) != expected:
            failures.append(
                f"prefix attestation: {contract_field} does not match /health"
            )

    prompts = request_payload.get("inputs")
    prompt_rows = contract.get("prompts")
    pairs = request_payload.get("prefix_pairs")
    prefix_rows = contract.get("prefixes")
    block_size = _integer(contract.get("block_size"))
    if block_size <= 0:
        failures.append("prefix attestation: block_size is invalid")
    if not isinstance(prompts, dict) or not isinstance(prompt_rows, dict):
        return failures + ["prefix attestation: prompt metadata is missing"]
    if set(prompt_rows) != set(prompts):
        failures.append("prefix attestation: prompt labels do not match request")
    for label, prompt in prompts.items():
        row = prompt_rows.get(label)
        if not isinstance(row, dict):
            failures.append(f"prefix attestation: prompt row {label} is missing")
            continue
        if row.get("input_sha256") != hashlib.sha256(
            str(prompt).encode()
        ).hexdigest():
            failures.append(
                f"prefix attestation: prompt {label} input digest is wrong"
            )
        if _integer(row.get("cache_prompt_token_count")) <= 1:
            failures.append(
                f"prefix attestation: prompt {label} token count is unusable"
            )
        if not _valid_path_free_sha256(
            row.get("cache_prompt_token_ids_sha256")
        ):
            failures.append(
                f"prefix attestation: prompt {label} token-vector digest is invalid"
            )
        discriminator_present = row.get(
            "generation_prompt_discriminator_present"
        )
        discriminator_sha = row.get(
            "generation_prompt_discriminator_sha256"
        )
        if not isinstance(discriminator_present, bool):
            failures.append(
                f"prefix attestation: prompt {label} generation-prompt "
                "discriminator presence is missing"
            )
        elif discriminator_present and not _valid_path_free_sha256(
            discriminator_sha
        ):
            failures.append(
                f"prefix attestation: prompt {label} generation-prompt "
                "discriminator digest is invalid"
            )
        elif not discriminator_present and discriminator_sha is not None:
            failures.append(
                f"prefix attestation: prompt {label} has a discriminator "
                "digest without a discriminator"
            )
    if not isinstance(pairs, dict) or not isinstance(prefix_rows, dict):
        return failures + ["prefix attestation: prefix rows are missing"]
    if set(prefix_rows) != set(pairs):
        failures.append("prefix attestation: prefix-pair labels do not match")
    for pair_name, labels in pairs.items():
        row = prefix_rows.get(pair_name)
        if not isinstance(row, dict):
            failures.append(
                f"prefix attestation: prefix row {pair_name} is missing"
            )
            continue
        if row.get("labels") != list(labels):
            failures.append(
                f"prefix attestation: prefix row {pair_name} labels are wrong"
            )
        left_prompt = prompt_rows.get(labels[0])
        right_prompt = prompt_rows.get(labels[1])
        if isinstance(left_prompt, dict) and isinstance(right_prompt, dict):
            expected_discriminator_present = left_prompt.get(
                "generation_prompt_discriminator_present"
            )
            expected_discriminator_sha = left_prompt.get(
                "generation_prompt_discriminator_sha256"
            )
            if (
                right_prompt.get("generation_prompt_discriminator_present")
                != expected_discriminator_present
                or right_prompt.get("generation_prompt_discriminator_sha256")
                != expected_discriminator_sha
            ):
                failures.append(
                    f"prefix attestation: prefix row {pair_name} prompt "
                    "generation discriminators differ"
                )
            if (
                row.get("generation_prompt_discriminator_present")
                != expected_discriminator_present
                or row.get("generation_prompt_discriminator_sha256")
                != expected_discriminator_sha
            ):
                failures.append(
                    f"prefix attestation: prefix row {pair_name} does not bind "
                    "the production generation discriminator"
                )
        lcp_tokens = _integer(row.get("longest_common_prefix_tokens"))
        reusable_tokens = _integer(row.get("reusable_prefix_tokens"))
        expected_blocks = _integer(row.get("expected_blocks"))
        if (
            block_size <= 0
            or lcp_tokens <= block_size
            or reusable_tokens != (lcp_tokens // block_size) * block_size
            or expected_blocks != reusable_tokens // block_size
            or expected_blocks <= 0
        ):
            failures.append(
                f"prefix attestation: prefix row {pair_name} block alignment is invalid"
            )
        for field in (
            "uncached_left_tokens",
            "uncached_right_tokens",
        ):
            if _integer(row.get(field)) <= 0:
                failures.append(
                    f"prefix attestation: prefix row {pair_name} {field} "
                    "does not preserve a suffix"
                )
        for field in (
            "prefix_token_vector_sha256",
            "block_chain_fingerprint_sha256",
            "terminal_block_fingerprint_sha256",
        ):
            if not _valid_path_free_sha256(row.get(field)):
                failures.append(
                    f"prefix attestation: prefix row {pair_name} {field} is invalid"
                )
        failures.extend(
            _validate_prefix_attestation_snapshot(
                row.get("l1"),
                layer="l1",
                expected_blocks=expected_blocks,
            )
        )
        failures.extend(
            _validate_prefix_attestation_snapshot(
                row.get("l2"),
                layer="l2",
                expected_blocks=expected_blocks,
            )
        )
    failures.extend(
        _prefix_attestation_forbidden_exposure(
            contract,
            prompt_values=tuple(str(prompt) for prompt in prompts.values()),
        )
    )
    return failures


def _fetch_prefix_attestation(
    *,
    base_url: str,
    model: str,
    prompts: dict[str, str],
    pairs: dict[str, tuple[str, str] | list[str]],
    timeout: int,
    health_attestation: dict[str, Any],
    request_controls: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    request_payload = _prefix_attestation_request(
        model,
        prompts,
        pairs,
        request_controls=request_controls,
    )
    try:
        contract = _json_post(
            f"{base_url}/v1/cache/prefix-attestation",
            request_payload,
            timeout,
            private_attestation=True,
        )
    except (OSError, ValueError, urllib.error.URLError) as exc:
        return {}, [
            "prefix attestation: source-owned endpoint is unavailable; "
            "required interface is POST /v1/cache/prefix-attestation "
            f"({exc})"
        ]
    return contract, _validate_prefix_attestation_contract(
        contract,
        request_payload=request_payload,
        health_attestation=health_attestation,
    )


def _prefix_binding(
    contract: dict[str, Any],
    pair_name: str,
) -> dict[str, Any]:
    row = (contract.get("prefixes") or {}).get(pair_name)
    if not isinstance(row, dict):
        return {}
    return {
        "schema": "vmlx-cache-prefix-binding-v1",
        "attestation_sha256": _canonical_sha256(contract),
        "request_sha256": contract.get("request_sha256"),
        "model_bundle_fingerprint_sha256": contract.get(
            "model_bundle_fingerprint_sha256"
        ),
        "cache_topology_fingerprint_sha256": contract.get(
            "cache_topology_fingerprint_sha256"
        ),
        "block_size": contract.get("block_size"),
        "prefix_token_vector_sha256": row.get(
            "prefix_token_vector_sha256"
        ),
        "block_chain_fingerprint_sha256": row.get(
            "block_chain_fingerprint_sha256"
        ),
        "terminal_block_fingerprint_sha256": row.get(
            "terminal_block_fingerprint_sha256"
        ),
        "generation_prompt_discriminator_present": row.get(
            "generation_prompt_discriminator_present"
        ),
        "generation_prompt_discriminator_sha256": row.get(
            "generation_prompt_discriminator_sha256"
        ),
        "longest_common_prefix_tokens": row.get(
            "longest_common_prefix_tokens"
        ),
        "reusable_prefix_tokens": row.get("reusable_prefix_tokens"),
        "uncached_left_tokens": row.get("uncached_left_tokens"),
        "uncached_right_tokens": row.get("uncached_right_tokens"),
        "expected_blocks": row.get("expected_blocks"),
        "snapshot_wall_time_ns": contract.get("snapshot_wall_time_ns"),
        "l1": row.get("l1"),
        "l2": row.get("l2"),
    }


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


def _stable_listener_pid(
    port: int,
    *,
    required_consecutive: int = 3,
    max_attempts: int = 20,
    interval_seconds: float = 0.05,
) -> tuple[int, list[list[int]]]:
    """Require a stable singleton listener while retaining transient evidence.

    Electron restarts can briefly overlap a retiring/inheriting process with
    the new backend.  A single lsof sample is therefore not an ownership
    attestation.  Accept only the same singleton PID in consecutive samples;
    persistent duplicates, gaps, or alternating owners still fail closed.
    """

    observations: list[list[int]] = []
    candidate: int | None = None
    consecutive = 0
    for attempt in range(max_attempts):
        pids = _listener_pids(port)
        observations.append(sorted(pids))
        if len(pids) == 1:
            observed = next(iter(pids))
            if observed == candidate:
                consecutive += 1
            else:
                candidate = observed
                consecutive = 1
            if consecutive >= required_consecutive:
                return observed, observations
        else:
            candidate = None
            consecutive = 0
        if attempt + 1 < max_attempts:
            time.sleep(interval_seconds)
    raise RuntimeError(
        f"expected one stable LISTEN PID on localhost:{port}; "
        f"observed PID sets {observations}"
    )


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

    pid, observations_before = _stable_listener_pid(port)
    started_at = _run_text(["/bin/ps", "-p", str(pid), "-o", "lstart="])
    command = _run_text(["/bin/ps", "-p", str(pid), "-o", "command="])
    cwd = _listener_cwd(pid)
    python_executable, launch_shape = _listener_launch(command, cwd)
    pid_after, observations_after = _stable_listener_pid(port)
    if pid_after != pid:
        raise RuntimeError(
            f"listener changed while observing localhost:{port}: "
            f"{pid} -> {pid_after}; observed PID sets "
            f"{observations_before + observations_after}"
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
        "stabilization": {
            "required_consecutive": 3,
            "max_attempts": 20,
            "interval_seconds": 0.05,
            "initial_observed_pid_sets": observations_before,
            "final_observed_pid_sets": observations_after,
        },
    }


def _observe_source_checkout() -> dict[str, Any]:
    """Record the checkout containing this harness and its exact Git tree."""
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
    tree = _run_text(
        [
            "/usr/bin/git",
            "-C",
            str(git_root),
            "rev-parse",
            "HEAD^{tree}",
        ]
    )
    if len(tree) != 40:
        raise RuntimeError(f"Git tree is not a full SHA: {tree!r}")
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
        "tree": tree,
        "dirty": bool(status_lines),
        "status_porcelain": status_lines,
        "status_sha256": hashlib.sha256(status_text.encode()).hexdigest(),
    }


def _compare_source_checkout_observations(
    before: dict[str, Any],
    after: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    for field in ("git_root", "head", "tree", "dirty", "status_sha256"):
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
    python_executable_hashes = {
        hashlib.sha256(
            str(observed_engine.get("python_executable") or "").encode()
        ).hexdigest()
    }
    # Python.framework/pyvenv launches can report the framework Python.app in
    # `ps`, while `sys.executable` inside the process truthfully resolves to the
    # checkout venv entrypoint.  Keep the gate strict by accepting only the venv
    # that belongs to the already-proven source checkout.
    checkout_venv_python = git_root / ".venv" / "bin" / "python3"
    if (
        observed_engine.get("launch_shape") == "python-module-vmlx-engine-cli"
        and checkout_venv_python.exists()
    ):
        python_executable_hashes.add(
            hashlib.sha256(str(checkout_venv_python.absolute()).encode()).hexdigest()
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
    }
    for field, expected_value in expected.items():
        if attestation.get(field) != expected_value:
            failures.append(
                f"provenance: /health {field} does not match the observed "
                "listener/source"
            )
    if (
        attestation.get("python_executable_fingerprint_sha256")
        not in python_executable_hashes
    ):
        failures.append(
            "provenance: /health python_executable_fingerprint_sha256 does not "
            "match the observed listener/source"
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
    """Wait for one request's post-eviction write fence to settle twice.

    Scheduler occupancy is intentionally observational here.  A terminal
    response is dispatched before architecture-native prompt-cache cleanup,
    and a later request may therefore be running while this request's exact
    physical write fence is already sealed and durable.  Requiring global
    scheduler idleness turns a request-correlated durability proof into a
    false failure for slow deferred-cleanup families such as MiniMax M3.
    """
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
        scheduler_cache = cache.get("scheduler_cache")
        if not isinstance(scheduler_cache, dict):
            scheduler_cache = {}
        global_budget = block_disk.get("global_budget")
        if not isinstance(global_budget, dict):
            global_budget = {}
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
        if scheduler_cache.get("strict_block_disk_write_fence") is not True:
            contract_failures.append(
                "engine was not launched with strict physical block-disk fences"
            )
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
            baseline_reconciliation_generation = _integer(
                baseline_counters.get(
                    "block_disk_cache.global_reconciliation_generation"
                )
            )
            fence_reconciliation_generation = _integer(
                matching_fence.get("global_reconciliation_generation")
            )
            if fence_reconciliation_generation <= baseline_reconciliation_generation:
                contract_failures.append(
                    "request write fence did not advance physical reconciliation"
                )
            if _integer(
                global_budget.get("reconciliation_generation")
            ) < fence_reconciliation_generation:
                contract_failures.append(
                    "managed-root telemetry is older than the request fence"
                )
        if global_budget.get("accounted") is not True:
            contract_failures.append("managed-root physical accounting is not settled")
        if global_budget.get("compliant") is not True:
            contract_failures.append("managed-root physical bytes are over limit")
        global_bytes_after = _integer(global_budget.get("bytes_after"))
        global_max_size_bytes = _integer(global_budget.get("max_size_bytes"))
        if global_max_size_bytes > 0 and not (
            0 <= global_bytes_after <= global_max_size_bytes
        ):
            contract_failures.append(
                "managed-root physical bytes exceed the finite configured limit"
            )
        if _integer(pipeline_snapshot.get("queue_depth")) != 0:
            contract_failures.append("block-disk write queue is not empty")
        if _integer(pipeline_snapshot.get("inflight")) != 0:
            contract_failures.append("block-disk writes are still in flight")
        if pipeline_snapshot.get("writer_alive") is not True:
            contract_failures.append("block-disk writer is not alive")
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
                "global_budget": global_budget,
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
    allow_native_matched_tail_replay: bool = False,
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
    reusable_prefix_tokens = cached_tokens
    replayed_tokens = _integer(execution.get("replayed_tokens"))
    matched_tokens = _integer(execution.get("matched_tokens"))
    checkpoint_tokens = _integer(execution.get("checkpoint_tokens"))
    memory_fit_partial = (
        replayed_tokens == 0
        and attempted_cached_tokens > cached_tokens > 0
        and matched_tokens == attempted_cached_tokens
        and isinstance(row.get("last_cache_reuse_partial"), dict)
    )
    if memory_fit_partial:
        reusable_prefix_tokens = attempted_cached_tokens
        scheduler_cache = row.get("scheduler_cache")
        block_size = (
            _integer(scheduler_cache.get("block_size"))
            if isinstance(scheduler_cache, dict)
            else 0
        )
        failures.extend(
            _validate_execution_prefix_bounds(
                row,
                {
                    "block_size": block_size,
                    "reusable_prefix_tokens": minimum_cached_tokens,
                    "longest_common_prefix_tokens": maximum_cached_tokens,
                },
                label=tag,
            )
        )
    elif replayed_tokens > 0 or matched_tokens > cached_tokens:
        native_cache = row.get("native_cache")
        native_replay_contract = bool(
            allow_native_matched_tail_replay
            and isinstance(native_cache, dict)
            and native_cache.get("family") == "deepseek_v4"
            and native_cache.get("cache_type") == "native_composite"
            and native_cache.get("schema") == "deepseek_v4_v10_delta"
            and (native_cache.get("generic_turboquant_kv") or {}).get("enabled")
            is False
        )
        if not native_replay_contract:
            failures.append(
                f"{tag}: matched-tail replay is not authorized by the native "
                "DSV4 delta-cache contract"
            )
        elif checkpoint_tokens != cached_tokens:
            failures.append(
                f"{tag}: checkpoint_tokens={checkpoint_tokens} does not equal "
                f"cached_tokens={cached_tokens}"
            )
        elif matched_tokens != checkpoint_tokens + replayed_tokens:
            failures.append(
                f"{tag}: matched_tokens={matched_tokens} does not equal "
                f"checkpoint_tokens+replayed_tokens="
                f"{checkpoint_tokens + replayed_tokens}"
            )
        elif matched_tokens > prompt_tokens:
            failures.append(
                f"{tag}: matched_tokens={matched_tokens} exceeds "
                f"prompt_tokens={prompt_tokens}"
            )
        else:
            reusable_prefix_tokens = matched_tokens

    if cached_tokens <= 0:
        failures.append(f"{tag}: cached_tokens must be positive")
    if reusable_prefix_tokens < minimum_cached_tokens:
        failures.append(
            f"{tag}: reusable prefix tokens={reusable_prefix_tokens} "
            "is below expected shared-prefix "
            f"floor {minimum_cached_tokens}"
        )
    if maximum_cached_tokens > 0 and reusable_prefix_tokens > maximum_cached_tokens:
        failures.append(
            f"{tag}: reusable prefix tokens={reusable_prefix_tokens} "
            "exceeds independent tokenizer "
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


def _validate_monotonic_counter_deltas(
    row: dict[str, Any],
    *,
    label: str,
) -> list[str]:
    """Reject counter resets inside one request-correlated observation."""

    deltas = row.get("health_counter_deltas")
    if not isinstance(deltas, dict):
        return []
    return [
        f"{label}: monotonic counter {key} has negative delta={value}"
        for key, value in sorted(deltas.items())
        if isinstance(value, (int, float)) and value < 0
    ]


def _validate_hybrid_ssm_tq4_hit(
    row: dict[str, Any],
    *,
    require_disk_origin: bool,
    label: str | None = None,
    require_tq4: bool = True,
    require_paged: bool = True,
) -> list[str]:
    """Require one accepted Qwen hybrid hit to include KV, SSM, and TQ truth.

    ``require_tq4=False`` selects the EXACT-KV contract for bundles where the
    engine's asymmetry guard deliberately disables TurboQuant storage (no
    calibrated jang_config.turboquant block -> live encode off): the hit must
    then attest storage quantization OFF (a TQ-on serve fails closed and must
    use the TQ4 profile). ``require_paged=False`` admits the disk-only lane,
    which can never attest a paged RAM pool.
    """

    tag = label or str(row.get("tag") or "<missing-tag>")
    failures: list[str] = []
    execution = row.get("last_cache_execution")
    if not isinstance(execution, dict):
        return [f"{tag}: hybrid SSM cache execution is missing"]

    response_id = str(row.get("response_id") or "")
    execution_request_id = str(execution.get("request_id") or "")
    if (
        row.get("response_id_consistent") is not True
        or not response_id
        or execution_request_id != response_id
    ):
        failures.append(
            f"{tag}: hybrid cache evidence is not request-correlated"
        )

    native_cache = row.get("native_cache")
    if not isinstance(native_cache, dict):
        native_cache = {}
    if native_cache.get("schema") != "hybrid_ssm_v1":
        failures.append(f"{tag}: native_cache schema is not hybrid_ssm_v1")
    if native_cache.get("cache_type") != "hybrid_ssm_typed":
        failures.append(f"{tag}: native_cache type is not hybrid_ssm_typed")
    if require_paged and native_cache.get("paged") is not True:
        failures.append(f"{tag}: native_cache does not attest paged RAM")
    if native_cache.get("block_disk_l2") is not True:
        failures.append(f"{tag}: native_cache does not attest block-disk L2")
    components = native_cache.get("components")
    if not isinstance(components, list) or not {
        "attention_kv",
        "ssm_companion_state",
    }.issubset(set(components)):
        failures.append(
            f"{tag}: native_cache components do not include attention KV and "
            "SSM companion state"
        )
    storage_quant = native_cache.get("attention_kv_storage_quantization")
    if not isinstance(storage_quant, dict):
        storage_quant = {}
    if require_tq4:
        if (
            storage_quant.get("enabled") is not True
            or storage_quant.get("codec") != "turboquant_native"
            or storage_quant.get("applies_to") != "attention_kv_layers_only"
            or _integer(storage_quant.get("bits")) != 4
            or _integer(storage_quant.get("value_bits")) != 4
        ):
            failures.append(
                f"{tag}: native_cache does not attest attention-only "
                "TurboQuant q4"
            )
    elif storage_quant.get("enabled") is not False:
        failures.append(
            f"{tag}: exact-KV contract requires storage quantization "
            "attested OFF (TQ-on serves must use the TQ4 profile)"
        )
    generic_tq = native_cache.get("generic_turboquant_kv")
    if not isinstance(generic_tq, dict):
        generic_tq = {}
    if require_tq4:
        if (
            generic_tq.get("enabled") is not True
            or generic_tq.get("reason") != "hybrid_attention_kv_only"
        ):
            failures.append(
                f"{tag}: native_cache does not attest Qwen hybrid-only "
                "TurboQuant"
            )

    cached_tokens = _integer(execution.get("cached_tokens"))
    attempted_cached_tokens = _integer(execution.get("attempted_cached_tokens"))
    prompt_tokens = _integer(execution.get("prompt_tokens"))
    uncached_prompt_tokens = _integer(execution.get("uncached_prompt_tokens"))
    generation_suffix_tokens = _integer(
        execution.get("generation_prompt_suffix_tokens")
    )
    if (
        attempted_cached_tokens <= 0
        or cached_tokens <= 0
        or cached_tokens > attempted_cached_tokens
    ):
        failures.append(
            f"{tag}: accepted cached tokens are not bound to a positive "
            "attempted prefix candidate"
        )
    if prompt_tokens <= cached_tokens or uncached_prompt_tokens <= 0:
        failures.append(
            f"{tag}: hybrid reuse is not partial-prefix reuse with an uncached tail"
        )
    if uncached_prompt_tokens != max(prompt_tokens - cached_tokens, 0):
        failures.append(
            f"{tag}: hybrid uncached tokens do not equal prompt minus cached"
        )
    if _integer(execution.get("prefill_tokens")) != (
        uncached_prompt_tokens + generation_suffix_tokens
    ):
        failures.append(
            f"{tag}: hybrid prefill does not equal uncached tail plus "
            "generation-prompt suffix"
        )
    ssm_companion = row.get("ssm_companion")
    if not isinstance(ssm_companion, dict):
        ssm_companion = {}
    lookup = ssm_companion.get("last_prefix_lookup")
    if not isinstance(lookup, dict):
        lookup = {}
    lookup_request_id = str(lookup.get("request_id") or "")
    if (
        not lookup_request_id
        or lookup_request_id != response_id
        or lookup_request_id != execution_request_id
    ):
        failures.append(
            f"{tag}: SSM companion prefix lookup is not request-correlated"
        )
    checkpoint_tokens = _integer(lookup.get("checkpoint_tokens"))
    max_len = _integer(lookup.get("max_len"))
    candidate_lengths = lookup.get("candidate_lengths")
    candidates_well_formed = (
        isinstance(candidate_lengths, list)
        and len(candidate_lengths) <= 20
        and all(type(candidate) is int for candidate in candidate_lengths)
    )
    if not candidates_well_formed:
        candidate_lengths = []
    attempted_candidate_lengths = lookup.get("attempted_candidate_lengths")
    attempts_well_formed = (
        isinstance(attempted_candidate_lengths, list)
        and 0 < len(attempted_candidate_lengths) <= 21
        and all(
            type(candidate) is int
            for candidate in attempted_candidate_lengths
        )
    )
    if not attempts_well_formed:
        attempted_candidate_lengths = []
    candidate_count = lookup.get("candidate_count")
    attempted_candidate_count = lookup.get("attempted_candidate_count")
    if (
        lookup.get("candidate_lengths_truncated") is not False
        or type(candidate_count) is not int
        or candidate_count != len(candidate_lengths)
    ):
        failures.append(
            f"{tag}: SSM candidate telemetry is truncated or count-mismatched"
        )
    if (
        lookup.get("attempted_candidate_lengths_truncated") is not False
        or type(attempted_candidate_count) is not int
        or attempted_candidate_count != len(attempted_candidate_lengths)
    ):
        failures.append(
            f"{tag}: SSM attempted-candidate telemetry is truncated or "
            "count-mismatched"
        )
    if lookup.get("matched") is not True:
        failures.append(f"{tag}: SSM companion prefix lookup did not match")
    if lookup.get("is_complete") is not True:
        failures.append(f"{tag}: SSM companion checkpoint is not complete")
    if checkpoint_tokens <= 0 or checkpoint_tokens != cached_tokens:
        failures.append(
            f"{tag}: SSM checkpoint_tokens={checkpoint_tokens} does not equal "
            f"accepted cached_tokens={cached_tokens}"
        )
    if max_len != attempted_cached_tokens:
        failures.append(
            f"{tag}: SSM prefix lookup max_len={max_len} does not equal "
            f"attempted_cached_tokens={attempted_cached_tokens}"
        )
    normalized_candidates = list(candidate_lengths)
    if (
        not candidates_well_formed
        or checkpoint_tokens not in normalized_candidates
        or not normalized_candidates
        or any(
            candidate <= 0 or candidate > attempted_cached_tokens
            for candidate in normalized_candidates
        )
    ):
        failures.append(
            f"{tag}: SSM candidate lengths are not bound to the attempted "
            "prefix and matched checkpoint"
        )
    normalized_attempts = list(attempted_candidate_lengths)
    if (
        not attempts_well_formed
        or normalized_attempts[-1:] != [checkpoint_tokens]
        or any(
            candidate <= 0 or candidate > attempted_cached_tokens
            for candidate in normalized_attempts
        )
        or any(
            candidate != attempted_cached_tokens
            and candidate not in normalized_candidates
            for candidate in normalized_attempts
        )
    ):
        failures.append(
            f"{tag}: SSM attempted candidate lengths are malformed, "
            "out of bounds, or do not terminate at the accepted checkpoint"
        )
    if lookup.get("source") not in {
        "exact_boundary_l1_or_l2",
        "l1_or_l2",
        "partial_boundary_disk_l2",
    }:
        failures.append(f"{tag}: SSM companion lookup source is not attested")

    detail = str(execution.get("cache_detail") or "")
    if require_paged:
        if str(execution.get("selection") or "").lower() != "paged":
            failures.append(
                f"{tag}: hybrid SSM proof did not select paged cache"
            )
        if "paged+ssm" not in detail.lower():
            failures.append(f"{tag}: cache_detail does not identify paged+ssm")
    else:
        # Disk-only lane: the hybrid hit must attest the block-disk selection
        # with its SSM companion — a paged selection here would mean the lane
        # under test is not the one that served.
        if str(execution.get("selection") or "").lower() not in {
            "block-disk",
            "paged",
        }:
            failures.append(
                f"{tag}: hybrid SSM proof selection is neither block-disk "
                "nor paged"
            )
        if "+ssm" not in detail.lower():
            failures.append(
                f"{tag}: cache_detail does not identify an SSM-typed hit"
            )
    if require_tq4:
        if "tq-native" not in detail.lower():
            failures.append(
                f"{tag}: cache_detail does not identify native TQ blocks"
            )
        if _integer(execution.get("tq_native_blocks")) <= 0:
            failures.append(
                f"{tag}: execution tq_native_blocks must be positive"
            )
    if execution.get("dequantized") is not True:
        failures.append(f"{tag}: dequantized is not true")
    if execution.get("dequantization_ok") is not True:
        failures.append(f"{tag}: dequantization_ok is not true")
    if require_disk_origin:
        if execution.get("disk_hit") is not True:
            failures.append(f"{tag}: restart hybrid refault disk_hit is not true")
        if "disk" not in detail.lower():
            failures.append(
                f"{tag}: restart hybrid cache_detail does not identify disk"
            )

    deltas = row.get("health_counter_deltas")
    if not isinstance(deltas, dict):
        return failures + [f"{tag}: hybrid cache counter deltas are missing"]
    for key in (
        "scheduler.hybrid_kv_without_ssm_hits",
        "scheduler.hybrid_kv_without_ssm_tokens",
    ):
        if key not in deltas:
            failures.append(f"{tag}: {key} delta is missing")
        elif _integer(deltas.get(key)) != 0:
            failures.append(f"{tag}: {key} increased during accepted hybrid reuse")
    if require_disk_origin:
        required_delta_keys = ["block_disk_cache.disk_hits"]
        if require_tq4:
            # Exact-KV bundles (asymmetry guard: TQ storage attested OFF)
            # persist plain KV records; tq_native_hits can never move there
            # and demanding it would fail every healthy exact refault.
            required_delta_keys.append("block_disk_cache.tq_native_hits")
        required_delta_keys.append("ssm_companion.disk.hits")
        for key in required_delta_keys:
            if key not in deltas:
                failures.append(f"{tag}: {key} delta is missing")
            elif _integer(deltas.get(key)) <= 0:
                failures.append(f"{tag}: {key} did not increase on restart refault")
        disk_blocks = _integer(execution.get("disk_blocks"))
        disk_hits = _integer(deltas.get("block_disk_cache.disk_hits"))
        if disk_blocks > 0 and disk_hits < disk_blocks:
            failures.append(
                f"{tag}: block-disk hit delta={disk_hits} is below "
                f"reconstructed disk_blocks={disk_blocks}"
            )
        tq_native_hits = _integer(
            deltas.get("block_disk_cache.tq_native_hits")
        )
        if require_tq4:
            tq_native_blocks = _integer(execution.get("tq_native_blocks"))
            if tq_native_blocks > 0 and tq_native_hits < tq_native_blocks:
                failures.append(
                    f"{tag}: TQ-native hit delta={tq_native_hits} is below "
                    f"reconstructed tq_native_blocks={tq_native_blocks}"
                )
        elif tq_native_hits > 0:
            # Fail closed: a TQ-native record hitting on an exact-KV serve
            # means two codecs share the token-prefix hash space.
            failures.append(
                f"{tag}: TQ-native hit delta={tq_native_hits} observed on an "
                "exact-KV (storage-quant OFF) serve"
            )
    return failures


def _tokenizer_prefix_floor(
    row: dict[str, Any],
    *,
    selector: str,
    require_partial: bool,
    token_contract: dict[str, Any],
    exact_complete_block_floor: bool = False,
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

    if require_partial or exact_complete_block_floor:
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
            # Native sparse caches also use this floor for exact prompts because
            # they may re-feed a short architecture-owned terminal boundary.
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


def _cache_contract_profile_from_health(health: dict[str, Any]) -> str:
    """Select only an attested architecture-owned cache contract."""

    topology_attestation = health.get("cache_topology_provenance")
    topology = (
        topology_attestation.get("configuration")
        if isinstance(topology_attestation, dict)
        else None
    )
    native = topology.get("native_cache") if isinstance(topology, dict) else None
    generic_tq = (
        native.get("generic_turboquant_kv")
        if isinstance(native, dict)
        else None
    )
    if (
        isinstance(native, dict)
        and native.get("family") == "deepseek_v4"
        and native.get("cache_type") == "native_composite"
        and native.get("schema") == "deepseek_v4_v10_delta"
        and isinstance(generic_tq, dict)
        and generic_tq.get("enabled") is False
        and (topology.get("turboquant_kv_cache") or {}).get("enabled") is False
        and (topology.get("kv_cache_quantization") or {}).get("enabled") is False
    ):
        return "deepseek_v4_native_delta"
    if (
        isinstance(native, dict)
        and native.get("family") == "qwen3_5"
        and native.get("schema") == "hybrid_ssm_v1"
    ):
        # Family + schema identify the Qwen hybrid contract. The TQ4 vs
        # exact-KV split keys on the ATTESTED storage-quantization state: the
        # asymmetry guard deliberately disables TurboQuant storage for
        # uncalibrated bundles (no jang_config.turboquant -> live encode off),
        # and such serves can never satisfy the TQ4 contract. Remaining typed
        # fields still validate later so malformed intended-Qwen attestations
        # fail closed instead of silently downgrading to generic KV.
        _storage_quant = (
            native.get("attention_kv_storage_quantization")
            if isinstance(native, dict)
            else None
        )
        if isinstance(_storage_quant, dict) and _storage_quant.get(
            "enabled"
        ) is False:
            return "qwen_hybrid_ssm_exact"
        return "qwen_hybrid_ssm_tq4"
    if (
        isinstance(native, dict)
        and native.get("family") == "minimax_m3"
        and native.get("cache_type") == "native_msa_sparse_kv"
        and native.get("schema") == "minimax_m3_msa_v1"
        and isinstance(generic_tq, dict)
        and generic_tq.get("enabled") is False
        and (topology.get("turboquant_kv_cache") or {}).get("enabled") is False
        and (topology.get("kv_cache_quantization") or {}).get("enabled") is False
    ):
        return "minimax_m3_sparse_block"
    return "generic"


def validate_cache_rows(
    phase: str,
    rows: list[dict[str, Any]],
    *,
    store_summary: dict[str, Any] | None = None,
    token_contract: dict[str, Any] | None = None,
    contract_profile: str = "generic",
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
    if contract_profile not in {
        "generic",
        "deepseek_v4_native_delta",
        "minimax_m3_sparse_block",
        "qwen_hybrid_ssm_tq4",
        "qwen_hybrid_ssm_exact",
    }:
        return [f"unsupported cache contract profile: {contract_profile}"]
    native_sparse = contract_profile == "minimax_m3_sparse_block"
    dsv4_native_delta = contract_profile == "deepseek_v4_native_delta"
    hybrid_ssm_tq4 = contract_profile in {
        "qwen_hybrid_ssm_tq4",
        "qwen_hybrid_ssm_exact",
    }
    hybrid_require_tq4 = contract_profile == "qwen_hybrid_ssm_tq4"
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
    prompt_offset = 0
    if native_sparse:
        prompts = token_contract.get("prompts")
        selectors = {
            "cold_a": "A",
            "warm_a": "A",
            "partial_b": "B",
            "restart_partial_c": "C",
            "restart_a": "A",
        }
        offsets: set[int] = set()
        suffix_allowances: list[int] = []
        if isinstance(prompts, dict):
            for tag in requirements:
                row = by_tag.get(tag)
                prompt = prompts.get(selectors[tag])
                execution = (
                    row.get("last_cache_execution")
                    if isinstance(row, dict)
                    else None
                )
                if not isinstance(prompt, dict) or not isinstance(execution, dict):
                    continue
                suffix_allowances.append(
                    _integer(prompt.get("generation_prompt_suffix_tokens"))
                )
                expected = _integer(prompt.get("cache_prompt_token_count"))
                observed = _integer(execution.get("prompt_tokens"))
                if expected > 1 and observed > 1:
                    offsets.add(observed - expected)
        if len(offsets) != 1:
            failures.append(
                f"{phase}: native sparse prompt offset is not stable: "
                f"{sorted(offsets)}"
            )
        else:
            prompt_offset = next(iter(offsets))
            suffix_allowance = min(suffix_allowances, default=0)
            if not 0 <= prompt_offset <= suffix_allowance:
                failures.append(
                    f"{phase}: native sparse prompt offset={prompt_offset} exceeds "
                    "minimum required-prompt template suffix allowance="
                    f"{suffix_allowance}"
                )
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
                expected_execution_tokens = 0
                count_failures: list[str] = []
                if expected_prompt_tokens <= 1:
                    row_failures.append(
                        f"{tag}: independent tokenizer prompt count is missing"
                    )
                elif native_sparse:
                    expected_execution_tokens = (
                        expected_prompt_tokens + prompt_offset
                    )
                    count_failures = []
                else:
                    expected_execution_tokens, count_failures = (
                        _expected_execution_prompt_tokens(
                            prompt_row,
                            execution,
                            tag=tag,
                        )
                    )
                row_failures.extend(count_failures)
                if (
                    expected_prompt_tokens > 1
                    and _integer(execution.get("prompt_tokens"))
                    != expected_execution_tokens
                ):
                    row_failures.append(
                        f"{tag}: execution prompt_tokens="
                        f"{_integer(execution.get('prompt_tokens'))} does not "
                        "match independent tokenizer execution count="
                        f"{expected_execution_tokens}"
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
                exact_complete_block_floor=native_sparse,
            )
            execution_count_failures: list[str] = []
            if native_sparse:
                expected_execution_tokens = expected_prompt_tokens + prompt_offset
            else:
                expected_execution_tokens, execution_count_failures = (
                    _expected_execution_prompt_tokens(
                        (token_contract.get("prompts") or {}).get(selector) or {},
                        row.get("last_cache_execution") or {},
                        tag=tag,
                    )
                )
            row_failures = _validate_hit_row(
                row,
                require_partial=requirement in {"partial", "disk_partial"},
                require_disk_origin=requirement == "disk_partial",
                minimum_cached_tokens=minimum_cached_tokens,
                maximum_cached_tokens=independent_lcp_tokens,
                expected_prompt_tokens=expected_execution_tokens,
                # Standard scheduler/TQ hits may be direct memory/prefix reuse.
                # Only restart-C must prove worker reconstruction from disk.
                allow_direct_reuse=requirement != "disk_partial",
                allow_native_matched_tail_replay=dsv4_native_delta,
            )
            if hybrid_ssm_tq4 and requirement in {"partial", "disk_partial"}:
                row_failures.extend(
                    _validate_hybrid_ssm_tq4_hit(
                        row,
                        require_disk_origin=requirement == "disk_partial",
                        require_tq4=hybrid_require_tq4,
                        require_paged=hybrid_require_tq4,
                    )
                )
            row_failures.extend(execution_count_failures)
            row_failures.extend(floor_failures)
            row["expected_shared_prefix_floor_tokens"] = minimum_cached_tokens
            row["independent_longest_common_prefix_tokens"] = (
                independent_lcp_tokens
            )
            row["independent_prompt_tokens"] = expected_prompt_tokens
        row_failures.extend(
            _validate_monotonic_counter_deltas(row, label=tag)
        )
        row["cache_contract_required"] = True
        row["cache_contract_profile"] = contract_profile
        if native_sparse:
            row["native_prompt_token_offset"] = prompt_offset
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


def _validate_prefix_binding(
    binding: Any,
    *,
    expected_fingerprint: str,
    expected_bundle_fingerprint: str,
    expected_topology_fingerprint: str,
    label: str,
) -> list[str]:
    failures: list[str] = []
    if not isinstance(binding, dict):
        return [f"{label}: source prefix binding is missing"]
    if binding.get("schema") != "vmlx-cache-prefix-binding-v1":
        failures.append(f"{label}: source prefix binding schema is invalid")
    for field in (
        "attestation_sha256",
        "request_sha256",
        "prefix_token_vector_sha256",
        "block_chain_fingerprint_sha256",
        "terminal_block_fingerprint_sha256",
    ):
        if not _valid_sha256(binding.get(field)):
            failures.append(f"{label}: {field} is not a SHA-256")
    discriminator_present = binding.get(
        "generation_prompt_discriminator_present"
    )
    discriminator_sha = binding.get("generation_prompt_discriminator_sha256")
    if not isinstance(discriminator_present, bool):
        failures.append(
            f"{label}: generation-prompt discriminator presence is missing"
        )
    elif discriminator_present and not _valid_sha256(discriminator_sha):
        failures.append(
            f"{label}: generation-prompt discriminator digest is invalid"
        )
    elif not discriminator_present and discriminator_sha is not None:
        failures.append(
            f"{label}: generation-prompt discriminator digest is unexpected"
        )
    if binding.get("block_chain_fingerprint_sha256") != expected_fingerprint:
        failures.append(f"{label}: prefix fingerprint does not match")
    if (
        binding.get("model_bundle_fingerprint_sha256")
        != expected_bundle_fingerprint
    ):
        failures.append(f"{label}: model-bundle fingerprint does not match")
    if (
        binding.get("cache_topology_fingerprint_sha256")
        != expected_topology_fingerprint
    ):
        failures.append(f"{label}: cache-topology fingerprint does not match")
    block_size = _integer(binding.get("block_size"))
    reusable_tokens = _integer(binding.get("reusable_prefix_tokens"))
    expected_blocks = _integer(binding.get("expected_blocks"))
    if (
        block_size <= 0
        or reusable_tokens <= 0
        or reusable_tokens % block_size != 0
        or expected_blocks != reusable_tokens // block_size
        or expected_blocks <= 0
    ):
        failures.append(f"{label}: reusable prefix/block count is invalid")
    if _integer(binding.get("uncached_left_tokens")) <= 0 or _integer(
        binding.get("uncached_right_tokens")
    ) <= 0:
        failures.append(f"{label}: source pair does not retain an uncached suffix")
    failures.extend(
        _validate_prefix_attestation_snapshot(
            binding.get("l1"),
            layer="l1",
            expected_blocks=expected_blocks,
        )
    )
    failures.extend(
        _validate_prefix_attestation_snapshot(
            binding.get("l2"),
            layer="l2",
            expected_blocks=expected_blocks,
        )
    )
    return failures


def _l2_binding_is_complete(binding: Any) -> bool:
    """Return whether one exact prefix chain is wholly readable from L2."""
    if not isinstance(binding, dict):
        return False
    expected_blocks = _integer(binding.get("expected_blocks"))
    l2 = binding.get("l2")
    if expected_blocks <= 0 or not isinstance(l2, dict):
        return False
    return bool(
        _integer(l2.get("expected_blocks")) == expected_blocks
        and _integer(l2.get("indexed_blocks")) == expected_blocks
        and _integer(l2.get("readable_blocks")) == expected_blocks
        and _integer(l2.get("contiguous_indexed_blocks")) == expected_blocks
        and _integer(l2.get("contiguous_readable_blocks")) == expected_blocks
        and _integer(l2.get("stale_index_blocks")) == 0
        and l2.get("terminal_indexed") is True
        and l2.get("terminal_readable") is True
    )


def _validate_complete_l2_binding(binding: Any, *, label: str) -> list[str]:
    if _l2_binding_is_complete(binding):
        return []
    return [f"{label}: exact complete readable L2 chain is not proven"]


def _validate_disk_refault_execution(
    execution: Any,
    *,
    label: str,
    contract_profile: str = "generic",
) -> list[str]:
    failures: list[str] = []
    if not isinstance(execution, dict):
        return [f"{label}: exact request execution is missing"]
    response_id = str(execution.get("response_id") or "")
    last = execution.get("last_cache_execution")
    if not response_id or not isinstance(last, dict):
        return [f"{label}: request-correlated cache execution is missing"]
    if str(last.get("request_id") or "") != response_id:
        failures.append(
            f"{label}: cache execution request_id does not match Responses id"
        )
    if execution.get("response_id_consistent") is not True:
        failures.append(f"{label}: Responses stream ID is not consistent")
    if execution.get("terminal_ok") is not True:
        failures.append(f"{label}: Responses stream did not complete")
    if execution.get("marker_ok") is not True:
        failures.append(f"{label}: expected visible marker is missing")
    if _integer(execution.get("cached_tokens")) <= 0:
        failures.append(f"{label}: no cached tokens were reported")
    if last.get("cache_reuse_applied") is not True:
        failures.append(f"{label}: cache reuse was not applied")
    if last.get("cache_outcome") != "hit":
        failures.append(f"{label}: cache outcome is not a hit; cold fallback is possible")
    if last.get("reconstructed") is not True:
        failures.append(f"{label}: block-disk reconstruction was not applied")
    if last.get("reconstruction_ok") is not True:
        failures.append(f"{label}: block-disk reconstruction did not succeed")
    if _integer(last.get("disk_blocks")) <= 0:
        failures.append(f"{label}: no block-disk blocks were loaded")
    if _integer(last.get("cached_tokens")) <= 0:
        failures.append(f"{label}: no block-disk tokens were restored")
    prompt_tokens = _integer(last.get("prompt_tokens"))
    uncached_tokens = _integer(last.get("uncached_prompt_tokens"))
    if prompt_tokens <= 0 or uncached_tokens <= 0:
        failures.append(
            f"{label}: refault did not retain a tokenizer-visible uncached suffix"
        )
    cache_detail = execution.get("cache_detail")
    if not isinstance(cache_detail, dict):
        cache_detail = {}
    origin = str(
        cache_detail.get("source")
        or cache_detail.get("origin")
        or last.get("cache_detail")
        or ""
    ).lower()
    if "disk" not in origin and "l2" not in origin:
        failures.append(f"{label}: cache origin is not block-disk")
    failures.extend(
        _validate_monotonic_counter_deltas(execution, label=label)
    )
    if contract_profile in {"qwen_hybrid_ssm_tq4", "qwen_hybrid_ssm_exact"}:
        failures.extend(
            _validate_hybrid_ssm_tq4_hit(
                execution,
                require_disk_origin=True,
                label=label,
                require_tq4=contract_profile == "qwen_hybrid_ssm_tq4",
                require_paged=contract_profile == "qwen_hybrid_ssm_tq4",
            )
        )
    return failures


def _validate_execution_prefix_bounds(
    execution: Any,
    binding: Any,
    *,
    label: str,
) -> list[str]:
    if not isinstance(execution, dict) or not isinstance(binding, dict):
        return [f"{label}: execution/prefix binding is missing"]
    failures: list[str] = []
    last = execution.get("last_cache_execution")
    if not isinstance(last, dict):
        return [f"{label}: request-correlated cache execution is missing"]
    response_id = str(execution.get("response_id") or "")
    attempted_tokens = _integer(last.get("attempted_cached_tokens"))
    cached_tokens = _integer(last.get("cached_tokens"))
    reported_cached_tokens = _integer(execution.get("cached_tokens"))
    reusable_floor = _integer(binding.get("reusable_prefix_tokens"))
    lcp_ceiling = _integer(binding.get("longest_common_prefix_tokens"))
    block_size = _integer(binding.get("block_size"))
    native_cache = execution.get("native_cache")
    if not isinstance(native_cache, dict):
        native_cache = {}
    dsv4_delta = (
        native_cache.get("family") == "deepseek_v4"
        and native_cache.get("schema") == "deepseek_v4_v10_delta"
    )
    bound_tokens = attempted_tokens
    bound_field = "attempted_cached_tokens"
    if dsv4_delta:
        checkpoint_tokens = _integer(last.get("checkpoint_tokens"))
        matched_tokens = _integer(last.get("matched_tokens"))
        replayed_tokens = _integer(last.get("replayed_tokens"))
        if checkpoint_tokens != attempted_tokens:
            failures.append(
                f"{label}: checkpoint_tokens={checkpoint_tokens} does not equal "
                f"attempted_cached_tokens={attempted_tokens}"
            )
        if matched_tokens != checkpoint_tokens + replayed_tokens:
            failures.append(
                f"{label}: matched_tokens={matched_tokens} does not equal "
                f"checkpoint_tokens+replayed_tokens="
                f"{checkpoint_tokens + replayed_tokens}"
            )
        bound_tokens = matched_tokens
        bound_field = "matched_tokens"
    if not reusable_floor <= bound_tokens <= lcp_ceiling:
        failures.append(
            f"{label}: {bound_field}={bound_tokens} is outside the exact "
            f"block-aligned prefix range [{reusable_floor}, {lcp_ceiling}]"
        )
    if block_size <= 0:
        failures.append(f"{label}: source block size is invalid")
    elif attempted_tokens <= 0 or attempted_tokens % block_size != 0:
        failures.append(
            f"{label}: attempted_cached_tokens={attempted_tokens} is not "
            f"positive and block-aligned to {block_size}"
        )
    if cached_tokens <= 0 or cached_tokens > attempted_tokens:
        failures.append(
            f"{label}: accepted cached_tokens={cached_tokens} is not within "
            f"(0, {attempted_tokens}]"
        )
    elif block_size > 0 and cached_tokens % block_size != 0:
        failures.append(
            f"{label}: accepted cached_tokens={cached_tokens} is not "
            f"block-aligned to {block_size}"
        )
    if reported_cached_tokens != cached_tokens:
        failures.append(
            f"{label}: Responses cached_tokens={reported_cached_tokens} does "
            f"not equal scheduler accepted cached_tokens={cached_tokens}"
        )
    if block_size <= 0 or attempted_tokens <= 0:
        return failures
    if cached_tokens == attempted_tokens:
        return failures

    partial = execution.get("last_cache_reuse_partial")
    if not isinstance(partial, dict):
        failures.append(
            f"{label}: reduced accepted prefix lacks request-correlated "
            "memory-fit telemetry"
        )
        return failures

    partial_request_id = str(partial.get("request_id") or "")
    last_request_id = str(last.get("request_id") or "")
    if (
        not response_id
        or partial_request_id != response_id
        or last_request_id != response_id
    ):
        failures.append(
            f"{label}: memory-fit telemetry is not correlated to the Responses request"
        )
    if partial.get("reason") != "insufficient_memory_for_full_cache_merge":
        failures.append(f"{label}: memory-fit telemetry reason is invalid")
    partial_original = _integer(partial.get("original_cached_tokens"))
    partial_used = _integer(partial.get("used_cached_tokens"))
    partial_dropped = _integer(partial.get("dropped_cached_tokens"))
    expected_dropped = attempted_tokens - cached_tokens
    if partial_original != attempted_tokens:
        failures.append(
            f"{label}: memory-fit original_cached_tokens={partial_original} "
            f"does not equal attempted_cached_tokens={attempted_tokens}"
        )
    if partial_used != cached_tokens:
        failures.append(
            f"{label}: memory-fit used_cached_tokens={partial_used} does not "
            f"equal accepted cached_tokens={cached_tokens}"
        )
    if partial_dropped != expected_dropped:
        failures.append(
            f"{label}: memory-fit dropped_cached_tokens={partial_dropped} does "
            f"not equal attempted-minus-accepted={expected_dropped}"
        )
    elif block_size > 0 and partial_dropped % block_size != 0:
        failures.append(
            f"{label}: memory-fit dropped_cached_tokens={partial_dropped} is "
            f"not block-aligned to {block_size}"
        )
    prompt_tokens = _integer(last.get("prompt_tokens"))
    uncached_tokens = _integer(last.get("uncached_prompt_tokens"))
    if _integer(partial.get("prompt_tokens")) != prompt_tokens:
        failures.append(f"{label}: memory-fit prompt token count does not match")
    if (
        _integer(partial.get("tail_tokens")) != uncached_tokens
        or uncached_tokens != max(prompt_tokens - cached_tokens, 0)
    ):
        failures.append(f"{label}: memory-fit tail token count does not match")

    def _number(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            return None
        return number

    available_bytes = _number(partial.get("available_bytes"))
    cache_bytes = _number(partial.get("cache_bytes"))
    budget_bytes = _number(partial.get("budget_bytes"))
    original_needed_bytes = _number(partial.get("original_needed_bytes"))
    used_cache_bytes = _number(partial.get("used_cache_bytes"))
    used_needed_bytes = _number(partial.get("used_needed_bytes"))
    multiplier = _number(partial.get("multiplier"))
    budget_fraction = _number(partial.get("budget_fraction"))
    if (
        available_bytes is None
        or available_bytes <= 0
        or cache_bytes is None
        or cache_bytes <= 0
        or budget_bytes is None
        or original_needed_bytes is None
        or used_cache_bytes is None
        or used_needed_bytes is None
        or multiplier is None
        or multiplier <= 0
        or budget_fraction is None
        or not 0.10 <= budget_fraction <= 0.95
    ):
        failures.append(f"{label}: exact memory-fit byte telemetry is invalid")
        return failures

    expected_budget_bytes = available_bytes * budget_fraction
    expected_original_needed_bytes = cache_bytes * multiplier
    expected_used_cache_bytes = cache_bytes * (
        float(cached_tokens) / float(attempted_tokens)
    )
    expected_used_needed_bytes = expected_used_cache_bytes * multiplier
    float_tolerance = 1e-9
    for field, observed, expected in (
        ("budget_bytes", budget_bytes, expected_budget_bytes),
        (
            "original_needed_bytes",
            original_needed_bytes,
            expected_original_needed_bytes,
        ),
        ("used_cache_bytes", used_cache_bytes, expected_used_cache_bytes),
        ("used_needed_bytes", used_needed_bytes, expected_used_needed_bytes),
    ):
        if abs(observed - expected) > max(1.0, abs(expected)) * float_tolerance:
            failures.append(
                f"{label}: memory-fit {field} does not match source telemetry"
            )
    if (
        original_needed_bytes <= budget_bytes
        or used_needed_bytes > budget_bytes
    ):
        failures.append(
            f"{label}: memory-fit budget does not prove full reuse exceeded "
            "the budget while accepted reuse fit"
        )

    bytes_per_token = max(cache_bytes / float(attempted_tokens), 1.0)
    cache_budget_bytes = (
        available_bytes * budget_fraction / float(multiplier)
    )
    expected_target = int(cache_budget_bytes / bytes_per_token)
    expected_target = min(expected_target, attempted_tokens - 1)
    expected_target = (expected_target // block_size) * block_size
    if expected_target < block_size:
        expected_target = 0
    cache_contract = str(partial.get("cache_contract") or "")
    exact_max_contracts = {
        "plain_kv",
        "turboquant_kv",
        "mixed_swa_kv",
    }
    architecture_safe_contracts = {
        "hybrid_ssm",
        "deepseek_v4_composite",
        "zaya_cca",
    }
    if cache_contract not in exact_max_contracts | architecture_safe_contracts:
        failures.append(
            f"{label}: memory-fit cache_contract={cache_contract!r} is unsupported"
        )
    elif cache_contract in exact_max_contracts and cached_tokens != expected_target:
        failures.append(
            f"{label}: accepted cached_tokens={cached_tokens} is not the "
            f"maximum memory-fit block-aligned prefix={expected_target}"
        )
    elif (
        cache_contract in architecture_safe_contracts
        and cached_tokens > expected_target
    ):
        failures.append(
            f"{label}: architecture-safe cached_tokens={cached_tokens} exceeds "
            f"the maximum memory-fit block-aligned prefix={expected_target}"
        )
    return failures


def _validate_strict_write_fence_proof(
    proof: Any,
    *,
    label: str,
) -> list[str]:
    """Require a fence-bound physical managed-root reconciliation."""
    if not isinstance(proof, dict):
        return [f"{label}: strict write-fence proof is missing"]
    failures: list[str] = []
    baseline_generation = _integer(
        proof.get("baseline_reconciliation_generation")
    )
    reconciliation_generation = _integer(
        proof.get("global_reconciliation_generation")
    )
    if proof.get("strict_physical_reconcile") is not True:
        failures.append(f"{label}: strict physical reconciliation is not proven")
    if reconciliation_generation <= baseline_generation:
        failures.append(f"{label}: reconciliation generation did not advance")
    if _integer(proof.get("global_accounting_generation")) <= 0:
        failures.append(f"{label}: global accounting generation is missing")
    bytes_after = _integer(proof.get("global_bytes_after"))
    max_size_bytes = _integer(proof.get("global_max_size_bytes"))
    if bytes_after < 0:
        failures.append(f"{label}: managed-root byte count is negative")
    if max_size_bytes > 0 and bytes_after > max_size_bytes:
        failures.append(f"{label}: managed-root physical bytes exceed the limit")
    return failures


def _validate_request_correlated_write_fence(
    proof: Any,
    *,
    label: str,
    require_disk_eviction: bool,
) -> list[str]:
    """Require one exact Responses request to own a settled durable write."""
    if not isinstance(proof, dict):
        return [f"{label}: request-correlated write-fence proof is missing"]
    failures: list[str] = []
    response_id = str(proof.get("response_id") or "")
    request_id = str(proof.get("request_id") or "")
    if (
        not response_id
        or request_id != response_id
        or proof.get("request_correlated") is not True
    ):
        failures.append(
            f"{label}: write fence is not bound to its Responses request_id"
        )
    if (
        proof.get("ok") is not True
        or proof.get("post_eviction_complete") is not True
        or proof.get("fence_sealed") is not True
        or _integer(proof.get("fence_completion_generation")) <= 0
    ):
        failures.append(f"{label}: write fence was not durably settled")
    if _integer(proof.get("disk_writes_delta")) <= 0:
        failures.append(f"{label}: write fence committed no block writes")
    if require_disk_eviction and _integer(
        proof.get("disk_evictions_delta")
    ) <= 0:
        failures.append(f"{label}: disk_evictions delta must be positive")
    failures.extend(_validate_strict_write_fence_proof(proof, label=label))
    return failures


def validate_l2_size_eviction_observation(
    observation: Any,
    *,
    expected_source_head: str,
    expected_source_tree: str,
    health_attestation: dict[str, Any],
    max_filler_requests: int,
) -> list[str]:
    """Fail closed unless exact old/recent chain identities prove L2 LRU."""
    if not isinstance(observation, dict):
        return ["L2 size eviction: observation is missing"]
    failures: list[str] = []
    if observation.get("schema") != L2_SIZE_EVICTION_SCHEMA:
        failures.append("L2 size eviction: schema is invalid")
    if observation.get("scenario") != "store-evict-refault":
        failures.append("L2 size eviction: scenario is invalid")
    if observation.get("source_head") != expected_source_head:
        failures.append("L2 size eviction: source HEAD does not match")
    if observation.get("source_tree") != expected_source_tree:
        failures.append("L2 size eviction: source tree does not match")
    bundle = health_attestation.get("model_bundle_provenance")
    topology = health_attestation.get("cache_topology_provenance")
    expected_bundle = (
        str(bundle.get("fingerprint_sha256") or "")
        if isinstance(bundle, dict)
        else ""
    )
    expected_topology = (
        str(topology.get("fingerprint_sha256") or "")
        if isinstance(topology, dict)
        else ""
    )
    if observation.get("model_bundle_fingerprint_sha256") != expected_bundle:
        failures.append("L2 size eviction: model-bundle fingerprint does not match")
    if observation.get("cache_topology_fingerprint_sha256") != expected_topology:
        failures.append("L2 size eviction: topology fingerprint does not match")

    old_fingerprint = str(
        observation.get("old_prefix_fingerprint_sha256") or ""
    )
    recent_fingerprint = str(
        observation.get("recent_prefix_fingerprint_sha256") or ""
    )
    if not _valid_sha256(old_fingerprint) or not _valid_sha256(
        recent_fingerprint
    ):
        failures.append("L2 size eviction: prefix fingerprints are invalid")
    elif old_fingerprint == recent_fingerprint:
        failures.append("L2 size eviction: old and recent prefixes are identical")

    saved_max = _integer(observation.get("saved_max_bytes"))
    l1_max = _integer(observation.get("l1_max_resident_bytes"))
    peak = _integer(observation.get("peak_observed_bytes"))
    final = _integer(observation.get("final_observed_bytes"))
    fillers = _integer(observation.get("bounded_filler_request_count"))
    post_refault_fillers = _integer(
        observation.get("post_refault_filler_request_count")
    )
    eviction_stage = str(observation.get("evicting_filler_stage") or "")
    recent_before_l1 = (
        (observation.get("recent_before") or {}).get("l1") or {}
    )
    disk_only_l1 = (
        recent_before_l1.get("backend_mode") == "block_disk_only"
        and recent_before_l1.get("disk_only") is True
        and recent_before_l1.get("paged_ram_enabled") is False
    )
    if saved_max <= 0:
        failures.append("L2 size eviction: configured disk bound is not positive")
    if disk_only_l1:
        if l1_max != 0:
            failures.append(
                "L2 size eviction: SSD-only effective L1 byte bound is not zero"
            )
        if observation.get("l1_l2_capacity_margin_ok") is not True:
            failures.append(
                "L2 size eviction: SSD-only topology has resident L1 payload"
            )
    else:
        if l1_max <= 0:
            failures.append(
                "L2 size eviction: configured L1 byte bound is not positive"
            )
        if (
            observation.get("l1_l2_capacity_margin_ok") is not True
            or l1_max <= 0
            or saved_max <= 0
            or l1_max * 2 >= saved_max
        ):
            failures.append(
                "L2 size eviction: live L1 byte bound does not leave the required "
                "2x margin below L2"
            )
    if not 0 <= peak <= saved_max:
        failures.append("L2 size eviction: peak bytes exceed configured bound")
    if not 0 <= final <= peak:
        failures.append("L2 size eviction: final bytes are invalid")
    if eviction_stage in {"pre-refault", "post-refault"}:
        if not 1 <= fillers <= min(256, max_filler_requests):
            failures.append("L2 size eviction: filler request count is unbounded")
        if eviction_stage == "post-refault" and post_refault_fillers <= 0:
            failures.append(
                "L2 size eviction: post-refault geometry has no eviction filler"
            )
        if eviction_stage == "pre-refault" and post_refault_fillers != 0:
            failures.append(
                "L2 size eviction: pre-refault geometry used a post-refault filler"
            )
    elif eviction_stage == "recent-store":
        if fillers != 0 or post_refault_fillers != 0:
            failures.append(
                "L2 size eviction: recent-store geometry unexpectedly used fillers"
            )
    else:
        failures.append("L2 size eviction: evicting stage is invalid")
    if observation.get("old_prefix_evicted") is not True:
        failures.append("L2 size eviction: old prefix was not evicted")
    if observation.get("recent_prefix_present") is not True:
        failures.append("L2 size eviction: recent prefix did not survive")
    if observation.get("recent_prefix_last_access_after_old") is not True:
        failures.append("L2 size eviction: recent LRU touch was not proven")
    binding_specs = (
        ("old_after_store", old_fingerprint),
        ("old_before", old_fingerprint),
        ("recent_before", recent_fingerprint),
        ("recent_pre_refault", recent_fingerprint),
        ("recent_post_refault", recent_fingerprint),
        ("old_after_durable_filler", old_fingerprint),
        ("recent_after_durable_filler", recent_fingerprint),
        ("old_final", old_fingerprint),
        ("recent_final", recent_fingerprint),
    )
    for label, fingerprint in binding_specs:
        failures.extend(
            _validate_prefix_binding(
                observation.get(label),
                expected_fingerprint=fingerprint,
                expected_bundle_fingerprint=expected_bundle,
                expected_topology_fingerprint=expected_topology,
                label=f"L2 size eviction {label}",
            )
        )

    old_after_store = observation.get("old_after_store")
    old_before = observation.get("old_before")
    recent_before = observation.get("recent_before")
    recent_pre = observation.get("recent_pre_refault")
    recent_post = observation.get("recent_post_refault")
    old_after_filler = observation.get("old_after_durable_filler")
    recent_after_filler = observation.get("recent_after_durable_filler")
    old_final = observation.get("old_final")
    recent_final = observation.get("recent_final")
    old_store_fence = observation.get("old_store_fence")
    recent_store_fence = observation.get("recent_store_fence")
    evicting_filler = observation.get("evicting_filler_fence")
    failures.extend(
        _validate_request_correlated_write_fence(
            old_store_fence,
            label="L2 size eviction old store",
            require_disk_eviction=False,
        )
    )
    failures.extend(
        _validate_request_correlated_write_fence(
            recent_store_fence,
            label="L2 size eviction recent store",
            require_disk_eviction=eviction_stage == "recent-store",
        )
    )
    failures.extend(
        _validate_request_correlated_write_fence(
            evicting_filler,
            label="L2 size eviction evicting write",
            require_disk_eviction=True,
        )
    )
    if isinstance(evicting_filler, dict):
        tag = str(evicting_filler.get("tag") or "")
        if eviction_stage == "recent-store" and tag != "l2_recent_store":
            failures.append(
                "L2 size eviction: recent-store geometry is not bound to the "
                "recent-store fence"
            )
        if (
            eviction_stage == "recent-store"
            and isinstance(recent_store_fence, dict)
            and evicting_filler.get("attestation_sha256")
            != recent_store_fence.get("attestation_sha256")
        ):
            failures.append(
                "L2 size eviction: evicting write is not the exact recent-store fence"
            )
        if (
            eviction_stage in {"pre-refault", "post-refault"}
            and not tag.startswith("l2_filler_")
        ):
            failures.append(
                f"L2 size eviction: {eviction_stage} geometry is not bound to a filler"
            )
    write_fences = observation.get("write_fences")
    if not isinstance(write_fences, list) or not write_fences:
        failures.append("L2 size eviction: strict write-fence rows are missing")
    else:
        fence_attestations = {
            str(write_fence.get("attestation_sha256") or "")
            for write_fence in write_fences
            if isinstance(write_fence, dict)
        }
        for index, write_fence in enumerate(write_fences):
            failures.extend(
                _validate_strict_write_fence_proof(
                    write_fence,
                    label=f"L2 size eviction write fence {index}",
                )
            )
        for label, proof in (
            ("old store", old_store_fence),
            ("recent store", recent_store_fence),
            ("evicting write", evicting_filler),
        ):
            attestation = (
                str(proof.get("attestation_sha256") or "")
                if isinstance(proof, dict)
                else ""
            )
            if not _valid_sha256(attestation) or attestation not in fence_attestations:
                failures.append(
                    f"L2 size eviction: {label} fence is not in the attested "
                    "write-fence sequence"
                )
            if (
                isinstance(proof, dict)
                and _integer(proof.get("global_max_size_bytes")) != saved_max
            ):
                failures.append(
                    f"L2 size eviction: {label} fence has the wrong byte limit"
                )
    if all(
        isinstance(item, dict)
        for item in (
            old_before,
            old_after_store,
            recent_before,
            recent_pre,
            recent_post,
            old_after_filler,
            recent_after_filler,
            old_final,
            recent_final,
        )
    ):
        failures.extend(
            _validate_complete_l2_binding(
                old_after_store,
                label="L2 size eviction old after its durable store",
            )
        )
        failures.extend(
            _validate_complete_l2_binding(
                recent_before,
                label="L2 size eviction recent after its durable store",
            )
        )
        if eviction_stage in {"pre-refault", "post-refault"}:
            failures.extend(
                _validate_complete_l2_binding(
                    old_before,
                    label=f"L2 size eviction old before {eviction_stage} filler",
                )
            )
        elif (old_before.get("l2") or {}).get("terminal_readable") is not False:
            failures.append(
                "L2 size eviction: recent-store geometry did not evict the old "
                "terminal"
            )
        old_access_time = _integer(
            (old_after_store.get("l2") or {}).get("terminal_last_accessed_ns")
        )
        recent_access_time = _integer(
            (recent_before.get("l2") or {}).get("terminal_last_accessed_ns")
        )
        if old_access_time <= 0 or recent_access_time <= old_access_time:
            failures.append(
                "L2 size eviction: old prefix was not strictly older than "
                "recent before filler"
            )
        recent_before_l1 = recent_before.get("l1") or {}
        disk_only = (
            recent_before_l1.get("backend_mode") == "block_disk_only"
            and recent_before_l1.get("disk_only") is True
            and recent_before_l1.get("paged_ram_enabled") is False
        )
        if disk_only:
            for label, binding in (
                ("recent_before", recent_before),
                ("recent_pre_refault", recent_pre),
                ("recent_post_refault", recent_post),
                ("recent_after_durable_filler", recent_after_filler),
                ("recent_final", recent_final),
            ):
                l1 = binding.get("l1") or {}
                if (
                    l1.get("backend_mode") != "block_disk_only"
                    or l1.get("disk_only") is not True
                    or l1.get("paged_ram_enabled") is not False
                    or _integer(l1.get("resident_payload_blocks_present")) != 0
                    or _integer(l1.get("resident_payload_bytes")) != 0
                ):
                    failures.append(
                        f"L2 size eviction: {label} does not truthfully "
                        "represent SSD-only state"
                    )
        elif eviction_stage in {"pre-refault", "post-refault"} and (
            recent_before_l1.get("terminal_resident_payload_present")
            is not True
        ):
            failures.append(
                "L2 size eviction: recent prefix was never resident in paged RAM"
            )
        elif eviction_stage == "recent-store" and (
            recent_before_l1.get("terminal_resident_payload_present")
            is not False
        ):
            failures.append(
                "L2 size eviction: recent-store geometry did not reach SSD-only state"
            )
        if (
            (recent_pre.get("l1") or {}).get(
                "terminal_resident_payload_present"
            )
            is not False
        ):
            failures.append(
                "L2 size eviction: recent prefix was not evicted from L1 before refault"
            )
        if (recent_pre.get("l2") or {}).get("terminal_readable") is not True:
            failures.append(
                "L2 size eviction: recent prefix was absent from L2 before refault"
            )
        for label, binding in (
            ("recent_pre_refault", recent_pre),
            ("recent_post_refault", recent_post),
            ("recent_after_durable_filler", recent_after_filler),
            ("recent_final", recent_final),
        ):
            failures.extend(
                _validate_complete_l2_binding(
                    binding,
                    label=f"L2 size eviction {label}",
                )
            )
        pre_access = _integer(
            (recent_pre.get("l2") or {}).get("terminal_access_count")
        )
        post_access = _integer(
            (recent_post.get("l2") or {}).get("terminal_access_count")
        )
        pre_time = _integer(
            (recent_pre.get("l2") or {}).get("terminal_last_accessed_ns")
        )
        post_time = _integer(
            (recent_post.get("l2") or {}).get("terminal_last_accessed_ns")
        )
        if post_access <= pre_access or post_time <= pre_time:
            failures.append(
                "L2 size eviction: real refault did not touch recent LRU metadata"
            )
        if (old_after_filler.get("l2") or {}).get(
            "terminal_readable"
        ) is not False:
            failures.append(
                "L2 size eviction: old prefix did not disappear after the "
                "durable evicting filler fence"
            )
        if (recent_after_filler.get("l2") or {}).get(
            "terminal_readable"
        ) is not True:
            failures.append(
                "L2 size eviction: recent prefix did not survive the durable "
                "evicting filler fence"
            )
        if (old_final.get("l2") or {}).get("terminal_readable") is not False:
            failures.append("L2 size eviction: old terminal block still exists")
        if (recent_final.get("l2") or {}).get("terminal_readable") is not True:
            failures.append("L2 size eviction: recent terminal block did not survive")
        for label, binding in (
            ("old_after_durable_filler", old_after_filler),
            ("recent_after_durable_filler", recent_after_filler),
            ("old_final", old_final),
            ("recent_final", recent_final),
        ):
            l2 = binding.get("l2") or {}
            observed_max = _integer(l2.get("store_max_size_bytes"))
            observed_size = _integer(l2.get("store_total_size_bytes"))
            if observed_max != saved_max or not 0 <= observed_size <= saved_max:
                failures.append(
                    f"L2 size eviction: {label} does not comply with the "
                    "configured byte limit"
                )
    failures.extend(
        _validate_disk_refault_execution(
            observation.get("recent_refault_execution"),
            label="L2 size eviction recent refault",
            contract_profile=_cache_contract_profile_from_health(
                health_attestation
            ),
        )
    )
    failures.extend(
        _validate_execution_prefix_bounds(
            observation.get("recent_refault_execution"),
            recent_pre,
            label="L2 size eviction recent refault",
        )
    )
    return failures


def validate_l2_restart_restore_observation(
    observation: Any,
    *,
    store_observation: Any,
    expected_source_head: str,
    expected_source_tree: str,
    health_attestation: dict[str, Any],
) -> list[str]:
    """Bind restart restore to the exact recent chain that survived eviction."""
    if not isinstance(observation, dict):
        return ["L2 restart restore: observation is missing"]
    failures: list[str] = []
    if observation.get("schema") != L2_RESTART_RESTORE_SCHEMA:
        failures.append("L2 restart restore: schema is invalid")
    if observation.get("scenario") != "restart-restore":
        failures.append("L2 restart restore: scenario is invalid")
    if observation.get("source_head") != expected_source_head:
        failures.append("L2 restart restore: source HEAD does not match")
    if observation.get("source_tree") != expected_source_tree:
        failures.append("L2 restart restore: source tree does not match")
    if not isinstance(store_observation, dict):
        return failures + ["L2 restart restore: store observation is missing"]
    recent_fingerprint = str(
        store_observation.get("recent_prefix_fingerprint_sha256") or ""
    )
    restart_fingerprint = str(
        observation.get("restart_probe_prefix_fingerprint_sha256") or ""
    )
    if not _valid_sha256(recent_fingerprint) or (
        restart_fingerprint != recent_fingerprint
    ):
        failures.append(
            "L2 restart restore: restart prefix is not the stored recent prefix"
        )
    bundle = health_attestation.get("model_bundle_provenance")
    topology = health_attestation.get("cache_topology_provenance")
    expected_bundle = (
        str(bundle.get("fingerprint_sha256") or "")
        if isinstance(bundle, dict)
        else ""
    )
    expected_topology = (
        str(topology.get("fingerprint_sha256") or "")
        if isinstance(topology, dict)
        else ""
    )
    for label in ("restart_pre", "restart_post"):
        failures.extend(
            _validate_prefix_binding(
                observation.get(label),
                expected_fingerprint=recent_fingerprint,
                expected_bundle_fingerprint=expected_bundle,
                expected_topology_fingerprint=expected_topology,
                label=f"L2 restart restore {label}",
            )
        )
    pre = observation.get("restart_pre")
    post = observation.get("restart_post")
    if isinstance(pre, dict) and isinstance(post, dict):
        if (
            (pre.get("l1") or {}).get("terminal_resident_payload_present")
            is not False
        ):
            failures.append(
                "L2 restart restore: prefix was already L1-resident before probe"
            )
        if (pre.get("l2") or {}).get("terminal_readable") is not True:
            failures.append(
                "L2 restart restore: exact prefix was absent from L2 before probe"
            )
        pre_access = _integer(
            (pre.get("l2") or {}).get("terminal_access_count")
        )
        post_access = _integer(
            (post.get("l2") or {}).get("terminal_access_count")
        )
        pre_time = _integer(
            (pre.get("l2") or {}).get("terminal_last_accessed_ns")
        )
        post_time = _integer(
            (post.get("l2") or {}).get("terminal_last_accessed_ns")
        )
        if post_access <= pre_access or post_time <= pre_time:
            failures.append(
                "L2 restart restore: real probe did not touch exact L2 prefix"
            )
    failures.extend(
        _validate_disk_refault_execution(
            observation.get("restart_execution"),
            label="L2 restart restore",
            contract_profile=_cache_contract_profile_from_health(
                health_attestation
            ),
        )
    )
    failures.extend(
        _validate_execution_prefix_bounds(
            observation.get("restart_execution"),
            pre,
            label="L2 restart restore",
        )
    )
    execution = observation.get("restart_execution")
    last = (
        execution.get("last_cache_execution")
        if isinstance(execution, dict)
        else {}
    )
    if not isinstance(last, dict):
        last = {}
    if _integer(observation.get("restart_restored_tokens")) != _integer(
        last.get("cached_tokens")
    ):
        failures.append(
            "L2 restart restore: restored-token summary is not source-bound"
        )
    if _integer(observation.get("restart_disk_blocks")) != _integer(
        last.get("disk_blocks")
    ):
        failures.append(
            "L2 restart restore: disk-block summary is not source-bound"
        )
    if _integer(observation.get("restart_uncached_tokens")) != (
        _integer(last.get("uncached_prompt_tokens"))
    ):
        failures.append(
            "L2 restart restore: uncached-token summary is not source-bound"
        )
    if observation.get("restart_restore_source") != "block-disk":
        failures.append("L2 restart restore: source is not block-disk")
    return failures


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


def _standard_cache_requests(
    phase: str,
    cache_scenario: str,
    prompts: dict[str, str],
) -> list[tuple[str, str, str]]:
    """Select ordinary A/B/C rows without contradicting L2 scenarios."""

    if cache_scenario == "restart-restore":
        # Its linked eviction phase proves that the old A/B/C chain is absent;
        # the dedicated recent-chain scenario below owns restart proof.
        return []
    if phase == "store":
        return [
            ("cold_a", prompts["A"], "A"),
            ("warm_a", prompts["A"], "A"),
            ("partial_b", prompts["B"], "B"),
        ]
    return [
        ("restart_partial_c", prompts["C"], "C"),
        ("restart_a", prompts["A"], "A"),
    ]


def _payload(
    model: str,
    prompt: str,
    *,
    request_controls: dict[str, Any] | None = None,
) -> dict[str, Any]:
    controls = (
        _cache_scenario_request_controls()
        if request_controls is None
        else request_controls
    )
    payload = {
        "model": model,
        "input": prompt,
        "stream": True,
        "store": False,
        # The release probe validates an exact marker of the form
        # CACHE-HIERARCHY-<32-byte nonce>-<selector>.  A 32-token cap can
        # truncate the marker itself on normal byte-fragmenting tokenizers and
        # turn a healthy cache hit into response.incomplete. Keep the cap small
        # enough to preserve the transport-only nature of the probe, but large
        # enough to allow the full marker or schema-valid tool call plus EOS.
        # A retained MiniMax-M2.7 warm-cache falsifier reached its correct
        # closing XML just beyond the 256-token cap.
        "max_output_tokens": 512,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 20,
        "enable_thinking": controls.get("enable_thinking", False),
    }
    for field in ("instructions", "tools", "tool_choice", "chat_template_kwargs"):
        if field in controls:
            payload[field] = controls[field]
    return payload


def _l2_identity_prompts(
    nonce: str,
    records: int,
) -> dict[str, str]:
    prompts: dict[str, str] = {}
    for identity in ("old", "recent"):
        prefix = "\n".join(
            [
                f"CACHE-IDENTITY {nonce}-l2-{identity}",
                _common_prefix(f"{nonce}-l2-{identity}", records),
            ]
        )
        stem = (
            f"{prefix}\nThe following selector is outside the shared cache "
            f"prefix. Reply exactly CACHE-HIERARCHY-{nonce}-L2-"
            f"{identity.upper()}-"
        )
        prompts[f"{identity}_store"] = f"{stem}STORE"
        prompts[f"{identity}_probe"] = f"{stem}PROBE"
        prompts[f"{identity}_restart"] = f"{stem}RESTART"
    return prompts


def _l2_filler_prompt(
    nonce: str,
    records: int,
    index: int,
) -> tuple[str, str]:
    marker = f"CACHE-HIERARCHY-{nonce}-FILLER-{index:03d}"
    identity = f"{nonce}-l2-filler-{index:03d}"
    prefix = "\n".join(
        [
            f"CACHE-IDENTITY {identity}",
            _common_prefix(identity, records),
        ]
    )
    return f"{prefix}\nReply exactly {marker}", marker


def _run_response_observation(
    *,
    base_url: str,
    model: str,
    tag: str,
    prompt: str,
    expected_marker: str,
    artifact_dir: Path,
    timeout: int,
    request_controls: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = _json_get(f"{base_url}/health", timeout)
    before_path = artifact_dir / f"{tag}.health-before.json"
    before_path.write_text(
        json.dumps(before, indent=2, sort_keys=True) + "\n"
    )
    payload = _payload(model, prompt, request_controls=request_controls)
    request_path = artifact_dir / f"{tag}.request.json"
    raw_path = artifact_dir / f"{tag}.sse"
    request_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    code, raw, elapsed = _post_sse(
        f"{base_url}/v1/responses",
        payload,
        timeout,
    )
    raw_path.write_text(raw)
    summary = _summarize(raw, elapsed, code)
    after = _json_get(f"{base_url}/health", timeout)
    after_path = artifact_dir / f"{tag}.health.json"
    after_path.write_text(
        json.dumps(after, indent=2, sort_keys=True) + "\n"
    )
    before_counters = _health_cache_counters(before)
    after_counters = _health_cache_counters(after)
    cache_contract_evidence = _health_cache_contract_evidence(after)
    summary.update(
        {
            "tag": tag,
            "expected_marker": expected_marker,
            "request_path": str(request_path),
            "raw_path": str(raw_path),
            "health_before_path": str(before_path),
            "health_path": str(after_path),
            "health_counters_before": before_counters,
            "health_counters_after": after_counters,
            "health_counter_deltas": _counter_deltas(
                before_counters,
                after_counters,
            ),
            "last_cache_execution": (after.get("scheduler") or {}).get(
                "last_cache_execution"
            ),
            "last_cache_reuse_partial": (after.get("scheduler") or {}).get(
                "last_cache_reuse_partial"
            ),
            "scheduler_cache": (
                (after.get("cache") or {}).get("scheduler_cache") or {}
            ),
            "block_disk_cache": (
                (after.get("cache") or {}).get("block_disk_cache") or {}
            ),
            **cache_contract_evidence,
        }
    )
    summary["marker_ok"] = _exact_cache_marker_observed(
        summary,
        expected_marker,
    )
    summary["terminal_ok"] = summary["terminal_events"] == [
        "response.completed"
    ]
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
    return summary, after


def _path_free_execution(row: dict[str, Any]) -> dict[str, Any]:
    last = row.get("last_cache_execution")
    if not isinstance(last, dict):
        last = {}
    partial = row.get("last_cache_reuse_partial")
    if not isinstance(partial, dict):
        partial = {}
    cache_detail = row.get("cache_detail")
    if not isinstance(cache_detail, dict):
        cache_detail = {}
    native_cache = row.get("native_cache")
    if not isinstance(native_cache, dict):
        native_cache = {}
    ssm_companion = row.get("ssm_companion")
    if not isinstance(ssm_companion, dict):
        ssm_companion = {}
    last_prefix_lookup = ssm_companion.get("last_prefix_lookup")
    if not isinstance(last_prefix_lookup, dict):
        last_prefix_lookup = {}
    ssm_disk = ssm_companion.get("disk")
    if not isinstance(ssm_disk, dict):
        ssm_disk = {}
    deltas = row.get("health_counter_deltas")
    if not isinstance(deltas, dict):
        deltas = {}
    retained_delta_keys = (
        "block_disk_cache.disk_hits",
        "block_disk_cache.tq_native_hits",
        "block_disk_cache.tq_native_writes",
        "scheduler.hybrid_kv_without_ssm_hits",
        "scheduler.hybrid_kv_without_ssm_tokens",
        "ssm_companion.disk.hits",
        "ssm_companion.disk.misses",
        "ssm_companion.disk.stores",
    )
    return {
        "tag": row.get("tag"),
        "response_id": row.get("response_id"),
        "response_id_consistent": row.get("response_id_consistent"),
        "status_code": row.get("status_code"),
        "terminal_ok": row.get("terminal_ok"),
        "marker_ok": row.get("marker_ok"),
        "cached_tokens": row.get("cached_tokens"),
        "cache_detail": {
            key: cache_detail.get(key)
            for key in (
                "source",
                "origin",
                "cache_type",
                "cached_tokens",
                "disk_blocks",
            )
            if key in cache_detail
        },
        "native_cache": _path_free_native_cache(native_cache),
        "ssm_companion": {
            "last_prefix_lookup": _path_free_prefix_lookup(last_prefix_lookup),
            "disk": {
                key: ssm_disk.get(key)
                for key in ("hits", "misses", "stores")
                if key in ssm_disk
            },
        },
        "health_counter_deltas": {
            key: deltas.get(key)
            for key in retained_delta_keys
            if key in deltas
        },
        "last_cache_execution": {
            key: last.get(key)
            for key in (
                "request_id",
                "cache_reuse_applied",
                "cache_outcome",
                "cache_detail",
                "selection",
                "prompt_tokens",
                "attempted_cached_tokens",
                "cached_tokens",
                "checkpoint_tokens",
                "matched_tokens",
                "replayed_tokens",
                "uncached_prompt_tokens",
                "prefill_tokens",
                "generation_prompt_suffix_tokens",
                "reconstructed",
                "reconstruction_ok",
                "dequantized",
                "dequantization_ok",
                "disk_hit",
                "disk_blocks",
                "tq_native_blocks",
            )
            if key in last
        },
        "last_cache_reuse_partial": {
            key: partial.get(key)
            for key in (
                "request_id",
                "reason",
                "cache_contract",
                "cache_format",
                "available_bytes",
                "cache_bytes",
                "budget_bytes",
                "original_needed_bytes",
                "used_cache_bytes",
                "used_needed_bytes",
                "original_needed_mb",
                "budget_mb",
                "available_mb",
                "original_cache_mb",
                "used_cache_mb",
                "used_needed_mb",
                "multiplier",
                "budget_fraction",
                "kv_cache_bits",
                "original_cached_tokens",
                "used_cached_tokens",
                "dropped_cached_tokens",
                "tail_tokens",
                "prompt_tokens",
                "cache_type",
            )
            if key in partial
        },
    }


def _wait_for_prefix_access_touch(
    *,
    base_url: str,
    model: str,
    prompts: dict[str, str],
    pairs: dict[str, tuple[str, str] | list[str]],
    pair_name: str,
    previous_binding: dict[str, Any],
    timeout: int,
    health_attestation: dict[str, Any],
    timeout_s: float,
    poll_interval_s: float,
    request_controls: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    deadline = time.monotonic() + max(0.0, timeout_s)
    previous_l2 = previous_binding.get("l2")
    if not isinstance(previous_l2, dict):
        previous_l2 = {}
    previous_count = _integer(previous_l2.get("terminal_access_count"))
    previous_time = _integer(previous_l2.get("terminal_last_accessed_ns"))
    last_contract: dict[str, Any] = {}
    last_failures: list[str] = []
    while True:
        contract, failures = _fetch_prefix_attestation(
            base_url=base_url,
            model=model,
            prompts=prompts,
            pairs=pairs,
            timeout=timeout,
            health_attestation=health_attestation,
            request_controls=request_controls,
        )
        last_contract = contract
        last_failures = failures
        binding = _prefix_binding(contract, pair_name)
        l2 = binding.get("l2")
        if not isinstance(l2, dict):
            l2 = {}
        if (
            not failures
            and _integer(l2.get("terminal_access_count")) > previous_count
            and _integer(l2.get("terminal_last_accessed_ns")) > previous_time
        ):
            return contract, []
        if time.monotonic() >= deadline:
            if not last_failures:
                last_failures = [
                    "prefix attestation: real cache hit did not update L2 "
                    "access metadata before timeout"
                ]
            return last_contract, last_failures
        time.sleep(poll_interval_s)


def _write_path_free_attestation(
    artifact_dir: Path,
    tag: str,
    contract: dict[str, Any],
) -> None:
    (artifact_dir / f"{tag}.prefix-attestation.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )


def _scenario_request_durability(
    *,
    row: dict[str, Any],
    base_url: str,
    timeout: int,
    durability_timeout: float,
    durability_poll_interval: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not (
        row.get("status_code") == 200
        and row.get("marker_ok") is True
        and row.get("terminal_ok") is True
        and row.get("response_id_consistent") is True
        and str(row.get("response_id") or "")
    ):
        return (
            {
                "ok": False,
                "skipped": True,
                "reason": "request did not satisfy the Responses contract",
            },
            {},
        )
    return _wait_for_store_durability(
        base_url=base_url,
        request_timeout=timeout,
        request_id=str(row["response_id"]),
        baseline_counters=row.get("health_counters_before") or {},
        timeout_s=durability_timeout,
        poll_interval_s=durability_poll_interval,
    )


def _path_free_durability_proof(
    row: dict[str, Any],
    durability: dict[str, Any],
) -> dict[str, Any]:
    """Keep the request/fence/counter facts needed for exact eviction proof."""
    fence = durability.get("matching_fence")
    if not isinstance(fence, dict):
        fence = {}
    deltas = durability.get("counter_deltas")
    if not isinstance(deltas, dict):
        deltas = {}
    response_id = str(row.get("response_id") or "")
    request_id = str(durability.get("request_id") or "")
    return {
        "tag": row.get("tag"),
        "response_id": response_id,
        "request_id": request_id,
        "request_correlated": bool(
            response_id
            and request_id == response_id
            and durability.get("exact_request_identity_proven") is True
        ),
        "ok": durability.get("ok") is True,
        "post_eviction_complete": fence.get("post_eviction_complete") is True,
        "fence_sealed": fence.get("sealed") is True,
        "fence_completion_generation": _integer(
            fence.get("completion_generation")
        ),
        "strict_physical_reconcile": bool(
            durability.get("ok") is True
            and _integer(fence.get("global_reconciliation_generation"))
            > _integer(
                (durability.get("baseline_counters") or {}).get(
                    "block_disk_cache.global_reconciliation_generation"
                )
            )
        ),
        "baseline_reconciliation_generation": _integer(
            (durability.get("baseline_counters") or {}).get(
                "block_disk_cache.global_reconciliation_generation"
            )
        ),
        "global_reconciliation_generation": _integer(
            fence.get("global_reconciliation_generation")
        ),
        "global_accounting_generation": _integer(
            fence.get("global_accounting_generation")
        ),
        "global_bytes_after": _integer(fence.get("global_bytes_after")),
        "global_max_size_bytes": _integer(fence.get("global_max_size_bytes")),
        "disk_writes_delta": _integer(
            deltas.get("block_disk_cache.disk_writes")
        ),
        "disk_evictions_delta": _integer(
            deltas.get("block_disk_cache.disk_evictions")
        ),
        "attestation_sha256": _canonical_sha256(durability),
    }


def _run_store_evict_refault_scenario(
    *,
    base_url: str,
    model: str,
    nonce: str,
    records: int,
    artifact_dir: Path,
    timeout: int,
    durability_timeout: float,
    durability_poll_interval: float,
    max_filler_requests: int,
    health_attestation: dict[str, Any],
    observed_source: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], dict[str, Any]]:
    """Prove L1 eviction, exact L2 refault, LRU touch, and bounded L2 eviction."""
    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    durability_rows: list[dict[str, Any]] = []
    prompts = _l2_identity_prompts(nonce, records)
    pairs: dict[str, tuple[str, str]] = {
        "old": ("old_store", "old_probe"),
        "recent": ("recent_store", "recent_probe"),
    }
    request_controls = _l2_scenario_request_controls()
    health_after: dict[str, Any] = {}
    old_after_store: dict[str, Any] = {}
    old_store_fence: dict[str, Any] = {}
    recent_store_fence: dict[str, Any] = {}

    for identity in ("old", "recent"):
        marker = (
            f"CACHE-HIERARCHY-{nonce}-L2-{identity.upper()}-STORE"
        )
        row, health_after = _run_response_observation(
            base_url=base_url,
            model=model,
            tag=f"l2_{identity}_store",
            prompt=prompts[f"{identity}_store"],
            expected_marker=marker,
            artifact_dir=artifact_dir,
            timeout=timeout,
            request_controls=request_controls,
        )
        rows.append(row)
        durability, durable_health = _scenario_request_durability(
            row=row,
            base_url=base_url,
            timeout=timeout,
            durability_timeout=durability_timeout,
            durability_poll_interval=durability_poll_interval,
        )
        durability_proof = _path_free_durability_proof(row, durability)
        durability_rows.append(durability_proof)
        if identity == "old":
            old_store_fence = durability_proof
        else:
            recent_store_fence = durability_proof
        if durable_health:
            health_after = durable_health
        if durability.get("ok") is not True:
            failures.append(
                f"{row['tag']}: exact write fence did not become durable"
            )
        if identity == "old":
            old_contract, old_attestation_failures = _fetch_prefix_attestation(
                base_url=base_url,
                model=model,
                prompts=prompts,
                pairs={"old": pairs["old"]},
                timeout=timeout,
                health_attestation=health_attestation,
                request_controls=request_controls,
            )
            failures.extend(old_attestation_failures)
            _write_path_free_attestation(
                artifact_dir,
                "l2_old_after_store",
                old_contract,
            )
            old_after_store = _prefix_binding(old_contract, "old")

    before_contract, before_failures = _fetch_prefix_attestation(
        base_url=base_url,
        model=model,
        prompts=prompts,
        pairs=pairs,
        timeout=timeout,
        health_attestation=health_attestation,
        request_controls=request_controls,
    )
    failures.extend(before_failures)
    _write_path_free_attestation(
        artifact_dir,
        "l2_before_eviction",
        before_contract,
    )
    old_before = _prefix_binding(before_contract, "old")
    recent_before = _prefix_binding(before_contract, "recent")
    old_fingerprint = str(
        old_after_store.get("block_chain_fingerprint_sha256") or ""
    )
    recent_fingerprint = str(
        recent_before.get("block_chain_fingerprint_sha256") or ""
    )
    cache_health = health_after.get("cache")
    if not isinstance(cache_health, dict):
        cache_health = {}
    cache_totals = cache_health.get("totals")
    if not isinstance(cache_totals, dict):
        cache_totals = {}
    l1_max_resident_bytes = _integer(
        cache_totals.get("l1_max_resident_bytes")
    )
    configured_l2_max_bytes = _integer(
        (recent_before.get("l2") or {}).get("store_max_size_bytes")
    )
    recent_before_l1 = recent_before.get("l1") or {}
    disk_only_l1 = (
        recent_before_l1.get("backend_mode") == "block_disk_only"
        and recent_before_l1.get("disk_only") is True
        and recent_before_l1.get("paged_ram_enabled") is False
    )
    if disk_only_l1:
        l1_l2_capacity_margin_ok = (
            configured_l2_max_bytes > 0
            and _integer(
                recent_before_l1.get("resident_payload_blocks_present")
            )
            == 0
            and _integer(recent_before_l1.get("resident_payload_bytes")) == 0
        )
    else:
        l1_l2_capacity_margin_ok = (
            l1_max_resident_bytes > 0
            and configured_l2_max_bytes > 0
            and l1_max_resident_bytes * 2 < configured_l2_max_bytes
        )
    peak_bytes = max(
        _integer(
            (old_after_store.get("l2") or {}).get("store_total_size_bytes")
        ),
        _integer((old_before.get("l2") or {}).get("store_total_size_bytes")),
        _integer(
            (recent_before.get("l2") or {}).get("store_total_size_bytes")
        ),
    )
    filler_count = 0
    pre_refault_contract = before_contract
    recent_pre_refault = recent_before
    evicting_filler_fence: dict[str, Any] = {}
    evicting_filler_stage: str | None = None
    old_after_durable_filler: dict[str, Any] = {}
    recent_after_durable_filler: dict[str, Any] = {}

    def _pre_refault_failure_observation() -> dict[str, Any]:
        return {
            "schema": L2_SIZE_EVICTION_SCHEMA,
            "scenario": "store-evict-refault",
            "source_head": observed_source.get("head"),
            "source_tree": observed_source.get("tree"),
            "model_bundle_fingerprint_sha256": (
                (health_attestation.get("model_bundle_provenance") or {}).get(
                    "fingerprint_sha256"
                )
            ),
            "cache_topology_fingerprint_sha256": (
                (health_attestation.get("cache_topology_provenance") or {}).get(
                    "fingerprint_sha256"
                )
            ),
            "saved_max_bytes": configured_l2_max_bytes,
            "l1_max_resident_bytes": l1_max_resident_bytes,
            "l1_l2_capacity_margin_ok": l1_l2_capacity_margin_ok,
            "peak_observed_bytes": peak_bytes,
            "bounded_filler_request_count": filler_count,
            "old_prefix_fingerprint_sha256": old_fingerprint,
            "recent_prefix_fingerprint_sha256": recent_fingerprint,
            "old_after_store": old_after_store,
            "old_before": old_before,
            "recent_before": recent_before,
            "recent_pre_refault": recent_pre_refault,
            "write_fences": durability_rows,
            "old_store_fence": old_store_fence,
            "recent_store_fence": recent_store_fence,
            "evicting_filler_stage": evicting_filler_stage,
            "evicting_filler_fence": evicting_filler_fence,
            "pre_refault_ready": False,
        }

    if not l1_l2_capacity_margin_ok:
        failures.append(
            "store-evict-refault: live L1 byte bound must be less than half "
            "the configured L2 byte bound before eviction fillers run"
        )
        return (
            _pre_refault_failure_observation(),
            rows,
            failures,
            health_after,
        )

    recent_store_eviction_ready = not _validate_request_correlated_write_fence(
        recent_store_fence,
        label="store-evict-refault recent store",
        require_disk_eviction=True,
    )
    while filler_count < max_filler_requests:
        old_pre_refault = _prefix_binding(pre_refault_contract, "old")
        old_pre_refault_l2 = old_pre_refault.get("l2")
        recent_l1 = recent_pre_refault.get("l1")
        if not isinstance(old_pre_refault_l2, dict):
            old_pre_refault_l2 = {}
        if not isinstance(recent_l1, dict):
            recent_l1 = {}
        recent_at_refault_boundary = bool(
            recent_l1.get("terminal_resident_payload_present") is False
            and _l2_binding_is_complete(recent_pre_refault)
        )
        if not _l2_binding_is_complete(recent_pre_refault):
            failures.append(
                "store-evict-refault: recent prefix was not completely readable "
                "from L2 before refault"
            )
            break
        if recent_at_refault_boundary:
            if old_pre_refault_l2.get("terminal_readable") is True:
                break
            if (
                old_pre_refault_l2.get("terminal_readable") is False
                and _l2_binding_is_complete(old_after_store)
                and recent_store_eviction_ready
            ):
                evicting_filler_fence = recent_store_fence
                evicting_filler_stage = "recent-store"
                old_after_durable_filler = old_pre_refault
                recent_after_durable_filler = recent_pre_refault
                break
            failures.append(
                "store-evict-refault: older prefix left L2 without an exact "
                "durable recent-store eviction proof"
            )
            break
        if old_pre_refault_l2.get("terminal_readable") is False:
            failures.append(
                "store-evict-refault: older prefix left L2 before the recent "
                "prefix reached the refault boundary"
            )
            break
        filler_prompt, marker = _l2_filler_prompt(
            nonce,
            records,
            filler_count,
        )
        row, health_after = _run_response_observation(
            base_url=base_url,
            model=model,
            tag=f"l2_filler_{filler_count:03d}",
            prompt=filler_prompt,
            expected_marker=marker,
            artifact_dir=artifact_dir,
            timeout=timeout,
            request_controls=request_controls,
        )
        rows.append(row)
        durability, durable_health = _scenario_request_durability(
            row=row,
            base_url=base_url,
            timeout=timeout,
            durability_timeout=durability_timeout,
            durability_poll_interval=durability_poll_interval,
        )
        durability_proof = _path_free_durability_proof(row, durability)
        durability_rows.append(durability_proof)
        filler_count += 1
        if durable_health:
            health_after = durable_health
        if durability.get("ok") is not True:
            failures.append(
                f"{row['tag']}: exact write fence did not become durable"
            )
            break
        pre_refault_contract, attestation_failures = (
            _fetch_prefix_attestation(
                base_url=base_url,
                model=model,
                prompts=prompts,
                pairs=pairs,
                timeout=timeout,
                health_attestation=health_attestation,
                request_controls=request_controls,
            )
        )
        failures.extend(attestation_failures)
        recent_pre_refault = _prefix_binding(
            pre_refault_contract,
            "recent",
        )
        old_after_candidate = _prefix_binding(
            pre_refault_contract,
            "old",
        )
        peak_bytes = max(
            peak_bytes,
            _integer(
                (recent_pre_refault.get("l2") or {}).get(
                    "store_total_size_bytes"
                )
            ),
        )
        if (
            not attestation_failures
            and durability_proof.get("ok") is True
            and durability_proof.get("post_eviction_complete") is True
            and _integer(durability_proof.get("disk_evictions_delta")) > 0
            and (old_after_candidate.get("l2") or {}).get(
                "terminal_readable"
            ) is False
            and (recent_pre_refault.get("l1") or {}).get(
                "terminal_resident_payload_present"
            ) is False
            and _l2_binding_is_complete(recent_pre_refault)
        ):
            evicting_filler_fence = durability_proof
            evicting_filler_stage = "pre-refault"
            old_after_durable_filler = old_after_candidate
            recent_after_durable_filler = recent_pre_refault
            break
        if attestation_failures:
            break

    _write_path_free_attestation(
        artifact_dir,
        "l2_recent_pre_refault",
        pre_refault_contract,
    )
    recent_l1 = recent_pre_refault.get("l1")
    old_pre_refault = _prefix_binding(pre_refault_contract, "old")
    old_pre_refault_l2 = old_pre_refault.get("l2")
    standard_pre_refault_ready = (
        isinstance(recent_l1, dict)
        and recent_l1.get("terminal_resident_payload_present") is False
        and _l2_binding_is_complete(recent_pre_refault)
        and isinstance(old_pre_refault_l2, dict)
        and old_pre_refault_l2.get("terminal_readable") is True
    )
    recent_store_pre_refault_ready = bool(
        evicting_filler_stage == "recent-store"
        and isinstance(recent_l1, dict)
        and recent_l1.get("terminal_resident_payload_present") is False
        and _l2_binding_is_complete(recent_pre_refault)
        and isinstance(old_pre_refault_l2, dict)
        and old_pre_refault_l2.get("terminal_readable") is False
        and _l2_binding_is_complete(old_after_store)
        and recent_store_eviction_ready
    )
    filler_pre_refault_ready = bool(
        evicting_filler_stage == "pre-refault"
        and isinstance(recent_l1, dict)
        and recent_l1.get("terminal_resident_payload_present") is False
        and _l2_binding_is_complete(recent_pre_refault)
        and isinstance(old_pre_refault_l2, dict)
        and old_pre_refault_l2.get("terminal_readable") is False
        and evicting_filler_fence
    )
    pre_refault_ready = (
        standard_pre_refault_ready
        or recent_store_pre_refault_ready
        or filler_pre_refault_ready
    )
    if not pre_refault_ready:
        if not any(
            failure.startswith("store-evict-refault:") for failure in failures
        ):
            failures.append(
                "store-evict-refault: bounded fillers did not evict the recent "
                "prefix from L1 while retaining it in L2"
            )
        return (
            _pre_refault_failure_observation(),
            rows,
            failures,
            health_after,
        )

    refault_row, health_after = _run_response_observation(
        base_url=base_url,
        model=model,
        tag="l2_recent_refault",
        prompt=prompts["recent_probe"],
        expected_marker=f"CACHE-HIERARCHY-{nonce}-L2-RECENT-PROBE",
        artifact_dir=artifact_dir,
        timeout=timeout,
        request_controls=request_controls,
    )
    rows.append(refault_row)
    post_refault_contract, touch_failures = _wait_for_prefix_access_touch(
        base_url=base_url,
        model=model,
        prompts=prompts,
        pairs=pairs,
        pair_name="recent",
        previous_binding=recent_pre_refault,
        timeout=timeout,
        health_attestation=health_attestation,
        timeout_s=durability_timeout,
        poll_interval_s=durability_poll_interval,
        request_controls=request_controls,
    )
    failures.extend(touch_failures)
    _write_path_free_attestation(
        artifact_dir,
        "l2_recent_post_refault",
        post_refault_contract,
    )
    recent_post_refault = _prefix_binding(
        post_refault_contract,
        "recent",
    )

    post_refault_filler_count = 0
    final_contract = post_refault_contract
    old_final = _prefix_binding(final_contract, "old")
    recent_final = _prefix_binding(final_contract, "recent")
    while (
        evicting_filler_stage not in {"pre-refault", "recent-store"}
        and filler_count < max_filler_requests
    ):
        old_l2 = old_final.get("l2")
        recent_l2 = recent_final.get("l2")
        if not isinstance(old_l2, dict):
            old_l2 = {}
        if not isinstance(recent_l2, dict):
            recent_l2 = {}
        if recent_l2.get("terminal_readable") is False:
            failures.append(
                "store-evict-refault: recent prefix left L2 after refault "
                "before the older prefix was evicted"
            )
            break
        if (
            old_l2.get("terminal_readable") is False
            and recent_l2.get("terminal_readable") is True
        ):
            if (
                post_refault_filler_count <= 0
                or not evicting_filler_fence
            ):
                failures.append(
                    "store-evict-refault: older prefix was already absent "
                    "before a post-refault eviction filler"
                )
            break
        filler_prompt, marker = _l2_filler_prompt(
            nonce,
            records,
            filler_count,
        )
        row, health_after = _run_response_observation(
            base_url=base_url,
            model=model,
            tag=f"l2_filler_{filler_count:03d}",
            prompt=filler_prompt,
            expected_marker=marker,
            artifact_dir=artifact_dir,
            timeout=timeout,
            request_controls=request_controls,
        )
        rows.append(row)
        durability, durable_health = _scenario_request_durability(
            row=row,
            base_url=base_url,
            timeout=timeout,
            durability_timeout=durability_timeout,
            durability_poll_interval=durability_poll_interval,
        )
        durability_proof = _path_free_durability_proof(row, durability)
        durability_rows.append(durability_proof)
        filler_count += 1
        post_refault_filler_count += 1
        if durable_health:
            health_after = durable_health
        if durability.get("ok") is not True:
            failures.append(
                f"{row['tag']}: exact write fence did not become durable"
            )
            break
        final_contract, attestation_failures = _fetch_prefix_attestation(
            base_url=base_url,
            model=model,
            prompts=prompts,
            pairs=pairs,
            timeout=timeout,
            health_attestation=health_attestation,
            request_controls=request_controls,
        )
        failures.extend(attestation_failures)
        old_final = _prefix_binding(final_contract, "old")
        recent_final = _prefix_binding(final_contract, "recent")
        if (
            not attestation_failures
            and durability_proof.get("ok") is True
            and durability_proof.get("post_eviction_complete") is True
            and _integer(durability_proof.get("disk_evictions_delta")) > 0
            and (old_final.get("l2") or {}).get("terminal_readable") is False
            and (recent_final.get("l2") or {}).get("terminal_readable") is True
            and not evicting_filler_fence
        ):
            evicting_filler_fence = durability_proof
            evicting_filler_stage = "post-refault"
            old_after_durable_filler = old_final
            recent_after_durable_filler = recent_final
        peak_bytes = max(
            peak_bytes,
            _integer(
                (recent_final.get("l2") or {}).get("store_total_size_bytes")
            ),
        )
        if attestation_failures:
            break

    _write_path_free_attestation(
        artifact_dir,
        "l2_final_eviction",
        final_contract,
    )
    old_final = _prefix_binding(final_contract, "old")
    recent_final = _prefix_binding(final_contract, "recent")
    old_l2 = old_final.get("l2")
    recent_l2 = recent_final.get("l2")
    if not isinstance(old_l2, dict):
        old_l2 = {}
    if not isinstance(recent_l2, dict):
        recent_l2 = {}
    saved_max = _integer(recent_l2.get("store_max_size_bytes"))
    final_bytes = _integer(recent_l2.get("store_total_size_bytes"))
    peak_bytes = max(peak_bytes, final_bytes)
    observation = {
        "schema": L2_SIZE_EVICTION_SCHEMA,
        "scenario": "store-evict-refault",
        "source_head": observed_source.get("head"),
        "source_tree": observed_source.get("tree"),
        "model_bundle_fingerprint_sha256": (
            (health_attestation.get("model_bundle_provenance") or {}).get(
                "fingerprint_sha256"
            )
        ),
        "cache_topology_fingerprint_sha256": (
            (health_attestation.get("cache_topology_provenance") or {}).get(
                "fingerprint_sha256"
            )
        ),
        "saved_max_bytes": saved_max,
        "l1_max_resident_bytes": l1_max_resident_bytes,
        "l1_l2_capacity_margin_ok": l1_l2_capacity_margin_ok,
        "peak_observed_bytes": peak_bytes,
        "final_observed_bytes": final_bytes,
        "bounded_filler_request_count": filler_count,
        "post_refault_filler_request_count": post_refault_filler_count,
        "evicting_filler_stage": evicting_filler_stage,
        "old_prefix_fingerprint_sha256": old_fingerprint,
        "recent_prefix_fingerprint_sha256": recent_fingerprint,
        "old_prefix_evicted": old_l2.get("terminal_readable") is False,
        "recent_prefix_present": recent_l2.get("terminal_readable") is True,
        "recent_prefix_last_access_after_old": (
            _integer((recent_post_refault.get("l2") or {}).get(
                "terminal_last_accessed_ns"
            ))
            > _integer(
                (old_after_store.get("l2") or {}).get(
                    "terminal_last_accessed_ns"
                )
            )
        ),
        "old_after_store": old_after_store,
        "old_before": old_before,
        "recent_before": recent_before,
        "recent_pre_refault": recent_pre_refault,
        "recent_post_refault": recent_post_refault,
        "evicting_filler_fence": evicting_filler_fence,
        "old_store_fence": old_store_fence,
        "recent_store_fence": recent_store_fence,
        "old_after_durable_filler": old_after_durable_filler,
        "recent_after_durable_filler": recent_after_durable_filler,
        "old_final": old_final,
        "recent_final": recent_final,
        "recent_refault_execution": _path_free_execution(refault_row),
        "write_fences": durability_rows,
    }
    failures.extend(
        validate_l2_size_eviction_observation(
            observation,
            expected_source_head=str(observed_source.get("head") or ""),
            expected_source_tree=str(observed_source.get("tree") or ""),
            health_attestation=health_attestation,
            max_filler_requests=max_filler_requests,
        )
    )
    return observation, rows, failures, health_after


def _run_restart_restore_scenario(
    *,
    base_url: str,
    model: str,
    nonce: str,
    records: int,
    artifact_dir: Path,
    timeout: int,
    durability_timeout: float,
    durability_poll_interval: float,
    health_attestation: dict[str, Any],
    observed_source: dict[str, Any],
    store_observation: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], dict[str, Any]]:
    failures: list[str] = []
    prompts_all = _l2_identity_prompts(nonce, records)
    prompts = {
        "recent_store": prompts_all["recent_store"],
        "recent_restart": prompts_all["recent_restart"],
    }
    pairs: dict[str, tuple[str, str]] = {
        "recent": ("recent_store", "recent_restart")
    }
    request_controls = _l2_scenario_request_controls()
    pre_contract, pre_failures = _fetch_prefix_attestation(
        base_url=base_url,
        model=model,
        prompts=prompts,
        pairs=pairs,
        timeout=timeout,
        health_attestation=health_attestation,
        request_controls=request_controls,
    )
    failures.extend(pre_failures)
    _write_path_free_attestation(
        artifact_dir,
        "l2_restart_pre",
        pre_contract,
    )
    restart_pre = _prefix_binding(pre_contract, "recent")
    row, health_after = _run_response_observation(
        base_url=base_url,
        model=model,
        tag="l2_restart_recent",
        prompt=prompts["recent_restart"],
        expected_marker=f"CACHE-HIERARCHY-{nonce}-L2-RECENT-RESTART",
        artifact_dir=artifact_dir,
        timeout=timeout,
        request_controls=request_controls,
    )
    post_contract, touch_failures = _wait_for_prefix_access_touch(
        base_url=base_url,
        model=model,
        prompts=prompts,
        pairs=pairs,
        pair_name="recent",
        previous_binding=restart_pre,
        timeout=timeout,
        health_attestation=health_attestation,
        timeout_s=durability_timeout,
        poll_interval_s=durability_poll_interval,
        request_controls=request_controls,
    )
    failures.extend(touch_failures)
    _write_path_free_attestation(
        artifact_dir,
        "l2_restart_post",
        post_contract,
    )
    restart_post = _prefix_binding(post_contract, "recent")
    execution = _path_free_execution(row)
    last = execution.get("last_cache_execution")
    if not isinstance(last, dict):
        last = {}
    observation = {
        "schema": L2_RESTART_RESTORE_SCHEMA,
        "scenario": "restart-restore",
        "source_head": observed_source.get("head"),
        "source_tree": observed_source.get("tree"),
        "model_bundle_fingerprint_sha256": (
            (health_attestation.get("model_bundle_provenance") or {}).get(
                "fingerprint_sha256"
            )
        ),
        "cache_topology_fingerprint_sha256": (
            (health_attestation.get("cache_topology_provenance") or {}).get(
                "fingerprint_sha256"
            )
        ),
        "restart_probe_prefix_fingerprint_sha256": restart_pre.get(
            "block_chain_fingerprint_sha256"
        ),
        "restart_restored_tokens": _integer(last.get("cached_tokens")),
        "restart_disk_blocks": _integer(last.get("disk_blocks")),
        "restart_uncached_tokens": _integer(
            last.get("uncached_prompt_tokens")
        ),
        "restart_restore_source": "block-disk",
        "restart_pre": restart_pre,
        "restart_post": restart_post,
        "restart_execution": execution,
    }
    failures.extend(
        validate_l2_restart_restore_observation(
            observation,
            store_observation=store_observation,
            expected_source_head=str(observed_source.get("head") or ""),
            expected_source_tree=str(observed_source.get("tree") or ""),
            health_attestation=health_attestation,
        )
    )
    return observation, [row], failures, health_after


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
    parser.add_argument(
        "--cache-scenario",
        choices=CACHE_SCENARIOS,
        default="standard",
        help=(
            "Run the standard store/probe contract plus explicit L2 "
            "eviction proof; restart-restore instead probes the recent chain "
            "that the linked eviction phase attested as surviving."
        ),
    )
    parser.add_argument(
        "--max-filler-requests",
        type=int,
        default=64,
        help="Hard bound for store-evict-refault filler generations (max 256).",
    )
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
    if not 1 <= args.max_filler_requests <= 256:
        parser.error("--max-filler-requests must be between 1 and 256")
    expected_phase = {
        "store-evict-refault": "store",
        "restart-restore": "probe",
    }.get(args.cache_scenario)
    if expected_phase is not None and args.phase != expected_phase:
        parser.error(
            f"--cache-scenario {args.cache_scenario} requires "
            f"--phase {expected_phase}"
        )

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
    cache_contract_profile = _cache_contract_profile_from_health(health_before)
    prefix = _common_prefix(args.nonce, args.records)
    prompts = _cache_prompts(prefix, args.nonce)
    requests = _standard_cache_requests(
        args.phase,
        args.cache_scenario,
        prompts,
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
        "schema": ARTIFACT_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": args.nonce,
        "phase": args.phase,
        "cache_scenario": args.cache_scenario,
        "nonce": args.nonce,
        "base_url": args.base_url,
        "model": args.model,
        "identity": identity,
        "cache_contract_profile": cache_contract_profile,
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
            "scenario_contract_ok": args.cache_scenario == "standard",
            "scenario_contract_failures": prerequisite_failures,
            "gate_ok": False,
            "verdict": "PARTIAL",
            "requests": [],
            "scenario_requests": [],
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
                "scenario_contract_ok": False,
                "scenario_contract_failures": linkage_failures,
                "gate_ok": False,
                "verdict": "PARTIAL",
                "requests": [],
                "scenario_requests": [],
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
        # The engine publishes the request-correlated cache-execution record
        # at finish processing, which can trail the SSE terminal by a
        # scheduler step — a single immediate read races it and captures the
        # PRIOR request's record (or none on the first request). Poll briefly
        # for the correlated record; keep the last read either way so a
        # genuine absence stays observable.
        _row_response_id = str(summary.get("response_id") or "")
        _lce_deadline = time.monotonic() + 6.0
        while _row_response_id and time.monotonic() < _lce_deadline:
            _lce = (health.get("scheduler") or {}).get("last_cache_execution")
            _lce_bound = (
                isinstance(_lce, dict)
                and str(_lce.get("request_id") or "") == _row_response_id
            )
            # The execution record can correlate EARLY (the admission-time
            # record lands before prefill) while the SSM companion lookup is
            # written at prefill — accepting on the execution match alone
            # captures a row with a dropped/absent lookup. Require BOTH
            # correlations when a companion block is exposed; a pure-miss
            # row that never produces a lookup exits at the deadline with
            # the execution-correlated read.
            _companion = (
                (health.get("cache") or {}).get("ssm_companion") or {}
            )
            _lookup = (
                _companion.get("last_prefix_lookup")
                if isinstance(_companion, dict)
                else None
            )
            _lookup_bound = (
                isinstance(_lookup, dict)
                and str(_lookup.get("request_id") or "") == _row_response_id
            )
            if _lce_bound and (_lookup_bound or not _companion):
                break
            if _lce_bound and time.monotonic() > _lce_deadline - 2.0:
                break
            time.sleep(0.25)
            health = _json_get(f"{args.base_url}/health", args.timeout)
        health_after = health
        health_path = args.artifact_dir / f"{tag}.health.json"
        health_path.write_text(json.dumps(health, indent=2, sort_keys=True) + "\n")
        health_counters_after = _health_cache_counters(health)
        health_counter_deltas = _counter_deltas(
            health_counters_before, health_counters_after
        )
        cache_contract_evidence = _health_cache_contract_evidence(health)
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
                "last_cache_reuse_partial": (health.get("scheduler") or {}).get(
                    "last_cache_reuse_partial"
                ),
                "scheduler_cache": (
                    (health.get("cache") or {}).get("scheduler_cache") or {}
                ),
                "block_disk_cache": (
                    (health.get("cache") or {}).get("block_disk_cache") or {}
                ),
                **cache_contract_evidence,
            }
        )
        summary["marker_ok"] = _exact_cache_marker_observed(
            summary,
            summary["expected_marker"],
        )
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

    validation_failures = (
        []
        if args.cache_scenario == "restart-restore"
        else validate_cache_rows(
            args.phase,
            rows,
            store_summary=store_summary,
            token_contract=tokenizer_lcp_contract,
            contract_profile=cache_contract_profile,
        )
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
    scenario_rows: list[dict[str, Any]] = []
    scenario_failures: list[str] = []
    l2_size_eviction_observation: dict[str, Any] | None = None
    l2_restart_restore_observation: dict[str, Any] | None = None
    scenario_prerequisites_ok = request_contract_ok and not validation_failures
    if args.cache_scenario != "standard" and not scenario_prerequisites_ok:
        scenario_failures.append(
            f"{args.cache_scenario}: skipped because the standard cache "
            "contract did not pass"
        )
    elif args.cache_scenario == "store-evict-refault":
        (
            l2_size_eviction_observation,
            scenario_rows,
            scenario_failures,
            scenario_health_after,
        ) = _run_store_evict_refault_scenario(
            base_url=args.base_url,
            model=args.model,
            nonce=args.nonce,
            records=args.records,
            artifact_dir=args.artifact_dir,
            timeout=args.timeout,
            durability_timeout=args.durability_timeout,
            durability_poll_interval=args.durability_poll_interval,
            max_filler_requests=args.max_filler_requests,
            health_attestation=health_attestation_before,
            observed_source=observed_source,
        )
        if scenario_health_after:
            health_after = scenario_health_after
    elif args.cache_scenario == "restart-restore":
        store_observation = (
            store_summary.get("l2_size_eviction_observation")
            if isinstance(store_summary, dict)
            else None
        )
        if not isinstance(store_observation, dict):
            scenario_failures.append(
                "restart-restore: store summary lacks the exact L2 eviction "
                "observation"
            )
        (
            l2_restart_restore_observation,
            scenario_rows,
            restart_failures,
            scenario_health_after,
        ) = _run_restart_restore_scenario(
            base_url=args.base_url,
            model=args.model,
            nonce=args.nonce,
            records=args.records,
            artifact_dir=args.artifact_dir,
            timeout=args.timeout,
            durability_timeout=args.durability_timeout,
            durability_poll_interval=args.durability_poll_interval,
            health_attestation=health_attestation_before,
            observed_source=observed_source,
            store_observation=store_observation,
        )
        scenario_failures.extend(restart_failures)
        if scenario_health_after:
            health_after = scenario_health_after
    scenario_request_contract_ok = all(
        row.get("status_code") == 200
        and row.get("marker_ok") is True
        and row.get("terminal_ok") is True
        for row in scenario_rows
    )
    if args.cache_scenario != "standard" and not scenario_rows:
        scenario_request_contract_ok = False
        no_rows_failure = (
            f"{args.cache_scenario}: scenario emitted no real Responses rows"
        )
        scenario_failures.append(no_rows_failure)
    validation_failures.extend(scenario_failures)
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
    request_contract_ok = request_contract_ok and scenario_request_contract_ok
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
        "scenario_contract_ok": not scenario_failures
        and scenario_request_contract_ok,
        "scenario_contract_failures": scenario_failures,
        "l2_size_eviction_observation": l2_size_eviction_observation,
        "l2_restart_restore_observation": l2_restart_restore_observation,
        "gate_ok": gate_ok,
        "verdict": "PASS" if gate_ok else "PARTIAL",
        "requests": rows,
        "scenario_requests": scenario_rows,
    }
    (args.artifact_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
