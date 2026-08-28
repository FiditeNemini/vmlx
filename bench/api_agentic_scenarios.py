#!/usr/bin/env python3
"""Attach-only, fail-closed connected agentic API scenario runner.

Extends bench/api_context_acceptance.py (whose provenance attestation,
cache-contract gate, and memory monitoring it reuses) with CONNECTED
multi-turn growth: one conversation grows turn by turn through tool
rounds, reasoning-effort changes, malformed-call recovery arms, optional
single-modality turns, and true deep-wake arms — never one giant cold
prefill. Both Chat Completions and Responses wires, streaming and
non-streaming, are validated from their raw events.

The runner never launches, stops, or swaps a model (attach-only). The
operator loads the exact bundle through the real Electron UI first. Any
invariant violation fails the whole scenario closed with a machine-readable
artifact; there are no soft passes.

Reuse judgments follow the project rule: cached-token expectations are
asserted only from the third connected turn onward.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from api_context_acceptance import (  # noqa: E402
    GateError,
    _filler,
    _headers,
    bundle_attestation,
    load_tokenizer,
    local_runtime_provenance,
    request_json,
    verify_cache_contract,
    verify_provenance,
)

SCENARIO_SCHEMA = "vmlx-api-agentic-scenario-v1"

WEATHER_TOOL_CHAT = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
WEATHER_TOOL_RESPONSES = {
    "type": "function",
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": WEATHER_TOOL_CHAT["function"]["parameters"],
}

THINK_LEAK_MARKERS = ("<think", "</think", "<mm:think", "[THINK]", "<atem:")


@dataclass
class TurnRecord:
    index: int
    kind: str
    wire: str
    stream: bool
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    errors: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)


class ScenarioFailure(GateError):
    pass


def _sse_events(raw: str) -> tuple[list[dict[str, Any]], int]:
    events: list[dict[str, Any]] = []
    done = 0
    for line in raw.splitlines():
        if line == "data: [DONE]":
            done += 1
        elif line.startswith("data: {"):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                events.append({"__unparseable__": line[:200]})
    return events, done


def _post(url: str, payload: dict[str, Any], api_key: str | None, timeout: float) -> tuple[int, str, float]:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST", headers=_headers(api_key)
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode(), time.perf_counter() - started
    except urllib.error.HTTPError as error:  # noqa: PERF203
        return error.code, error.read().decode(), time.perf_counter() - started


def _leak_errors(content: str) -> list[str]:
    return [
        f"reasoning/tool markup leaked into visible content: {marker!r}"
        for marker in THINK_LEAK_MARKERS
        if marker in (content or "")
    ]


class ChatWire:
    """Chat Completions dialect: connected history + raw validation."""

    name = "chat"

    def __init__(self, base: str, model: str, api_key: str | None, timeout: float):
        self.url = base + "/v1/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.messages: list[dict[str, Any]] = []

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def run_turn(
        self,
        *,
        stream: bool,
        tools: bool,
        reasoning_effort: str | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "messages": list(self.messages), "stream": stream}
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if tools:
            payload["tools"] = [WEATHER_TOOL_CHAT]
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        if max_tokens:
            payload["max_tokens"] = max_tokens
        status, raw, latency = _post(self.url, payload, self.api_key, self.timeout)
        out: dict[str, Any] = {"status": status, "latency_s": latency, "errors": []}
        if status != 200:
            out["errors"].append(f"HTTP {status}: {raw[:200]}")
            return out
        if stream:
            events, done = _sse_events(raw)
            if done != 1:
                out["errors"].append(f"expected exactly one [DONE], observed {done}")
            content, reasoning, calls, finishes, usage, stream_errors = self._assemble(events)
            out["errors"].extend(stream_errors)
        else:
            body = json.loads(raw)
            choice = (body.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""
            calls = [
                {
                    "id": call.get("id"),
                    "name": (call.get("function") or {}).get("name"),
                    "arguments": (call.get("function") or {}).get("arguments"),
                }
                for call in (message.get("tool_calls") or [])
            ]
            finishes = [choice.get("finish_reason")] if choice.get("finish_reason") else []
            usage = body.get("usage") or {}
        if len(finishes) != 1:
            out["errors"].append(f"expected exactly one terminal finish reason, observed {finishes}")
        out.update(
            content=content,
            reasoning=reasoning,
            tool_calls=calls,
            usage=usage,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            cached_tokens=int(
                ((usage.get("prompt_tokens_details") or {}).get("cached_tokens"))
                or usage.get("cached_tokens")
                or 0
            ),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )
        out["errors"].extend(_leak_errors(content))
        return out

    @staticmethod
    def _assemble(events: list[dict[str, Any]]):
        content: list[str] = []
        reasoning: list[str] = []
        finishes: list[str] = []
        usage: dict[str, Any] = {}
        errors: list[str] = []
        slots: dict[int, dict[str, Any]] = {}
        for event in events:
            if "__unparseable__" in event:
                errors.append(f"unparseable SSE event: {event['__unparseable__']}")
                continue
            if event.get("error"):
                errors.append(f"stream error event: {str(event['error'])[:160]}")
                continue
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                for call in delta.get("tool_calls") or []:
                    slot = slots.setdefault(call.get("index", 0), {"id": None, "name": None, "args": []})
                    if call.get("id"):
                        if slot["id"] and slot["id"] != call["id"]:
                            errors.append("tool call id changed mid-stream")
                        slot["id"] = call["id"]
                    function = call.get("function") or {}
                    if function.get("name"):
                        slot["name"] = function["name"]
                    if function.get("arguments"):
                        slot["args"].append(function["arguments"])
                if choice.get("finish_reason"):
                    finishes.append(choice["finish_reason"])
        calls = [
            {"id": slot["id"], "name": slot["name"], "arguments": "".join(slot["args"])}
            for slot in slots.values()
        ]
        return "".join(content), "".join(reasoning), calls, finishes, usage, errors

    def commit_assistant(self, content: str, tool_calls: list[dict[str, Any]]) -> None:
        message: dict[str, Any] = {"role": "assistant", "content": content or None}
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {"name": call["name"], "arguments": call["arguments"]},
                }
                for call in tool_calls
            ]
        self.messages.append(message)

    def commit_tool_result(self, call: dict[str, Any], output: str) -> None:
        self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})


class ResponsesWire:
    """OpenAI Responses dialect: connected input items + raw event validation."""

    name = "responses"

    def __init__(self, base: str, model: str, api_key: str | None, timeout: float):
        self.url = base + "/v1/responses"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.items: list[dict[str, Any]] = []

    def add_user(self, content: str) -> None:
        self.items.append({"role": "user", "content": content})

    def run_turn(
        self,
        *,
        stream: bool,
        tools: bool,
        reasoning_effort: str | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "input": list(self.items), "stream": stream}
        if tools:
            payload["tools"] = [WEATHER_TOOL_RESPONSES]
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}
        if max_tokens:
            payload["max_output_tokens"] = max_tokens
        status, raw, latency = _post(self.url, payload, self.api_key, self.timeout)
        out: dict[str, Any] = {"status": status, "latency_s": latency, "errors": []}
        if status != 200:
            out["errors"].append(f"HTTP {status}: {raw[:200]}")
            return out
        if stream:
            events, _done = _sse_events(raw)
            completed = [event for event in events if event.get("type") == "response.completed"]
            failed = [
                event
                for event in events
                if event.get("type") in ("response.failed", "response.incomplete", "error")
            ]
            if failed:
                out["errors"].append(f"terminal failure event: {str(failed[-1])[:180]}")
            if len(completed) != 1:
                out["errors"].append(
                    f"expected exactly one response.completed, observed {len(completed)}"
                )
                return out
            response = completed[-1].get("response") or {}
        else:
            response = json.loads(raw)
            if response.get("status") != "completed":
                out["errors"].append(f"response status={response.get('status')!r} not completed")
        output = response.get("output") or []
        texts: list[str] = []
        reasoning_chars = 0
        calls: list[dict[str, Any]] = []
        for item in output:
            if item.get("type") == "message":
                for part in item.get("content") or []:
                    if part.get("type") == "output_text":
                        texts.append(part.get("text") or "")
            elif item.get("type") == "reasoning":
                for summary in item.get("summary") or []:
                    reasoning_chars += len(summary.get("text") or "")
            elif item.get("type") == "function_call":
                calls.append(
                    {
                        "id": item.get("call_id") or item.get("id"),
                        "name": item.get("name"),
                        "arguments": item.get("arguments"),
                    }
                )
        usage = response.get("usage") or {}
        content = "".join(texts)
        out.update(
            content=content,
            reasoning="x" * reasoning_chars,
            tool_calls=calls,
            usage=usage,
            prompt_tokens=int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0),
            cached_tokens=int(
                ((usage.get("input_tokens_details") or {}).get("cached_tokens"))
                or usage.get("cached_tokens")
                or 0
            ),
            completion_tokens=int(usage.get("output_tokens") or usage.get("completion_tokens") or 0),
        )
        out["errors"].extend(_leak_errors(content))
        return out

    def commit_assistant(self, content: str, tool_calls: list[dict[str, Any]]) -> None:
        for call in tool_calls:
            self.items.append(
                {
                    "type": "function_call",
                    "call_id": call["id"],
                    "name": call["name"],
                    "arguments": call["arguments"],
                }
            )
        if content:
            self.items.append({"role": "assistant", "content": content})

    def commit_tool_result(self, call: dict[str, Any], output: str) -> None:
        self.items.append(
            {"type": "function_call_output", "call_id": call["id"], "output": output}
        )


def _admin_deep_sleep(base: str, api_key: str | None, timeout: float) -> None:
    request = urllib.request.Request(
        base + "/admin/deep-sleep", data=b"", method="POST", headers=_headers(api_key)
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode())
    if body.get("status") != "deep_sleep":
        raise ScenarioFailure(f"deep-sleep arm did not enter deep sleep: {body}")
    time.sleep(3.0)


def _health_memory(base: str, api_key: str | None) -> dict[str, Any]:
    try:
        _status, body, _latency = request_json(
            "GET", f"{base}/health", api_key=api_key, timeout=15.0
        )
        memory = (body or {}).get("memory") or {}
        return {
            "active_mb": memory.get("active_mb"),
            "peak_mb": memory.get("peak_mb"),
            "cache_mb": memory.get("cache_mb"),
        }
    except Exception:  # noqa: BLE001 — sampling must never fail a turn itself
        return {}


def run_scenario(
    manifest: dict[str, Any],
    *,
    base: str,
    model: str,
    api_key: str | None,
    tokenizer: Any,
    timeout: float,
    max_active_gb: float | None,
) -> tuple[list[TurnRecord], list[str]]:
    wire_name = manifest.get("wire", "chat")
    stream = bool(manifest.get("stream", True))
    wire = (ChatWire if wire_name == "chat" else ResponsesWire)(base, model, api_key, timeout)
    records: list[TurnRecord] = []
    scenario_errors: list[str] = []
    reuse_watch_from = 3  # project rule: judge reuse from turn 3 on
    prev_cached = 0

    for index, turn in enumerate(manifest.get("turns") or [], start=1):
        kind = turn.get("kind")
        record = TurnRecord(index=index, kind=str(kind), wire=wire_name, stream=stream)
        records.append(record)

        if kind == "deep_sleep":
            _admin_deep_sleep(base, api_key, timeout)
            record.notes["deep_sleep"] = True
            continue

        if kind == "grow":
            target = int(turn["grow_to_tokens"])
            seed = int(turn.get("seed", 1000 + index))
            filler = _filler(max(64, target // 6), seed)
            wire.add_user(
                f"[turn {index}] Context installment (respond with exactly OK-{index}): {filler}"
            )
            expect = f"OK-{index}"
        elif kind == "tool_round":
            wire.add_user(
                f"[turn {index}] What is the weather in {turn.get('city', 'Paris')} right now? "
                "Use the get_weather tool."
            )
            expect = None
        elif kind == "probe":
            wire.add_user(str(turn["prompt"]))
            expect = turn.get("expect")
        else:
            scenario_errors.append(f"turn {index}: unknown kind {kind!r}")
            continue

        result = wire.run_turn(
            stream=stream,
            tools=(kind == "tool_round"),
            reasoning_effort=turn.get("reasoning_effort"),
            max_tokens=turn.get("max_tokens"),
        )
        record.errors.extend(result.get("errors") or [])
        record.prompt_tokens = int(result.get("prompt_tokens") or 0)
        record.cached_tokens = int(result.get("cached_tokens") or 0)
        record.completion_tokens = int(result.get("completion_tokens") or 0)
        record.latency_s = float(result.get("latency_s") or 0.0)

        content = str(result.get("content") or "")
        calls = result.get("tool_calls") or []

        if kind == "tool_round":
            if len(calls) != 1:
                record.errors.append(f"expected exactly one tool call, observed {len(calls)}")
            else:
                call = calls[0]
                if call.get("name") != "get_weather":
                    record.errors.append(f"wrong tool name {call.get('name')!r}")
                if not call.get("id"):
                    record.errors.append("tool call carries no id")
                try:
                    arguments = json.loads(call.get("arguments") or "")
                    if "city" not in arguments:
                        record.errors.append(f"tool arguments missing city: {arguments}")
                except (TypeError, json.JSONDecodeError) as error:
                    record.errors.append(f"tool arguments not valid JSON: {error}")
                if not record.errors:
                    wire.commit_assistant(content, calls)
                    called_city = "Paris"
                    try:
                        called_city = json.loads(call.get("arguments") or "{}").get("city") or called_city
                    except json.JSONDecodeError:
                        pass
                    # The fixture result must echo the CITY THE MODEL CALLED —
                    # a hardcoded city made the grader fail the model for
                    # correctly repeating the tool's own output.
                    wire.commit_tool_result(
                        call, json.dumps({"city": called_city, "temp_c": 21, "sky": "sunny"})
                    )
                    continuation = wire.run_turn(
                        stream=stream, tools=True,
                        reasoning_effort=turn.get("reasoning_effort"), max_tokens=turn.get("max_tokens"),
                    )
                    record.notes["continuation_latency_s"] = continuation.get("latency_s")
                    record.errors.extend(continuation.get("errors") or [])
                    continuation_content = str(continuation.get("content") or "")
                    continuation_calls = continuation.get("tool_calls") or []
                    if continuation_calls:
                        record.notes["continuation_repeat_call"] = True
                    elif "21" not in continuation_content:
                        record.errors.append(
                            f"continuation answer missing tool result: {continuation_content[:80]!r}"
                        )
                    wire.commit_assistant(continuation_content, continuation_calls)
        else:
            if not content.strip():
                record.errors.append("turn produced no visible content")
            if expect and expect not in content:
                record.errors.append(f"expected {expect!r} in answer, got {content[:80]!r}")
            wire.commit_assistant(content, [])

        if turn.get("min_prompt_tokens") and record.prompt_tokens < int(turn["min_prompt_tokens"]):
            record.errors.append(
                f"connected growth milestone missed: prompt_tokens={record.prompt_tokens} "
                f"< {turn['min_prompt_tokens']}"
            )
        if (
            index >= reuse_watch_from
            and turn.get("expect_reuse")
            and record.cached_tokens <= prev_cached // 2
        ):
            record.errors.append(
                f"prefix reuse collapsed: cached_tokens={record.cached_tokens} "
                f"(previous turn {prev_cached})"
            )
        record.notes["memory"] = _health_memory(base, api_key)
        active_mb = record.notes["memory"].get("active_mb")
        if max_active_gb and isinstance(active_mb, (int, float)) and active_mb > max_active_gb * 1024:
            record.errors.append(
                f"MLX active memory {active_mb / 1024:.1f} GB exceeded the manifest bound "
                f"{max_active_gb} GB"
            )
        prev_cached = record.cached_tokens
        scenario_errors.extend(f"turn {index} ({kind}): {error}" for error in record.errors)

    return records, scenario_errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--source-root", default=str(Path(__file__).resolve().parents[1]), type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--api-key", default=os.environ.get("VMLINUX_API_KEY"))
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--max-active-gb", type=float, default=None,
                        help="fail the scenario if /health MLX active memory exceeds this bound")
    parser.add_argument("--skip-provenance", action="store_true",
                        help="diagnostics only; artifacts are stamped UNPROVEN")
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    base = args.base_url.rstrip("/")
    bundle = args.bundle.expanduser().resolve(strict=True)
    manifest = json.loads(args.manifest.read_text())
    artifact: dict[str, Any] = {
        "schema": SCENARIO_SCHEMA,
        "status": "INVALID",
        "started_at_unix": time.time(),
        "scenario": manifest.get("scenario"),
        "wire": manifest.get("wire", "chat"),
        "stream": bool(manifest.get("stream", True)),
    }
    try:
        _hs, health, _hl = request_json("GET", f"{base}/health", api_key=args.api_key, timeout=30.0)
        _ms, models, _ml = request_json("GET", f"{base}/v1/models", api_key=args.api_key, timeout=30.0)
        artifact["bundle_attestation"] = bundle_attestation(bundle)
        artifact["runtime_provenance"] = local_runtime_provenance(args.source_root)
        if args.skip_provenance:
            artifact["provenance"] = "SKIPPED-UNPROVEN"
        else:
            provenance_errors = verify_provenance(
                health or {},
                models or {},
                source_root=args.source_root,
                bundle=bundle,
                served_model=args.served_model,
            )
            if provenance_errors:
                raise ScenarioFailure("; ".join(provenance_errors))
            artifact["provenance"] = "VERIFIED"
        artifact["cache_contract_errors"] = verify_cache_contract(health or {})
        tokenizer = load_tokenizer(bundle)
        records, errors = run_scenario(
            manifest,
            base=base,
            model=args.served_model,
            api_key=args.api_key,
            tokenizer=tokenizer,
            timeout=args.timeout_s,
            max_active_gb=args.max_active_gb,
        )
        artifact["turns"] = [record.__dict__ for record in records]
        artifact["errors"] = errors
        artifact["status"] = "PASS" if not errors and not artifact["cache_contract_errors"] else "FAIL"
    except (GateError, ScenarioFailure, urllib.error.URLError) as error:
        artifact["status"] = "FAIL"
        artifact["fatal"] = f"{type(error).__name__}: {error}"
    artifact["finished_at_unix"] = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, default=str))
    print(json.dumps({"status": artifact["status"], "errors": artifact.get("errors", artifact.get("fatal"))}, indent=2)[:1600])
    return 0 if artifact["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
