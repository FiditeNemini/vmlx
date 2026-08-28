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
import subprocess
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
        out: dict[str, Any] = {
            "status": status, "latency_s": latency, "errors": [],
            "raw": raw, "request_payload": payload,
        }
        if status != 200:
            out["errors"].append(f"HTTP {status}: {raw[:200]}")
            return out
        if stream:
            events, done = _sse_events(raw)
            if done != 1:
                out["errors"].append(f"expected exactly one [DONE], observed {done}")
            content, reasoning, calls, finishes, usage, stream_errors = self._assemble(events)
            out["response_id"] = next((e.get("id") for e in events if e.get("id")), None)
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
            out["response_id"] = body.get("id")
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

    def commit_assistant(self, content: str, tool_calls: list[dict[str, Any]], reasoning_items=None) -> None:
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
        out: dict[str, Any] = {
            "status": status, "latency_s": latency, "errors": [],
            "raw": raw, "request_payload": payload,
        }
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
        out["response_id"] = response.get("id")
        out["reasoning_items"] = [item for item in output if item.get("type") == "reasoning"]
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

    def commit_assistant(self, content: str, tool_calls: list[dict[str, Any]], reasoning_items=None) -> None:
        # Realistic Responses replay: OpenAI clients replay reasoning items
        # ahead of the function_call items they justified.
        for item in reasoning_items or []:
            self.items.append(item)
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


def confirm_deep_sleep(
    base: str,
    api_key: str | None,
    *,
    poll=None,
    timeout_s: float = 45.0,
    interval_s: float = 0.5,
) -> list[str]:
    """Enter deep sleep and CONFIRM it from lifecycle health.

    Never a fixed arbitrary delay: the arm passes only when /health reports
    status standby_deep with model_loaded false within the timeout. `poll`
    is injectable for regression tests.
    """
    if poll is None:
        def poll():
            _s, body, _l = request_json("GET", f"{base}/health", api_key=api_key, timeout=10.0)
            return body or {}
    request = urllib.request.Request(
        base + "/admin/deep-sleep", data=b"", method="POST", headers=_headers(api_key)
    )
    with urllib.request.urlopen(request, timeout=20.0) as response:
        body = json.loads(response.read().decode())
    if body.get("status") != "deep_sleep":
        return [f"deep-sleep endpoint did not enter deep sleep: {body}"]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        health = poll()
        progress = health.get("load_progress") or {}
        if str(health.get("status", "")).startswith("standby_deep") and not progress.get("model_loaded", True):
            return []
        time.sleep(interval_s)
    return [f"health never confirmed standby_deep/model_loaded=false within {timeout_s}s"]


class ContinuousSampler:
    """Samples /health memory gauges plus engine PID count and port owner
    CONTINUOUSLY while a turn runs — a single post-turn sample misses
    transient peaks (the 128%/140 GB class)."""

    def __init__(self, base: str, api_key: str | None, port: int, interval_s: float = 0.5):
        import threading

        self.base = base
        self.api_key = api_key
        self.port = port
        self.interval_s = interval_s
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _engine_pids(self) -> list[int]:
        try:
            out = subprocess.run(
                ["pgrep", "-f", "vmlx_engine.cli serve"], capture_output=True, text=True, timeout=3
            ).stdout
            return [int(x) for x in out.split()]
        except Exception:  # noqa: BLE001
            return []

    def _run(self) -> None:
        while not self._stop.is_set():
            sample: dict[str, Any] = {"t": time.time(), "engine_pids": self._engine_pids()}
            try:
                _s, body, _l = request_json(
                    "GET", f"{self.base}/health", api_key=self.api_key, timeout=5.0
                )
                memory = (body or {}).get("memory") or {}
                sample.update(
                    active_mb=memory.get("active_mb"),
                    peak_mb=memory.get("peak_mb"),
                    cache_mb=memory.get("cache_mb"),
                    status=(body or {}).get("status"),
                )
            except Exception:  # noqa: BLE001
                sample["health_error"] = True
            self.samples.append(sample)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "ContinuousSampler":
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def summary(self) -> dict[str, Any]:
        actives = [s["active_mb"] for s in self.samples if isinstance(s.get("active_mb"), (int, float))]
        pid_sets = {tuple(s.get("engine_pids") or []) for s in self.samples}
        return {
            "samples": len(self.samples),
            "active_mb_min": min(actives) if actives else None,
            "active_mb_max": max(actives) if actives else None,
            "engine_pid_sets": [list(p) for p in pid_sets],
            "single_engine_throughout": all(len(s.get("engine_pids") or []) <= 1 for s in self.samples),
        }


def grade_tool_call(calls: list[dict[str, Any]], expected_city: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Strict round-1 tool grading: exact name, normalized args, the
    MANIFEST city (never blessing whichever city the model picked), one
    call, stable id. Pure — regression-tested offline."""
    errors: list[str] = []
    if len(calls) != 1:
        return None, [f"expected exactly one tool call, observed {len(calls)}"]
    call = calls[0]
    if call.get("name") != "get_weather":
        errors.append(f"wrong tool name {call.get('name')!r}")
    if not call.get("id"):
        errors.append("tool call carries no id")
    try:
        arguments = json.loads(call.get("arguments") or "")
    except (TypeError, json.JSONDecodeError) as error:
        return None, errors + [f"tool arguments not valid JSON: {error}"]
    called_city = str(arguments.get("city", "")).strip()
    if called_city.lower() != expected_city.strip().lower():
        errors.append(
            f"model called city {called_city!r} but the manifest requested {expected_city!r}"
        )
    return (call if not errors else None), errors


def expected_block_reuse(previous_tokens: list[int], current_tokens: list[int], block_tokens: int) -> int:
    """Expected safe restore from the tokenized longest common prefix — the
    grading oracle, instead of comparing against the previous turn's cached
    count.

    Two separate engine mechanisms apply, never composed as
    block_floor(lcp - 1) (internally inconsistent: block_floor(6140) with a
    64-token block is 6080, not 6140 -- caught 2026-08-28). When the current
    render is identical all the way through the predecessor's own last
    token (lcp >= len(previous_tokens) - 1), the engine's documented N-1
    partial-terminal-block index applies and the safe extent is exactly
    len(previous_tokens) - 1, unaligned to any block boundary. Otherwise
    (content diverges before the predecessor's end -- e.g. a tool schema
    injected mid-conversation) only full chain-hash blocks are safe.
    """
    limit = min(len(previous_tokens), len(current_tokens))
    lcp = 0
    while lcp < limit and previous_tokens[lcp] == current_tokens[lcp]:
        lcp += 1
    predecessor_terminal = len(previous_tokens) - 1
    if predecessor_terminal >= 0 and lcp >= predecessor_terminal:
        return predecessor_terminal
    return (lcp // block_tokens) * block_tokens


def fit_filler_to_target(
    count_tokens,
    render_overhead_tokens: int,
    current_tokens: int,
    target_tokens: int,
    seed: int,
    tolerance: int,
) -> str:
    """Tokenizer-exact growth: size the filler so the RENDERED prompt lands
    within tolerance of the milestone (the 8192-target turn that actually
    rendered 19,712 tokens is the failure this replaces)."""
    needed = max(32, target_tokens - current_tokens - render_overhead_tokens)
    words = max(16, needed // 2)
    filler = _filler(words, seed)
    for _ in range(6):
        measured = count_tokens(filler)
        if abs(measured - needed) <= max(16, tolerance // 4):
            break
        words = max(16, int(words * needed / max(1, measured)))
        filler = _filler(words, seed)
    return filler


def _chat_projection(wire) -> list[dict[str, Any]]:
    """Chat-shaped view of the connected history for client-side token
    counting (recorded as an approximation for the Responses wire)."""
    if isinstance(wire, ChatWire):
        return list(wire.messages)
    projected: list[dict[str, Any]] = []
    for item in wire.items:
        if item.get("type") == "function_call":
            projected.append({
                "role": "assistant", "content": "",
                "tool_calls": [{"id": item.get("call_id"), "type": "function",
                                 "function": {"name": item.get("name"), "arguments": item.get("arguments")}}],
            })
        elif item.get("type") == "function_call_output":
            projected.append({"role": "tool", "tool_call_id": item.get("call_id"), "content": item.get("output")})
        elif item.get("type") == "reasoning":
            continue
        else:
            projected.append({"role": item.get("role", "user"), "content": item.get("content", "")})
    return projected


def run_scenario(
    manifest: dict[str, Any],
    *,
    base: str,
    model: str,
    api_key: str | None,
    tokenizer: Any,
    timeout: float,
    max_active_gb: float | None,
    block_tokens: int,
) -> tuple[list[TurnRecord], list[str]]:
    wire_name = manifest.get("wire", "chat")
    default_stream = bool(manifest.get("stream", True))
    wire = (ChatWire if wire_name == "chat" else ResponsesWire)(base, model, api_key, timeout)
    port = int(base.rsplit(":", 1)[-1])
    records: list[TurnRecord] = []
    scenario_errors: list[str] = []
    reuse_watch_from = 3  # project rule: judge reuse from turn 3 on
    previous_render_tokens: list[int] = []

    def render_tokens() -> list[int]:
        try:
            return list(
                tokenizer.apply_chat_template(
                    _chat_projection(wire), add_generation_prompt=True, tokenize=True
                )
            )
        except Exception:  # noqa: BLE001
            return []

    def count_tokens(text: str) -> int:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return len(tokenizer.encode(text))

    replay_reasoning = bool(manifest.get("replay_reasoning", True))

    def _reasoning(result_dict):
        return result_dict.get("reasoning_items") if replay_reasoning else None

    for index, turn in enumerate(manifest.get("turns") or [], start=1):
        kind = turn.get("kind")
        if turn.get("delay_s"):
            # Explicit inter-turn delay arm (store-latency investigations).
            time.sleep(float(turn["delay_s"]))
        stream = bool(turn.get("stream", default_stream))
        record = TurnRecord(index=index, kind=str(kind), wire=wire_name, stream=stream)
        records.append(record)

        if kind == "deep_sleep":
            record.errors.extend(confirm_deep_sleep(base, api_key))
            record.notes["deep_sleep_confirmed"] = not record.errors
            scenario_errors.extend(f"turn {index} (deep_sleep): {e}" for e in record.errors)
            continue

        expect = None
        expected_city = None
        if kind == "grow":
            target = int(turn["grow_to_tokens"])
            tolerance = int(turn.get("growth_tolerance", 512))
            seed = int(turn.get("seed", 1000 + index))
            current = len(render_tokens())
            preamble = f"[turn {index}] Context installment (respond with exactly OK-{index}): "
            filler = fit_filler_to_target(
                count_tokens, count_tokens(preamble) + 32, current, target, seed, tolerance
            )
            wire.add_user(preamble + filler)
            expect = f"OK-{index}"
            record.notes["growth_target"] = target
            record.notes["growth_tolerance"] = tolerance
        elif kind == "tool_round":
            expected_city = str(turn.get("city", "Paris"))
            wire.add_user(
                f"[turn {index}] What is the weather in {expected_city} right now? "
                "Use the get_weather tool."
            )
        elif kind == "probe":
            wire.add_user(str(turn["prompt"]))
            expect = turn.get("expect")
        else:
            scenario_errors.append(f"turn {index}: unknown kind {kind!r}")
            continue

        request_render = render_tokens()
        expected_reuse = expected_block_reuse(previous_render_tokens, request_render, block_tokens)
        record.notes["client_render_tokens"] = len(request_render)
        record.notes["expected_block_reuse_approx"] = expected_reuse
        record.notes["render_is_approximation"] = wire_name != "chat"

        with ContinuousSampler(base, api_key, port) as sampler:
            result = wire.run_turn(
                stream=stream,
                tools=(kind == "tool_round"),
                reasoning_effort=turn.get("reasoning_effort"),
                max_tokens=turn.get("max_tokens"),
            )
        record.notes["memory"] = sampler.summary()
        if not record.notes["memory"]["single_engine_throughout"]:
            record.errors.append("more than one engine PID observed during the turn")
        record.errors.extend(result.get("errors") or [])
        record.prompt_tokens = int(result.get("prompt_tokens") or 0)
        record.cached_tokens = int(result.get("cached_tokens") or 0)
        record.completion_tokens = int(result.get("completion_tokens") or 0)
        record.latency_s = float(result.get("latency_s") or 0.0)
        record.notes["raw"] = {
            "request_payload": result.get("request_payload"),
            "response_raw": result.get("raw"),
            "response_id": result.get("response_id"),
        }

        content = str(result.get("content") or "")
        calls = result.get("tool_calls") or []

        if kind == "grow":
            target = int(turn["grow_to_tokens"])
            tolerance = int(turn.get("growth_tolerance", 512))
            if record.prompt_tokens and abs(record.prompt_tokens - target) > tolerance:
                record.errors.append(
                    f"growth milestone missed: server prompt_tokens={record.prompt_tokens} "
                    f"not within {tolerance} of target {target}"
                )

        if kind == "tool_round":
            call, tool_errors = grade_tool_call(calls, expected_city or "")
            record.errors.extend(tool_errors)
            if call is not None:
                wire.commit_assistant(content, calls, reasoning_items=_reasoning(result))
                wire.commit_tool_result(
                    call, json.dumps({"city": expected_city, "temp_c": 21, "sky": "sunny"})
                )
                continuation_render = render_tokens()
                continuation_expected = expected_block_reuse(
                    request_render, continuation_render, block_tokens
                )
                with ContinuousSampler(base, api_key, port) as cont_sampler:
                    continuation = wire.run_turn(
                        stream=stream, tools=True,
                        reasoning_effort=turn.get("reasoning_effort"),
                        max_tokens=turn.get("max_tokens"),
                    )
                cont_record: dict[str, Any] = {
                    "prompt_tokens": int(continuation.get("prompt_tokens") or 0),
                    "cached_tokens": int(continuation.get("cached_tokens") or 0),
                    "completion_tokens": int(continuation.get("completion_tokens") or 0),
                    "latency_s": float(continuation.get("latency_s") or 0.0),
                    "errors": list(continuation.get("errors") or []),
                    "memory": cont_sampler.summary(),
                    "expected_block_reuse_approx": continuation_expected,
                    "raw": {
                        "request_payload": continuation.get("request_payload"),
                        "response_raw": continuation.get("raw"),
                        "response_id": continuation.get("response_id"),
                    },
                }
                record.notes["continuation"] = cont_record
                record.errors.extend(cont_record["errors"])
                continuation_content = str(continuation.get("content") or "")
                continuation_calls = continuation.get("tool_calls") or []
                if continuation_calls:
                    record.errors.append(
                        f"continuation unexpectedly called tools again: "
                        f"{[c.get('name') for c in continuation_calls]}"
                    )
                elif "21" not in continuation_content:
                    record.errors.append(
                        f"continuation answer missing tool result: {continuation_content[:80]!r}"
                    )
                wire.commit_assistant(continuation_content, continuation_calls,
                                       reasoning_items=_reasoning(continuation))
                previous_render_tokens = render_tokens()
        else:
            if not content.strip():
                record.errors.append("turn produced no visible content")
            if expect and expect not in content:
                record.errors.append(f"expected {expect!r} in answer, got {content[:80]!r}")
            wire.commit_assistant(content, [], reasoning_items=_reasoning(result))
            previous_render_tokens = render_tokens()

        if index >= reuse_watch_from and turn.get("expect_reuse"):
            slack = 2 * block_tokens
            if record.cached_tokens + slack < expected_reuse:
                record.errors.append(
                    f"reuse below block-aligned expectation: cached={record.cached_tokens}, "
                    f"expected≈{expected_reuse} (client-render approximation)"
                )

        memory_max = record.notes["memory"].get("active_mb_max")
        if max_active_gb and isinstance(memory_max, (int, float)) and memory_max > max_active_gb * 1024:
            record.errors.append(
                f"MLX active memory {memory_max / 1024:.1f} GB exceeded the manifest bound {max_active_gb} GB"
            )
        scenario_errors.extend(f"turn {index} ({kind}): {e}" for e in record.errors)

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
    parser.add_argument("--block-tokens", type=int, default=64,
                        help="cache block size for the expected complete-block reuse oracle")
    parser.add_argument("--skip-provenance", action="store_true",
                        help="diagnostics only; artifacts are stamped UNPROVEN")
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    base = args.base_url.rstrip("/")
    bundle = args.bundle.expanduser().resolve(strict=True)
    manifest = json.loads(args.manifest.read_text())
    def _sha256_file(path: Path) -> str:
        import hashlib

        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _engine_processes() -> list[dict[str, Any]]:
        try:
            out = subprocess.run(
                ["pgrep", "-fl", "vmlx_engine.cli serve"], capture_output=True, text=True, timeout=5
            ).stdout
            processes = []
            for line in out.splitlines():
                pid, _, argv = line.partition(" ")
                full = subprocess.run(
                    ["ps", "-o", "command=", "-p", pid], capture_output=True, text=True, timeout=5
                ).stdout.strip()
                processes.append({"pid": int(pid), "argv": full or argv})
            return processes
        except Exception:  # noqa: BLE001
            return []

    artifact: dict[str, Any] = {
        "schema": SCENARIO_SCHEMA,
        "status": "INVALID",
        "started_at_unix": time.time(),
        "scenario": manifest.get("scenario"),
        "wire": manifest.get("wire", "chat"),
        "stream": bool(manifest.get("stream", True)),
        "git_head": subprocess.run(
            ["git", "-C", str(args.source_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip(),
        "harness_sha256": _sha256_file(Path(__file__).resolve()),
        "manifest_sha256": _sha256_file(args.manifest.resolve()),
        "python_executable": sys.executable,
        "engine_processes": _engine_processes(),
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
        try:
            import importlib.util

            spec = importlib.util.find_spec("vmlx_engine")
            artifact["imported_engine_path"] = getattr(spec, "origin", None)
        except Exception:  # noqa: BLE001
            artifact["imported_engine_path"] = None
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
            block_tokens=args.block_tokens,
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
