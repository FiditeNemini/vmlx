# SPDX-License-Identifier: Apache-2.0
"""Production-route mid-stream failure contracts for Anthropic and Ollama."""

from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient


def _ndjson_rows(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _sse_rows(text: str) -> list[dict]:
    rows: list[dict] = []
    for line in text.splitlines():
        if line.startswith("data: "):
            rows.append(json.loads(line.removeprefix("data: ")))
    return rows


def test_anthropic_and_ollama_midstream_failures_end_as_errors_then_recover(
    monkeypatch,
):
    """Every compatibility adapter must preserve partial text, then fail truthfully."""
    import vmlx_engine.server as server
    from vmlx_engine.engine.base import GenerationOutput

    class FailureEngine:
        is_mllm = False
        preserve_native_tool_format = False
        tokenizer = SimpleNamespace(has_thinking=False)

        async def stream_chat(self, *, messages, **kwargs):
            prompt = str(messages[-1].get("content", ""))
            if "FAIL" in prompt:
                yield GenerationOutput(
                    text="CHAT-PARTIAL",
                    new_text="CHAT-PARTIAL",
                    tokens=[1],
                    prompt_tokens=5,
                    completion_tokens=1,
                    finished=False,
                    finish_reason=None,
                )
                raise RuntimeError("CHAT MIDSTREAM PROBE FAILURE")
            yield GenerationOutput(
                text="CHAT-RECOVERY-OK",
                new_text="CHAT-RECOVERY-OK",
                tokens=[1],
                prompt_tokens=6,
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
            )

        async def stream_generate(self, *, prompt, **kwargs):
            if "FAIL" in prompt:
                yield GenerationOutput(
                    text="RAW-PARTIAL",
                    new_text="RAW-PARTIAL",
                    tokens=[1],
                    prompt_tokens=4,
                    completion_tokens=1,
                    finished=False,
                    finish_reason=None,
                )
                raise RuntimeError("RAW MIDSTREAM PROBE FAILURE")
            yield GenerationOutput(
                text="RAW-RECOVERY-OK",
                new_text="RAW-RECOVERY-OK",
                tokens=[1],
                prompt_tokens=4,
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
            )

        async def abort_request(self, request_id):
            return True

    monkeypatch.setattr(server, "_engine", FailureEngine())
    monkeypatch.setattr(server, "_model_path", None)
    monkeypatch.setattr(server, "_model_name", "adapter-failure-test")
    monkeypatch.setattr(server, "_served_model_name", None)
    monkeypatch.setattr(server, "_reasoning_parser", None)
    monkeypatch.setattr(server, "_tool_call_parser", None)
    monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", True)
    monkeypatch.setattr(server, "_mcp_manager", None)
    monkeypatch.setattr(server, "_api_key", None, raising=False)
    monkeypatch.setattr(server, "_default_timeout", 5.0)
    monkeypatch.setattr(server, "_default_temperature", None)
    monkeypatch.setattr(server, "_default_top_p", None)
    monkeypatch.setattr(server, "_default_top_k", None)
    monkeypatch.setattr(server, "_default_min_p", None)
    monkeypatch.setattr(server, "_default_repetition_penalty", None)
    monkeypatch.setattr(server, "_default_max_tokens", 64)
    monkeypatch.setattr(server, "_default_max_tokens_explicit", True, raising=False)
    monkeypatch.setattr(server, "_max_prompt_tokens", 4096)
    server._jang_sampling_defaults_cache.clear()
    server._generation_defaults_cache.clear()

    client = TestClient(server.app, raise_server_exceptions=False)

    anthropic_fail = client.post(
        "/v1/messages",
        json={
            "model": "adapter-failure-test",
            "max_tokens": 16,
            "stream": True,
            "messages": [{"role": "user", "content": "FAIL ANTHROPIC"}],
        },
    )
    anthropic_rows = _sse_rows(anthropic_fail.text)
    assert anthropic_fail.status_code == 200
    assert any(
        row.get("delta", {}).get("text") == "CHAT-PARTIAL"
        for row in anthropic_rows
    )
    assert anthropic_rows[-1]["type"] == "error"
    assert "CHAT MIDSTREAM PROBE FAILURE" in anthropic_rows[-1]["error"]["message"]
    assert not any(row.get("type") == "message_stop" for row in anthropic_rows)

    anthropic_recovery = client.post(
        "/v1/messages",
        json={
            "model": "adapter-failure-test",
            "max_tokens": 16,
            "stream": True,
            "messages": [{"role": "user", "content": "RECOVER ANTHROPIC"}],
        },
    )
    recovery_rows = _sse_rows(anthropic_recovery.text)
    assert any(
        row.get("delta", {}).get("text") == "CHAT-RECOVERY-OK"
        for row in recovery_rows
    )
    assert recovery_rows[-1]["type"] == "message_stop"

    ollama_cases = (
        (
            "/api/chat",
            {
                "model": "adapter-failure-test",
                "stream": True,
                "messages": [{"role": "user", "content": "FAIL CHAT"}],
            },
            "CHAT-PARTIAL",
            "CHAT MIDSTREAM PROBE FAILURE",
        ),
        (
            "/api/generate",
            {
                "model": "adapter-failure-test",
                "stream": True,
                "prompt": "FAIL TEMPLATED",
            },
            "CHAT-PARTIAL",
            "CHAT MIDSTREAM PROBE FAILURE",
        ),
        (
            "/api/generate",
            {
                "model": "adapter-failure-test",
                "stream": True,
                "raw": True,
                "prompt": "FAIL RAW",
            },
            "RAW-PARTIAL",
            "RAW MIDSTREAM PROBE FAILURE",
        ),
    )
    for endpoint, body, partial, failure in ollama_cases:
        response = client.post(endpoint, json=body)
        rows = _ndjson_rows(response.text)
        assert response.status_code == 200
        assert partial in response.text
        assert rows[-1] == {
            "error": next(
                row["error"] for row in rows if failure in row.get("error", "")
            )
        }
        assert failure in rows[-1]["error"]
        assert not any(row.get("done") is True for row in rows)

    ollama_recovery = client.post(
        "/api/chat",
        json={
            "model": "adapter-failure-test",
            "stream": True,
            "messages": [{"role": "user", "content": "RECOVER CHAT"}],
        },
    )
    ollama_recovery_rows = _ndjson_rows(ollama_recovery.text)
    assert "CHAT-RECOVERY-OK" in ollama_recovery.text
    assert ollama_recovery_rows[-1]["done"] is True
    assert "error" not in ollama_recovery_rows[-1]
