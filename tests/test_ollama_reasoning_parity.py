# SPDX-License-Identifier: Apache-2.0
"""Focused contracts for Ollama streaming reasoning-policy parity."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


def _run_streaming_ollama_chat(
    monkeypatch,
    *,
    family_name: str,
    model_type: str,
    body: dict,
    reasoning_parser=None,
    preserve_native_tool_format: bool = False,
) -> dict:
    import vmlx_engine.server as server

    config = SimpleNamespace(
        family_name=family_name,
        model_type=model_type,
        supports_thinking=True,
        supports_instruct_mode=True,
        reasoning_parser=None,
        think_in_template=False,
        architecture_hints={},
    )

    class Registry:
        def lookup(self, _model_key):
            return config

    captured: dict = {}

    async def fake_stream_chat_completion(
        _engine,
        messages,
        _request,
        *,
        fastapi_request=None,
        **kwargs,
    ):
        captured["messages"] = copy.deepcopy(messages)
        captured["kwargs"] = copy.deepcopy(kwargs)
        yield (
            'data: {"id":"chatcmpl-test","object":"chat.completion.chunk",'
            '"created":1,"model":"test","choices":[{"index":0,'
            '"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
        )
        yield (
            'data: {"id":"chatcmpl-test","object":"chat.completion.chunk",'
            '"created":1,"model":"test","choices":[{"index":0,'
            '"delta":{},"finish_reason":"stop"}],"usage":'
            '{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}\n\n'
        )
        yield "data: [DONE]\n\n"

    fake_engine = SimpleNamespace(
        is_mllm=False,
        tokenizer=SimpleNamespace(has_thinking=False),
        preserve_native_tool_format=preserve_native_tool_format,
    )
    monkeypatch.setattr(server, "_engine", fake_engine)
    monkeypatch.setattr(server, "_model_path", None)
    monkeypatch.setattr(server, "_model_name", body["model"])
    monkeypatch.setattr(server, "_served_model_name", None)
    monkeypatch.setattr(server, "_reasoning_parser", reasoning_parser)
    monkeypatch.setattr(server, "_tool_call_parser", None)
    monkeypatch.setattr(server, "_mcp_manager", None)
    monkeypatch.setattr(server, "_api_key", None, raising=False)
    monkeypatch.setattr(server, "_default_enable_thinking", None)
    monkeypatch.setattr(server, "_default_temperature", 0.7)
    monkeypatch.setattr(server, "_default_top_p", 0.95)
    monkeypatch.setattr(server, "_default_top_k", None)
    monkeypatch.setattr(server, "_default_min_p", None)
    monkeypatch.setattr(server, "_default_repetition_penalty", None)
    monkeypatch.setattr(server, "_default_max_tokens", 64)
    monkeypatch.setattr(server, "_default_max_tokens_explicit", True, raising=False)
    monkeypatch.setattr(server, "_max_prompt_tokens", 0)
    monkeypatch.setattr(server, "stream_chat_completion", fake_stream_chat_completion)
    monkeypatch.setattr(
        "vmlx_engine.model_config_registry.get_model_config_registry",
        lambda: Registry(),
    )
    server._jang_sampling_defaults_cache.clear()
    server._generation_defaults_cache.clear()

    response = TestClient(server.app).post("/api/chat", json=body)

    assert response.status_code == 200
    assert '"done": true' in response.text or '"done":true' in response.text
    assert captured, "streaming route did not hand the request to generation"
    return captured


@pytest.mark.parametrize(
    ("think", "expected_mode"),
    [(True, "enabled"), (False, "disabled")],
)
def test_ollama_streaming_normalizes_minimax_m3_thinking_mode(
    monkeypatch,
    think,
    expected_mode,
):
    captured = _run_streaming_ollama_chat(
        monkeypatch,
        family_name="minimax_m3",
        model_type="minimax_m3",
        body={
            "model": "minimax-m3-test",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "think": think,
        },
    )

    kwargs = captured["kwargs"]
    assert kwargs["enable_thinking"] is think
    assert kwargs["chat_template_kwargs"] == {"thinking_mode": expected_mode}


@pytest.mark.parametrize("think", [True, False])
def test_ollama_streaming_normalizes_openpangu_thinking(monkeypatch, think):
    captured = _run_streaming_ollama_chat(
        monkeypatch,
        family_name="openpangu_v2",
        model_type="openpangu_v2",
        body={
            "model": "openpangu-test",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "think": think,
        },
    )

    kwargs = captured["kwargs"]
    assert kwargs["enable_thinking"] is think
    assert kwargs["chat_template_kwargs"] == {"thinking": think}


@pytest.mark.parametrize(
    ("think", "expected_effort"),
    [(True, "high"), (False, "none")],
)
def test_ollama_streaming_maps_mistral4_bool_to_reasoning_effort(
    monkeypatch,
    think,
    expected_effort,
):
    class MistralReasoningParser:
        pass

    captured = _run_streaming_ollama_chat(
        monkeypatch,
        family_name="mistral4",
        model_type="mistral4",
        reasoning_parser=MistralReasoningParser(),
        body={
            "model": "mistral4-test",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "think": think,
        },
    )

    kwargs = captured["kwargs"]
    assert kwargs["enable_thinking"] is think
    assert kwargs["chat_template_kwargs"]["reasoning_effort"] == expected_effort
    if think:
        assert kwargs["reasoning_effort"] == "high"
    else:
        assert "reasoning_effort" not in kwargs


def test_ollama_streaming_think_false_strips_historical_private_reasoning(
    monkeypatch,
):
    captured = _run_streaming_ollama_chat(
        monkeypatch,
        family_name="qwen3",
        model_type="qwen3",
        body={
            "model": "qwen-test",
            "messages": [
                {"role": "user", "content": "first"},
                {
                    "role": "assistant",
                    "thinking": "PRIVATE-PLAN-ONE",
                    "content": "visible answer",
                },
                {
                    "role": "assistant",
                    "thinking": "PRIVATE-PLAN-TWO",
                    "content": "",
                },
                {"role": "user", "content": "second"},
            ],
            "stream": True,
            "think": False,
        },
    )

    assert captured["messages"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "visible answer"},
        {"role": "user", "content": "second"},
    ]
    assert "PRIVATE-PLAN" not in repr(captured["messages"])


@pytest.mark.parametrize("preserve_native_tool_format", [True, False])
def test_ollama_streaming_respects_engine_native_tool_history_format(
    monkeypatch,
    preserve_native_tool_format,
):
    captured = _run_streaming_ollama_chat(
        monkeypatch,
        family_name="hunyuan_v1" if preserve_native_tool_format else "qwen3",
        model_type="hy_v3" if preserve_native_tool_format else "qwen3",
        preserve_native_tool_format=preserve_native_tool_format,
        body={
            "model": "tool-history-test",
            "messages": [
                {"role": "user", "content": "inspect the package"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "file_info",
                                "arguments": {"path": "panel/package.json"},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "Size: 5.2 KB",
                },
                {"role": "user", "content": "now run pwd"},
            ],
            "stream": True,
        },
    )

    messages = captured["messages"]
    if preserve_native_tool_format:
        assert messages[1]["role"] == "assistant"
        assert messages[1]["tool_calls"][0]["function"] == {
            "name": "file_info",
            "arguments": {"path": "panel/package.json"},
        }
        assert messages[2] == {
            "role": "tool",
            "content": "Size: 5.2 KB",
            "tool_call_id": "",
        }
    else:
        assert messages[1]["role"] == "assistant"
        assert "tool_calls" not in messages[1]
        assert "[Calling tool: file_info(" in messages[1]["content"]
        assert messages[2]["role"] == "user"
        assert "[Tool Result ()]" in messages[2]["content"]
        assert "Size: 5.2 KB" in messages[2]["content"]
