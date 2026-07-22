# SPDX-License-Identifier: Apache-2.0
"""Ollama adapter parity tests."""

from __future__ import annotations

import json


def test_ollama_generate_default_uses_chat_template_request_shape():
    from vmlx_engine.api.ollama_adapter import ollama_generate_to_openai_chat

    req = ollama_generate_to_openai_chat(
        {
            "model": "zaya",
            "system": "Be terse.",
            "prompt": "What is the capital of France?",
            "stream": False,
            "format": "json",
            "options": {
                "num_predict": 16,
                "temperature": 0,
                "top_p": 1,
                "top_k": 40,
                "min_p": 0.02,
                "repeat_penalty": 1.1,
            },
        }
    )

    assert req["messages"] == [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    assert req["stream"] is False
    assert req["max_tokens"] == 16
    assert req["temperature"] == 0
    assert req["top_p"] == 1
    assert req["top_k"] == 40
    assert req["min_p"] == 0.02
    assert req["repetition_penalty"] == 1.1
    assert "enable_thinking" not in req
    assert req["response_format"] == {"type": "json_object"}


def test_ollama_chat_omits_disabled_top_k_sentinels():
    from vmlx_engine.api.ollama_adapter import ollama_chat_to_openai

    for sentinel in (0, -1):
        req = ollama_chat_to_openai(
            {
                "model": "hy3",
                "messages": [{"role": "user", "content": "hi"}],
                "options": {"top_k": sentinel},
            }
        )
        assert "top_k" not in req


def test_ollama_chat_translates_video_and_audio_extensions():
    from vmlx_engine.api.ollama_adapter import ollama_chat_to_openai

    req = ollama_chat_to_openai(
        {
            "model": "omni",
            "messages": [{
                "role": "user",
                "content": "Inspect both.",
                "videos": ["AAAAIGZ0eXA="],
                "audio": "SUQz",
            }],
        }
    )

    assert req["messages"] == [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Inspect both."},
            {"type": "video_url", "video_url": {
                "url": "data:video/mp4;base64,AAAAIGZ0eXA=",
            }},
            {"type": "audio_url", "audio_url": {
                "url": "data:audio/wav;base64,SUQz",
            }},
        ],
    }]


def test_ollama_generate_omits_disabled_top_k_sentinels():
    from vmlx_engine.api.ollama_adapter import (
        ollama_generate_to_openai,
        ollama_generate_to_openai_chat,
    )

    for convert in (ollama_generate_to_openai, ollama_generate_to_openai_chat):
        for sentinel in (0, -1):
            req = convert(
                {
                    "model": "hy3",
                    "prompt": "hi",
                    "options": {"top_k": sentinel},
                }
            )
            assert "top_k" not in req


def test_ollama_chat_omits_non_positive_num_predict_sentinels():
    from vmlx_engine.api.ollama_adapter import ollama_chat_to_openai

    for sentinel in (0, -1, -2):
        req = ollama_chat_to_openai(
            {
                "model": "hy3",
                "messages": [{"role": "user", "content": "hi"}],
                "options": {"num_predict": sentinel},
            }
        )
        assert "max_tokens" not in req


def test_ollama_generate_omits_non_positive_num_predict_sentinels():
    from vmlx_engine.api.ollama_adapter import (
        ollama_generate_to_openai,
        ollama_generate_to_openai_chat,
    )

    for convert in (ollama_generate_to_openai, ollama_generate_to_openai_chat):
        for sentinel in (0, -1, -2):
            req = convert(
                {
                    "model": "hy3",
                    "prompt": "hi",
                    "options": {"num_predict": sentinel},
                }
            )
            assert "max_tokens" not in req


def test_ollama_chat_omits_enable_thinking_when_think_is_omitted():
    from vmlx_engine.api.ollama_adapter import ollama_chat_to_openai

    req = ollama_chat_to_openai(
        {
            "model": "zaya",
            "messages": [{"role": "user", "content": "hi"}],
        }
    )

    assert "enable_thinking" not in req


def test_ollama_chat_accepts_enable_thinking_extension():
    from vmlx_engine.api.ollama_adapter import ollama_chat_to_openai

    req = ollama_chat_to_openai(
        {
            "model": "zaya",
            "messages": [{"role": "user", "content": "hi"}],
            "enable_thinking": False,
        }
    )

    assert req["enable_thinking"] is False


def test_ollama_native_think_beats_enable_thinking_extension():
    from vmlx_engine.api.ollama_adapter import ollama_chat_to_openai

    req = ollama_chat_to_openai(
        {
            "model": "zaya",
            "messages": [{"role": "user", "content": "hi"}],
            "think": True,
            "enable_thinking": False,
        }
    )

    assert req["enable_thinking"] is True


def test_ollama_native_think_false_disables_reasoning():
    from vmlx_engine.api.ollama_adapter import ollama_chat_to_openai

    req = ollama_chat_to_openai(
        {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hi"}],
            "think": False,
        }
    )

    assert req["enable_thinking"] is False


def test_ollama_chat_drops_reasoning_effort_when_native_think_false():
    from vmlx_engine.api.ollama_adapter import ollama_chat_to_openai

    req = ollama_chat_to_openai(
        {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hi"}],
            "think": False,
            "reasoning_effort": "max",
        }
    )

    assert req["enable_thinking"] is False
    assert "reasoning_effort" not in req


def test_ollama_generate_chat_native_think_false_disables_reasoning():
    from vmlx_engine.api.ollama_adapter import ollama_generate_to_openai_chat

    req = ollama_generate_to_openai_chat(
        {"model": "qwen", "prompt": "hi", "think": False}
    )

    assert req["enable_thinking"] is False


def test_ollama_generate_chat_accepts_enable_thinking_extension():
    from vmlx_engine.api.ollama_adapter import ollama_generate_to_openai_chat

    req = ollama_generate_to_openai_chat(
        {"model": "zaya", "prompt": "hi", "enable_thinking": False}
    )

    assert req["enable_thinking"] is False


def test_ollama_generate_chat_drops_reasoning_effort_when_template_kwargs_disable_thinking():
    from vmlx_engine.api.ollama_adapter import ollama_generate_to_openai_chat

    req = ollama_generate_to_openai_chat(
        {
            "model": "qwen",
            "prompt": "hi",
            "reasoning_effort": "high",
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )

    assert req["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_effort" not in req


def test_ollama_generate_raw_keeps_completion_request_shape():
    from vmlx_engine.api.ollama_adapter import ollama_generate_to_openai

    req = ollama_generate_to_openai(
        {
            "model": "base",
            "prompt": "raw text",
            "stream": False,
            "options": {
                "num_predict": 4,
                "temperature": 0,
                "top_p": 0.9,
                "top_k": 20,
                "min_p": 0.01,
                "repeat_penalty": 1.05,
            },
        }
    )

    assert req["prompt"] == "raw text"
    assert "messages" not in req
    assert req["max_tokens"] == 4
    assert req["temperature"] == 0
    assert req["top_p"] == 0.9
    assert req["top_k"] == 20
    assert req["min_p"] == 0.01
    assert req["repetition_penalty"] == 1.05


def test_chat_response_converts_to_ollama_generate_shape():
    from vmlx_engine.api.ollama_adapter import (
        openai_chat_response_to_ollama_generate,
    )

    out = openai_chat_response_to_ollama_generate(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "Paris",
                        "reasoning_content": "I know this.",
                    },
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        },
        "zaya",
    )

    assert out["response"] == "Paris"
    assert out["thinking"] == "I know this."
    assert out["done"] is True
    assert out["done_reason"] == "stop"
    assert out["prompt_eval_count"] == 3
    assert out["eval_count"] == 2


def test_chat_stream_chunk_converts_to_ollama_generate_ndjson():
    from vmlx_engine.api.ollama_adapter import (
        openai_chat_chunk_to_ollama_generate_ndjson,
    )

    line = "data: " + json.dumps(
        {
            "choices": [
                {
                    "delta": {"content": "Pa"},
                    "finish_reason": None,
                }
            ]
        }
    )

    out = json.loads(openai_chat_chunk_to_ollama_generate_ndjson(line, "zaya"))

    assert out["response"] == "Pa"
    assert out["done"] is False


def test_ollama_chat_preserves_two_tool_results_for_continuation():
    from vmlx_engine.api.ollama_adapter import ollama_chat_to_openai

    messages = [
        {"role": "user", "content": "use both tools"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "file_info",
                        "arguments": {"path": "panel/package.json"},
                    }
                },
                {
                    "function": {
                        "name": "run_command",
                        "arguments": {"command": "pwd"},
                    }
                },
            ],
        },
        {"role": "tool", "tool_name": "file_info", "content": "Size: 5.2 KB"},
        {
            "role": "tool",
            "tool_name": "run_command",
            "content": "/Users/example/mlx/vllm-mlx",
        },
    ]

    converted = ollama_chat_to_openai(
        {"model": "minimax", "messages": messages, "tools": [{"type": "function"}]}
    )

    assert converted["messages"] == messages
    assert converted["tools"] == [{"type": "function"}]


def test_ollama_chat_preserves_prior_thinking_privately_for_tool_continuation():
    from vmlx_engine.api.ollama_adapter import ollama_chat_to_openai

    converted = ollama_chat_to_openai(
        {
            "model": "laguna",
            "messages": [
                {"role": "user", "content": "inspect the file"},
                {
                    "role": "assistant",
                    "thinking": "I should call file_info.",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "file_info",
                                "arguments": {"path": "panel/package.json"},
                            }
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_name": "file_info",
                    "content": "Size: 5.2 KB",
                },
            ],
        }
    )

    assistant = converted["messages"][1]
    assert "thinking" not in assistant
    assert assistant["reasoning_content"] == "I should call file_info."
    assert assistant["content"] == ""


def test_ollama_stream_terminal_preserves_two_object_argument_tool_calls():
    from vmlx_engine.api.ollama_adapter import openai_chat_chunk_to_ollama_ndjson

    line = "data: " + json.dumps(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_file",
                                "function": {
                                    "name": "file_info",
                                    "arguments": '{"path":"panel/package.json"}',
                                },
                            },
                            {
                                "index": 1,
                                "id": "call_pwd",
                                "function": {
                                    "name": "run_command",
                                    "arguments": '{"command":"pwd"}',
                                },
                            },
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 304, "completion_tokens": 67},
        }
    )

    out = json.loads(openai_chat_chunk_to_ollama_ndjson(line, "minimax"))

    assert out["done"] is True
    assert out["done_reason"] == "tool_calls"
    assert out["prompt_eval_count"] == 304
    assert out["eval_count"] == 67
    assert out["message"]["tool_calls"] == [
        {
            "function": {
                "name": "file_info",
                "arguments": {"path": "panel/package.json"},
            }
        },
        {"function": {"name": "run_command", "arguments": {"command": "pwd"}}},
    ]


def test_hy3_answer_pass_streams_final_text_as_ollama_content():
    """Hy3's bounded retry must leave the high-effort thinking rail.

    If ``reasoning_effort=high`` survives from the first pass, the parser sends
    the final answer as ``reasoning_content`` and Ollama misroutes it to
    ``message.thinking``.  The direct-rail retry produces a content delta; the
    adapter must preserve that channel while keeping genuine reasoning separate.
    """
    from vmlx_engine import server
    from vmlx_engine.api.ollama_adapter import openai_chat_chunk_to_ollama_ndjson

    answer_kwargs = {
        "enable_thinking": True,
        "reasoning_effort": "high",
        "chat_template_kwargs": {"reasoning_effort": "high"},
    }
    assert "hy_v3" in server._REASONING_ANSWER_PASS_FAMILIES
    server._force_answer_pass_direct_rail(
        answer_kwargs,
        family_name="hy_v3",
    )
    assert answer_kwargs == {
        "enable_thinking": False,
        "reasoning_effort": "no_think",
        "chat_template_kwargs": {"reasoning_effort": "no_think"},
    }

    reasoning_line = "data: " + json.dumps(
        {
            "choices": [
                {
                    "delta": {"reasoning_content": "internal rail"},
                    "finish_reason": None,
                }
            ]
        }
    )
    answer_line = "data: " + json.dumps(
        {
            "choices": [
                {
                    "delta": {"content": "FINAL-CHECK"},
                    "finish_reason": None,
                }
            ]
        }
    )
    reasoning = json.loads(openai_chat_chunk_to_ollama_ndjson(reasoning_line, "hy3"))
    answer = json.loads(openai_chat_chunk_to_ollama_ndjson(answer_line, "hy3"))

    assert reasoning["message"] == {
        "role": "assistant",
        "content": "",
        "thinking": "internal rail",
    }
    assert answer["message"] == {
        "role": "assistant",
        "content": "FINAL-CHECK",
    }


def test_ollama_terminal_merge_defers_length_until_after_answer_and_usage():
    from vmlx_engine.api.ollama_adapter import merge_ollama_stream_terminal

    provisional = {
        "model": "hy3",
        "message": {"role": "assistant", "content": "", "thinking": "reason"},
        "done": True,
        "done_reason": "length",
    }
    answer_stop = {
        "model": "hy3",
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "stop",
    }
    usage = {
        "model": "hy3",
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "stop",
        "eval_count": 428,
        "prompt_eval_count": 97,
    }

    merged = merge_ollama_stream_terminal(provisional, answer_stop)
    merged = merge_ollama_stream_terminal(merged, usage)

    assert merged["done"] is True
    assert merged["done_reason"] == "stop"
    assert merged["eval_count"] == 428
    assert merged["prompt_eval_count"] == 97
    assert merged["message"]["thinking"] == "reason"


def test_ollama_terminal_merge_preserves_truncated_answer_after_usage():
    from vmlx_engine.api.ollama_adapter import merge_ollama_stream_terminal

    answer_length = {
        "model": "laguna",
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "length",
    }
    usage = {
        "model": "laguna",
        "message": {"role": "assistant", "content": ""},
        "done": True,
        # Adapter-created usage terminals default to stop; they are accounting,
        # not a second model finish event.
        "done_reason": "stop",
        "eval_count": 560,
        "prompt_eval_count": 73,
    }

    merged = merge_ollama_stream_terminal(answer_length, usage)

    assert merged["done"] is True
    assert merged["done_reason"] == "length"
    assert merged["eval_count"] == 560
    assert merged["prompt_eval_count"] == 73


def test_ollama_generate_terminal_merge_retains_usage_on_single_done_row():
    from vmlx_engine.api.ollama_adapter import (
        merge_ollama_generate_stream_terminal,
    )

    finish = {
        "model": "minimax",
        "created_at": "2026-07-19T00:00:00.000Z",
        "response": "",
        "done": True,
        "done_reason": "stop",
    }
    usage = {
        "model": "minimax",
        "created_at": "2026-07-19T00:00:01.000Z",
        "response": "",
        "done": True,
        "done_reason": "stop",
        "eval_count": 215,
        "prompt_eval_count": 78,
    }

    merged = merge_ollama_generate_stream_terminal(finish, usage)

    assert merged["done"] is True
    assert merged["done_reason"] == "stop"
    assert merged["eval_count"] == 215
    assert merged["prompt_eval_count"] == 78


def test_ollama_generate_terminal_merge_preserves_length_after_usage():
    from vmlx_engine.api.ollama_adapter import (
        merge_ollama_generate_stream_terminal,
    )

    finish = {
        "model": "laguna",
        "created_at": "2026-07-22T00:00:00.000Z",
        "response": "",
        "done": True,
        "done_reason": "length",
    }
    usage = {
        "model": "laguna",
        "created_at": "2026-07-22T00:00:01.000Z",
        "response": "",
        "done": True,
        "done_reason": "stop",
        "eval_count": 560,
        "prompt_eval_count": 73,
    }

    merged = merge_ollama_generate_stream_terminal(finish, usage)

    assert merged["done"] is True
    assert merged["done_reason"] == "length"
    assert merged["eval_count"] == 560
    assert merged["prompt_eval_count"] == 73


def test_ollama_chat_stream_maps_upstream_error_to_native_error_row():
    from vmlx_engine.api.ollama_adapter import openai_chat_chunk_to_ollama_ndjson

    line = "data: " + json.dumps(
        {
            "error": {
                "type": "server_error",
                "message": "CHAT MIDSTREAM PROBE FAILURE",
                "code": "internal_error",
            }
        }
    )

    assert json.loads(openai_chat_chunk_to_ollama_ndjson(line, "probe")) == {
        "error": "CHAT MIDSTREAM PROBE FAILURE"
    }


def test_ollama_templated_generate_stream_preserves_native_error_row():
    from vmlx_engine.api.ollama_adapter import (
        openai_chat_chunk_to_ollama_generate_ndjson,
    )

    line = "data: " + json.dumps(
        {"error": {"type": "server_error", "message": "TEMPLATE FAILURE"}}
    )

    assert json.loads(
        openai_chat_chunk_to_ollama_generate_ndjson(line, "probe")
    ) == {"error": "TEMPLATE FAILURE"}


def test_ollama_raw_generate_stream_maps_upstream_error_to_native_error_row():
    from vmlx_engine.api.ollama_adapter import (
        openai_completion_chunk_to_ollama_ndjson,
    )

    line = "data: " + json.dumps(
        {"error": {"type": "server_error", "message": "RAW FAILURE"}}
    )

    assert json.loads(openai_completion_chunk_to_ollama_ndjson(line, "probe")) == {
        "error": "RAW FAILURE"
    }
