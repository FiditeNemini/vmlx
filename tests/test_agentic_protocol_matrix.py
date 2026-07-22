# SPDX-License-Identifier: Apache-2.0
"""Pure contracts for the reusable four-protocol agentic matrix runner."""

import copy
import json
from pathlib import Path

import pytest

from tests.cross_matrix import run_agentic_protocol_matrix as matrix


def _round(call_id: str = "call_1", name: str = "file_info", arguments=None):
    return {
        "content": "",
        "reasoning": "private",
        "tool_calls": [
            {
                "index": 0,
                "id": call_id,
                "name": name,
                "arguments": arguments or {"path": matrix.FILE_INFO_PATH},
            }
        ],
    }


def _execution(call_id: str = "call_1", name: str = "file_info"):
    arguments = (
        {"path": matrix.FILE_INFO_PATH}
        if name == "file_info"
        else {"command": matrix.PWD_COMMAND}
    )
    return {
        "name": name,
        "call_id": call_id,
        "arguments": arguments,
        "result": {"ok": True},
        "output": '{"ok":true}',
    }


def test_fragmented_tool_assembler_reconstructs_split_name_and_arguments():
    assembler = matrix.FragmentedToolAssembler()
    assembler.add(
        0,
        call_id="call_split",
        name="file_",
        arguments='{"path":"panel/',
    )
    assembler.add(0, name="info", arguments='package.json"}')

    assert assembler.calls() == [
        {
            "index": 0,
            "id": "call_split",
            "name": "file_info",
            "arguments": {"path": "panel/package.json"},
        }
    ]


def test_anthropic_empty_tool_start_does_not_mask_later_json_fragments():
    collector = matrix.EventCollector(protocol="anthropic", started=0.0)
    matrix._parse_stream_object(
        "anthropic",
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_split",
                "name": "file_info",
                "input": {},
            },
        },
        "content_block_start",
        collector,
        1.0,
    )
    matrix._parse_stream_object(
        "anthropic",
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {
                "type": "input_json_delta",
                "partial_json": '{"path":"panel/package.json"}',
            },
        },
        "content_block_delta",
        collector,
        2.0,
    )

    assert collector.tools.calls() == [
        {
            "index": 2,
            "id": "toolu_split",
            "name": "file_info",
            "arguments": {"path": "panel/package.json"},
        }
    ]


def test_responses_argument_delta_never_replaces_call_id_with_item_id():
    collector = matrix.EventCollector(protocol="responses", started=0.0)
    matrix._parse_stream_object(
        "responses",
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {
                "id": "fc_item",
                "type": "function_call",
                "call_id": "call_real",
                "name": "file_info",
                "arguments": "",
            },
        },
        "response.output_item.added",
        collector,
        1.0,
    )
    matrix._parse_stream_object(
        "responses",
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_item",
            "output_index": 1,
            "delta": '{"path":"panel/package.json"}',
        },
        "response.function_call_arguments.delta",
        collector,
        2.0,
    )
    matrix._parse_stream_object(
        "responses",
        {
            "type": "response.function_call_arguments.done",
            "item_id": "fc_item",
            "output_index": 1,
            "arguments": '{"path":"panel/package.json"}',
        },
        "response.function_call_arguments.done",
        collector,
        3.0,
    )
    assert collector.tools.calls()[0]["id"] == "call_real"


@pytest.mark.parametrize("protocol", ["chat", "anthropic", "ollama"])
def test_history_after_tool_preserves_native_call_result_adjacency(protocol):
    history = [{"role": "user", "content": "use the tool"}]
    updated = matrix.history_after_tool(
        protocol,
        history,
        _round(),
        _execution(),
        "continue",
    )

    assert updated[0] == history[0]
    assert updated[1]["role"] == "assistant"
    if protocol == "chat":
        assert updated[1]["tool_calls"][0]["id"] == "call_1"
        assert updated[2] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "file_info",
            "content": '{"ok":true}',
        }
    elif protocol == "anthropic":
        thinking = updated[1]["content"][0]
        tool_use = updated[1]["content"][1]
        tool_result = updated[2]["content"][0]
        assert thinking == {
            "type": "thinking",
            "thinking": "private",
            "signature": "dm1seA==",
        }
        assert tool_use["type"] == "tool_use"
        assert tool_use["id"] == "call_1"
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == "call_1"
    else:
        assert updated[1]["tool_calls"][0]["function"]["arguments"] == {
            "path": "panel/package.json"
        }
        assert updated[2] == {
            "role": "tool",
            "tool_name": "file_info",
            "content": '{"ok":true}',
        }


def test_responses_history_carries_real_output_then_latest_user_instruction():
    updated = matrix.history_after_tool(
        "responses",
        [],
        _round(),
        _execution(),
        "continue",
    )

    assert updated == [
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"ok":true}',
        },
        {"role": "user", "content": "continue"},
    ]


@pytest.mark.parametrize(
    ("protocol", "terminals", "stream", "expect_tool", "expected"),
    [
        ("chat", ["tool_calls", "DONE"], True, True, True),
        ("chat", ["stop", "DONE"], True, False, True),
        ("chat", ["stop", "DONE", "DONE"], True, False, False),
        ("responses", ["response.completed"], True, True, True),
        ("responses", ["response.incomplete"], True, False, False),
        ("anthropic", ["tool_use", "message_stop"], True, True, True),
        ("anthropic", ["end_turn", "message_stop"], True, False, True),
        ("ollama", ["tool_calls"], True, True, True),
        ("ollama", ["stop"], True, True, False),
        ("ollama", ["stop"], False, False, True),
    ],
)
def test_terminal_classification(protocol, terminals, stream, expect_tool, expected):
    result = matrix.classify_terminal(
        protocol,
        terminals,
        stream=stream,
        expect_tool=expect_tool,
    )

    assert result["pass"] is expected


@pytest.mark.parametrize(
    ("protocol", "mode", "payload", "expected"),
    [
        (
            "chat",
            "stream",
            {
                "delta_events_before_abort": 3,
                "cancel_status": 200,
                "terminals_before_abort": [],
                "idle_after_abort": {"idle": True},
            },
            True,
        ),
        (
            "responses",
            "stream",
            {
                "delta_events_before_abort": 3,
                "cancel_status": 404,
                "terminals_before_abort": [],
                "idle_after_abort": {"idle": True},
            },
            False,
        ),
        (
            "anthropic",
            "stream",
            {
                "delta_events_before_abort": 3,
                "cancel_status": None,
                "terminals_before_abort": [],
                "idle_after_abort": {"idle": True},
            },
            True,
        ),
        (
            "ollama",
            "stream",
            {
                "delta_events_before_abort": 3,
                "cancel_status": None,
                "terminals_before_abort": ["stop"],
                "idle_after_abort": {"idle": True},
            },
            False,
        ),
        (
            "chat",
            "nonstream",
            {"idle_after_disconnect": {"idle": True}},
            True,
        ),
    ],
)
def test_abort_classification_requires_real_cancel_and_no_false_terminal(
    protocol, mode, payload, expected
):
    assert matrix.classify_abort(protocol, mode, payload, 3)["pass"] is expected


def test_allowlist_rejects_path_command_and_extra_argument_variants():
    valid_file = {
        "id": "call_file",
        "name": "file_info",
        "arguments": {"path": "panel/package.json"},
    }
    valid_pwd = {
        "id": "call_pwd",
        "name": "run_command",
        "arguments": {"command": "pwd"},
    }
    assert matrix.validate_allowlisted_call(valid_file, "file_info") == (True, "")
    assert matrix.validate_allowlisted_call(valid_pwd, "run_command") == (True, "")

    for invalid, name in [
        ({**valid_file, "arguments": {"path": "pyproject.toml"}}, "file_info"),
        ({**valid_file, "arguments": {"path": "panel/package.json", "extra": 1}}, "file_info"),
        ({**valid_pwd, "arguments": {"command": "pwd; env"}}, "run_command"),
    ]:
        assert matrix.validate_allowlisted_call(invalid, name)[0] is False


def test_execute_allowlisted_tools_uses_real_repo_state(tmp_path: Path):
    package = tmp_path / "panel" / "package.json"
    package.parent.mkdir()
    package.write_text('{"name":"fixture"}\n')

    file_result = matrix.execute_allowlisted_tool(
        tmp_path,
        {
            "id": "call_file",
            "name": "file_info",
            "arguments": {"path": "panel/package.json"},
        },
    )
    pwd_result = matrix.execute_allowlisted_tool(
        tmp_path,
        {
            "id": "call_pwd",
            "name": "run_command",
            "arguments": {"command": "pwd"},
        },
    )

    assert file_result["result"]["size_bytes"] == package.stat().st_size
    assert file_result["result"]["path"] == "panel/package.json"
    assert pwd_result["result"] == {
        "command": "pwd",
        "stdout": str(tmp_path),
        "exit_code": 0,
    }
    assert matrix._human_size(5284) == "5.2 KB"


def test_sanitized_round_keeps_timing_and_hashes_but_drops_private_reasoning():
    private = "PRIVATE-REASONING-MUST-NOT-BE-SERIALIZED"
    row = {
        "status_code": 200,
        "elapsed_ms": 12.5,
        "response_id": "resp_safe",
        "reasoning": private,
        "content": "VISIBLE-DONE",
        "tool_calls": [],
        "terminals": ["response.completed"],
        "events": [
            {
                "at_ms": 4.0,
                "channel": "reasoning",
                "kind": "response.reasoning_summary_text.delta",
                "chars": len(private),
                "sha256": matrix._sha256(private),
            }
        ],
    }

    sanitized = matrix._sanitized_round(row)
    encoded = json.dumps(sanitized)
    assert private not in encoded
    assert sanitized["reasoning_chars"] == len(private)
    assert sanitized["reasoning_sha256"] == matrix._sha256(private)
    assert sanitized["events"][0]["at_ms"] == 4.0
    assert sanitized["content"] == "VISIBLE-DONE"


@pytest.mark.parametrize(
    "visible",
    [
        "<think>private</think>",
        '<tool_call>{"name":"file_info"}</tool_call>',
        "<function=file_info><parameter=path>panel/package.json",
        "<|tool_call|>file_info",
        "[TOOL_CALLS] file_info",
        "<｜tool▁calls▁begin｜>",
        '<｜DSML｜invoke name="file_info">',
        "<minimax:tool_call>",
        "```tool_code",
    ],
)
def test_visible_control_markup_rejects_reasoning_and_tool_protocol_leaks(visible):
    assert matrix._contains_control_markup(visible)


@pytest.mark.parametrize(
    ("protocol", "payload", "event_name", "error_kind"),
    [
        ("responses", {"type": "error", "error": {"message": "boom"}}, "error", "error"),
        ("anthropic", {"type": "error", "error": {"message": "boom"}}, "error", "error"),
        ("ollama", {"error": "boom"}, None, "ollama.error"),
    ],
)
def test_protocol_error_events_are_retained_as_fail_closed_evidence(
    protocol, payload, event_name, error_kind
):
    collector = matrix.EventCollector(protocol=protocol, started=0.0)
    matrix._parse_stream_object(protocol, payload, event_name, collector, 1.0)
    assert collector.errors[0]["kind"] == error_kind


def test_final_synthesis_instruction_does_not_leak_expected_result_values():
    prompt = matrix.final_synthesis_instruction("direct", "responses", "stream")
    assert "SIZE=<copy size_human" in prompt
    assert "PWD=<copy stdout" in prompt
    assert "5.2 KB" not in prompt
    assert "/Users/example" not in prompt


def test_direct_and_gateway_synthesis_prompts_are_byte_identical():
    direct = matrix.final_synthesis_instruction("direct", "anthropic", "stream")
    gateway = matrix.final_synthesis_instruction("gateway", "anthropic", "stream")

    assert direct == gateway
    assert "DIRECT" not in direct
    assert "GATEWAY" not in direct


def test_request_metadata_hashes_full_body_and_normalizes_only_tool_ids():
    base = {
        "model": "served-model",
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_one",
                        "type": "function",
                        "function": {"name": "file_info", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_one", "content": "ok"},
        ],
        "stream": True,
        "max_tokens": 512,
        "enable_thinking": True,
        "tool_choice": {"type": "function", "function": {"name": "run_command"}},
    }
    other = copy.deepcopy(base)
    other["messages"][0]["tool_calls"][0]["id"] = "call_two"
    other["messages"][1]["tool_call_id"] = "call_two"

    first = matrix._request_public(2, base)
    second = matrix._request_public(2, other)

    assert first["body_sha256"] != second["body_sha256"]
    assert first["canonical_body_sha256"] == second["canonical_body_sha256"]
    assert first["tool_choice"] == base["tool_choice"]
    assert first["max_output_tokens"] == 512


def test_first_tool_prompt_is_base_independent():
    prompt = matrix.first_tool_instruction("anthropic", "stream")

    assert prompt == matrix.first_tool_instruction("anthropic", "stream")
    assert "direct" not in prompt.lower()
    assert "gateway" not in prompt.lower()
    assert "agentic/" not in prompt.lower()


def test_build_request_uses_native_tool_choice_shapes_without_hosts():
    chat = matrix.build_request(
        "chat",
        "served-model",
        "stream",
        1,
        history=[{"role": "user", "content": "x"}],
        instructions="x",
    )
    responses = matrix.build_request(
        "responses",
        "served-model",
        "stream",
        1,
        history="x",
        instructions="x",
    )
    anthropic = matrix.build_request(
        "anthropic",
        "served-model",
        "nonstream",
        1,
        history=[{"role": "user", "content": "x"}],
        instructions="x",
    )
    ollama = matrix.build_request(
        "ollama",
        "served-model",
        "stream",
        1,
        history=[{"role": "user", "content": "x"}],
        instructions="x",
    )

    assert chat["tool_choice"] == {
        "type": "function",
        "function": {"name": "file_info"},
    }
    assert responses["tool_choice"] == {"type": "function", "name": "file_info"}
    assert anthropic["tool_choice"] == {"type": "any"}
    assert "tool_choice" not in ollama
    assert [tool["function"]["name"] for tool in ollama["tools"]] == ["file_info"]

    assert matrix.tool_choice("chat", "stream", 2, "explicit") == {
        "type": "function",
        "function": {"name": "run_command"},
    }
    assert matrix.tool_choice("responses", "stream", 2, "explicit") == {
        "type": "function",
        "name": "run_command",
    }
    assert matrix.tool_choice("anthropic", "stream", 2, "explicit") == {
        "type": "tool",
        "name": "run_command",
    }


def test_import_safe_parser_requires_caller_supplied_model_and_base(tmp_path: Path):
    parser = matrix.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--output", str(tmp_path / "out.json")])

    args = parser.parse_args(
        [
            "--base-url",
            "direct=http://127.0.0.1:8000",
            "--base-url",
            "gateway=http://127.0.0.1:8088",
            "--model",
            "served-model",
            "--output",
            str(tmp_path / "out.json"),
        ]
    )
    assert args.model == "served-model"
    assert args.base_url == [
        "direct=http://127.0.0.1:8000",
        "gateway=http://127.0.0.1:8088",
    ]
