# SPDX-License-Identifier: Apache-2.0
"""Pure contracts for the reusable four-protocol agentic matrix runner."""

import copy
import hashlib
import json
import stat
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

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
        (
            {**valid_file, "arguments": {"path": "panel/package.json", "extra": 1}},
            "file_info",
        ),
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
        (
            "responses",
            {"type": "error", "error": {"message": "boom"}},
            "error",
            "error",
        ),
        (
            "anthropic",
            {"type": "error", "error": {"message": "boom"}},
            "error",
            "error",
        ),
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
    assert args.raw_artifact_dir is None


def test_raw_capture_rejects_worktree_and_symlink_escape(
    tmp_path: Path,
):
    worktree = tmp_path / "repo"
    worktree.mkdir()

    with pytest.raises(ValueError, match="outside every Git worktree"):
        matrix.DecompressedParserInputCaptureRecorder(
            worktree / "captures",
            worktree,
            run_id="rejected",
        )
    assert not (worktree / "captures").exists()

    link = tmp_path / "repo-link"
    link.symlink_to(worktree, target_is_directory=True)
    with pytest.raises(ValueError, match="outside every Git worktree"):
        matrix.DecompressedParserInputCaptureRecorder(
            link / "captures",
            worktree,
            run_id="rejected-link",
        )


def test_raw_capture_rejects_apfs_case_alias_and_sibling_git_worktree(
    tmp_path: Path,
):
    worktree = tmp_path / "GuardedRepo"
    worktree.mkdir()
    case_alias = tmp_path / "guardedrepo"
    if case_alias.exists() and case_alias.samefile(worktree):
        with pytest.raises(ValueError, match="outside every Git worktree"):
            matrix.DecompressedParserInputCaptureRecorder(
                case_alias / "captures",
                worktree,
                run_id="rejected-case-alias",
            )
        assert not (worktree / "captures").exists()

    sibling_repo = tmp_path / "sibling-public-repo"
    subprocess.run(
        ["git", "init", "-q", str(sibling_repo)],
        check=True,
        capture_output=True,
    )
    with pytest.raises(ValueError, match="outside every Git worktree"):
        matrix.DecompressedParserInputCaptureRecorder(
            sibling_repo / "captures",
            worktree,
            run_id="rejected-sibling",
        )
    assert not (sibling_repo / "captures").exists()


def test_opt_in_capture_preserves_parser_input_bytes_and_allowlists_metadata(
    monkeypatch,
    tmp_path: Path,
):
    secret = "SECRET-AUTH-VALUE"
    worktree = tmp_path / "repo"
    worktree.mkdir()
    capture_root = tmp_path / "private-captures"
    response_body = (
        b'data: {"id":"chat_1","choices":[{"delta":'
        b'{"reasoning_content":"private"},"finish_reason":null}]}\r\n\r\n'
        b'data: {"id":"chat_1","choices":[{"delta":'
        b'{"content":"$43 \\\\times 2"},"finish_reason":"stop"}]}\r\n\r\n'
        b"data: [DONE]\r\n\r\n"
    )

    class FakeRaw:
        def __init__(self):
            self.headers = {
                "Content-Type": "text/event-stream",
                "Set-Cookie": f"session={secret}",
                "X-Trace-Id": "trace-safe",
                "Location": f"https://example.invalid/callback?token={secret}",
            }
            self.closed = False

        def stream(self, _amount, decode_content=None):
            assert decode_content is True
            yield response_body[:37]
            yield response_body[37:91]
            yield response_body[91:]

        def close(self):
            self.closed = True

    payload = {
        "model": "served-model",
        "messages": [{"role": "user", "content": "private prompt"}],
        "stream": True,
        "max_tokens": 32,
    }
    prepared = requests.Request(
        "POST",
        "http://user:password@127.0.0.1:8000/v1/chat/completions?key=secret",
        headers={
            "Authorization": f"Bearer {secret}",
            "X-Api-Key": secret,
            "Content-Type": "application/json",
        },
        json=payload,
    ).prepare()
    response = requests.Response()
    response.status_code = 200
    response.reason = "OK"
    response.raw = FakeRaw()
    response.request = prepared
    response.encoding = "utf-8"

    def fake_post(*_args, **_kwargs):
        return response

    monkeypatch.setattr(matrix.requests, "post", fake_post)
    recorder = matrix.DecompressedParserInputCaptureRecorder(
        capture_root,
        worktree,
        run_id="unit-test",
    )
    recorder.configure_expected([("direct", "chat", "stream-round1")])
    client = matrix.ProtocolClient(
        "http://127.0.0.1:8000",
        secret,
        30,
        base_label="direct",
        raw_recorder=recorder,
    )

    result = client.send(
        "chat",
        payload,
        True,
        capture_label="stream-round1",
    )

    assert result["reasoning"] == "private"
    assert result["content"] == "$43 \\times 2"
    capture_summary = recorder.finalize()
    body_path = next(recorder.run_dir.glob("*.decompressed-parser-input.bin"))
    metadata_path = next(recorder.run_dir.glob("*.metadata.json"))
    assert body_path.read_bytes() == response_body
    metadata_text = metadata_path.read_text()
    assert secret not in metadata_text
    assert "password" not in metadata_text
    assert "?key=secret" not in metadata_text

    metadata = json.loads(metadata_text)
    assert metadata["capture_layer"] == matrix.CAPTURE_LAYER
    assert metadata["capture_semantics"] == matrix.CAPTURE_SEMANTICS
    assert metadata["request"]["url"] == ("http://127.0.0.1:8000/v1/chat/completions")
    assert metadata["request"]["body_bytes"] == len(prepared.body)
    assert (
        metadata["request"]["body_sha256"] == hashlib.sha256(prepared.body).hexdigest()
    )
    assert metadata["request"]["payload"]["top_level_fields"] == [
        "max_tokens",
        "messages",
        "model",
        "stream",
    ]
    assert {
        row["name"].lower(): row["value"] for row in metadata["request"]["headers"]
    }["authorization"] == "<redacted>"
    assert {
        row["name"].lower(): row["value"] for row in metadata["response"]["headers"]
    }["set-cookie"] == "<redacted>"
    assert {
        row["name"].lower(): row["value"] for row in metadata["response"]["headers"]
    }["location"] == "<redacted>"
    assert {
        row["name"].lower(): row["value"] for row in metadata["response"]["headers"]
    }["x-trace-id"] == "<redacted>"
    assert {
        row["name"].lower(): row["value"] for row in metadata["response"]["headers"]
    }["content-type"] == "text/event-stream"
    assert metadata["response"]["status_code"] == 200
    assert metadata["response"]["body_bytes"] == len(response_body)
    assert (
        metadata["response"]["body_sha256"] == hashlib.sha256(response_body).hexdigest()
    )
    assert metadata["response"]["first_byte_ms"] is not None
    assert (
        metadata["response"]["completed_ms"] >= (metadata["response"]["first_byte_ms"])
    )
    assert stat.S_IMODE(body_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(metadata_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(recorder.manifest_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(recorder.run_dir.stat().st_mode) == 0o700
    assert capture_summary["expected"] == 1
    assert capture_summary["started"] == 1
    assert capture_summary["finished"] == 1
    assert capture_summary["errors"] == 0
    assert capture_summary["complete"] is True


@pytest.mark.parametrize("base_label", ["direct", "gateway"])
@pytest.mark.parametrize(
    ("protocol", "response_body"),
    [
        (
            "chat",
            b'data: {"choices":[{"delta":{"content":"ok"},'
            b'"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
        ),
        (
            "responses",
            b"event: response.output_text.delta\n"
            b'data: {"type":"response.output_text.delta","delta":"ok"}\n\n'
            b"event: response.completed\n"
            b'data: {"type":"response.completed","response":{"id":"r1"}}\n\n',
        ),
        (
            "anthropic",
            b"event: content_block_delta\n"
            b'data: {"type":"content_block_delta","index":0,"delta":'
            b'{"type":"text_delta","text":"ok"}}\n\n'
            b"event: message_delta\n"
            b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n\n'
            b"event: message_stop\n"
            b'data: {"type":"message_stop"}\n\n',
        ),
        (
            "ollama",
            b'{"message":{"content":"ok"},"done":true,"done_reason":"stop"}\n',
        ),
    ],
)
def test_parser_input_capture_covers_each_protocol_and_base(
    monkeypatch,
    tmp_path: Path,
    base_label: str,
    protocol: str,
    response_body: bytes,
):
    worktree = tmp_path / "repo"
    worktree.mkdir()

    class FakeRaw:
        headers = {"Content-Type": "application/octet-stream"}

        def stream(self, _amount, decode_content=None):
            assert decode_content is True
            midpoint = len(response_body) // 2
            yield response_body[:midpoint]
            yield response_body[midpoint:]

        def close(self):
            return None

    payload = {"model": "served-model", "stream": True}
    prepared = requests.Request(
        "POST",
        f"http://127.0.0.1:8000{matrix.ProtocolClient.route(protocol)}",
        json=payload,
    ).prepare()
    response = requests.Response()
    response.status_code = 200
    response.reason = "OK"
    response.raw = FakeRaw()
    response.request = prepared
    response.encoding = "utf-8"
    monkeypatch.setattr(matrix.requests, "post", lambda *_args, **_kwargs: response)

    recorder = matrix.DecompressedParserInputCaptureRecorder(
        tmp_path / "private-captures",
        worktree,
        run_id=f"{base_label}-{protocol}",
    )
    recorder.configure_expected([(base_label, protocol, "stream-flow-round1")])
    client = matrix.ProtocolClient(
        "http://127.0.0.1:8000",
        None,
        30,
        base_label=base_label,
        raw_recorder=recorder,
    )

    result = client.send(
        protocol,
        payload,
        True,
        capture_label="stream-flow-round1",
    )

    assert result["content"] == "ok"
    summary = recorder.finalize()
    body_path = next(recorder.run_dir.glob("*.decompressed-parser-input.bin"))
    metadata_path = next(recorder.run_dir.glob("*.metadata.json"))
    assert body_path.read_bytes() == response_body
    metadata = json.loads(metadata_path.read_text())
    assert metadata["base_label"] == base_label
    assert metadata["protocol"] == protocol
    assert metadata["request"]["payload"]["model"] == "served-model"
    assert summary["complete"] is True


def test_capture_setup_failure_closes_response(monkeypatch):
    class FailingRecorder:
        def begin(self, **_kwargs):
            raise OSError("capture unavailable")

    response = requests.Response()
    response.status_code = 200
    response.raw = SimpleNamespace()
    response.request = requests.Request(
        "POST",
        "http://127.0.0.1:8000/v1/chat/completions",
        json={"model": "served-model", "stream": True},
    ).prepare()
    closed = False

    def close_response():
        nonlocal closed
        closed = True

    response.close = close_response
    monkeypatch.setattr(
        matrix.requests,
        "post",
        lambda *_args, **_kwargs: response,
    )
    client = matrix.ProtocolClient(
        "http://127.0.0.1:8000",
        None,
        30,
        raw_recorder=FailingRecorder(),
    )

    with pytest.raises(OSError, match="capture unavailable"):
        client.send(
            "chat",
            {"model": "served-model", "stream": True},
            True,
        )
    assert closed is True


def test_abort_capture_setup_failure_closes_response(monkeypatch):
    class FailingRecorder:
        def begin(self, **_kwargs):
            raise OSError("capture unavailable")

    response = requests.Response()
    response.status_code = 200
    response.raw = SimpleNamespace()
    response.request = requests.Request(
        "POST",
        "http://127.0.0.1:8000/v1/chat/completions",
        json={"model": "served-model", "stream": True},
    ).prepare()
    closed = False

    def close_response():
        nonlocal closed
        closed = True

    response.close = close_response
    monkeypatch.setattr(
        matrix.requests,
        "post",
        lambda *_args, **_kwargs: response,
    )
    client = matrix.ProtocolClient(
        "http://127.0.0.1:8000",
        None,
        30,
        raw_recorder=FailingRecorder(),
    )

    with pytest.raises(OSError, match="capture unavailable"):
        matrix.abort_stream_after_deltas(
            client,
            "chat",
            {
                "model": "served-model",
                "messages": [{"role": "user", "content": "final"}],
                "stream": True,
            },
            health_url="http://127.0.0.1:8000/health",
            minimum_deltas=1,
        )
    assert closed is True


def test_capture_session_setup_failure_removes_partial_body(tmp_path: Path):
    class ExplodingHeaders:
        def items(self):
            raise RuntimeError("header enumeration failed")

    worktree = tmp_path / "repo"
    worktree.mkdir()
    recorder = matrix.DecompressedParserInputCaptureRecorder(
        tmp_path / "private-captures",
        worktree,
        run_id="setup-failure",
    )
    recorder.configure_expected([("direct", "chat", "stream-round1")])
    response = requests.Response()
    response.status_code = 200
    response.raw = SimpleNamespace(headers=ExplodingHeaders())
    response.request = requests.Request(
        "POST",
        "http://127.0.0.1:8000/v1/chat/completions",
        json={"model": "served-model", "stream": True},
    ).prepare()

    with pytest.raises(RuntimeError, match="header enumeration failed"):
        recorder.begin(
            base_label="direct",
            protocol="chat",
            capture_label="stream-round1",
            payload={"model": "served-model", "stream": True},
            response=response,
            started=time.monotonic(),
            started_at="2026-07-24T00:00:00+00:00",
        )

    assert not list(recorder.run_dir.glob("*.decompressed-parser-input.bin"))
    assert not list(recorder.run_dir.glob("*.metadata.json"))
    summary = recorder.finalize()
    assert summary["started"] == 1
    assert summary["finished"] == 0
    assert summary["errors"] == 1
    assert summary["complete"] is False


def test_capture_finish_failure_removes_body_and_partial_metadata(
    monkeypatch,
    tmp_path: Path,
):
    worktree = tmp_path / "repo"
    worktree.mkdir()
    recorder = matrix.DecompressedParserInputCaptureRecorder(
        tmp_path / "private-captures",
        worktree,
        run_id="finish-failure",
    )
    recorder.configure_expected([("direct", "chat", "stream-round1")])
    response = requests.Response()
    response.status_code = 200
    response.raw = SimpleNamespace(headers={"Content-Type": "text/event-stream"})
    response.request = requests.Request(
        "POST",
        "http://127.0.0.1:8000/v1/chat/completions",
        json={"model": "served-model", "stream": True},
    ).prepare()
    capture = recorder.begin(
        base_label="direct",
        protocol="chat",
        capture_label="stream-round1",
        payload={"model": "served-model", "stream": True},
        response=response,
        started=time.monotonic(),
        started_at="2026-07-24T00:00:00+00:00",
    )
    capture.write(b"sensitive decompressed parser input")
    artifact = recorder.current_summary()["routes"][0]["artifacts"][0]
    body_path = recorder.run_dir / artifact["body_file"]
    metadata_path = recorder.run_dir / artifact["metadata_file"]
    real_fdopen = matrix.os.fdopen

    class FailingMetadataFile:
        def __init__(self, descriptor: int, mode: str) -> None:
            self._file = real_fdopen(descriptor, mode)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self._file.close()

        def write(self, data: bytes) -> None:
            self._file.write(data[:8])
            self._file.flush()
            raise OSError("metadata write failed")

    monkeypatch.setattr(matrix.os, "fdopen", FailingMetadataFile)
    with pytest.raises(OSError, match="metadata write failed"):
        capture.finish(completed_ms=1.0)

    assert not body_path.exists()
    assert not metadata_path.exists()
    monkeypatch.setattr(matrix.os, "fdopen", real_fdopen)
    summary = recorder.finalize()
    assert summary["started"] == 1
    assert summary["finished"] == 0
    assert summary["errors"] == 1
    assert summary["complete"] is False


def _synthetic_terminals(protocol: str, *, expect_tool: bool) -> list[str]:
    if protocol == "chat":
        return ["tool_calls" if expect_tool else "stop", "DONE"]
    if protocol == "responses":
        return ["response.completed"]
    if protocol == "anthropic":
        return ["tool_use" if expect_tool else "end_turn", "message_stop"]
    return ["tool_calls" if expect_tool else "stop"]


def _finish_synthetic_capture(
    client,
    protocol: str,
    capture_label: str,
    payload: dict,
    *,
    omitted_label: str | None,
) -> None:
    recorder = client.raw_recorder
    if recorder is None or capture_label == omitted_label:
        return
    response = requests.Response()
    response.status_code = 200
    response.raw = SimpleNamespace(headers={"Content-Type": "text/event-stream"})
    response.request = requests.Request(
        "POST",
        client.base_url + client.route(protocol),
        json=payload,
    ).prepare()
    capture = recorder.begin(
        base_label=client.base_label,
        protocol=protocol,
        capture_label=capture_label,
        payload=payload,
        response=response,
        started=time.monotonic(),
        started_at="2026-07-24T00:00:00+00:00",
    )
    capture.write(f"{client.base_label}/{protocol}/{capture_label}\n".encode())
    capture.finish(completed_ms=1.0)


@pytest.mark.parametrize(
    (
        "capture_enabled",
        "omitted_label",
        "expected_pass",
        "expected_started",
    ),
    [
        (True, None, True, 40),
        (True, "stream-flow-round2", False, 32),
        (False, None, True, 0),
    ],
)
def test_run_matrix_binds_full_stream_capture_manifest_into_pass(
    monkeypatch,
    tmp_path: Path,
    capture_enabled: bool,
    omitted_label: str | None,
    expected_pass: bool,
    expected_started: int,
):
    repo_root = tmp_path / "repo"
    package_json = repo_root / matrix.FILE_INFO_PATH
    package_json.parent.mkdir(parents=True)
    package_json.write_text("{}\n")
    subprocess.run(
        ["git", "init", "-q", str(repo_root)],
        check=True,
        capture_output=True,
    )
    raw_root = tmp_path / "private-captures"
    size_human = matrix._human_size(package_json.stat().st_size)

    def fake_send(
        client,
        protocol,
        payload,
        stream,
        *,
        capture_label="request",
    ):
        assert stream is True
        _finish_synthetic_capture(
            client,
            protocol,
            capture_label,
            payload,
            omitted_label=omitted_label,
        )
        response_id = f"{client.base_label}-{protocol}-{capture_label}"
        result = {
            "status_code": 200,
            "elapsed_ms": 1.0,
            "response_id": response_id,
            "reasoning": "",
            "content": "",
            "tool_calls": [],
            "terminals": [],
            "errors": [],
            "events": [],
        }
        if capture_label == "stream-flow-round1":
            result["tool_calls"] = [
                {
                    "index": 0,
                    "id": f"{response_id}-call",
                    "name": "file_info",
                    "arguments": {"path": matrix.FILE_INFO_PATH},
                }
            ]
            result["terminals"] = _synthetic_terminals(
                protocol,
                expect_tool=True,
            )
        elif capture_label == "stream-flow-round2":
            result["tool_calls"] = [
                {
                    "index": 0,
                    "id": f"{response_id}-call",
                    "name": "run_command",
                    "arguments": {"command": matrix.PWD_COMMAND},
                }
            ]
            result["terminals"] = _synthetic_terminals(
                protocol,
                expect_tool=True,
            )
        elif capture_label == "stream-flow-round3":
            result["content"] = (
                f"AGENTIC-{protocol.upper()}-STREAM-DONE "
                f"SIZE={size_human} PWD={repo_root}"
            )
            result["terminals"] = _synthetic_terminals(
                protocol,
                expect_tool=False,
            )
            result["events"] = [
                {
                    "at_ms": 1.0,
                    "channel": "content",
                    "kind": "synthetic.delta.1",
                },
                {
                    "at_ms": 2.0,
                    "channel": "content",
                    "kind": "synthetic.delta.2",
                },
            ]
        elif capture_label == "stream-recovery":
            result["content"] = (
                f"RECOVERY-{client.base_label.upper()}-{protocol.upper()}-STREAM-DONE"
            )
            result["terminals"] = _synthetic_terminals(
                protocol,
                expect_tool=False,
            )
        else:
            raise AssertionError(f"unexpected capture label: {capture_label}")

        if protocol == "responses" and capture_label in {
            "stream-flow-round1",
            "stream-flow-round2",
        }:
            result["events"] = [
                {
                    "at_ms": 1.0,
                    "channel": "tool",
                    "kind": "response.output_item.added",
                },
                {
                    "at_ms": 2.0,
                    "channel": "tool",
                    "kind": "response.function_call_arguments.done",
                },
                {
                    "at_ms": 3.0,
                    "channel": "tool",
                    "kind": "response.output_item.done",
                },
            ]
        return result

    def fake_abort(
        client,
        protocol,
        payload,
        *,
        health_url,
        minimum_deltas,
    ):
        del health_url
        _finish_synthetic_capture(
            client,
            protocol,
            "stream-abort",
            payload,
            omitted_label=omitted_label,
        )
        return {
            "status_code": 200,
            "closed_at_ms": 1.0,
            "response_id": f"{client.base_label}-{protocol}-abort",
            "delta_events_before_abort": minimum_deltas,
            "cancel_status": (200 if protocol in {"chat", "responses"} else None),
            "cancel_body_sha256": "",
            "terminals_before_abort": [],
            "events": [],
            "idle_after_abort": {"idle": True},
        }

    monkeypatch.setattr(matrix.ProtocolClient, "send", fake_send)
    monkeypatch.setattr(
        matrix,
        "abort_stream_after_deltas",
        fake_abort,
    )
    argv = [
        "--base-url",
        "direct=http://direct.invalid",
        "--base-url",
        "gateway=http://gateway.invalid",
        "--model",
        "served-model",
        "--repo-root",
        str(repo_root),
        "--output",
        str(tmp_path / "result.json"),
        "--mode",
        "stream",
        "--no-enable-thinking",
    ]
    if capture_enabled:
        argv.extend(("--raw-artifact-dir", str(raw_root)))
    args = matrix.build_parser().parse_args(argv)

    result = matrix.run_matrix(args)

    assert result["checks"]["all_flows_pass"] is True
    assert result["checks"]["all_abort_recovery_pass"] is True
    assert result["raw_capture"]["enabled"] is capture_enabled
    assert result["raw_capture"]["started"] == expected_started
    assert result["pass"] is expected_pass
    if not capture_enabled:
        assert result["raw_capture"]["reason"] == ("--raw-artifact-dir not supplied")
        assert result["checks"]["raw_capture_complete"] is True
        return

    assert result["raw_capture"]["expected"] == 40
    assert result["raw_capture"]["finished"] == expected_started
    assert result["raw_capture"]["errors"] == 0
    assert result["checks"]["raw_capture_complete"] is expected_pass
    manifest_path = next(raw_root.glob("*/manifest.json"))
    manifest_bytes = manifest_path.read_bytes()
    assert (
        hashlib.sha256(manifest_bytes).hexdigest()
        == (result["raw_capture"]["manifest_sha256"])
    )
    manifest = json.loads(manifest_bytes)
    assert manifest["expected"] == 40
    assert manifest["started"] == expected_started
    assert manifest["finished"] == expected_started
    assert manifest["complete"] is expected_pass
    if expected_pass:
        assert len(manifest["routes"]) == 40
        assert all(row["expected"] == 1 for row in manifest["routes"])
        assert all(row["started"] == 1 for row in manifest["routes"])
        assert all(row["finished"] == 1 for row in manifest["routes"])
        assert {row["capture_label"] for row in manifest["routes"]} == {
            "stream-flow-round1",
            "stream-flow-round2",
            "stream-flow-round3",
            "stream-abort",
            "stream-recovery",
        }


def test_run_matrix_rejects_nonstream_only_private_capture_before_creation(
    tmp_path: Path,
):
    repo_root = tmp_path / "repo"
    package_json = repo_root / matrix.FILE_INFO_PATH
    package_json.parent.mkdir(parents=True)
    package_json.write_text("{}\n")
    subprocess.run(
        ["git", "init", "-q", str(repo_root)],
        check=True,
        capture_output=True,
    )
    raw_root = tmp_path / "private-captures"
    args = matrix.build_parser().parse_args(
        [
            "--base-url",
            "direct=http://direct.invalid",
            "--base-url",
            "gateway=http://gateway.invalid",
            "--model",
            "served-model",
            "--repo-root",
            str(repo_root),
            "--output",
            str(tmp_path / "result.json"),
            "--mode",
            "nonstream",
            "--raw-artifact-dir",
            str(raw_root),
        ]
    )

    with pytest.raises(
        ValueError,
        match="captures streaming parser-input bytes and requires --mode stream",
    ):
        matrix.run_matrix(args)
    assert not raw_root.exists()
    assert "requires --mode stream" in matrix.build_parser().format_help()
