# SPDX-License-Identifier: Apache-2.0
"""Tool-parser terminals must retain normal cache-persistence ownership."""

import asyncio
import json
from types import SimpleNamespace

import pytest


_TOOL = {
    "type": "function",
    "function": {
        "name": "file_info",
        "description": "Inspect a file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}

_CALL_DELTAS = [
    "reasoning before the call ",
    "<tool_call>",
    "<function=file_info>",
    "<parameter=path>panel/package.json</parameter>",
    "</function>",
    "</tool_call>",
]


class _GracefulToolEngine:
    tokenizer = SimpleNamespace(has_thinking=False)

    def __init__(self):
        self.graceful_stops: list[str] = []
        self.aborts: list[str] = []
        self.durable = False
        self.chunks_consumed = 0

    async def request_graceful_stop(self, request_id: str) -> bool:
        self.graceful_stops.append(request_id)
        return True

    async def abort_request(self, request_id: str) -> bool:
        self.aborts.append(request_id)
        return True

    async def stream_chat(self, *, messages, **kwargs):
        from vmlx_engine.engine.base import GenerationOutput

        text = ""
        for idx, delta in enumerate(_CALL_DELTAS, start=1):
            text += delta
            self.chunks_consumed += 1
            yield GenerationOutput(
                text=text,
                new_text=delta,
                tokens=[idx],
                prompt_tokens=128,
                completion_tokens=idx,
                finished=False,
            )

        # The parser has requested a graceful stop by the time the async
        # generator resumes. This token represents the model-worker drain and
        # must never leak through either API surface.
        assert self.graceful_stops
        self.chunks_consumed += 1
        yield GenerationOutput(
            text=text + "INTERNAL_POST_CALL_DRAIN",
            new_text="INTERNAL_POST_CALL_DRAIN",
            tokens=[90],
            prompt_tokens=128,
            completion_tokens=len(_CALL_DELTAS) + 1,
            finished=False,
        )

        # A real engine exposes this object only after its cache durability
        # barrier. Mark the same observable ordering in the fake.
        self.durable = True
        self.chunks_consumed += 1
        yield GenerationOutput(
            text=text + "INTERNAL_POST_CALL_DRAIN!",
            new_text="!",
            tokens=[91],
            prompt_tokens=128,
            completion_tokens=len(_CALL_DELTAS) + 2,
            finished=True,
            finish_reason="stop",
        )


def _set_qwen_tool_globals(monkeypatch, server) -> None:
    monkeypatch.setattr(server, "_default_timeout", 5.0)
    monkeypatch.setattr(server, "_model_name", "bonsai-test")
    monkeypatch.setattr(server, "_model_path", None)
    monkeypatch.setattr(server, "_reasoning_parser", None)
    monkeypatch.setattr(server, "_tool_call_parser", "qwen")
    monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", False)


@pytest.mark.asyncio
async def test_chat_tool_terminal_waits_for_graceful_engine_finish(monkeypatch):
    import vmlx_engine.server as server
    from vmlx_engine.api.models import ChatCompletionRequest

    _set_qwen_tool_globals(monkeypatch, server)
    engine = _GracefulToolEngine()
    request = ChatCompletionRequest(
        model="bonsai-test",
        messages=[
            {
                "role": "user",
                "content": (
                    "Call the built-in file_info tool exactly once with path "
                    "panel/package.json."
                ),
            }
        ],
        stream=True,
        tools=[_TOOL],
        tool_choice={"type": "function", "function": {"name": "file_info"}},
    )

    raw = []
    async for chunk in server.stream_chat_completion(
        engine,
        request.messages,
        request,
        fastapi_request=None,
        response_id="chatcmpl-cache-durable",
    ):
        # The parsed tool terminal may only appear after the fake's real engine
        # terminal crossed its durability point.
        if '"finish_reason":"tool_calls"' in chunk.replace(" ", ""):
            assert engine.durable
        raw.append(chunk)

    payload = "".join(raw)
    assert engine.graceful_stops == ["chatcmpl-cache-durable"]
    assert engine.aborts == []
    assert engine.chunks_consumed == len(_CALL_DELTAS) + 2
    assert "INTERNAL_POST_CALL_DRAIN" not in payload
    assert '"finish_reason":"tool_calls"' in payload.replace(" ", "")
    assert '"name":"file_info"' in payload.replace(" ", "")


@pytest.mark.asyncio
async def test_responses_tool_terminal_waits_for_graceful_engine_finish(monkeypatch):
    import vmlx_engine.server as server
    from vmlx_engine.api.models import ResponsesRequest

    _set_qwen_tool_globals(monkeypatch, server)
    server._responses_history.clear()
    engine = _GracefulToolEngine()
    request = ResponsesRequest(
        model="bonsai-test",
        input=(
            "Call the built-in file_info tool exactly once with path "
            "panel/package.json."
        ),
        stream=True,
        tools=[
            {
                "type": "function",
                "name": "file_info",
                "description": "Inspect a file.",
                "parameters": _TOOL["function"]["parameters"],
            }
        ],
        tool_choice={"type": "function", "name": "file_info"},
    )

    events = []
    async for chunk in server.stream_responses_api(
        engine,
        [{"role": "user", "content": request.input}],
        request,
        fastapi_request=None,
        response_id="resp_cache_durable",
    ):
        for line in chunk.splitlines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line.removeprefix("data: "))
            if event.get("type") == "response.completed":
                assert engine.durable
            events.append(event)

    assert engine.graceful_stops == ["resp_cache_durable"]
    assert engine.aborts == []
    assert engine.chunks_consumed == len(_CALL_DELTAS) + 2
    assert "INTERNAL_POST_CALL_DRAIN" not in json.dumps(events)
    function_items = [
        event["item"]
        for event in events
        if event.get("type") == "response.output_item.done"
        and event.get("item", {}).get("type") == "function_call"
    ]
    assert len(function_items) == 1
    assert function_items[0]["name"] == "file_info"
    assert any(event.get("type") == "response.completed" for event in events)


def test_mllm_scheduler_graceful_stop_keeps_request_owned():
    from vmlx_engine.mllm_scheduler import MLLMScheduler
    from vmlx_engine.request import RequestStatus

    class _Generator:
        def __init__(self):
            self.requested = []

        def request_graceful_stop(self, uid):
            self.requested.append(uid)
            return True

    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler._queue_lock = asyncio.Lock()
    # Production uses threading locks. asyncio.Lock cannot be used as a sync
    # context manager, so replace both with the exact lock class here.
    import threading

    scheduler._queue_lock = threading.Lock()
    scheduler._batch_lock = threading.Lock()
    request = SimpleNamespace(status=RequestStatus.RUNNING)
    scheduler.running = {"req-1": request}
    scheduler.request_id_to_uid = {"req-1": 17}
    scheduler.batch_generator = _Generator()

    assert scheduler.request_graceful_stop("req-1") is True
    assert scheduler.batch_generator.requested == [17]
    assert scheduler.running["req-1"] is request
    assert request.status == RequestStatus.RUNNING


def test_mllm_scheduler_graceful_stop_drains_terminalizing_collector():
    """A detached generator row with an owned output queue must not abort."""
    import threading

    from vmlx_engine.mllm_scheduler import MLLMScheduler
    from vmlx_engine.request import RequestStatus

    class _Generator:
        def request_graceful_stop(self, _uid):
            return False

    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler._queue_lock = threading.Lock()
    scheduler._batch_lock = threading.Lock()
    request = SimpleNamespace(status=RequestStatus.RUNNING)
    scheduler.running = {"req-terminalizing": request}
    scheduler.request_id_to_uid = {"req-terminalizing": 23}
    scheduler.output_queues = {"req-terminalizing": asyncio.Queue()}
    scheduler.batch_generator = _Generator()

    assert scheduler.request_graceful_stop("req-terminalizing") is True
    assert scheduler.running["req-terminalizing"] is request
    assert request.status == RequestStatus.RUNNING

    request.status = RequestStatus.FINISHED_STOPPED
    assert scheduler.request_graceful_stop("req-terminalizing") is True


def test_qwen4_specific_contract_is_render_only_and_append_only():
    from vmlx_engine.engine.batched import BatchedEngine

    original = [
        {"role": "system", "content": "stable system catalog"},
        {"role": "user", "content": "inspect panel/package.json"},
    ]
    rendered = BatchedEngine._append_latest_user_tool_contract(
        original,
        "file_info",
    )

    assert original[-1]["content"] == "inspect panel/package.json"
    assert rendered[0] == original[0]
    assert rendered[-1]["content"].startswith(original[-1]["content"])
    assert "must be file_info" in rendered[-1]["content"]

    continuation = [
        *original,
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
        {"role": "tool", "content": "size=5.3 KB"},
        {"role": "user", "content": "Now run pwd."},
    ]
    rerendered = BatchedEngine._append_latest_user_tool_contract(
        continuation,
        "run_command",
    )
    assert rerendered[1]["content"] == rendered[1]["content"]
    assert "must be run_command" in rerendered[-1]["content"]
    assert continuation[1]["content"] == original[1]["content"]


@pytest.mark.asyncio
async def test_qwen4_specific_choice_keeps_catalog_but_restricts_parser(
    monkeypatch,
):
    """Chat and Responses render both tools but authorize only the selection."""
    import copy

    from fastapi import HTTPException

    import vmlx_engine.server as server
    from vmlx_engine.api.models import (
        ChatCompletionRequest,
        Message,
        ResponsesRequest,
    )
    from vmlx_engine.engine.base import GenerationOutput

    class _Engine:
        is_mllm = False
        preserve_native_tool_format = False
        tokenizer = SimpleNamespace(has_thinking=False)

    captures: list[dict] = []

    async def _fake_await_chat(*_args, **kwargs):
        captures.append(copy.deepcopy(kwargs["chat_kwargs"]))
        return GenerationOutput(
            text="plain answer",
            raw_text="plain answer",
            prompt_tokens=3,
            completion_tokens=2,
            finish_reason="stop",
        )

    monkeypatch.setattr(server, "_engine", _Engine())
    monkeypatch.setattr(server, "_model_path", None)
    monkeypatch.setattr(server, "_model_name", "qwen38-flash-next-test")
    monkeypatch.setattr(server, "_served_model_name", "qwen38-flash-next-test")
    monkeypatch.setattr(server, "_model_type", "llm")
    monkeypatch.setattr(server, "_reasoning_parser", None)
    monkeypatch.setattr(server, "_tool_call_parser", "qwen")
    monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", False)
    monkeypatch.setattr(server, "_mcp_manager", None)
    monkeypatch.setattr(server, "_default_timeout", 5.0)
    monkeypatch.setattr(server, "_is_loaded_qwen4_exp_model", lambda _model="": True)
    monkeypatch.setattr(server, "_await_chat_with_disconnect_abort", _fake_await_chat)

    second_tool = {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
    chat_tools = [_TOOL, second_tool]
    chat_request = ChatCompletionRequest(
        model="qwen38-flash-next-test",
        messages=[Message(role="user", content="Inspect the file.")],
        tools=chat_tools,
        tool_choice={
            "type": "function",
            "function": {"name": "file_info"},
        },
    )
    with pytest.raises(HTTPException):
        await server.create_chat_completion(
            chat_request,
            fastapi_request=None,
        )
    chat_capture = captures[-1]
    assert [tool["function"]["name"] for tool in chat_capture["tools"]] == [
        "file_info",
        "run_command",
    ]
    assert chat_capture["_vmlx_qwen4_specific_tool_name"] == "file_info"
    assert "tool_choice" not in chat_capture.get("chat_template_kwargs", {})
    assert [
        server._tool_definition_name(tool)
        for tool in server._effective_tools_for_tool_parsing(chat_request)
    ] == ["file_info"]

    response_tools = [
        {
            "type": "function",
            "name": tool["function"]["name"],
            "description": tool["function"].get("description", ""),
            "parameters": tool["function"]["parameters"],
        }
        for tool in chat_tools
    ]
    responses_request = ResponsesRequest(
        model="qwen38-flash-next-test",
        input="Run the command.",
        tools=response_tools,
        tool_choice={"type": "function", "name": "run_command"},
    )
    with pytest.raises(HTTPException):
        await server.create_response(
            responses_request,
            fastapi_request=None,
        )
    responses_capture = captures[-1]
    assert [tool["function"]["name"] for tool in responses_capture["tools"]] == [
        "file_info",
        "run_command",
    ]
    assert responses_capture["_vmlx_qwen4_specific_tool_name"] == "run_command"
    assert "tool_choice" not in responses_capture.get("chat_template_kwargs", {})
    assert [
        server._tool_definition_name(tool)
        for tool in server._effective_tools_for_tool_parsing(responses_request)
    ] == ["run_command"]
