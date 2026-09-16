# SPDX-License-Identifier: Apache-2.0
"""A tool's execution error is data for the next generation, not a lost error."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from vmlx_engine.mcp.client import MCPClient
from vmlx_engine.mcp.types import MCPServerConfig, MCPServerState, MCPToolResult


@pytest.mark.parametrize("detail", ["missing field: tax", "  indented\nline\\n\n", "null", "拒否: field"])
def test_error_content_is_preserved_for_api_audit_and_continuation(detail):
    result = MCPToolResult("read", detail, is_error=True)
    assert result.content == detail
    assert result.error_message == detail
    assert result.to_message("call-1") == {
        "role": "tool", "tool_call_id": "call-1", "content": f"Error: {detail}",
    }


def test_explicit_error_message_takes_precedence_without_rewriting_content():
    result = MCPToolResult("read", {"reason": "details"}, True, "  denied\n")
    assert result.error_message == "  denied\n"
    assert result.content == {"reason": "details"}
    assert result.to_message("c")["content"] == "Error:   denied\n"


@pytest.mark.parametrize("content", [None, "", " \n\t"])
def test_missing_error_detail_does_not_become_python_none(content):
    result = MCPToolResult("read", content, is_error=True)
    assert result.content == content
    assert result.error_message == "Unknown error"
    assert result.to_message("c")["content"] == "Error: Unknown error"


def test_blank_error_message_falls_back_to_actual_content():
    assert MCPToolResult("read", "missing", True, " \n").error_message == "missing"


@pytest.mark.parametrize("content", [{"field": None, "literal": "null"}, ["a", 2], False, 0])
def test_structured_error_content_keeps_its_types(content):
    result = MCPToolResult("read", content, is_error=True)
    assert result.content == content
    assert json.loads(result.error_message) == content
    assert result.to_message("c")["content"] == f"Error: {json.dumps(content)}"


@pytest.mark.parametrize("content", [None, "", "  a\n", {"n": None}, [1, "null"], False, 0])
def test_success_serialization_is_unchanged(content):
    result = MCPToolResult("read", content)
    assert result.error_message is None
    expected = content if isinstance(content, str) else json.dumps(content)
    assert result.to_message("same-id") == {
        "role": "tool", "tool_call_id": "same-id", "content": expected,
    }


@pytest.mark.parametrize("error_field", ["isError", "is_error"])
@pytest.mark.parametrize("is_error", [True, False])
def test_real_client_preserves_sdk_error_content_and_call_arguments(error_field, is_error):
    detail = "  Missing field: tax\nUse the existing net field.\n"
    sdk_result = SimpleNamespace(
        content=[SimpleNamespace(text=detail)], **{error_field: is_error},
    )
    client = MCPClient(MCPServerConfig(name="local", command="python3"))
    client._state = MCPServerState.CONNECTED
    client._session = SimpleNamespace(call_tool=AsyncMock(return_value=sdk_result))
    arguments = {"field": None, "literal": "null"}
    result = asyncio.run(client.call_tool("read", arguments))
    client._session.call_tool.assert_awaited_once_with("read", arguments)
    assert result.content == detail
    assert result.is_error is is_error
    assert result.error_message == (detail if is_error else None)
    assert result.to_message("sdk-call")["content"] == (f"Error: {detail}" if is_error else detail)
