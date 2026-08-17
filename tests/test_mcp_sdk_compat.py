# SPDX-License-Identifier: Apache-2.0
"""MCP SDK 1.x/2.x compatibility pins.

Field report 2026-08-16 (external user, confirmed by reading source): the app
bundles MCP SDK 2.0.0 while the dev venv had 1.26.0. 2.0.0 renamed the public
surface to snake_case, so EVERY http-transport MCP server failed for users
while every test passed locally, and stdio servers spawned fine then reported
0 tools.

The nasty part was the silent ones. `inputSchema` and `isError` were read
behind `hasattr` guards with defaults, so on 2.0.0 they did not raise — a
tool's schema quietly became `{}` (the model cannot call a tool with no
schema, which is precisely "the model couldn't see them") and `isError`
quietly became False (a FAILED tool call reported as success).

These tests assert against the SDK that is actually installed, so a future
rename fails here instead of in a user's app.
"""

import importlib.metadata

import pytest

from vmlx_engine.mcp import client as mcp_client


def test_streamable_http_factory_resolves_on_the_installed_sdk():
    """The http transport must find its factory under either SDK major."""
    factory = mcp_client._resolve_streamable_http_client()
    assert callable(factory)
    assert factory.__name__ in mcp_client._STREAMABLE_HTTP_FACTORY_NAMES


def test_streamable_http_resolution_reports_what_it_looked_for():
    """A miss must name the symbols tried, not guess at the cause.

    The original message said "Upgrade with: pip install -U mcp" — the exact
    opposite of the real problem, which was an SDK that was too NEW.
    """
    import mcp.client.streamable_http as real_mod

    # `import a.b.c as x` resolves through the PARENT package attribute, so
    # swapping sys.modules alone is not enough — delete the factory names off
    # the real module and restore them, which is what a renamed SDK looks like.
    saved = {
        name: getattr(real_mod, name)
        for name in mcp_client._STREAMABLE_HTTP_FACTORY_NAMES
        if hasattr(real_mod, name)
    }
    for name in saved:
        delattr(real_mod, name)
    try:
        with pytest.raises(ImportError) as exc:
            mcp_client._resolve_streamable_http_client()
        for name in mcp_client._STREAMABLE_HTTP_FACTORY_NAMES:
            assert name in str(exc.value)
    finally:
        for name, value in saved.items():
            setattr(real_mod, name, value)


class _Only2x:
    protocol_version = "2025-06-18"
    input_schema = {"type": "object"}
    is_error = True


class _Only1x:
    protocolVersion = "2024-11-05"
    inputSchema = {"type": "object"}
    isError = True


@pytest.mark.parametrize("obj", [_Only2x(), _Only1x()])
def test_sdk_attr_reads_either_naming(obj):
    assert mcp_client._sdk_attr(obj, "protocol_version", "protocolVersion")
    assert mcp_client._sdk_attr(obj, "input_schema", "inputSchema") == {
        "type": "object"
    }
    assert mcp_client._sdk_attr(obj, "is_error", "isError") is True


def test_sdk_attr_prefers_the_2x_name():
    class Both:
        input_schema = {"which": "2x"}
        inputSchema = {"which": "1x"}

    assert mcp_client._sdk_attr(Both(), "input_schema", "inputSchema") == {"which": "2x"}


def test_sdk_attr_returns_default_when_neither_exists():
    assert mcp_client._sdk_attr(object(), "a", "b", default="fallback") == "fallback"


def test_load_bearing_fields_exist_on_the_installed_sdk():
    """Guard the four renamed fields against the REAL installed SDK.

    A silent default here is worse than a crash: empty schema hides tools and
    a False is_error turns a failure into a success.
    """
    import mcp.types as t

    version = importlib.metadata.version("mcp")

    tool_fields = set(t.Tool.model_fields)
    assert tool_fields & {"input_schema", "inputSchema"}, (version, tool_fields)

    result_fields = set(t.CallToolResult.model_fields)
    assert result_fields & {"is_error", "isError"}, (version, result_fields)

    init_fields = set(t.InitializeResult.model_fields)
    assert init_fields & {"protocol_version", "protocolVersion"}, (version, init_fields)
    assert init_fields & {"server_info", "serverInfo"}, (version, init_fields)


def test_client_source_has_no_unguarded_camelcase_sdk_reads():
    """Source-level guard so a future edit cannot reintroduce the bug.

    Every SDK field read must go through _sdk_attr; a bare `result.isError`
    works on 1.x and silently misreads on 2.x, which is how this shipped.
    """
    import inspect
    import re

    src = inspect.getsource(mcp_client)
    # Strip comments and docstrings' mentions of the old names: this checks
    # CODE, and the module documents the rename on purpose.
    code_only = "\n".join(
        line.split("#", 1)[0] for line in src.splitlines()
    )
    for bad in ("result.protocolVersion", "result.serverInfo",
                "tool.inputSchema", "result.isError"):
        assert bad not in code_only, f"unguarded SDK read reintroduced: {bad}"
