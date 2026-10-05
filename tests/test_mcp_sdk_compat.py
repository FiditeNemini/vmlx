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


@pytest.mark.parametrize("failure", [None, "enter", "body"])
def test_http_transport_sdk_client_headers_and_cleanup(monkeypatch, failure):
    import asyncio
    from contextlib import asynccontextmanager
    import mcp.shared._httpx_utils as utils

    events = []
    clients = []
    real_create = utils.create_mcp_http_client

    def create(**kwargs):
        client = real_create(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(utils, "create_mcp_http_client", create)

    @asynccontextmanager
    async def transport(url, *, http_client):
        assert url == "http://localhost/mcp"
        assert http_client.headers["X-Test-Auth"] == "fixture"
        assert not http_client.is_closed
        assert http_client.timeout.connect == utils.MCP_DEFAULT_TIMEOUT
        assert http_client.timeout.read == utils.MCP_DEFAULT_SSE_READ_TIMEOUT
        if failure == "enter":
            raise RuntimeError("transport entry failure")
        try:
            yield ("read", "write")
        finally:
            assert not http_client.is_closed
            events.append("transport closed before client")

    async def run():
        async with mcp_client._streamable_http_transport(
            transport, "http://localhost/mcp", {"X-Test-Auth": "fixture"}
        ) as streams:
            assert streams == ("read", "write")
            if failure == "body":
                raise RuntimeError("session failure")

    if failure:
        with pytest.raises(RuntimeError):
            asyncio.run(run())
    else:
        asyncio.run(run())
    assert len(clients) == 1 and clients[0].is_closed
    assert events == ([] if failure == "enter" else ["transport closed before client"])


def test_http_transport_legacy_headers_and_no_header_defaults():
    import asyncio
    from contextlib import asynccontextmanager

    calls = []

    @asynccontextmanager
    async def legacy(url, headers=None):
        calls.append((url, headers))
        yield ("read", "write", None)

    async def run():
        for headers in ({"X-Test-Auth": "fixture"}, None):
            async with mcp_client._streamable_http_transport(legacy, "url", headers):
                pass

    asyncio.run(run())
    assert calls == [("url", {"X-Test-Auth": "fixture"}), ("url", None)]


def test_installed_http_factory_accepts_sdk_http_client_without_network():
    import inspect
    from mcp.shared._httpx_utils import create_mcp_http_client
    import asyncio

    factory = mcp_client._resolve_streamable_http_client()
    signature = inspect.signature(factory)

    async def run():
        if "http_client" in signature.parameters:
            async with create_mcp_http_client(headers={"X-Test-Auth": "fixture"}) as client:
                signature.bind("http://localhost/mcp", http_client=client)
                assert client.headers["X-Test-Auth"] == "fixture"
        else:
            signature.bind("http://localhost/mcp", headers={"X-Test-Auth": "fixture"})

    asyncio.run(run())


def test_real_http_sdk_auth_schema_results_and_cleanup():
    """Actual SDK transport against bounded loopback JSON-RPC, without models."""
    import asyncio
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from vmlx_engine.mcp.types import MCPServerConfig, MCPTransport

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, payload=None):
            data = json.dumps(payload).encode() if payload is not None else b""
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Mcp-Session-Id", "fixture-session")
            self.end_headers()
            self.wfile.write(data)

        def authorized(self):
            requests.append((self.command, self.headers.get("Authorization")))
            if self.headers.get("Authorization") != "Bearer fixture-only":
                self.reply(401)
                return False
            return True

        def do_GET(self):
            if self.authorized():
                self.reply(405)

        def do_DELETE(self):
            if self.authorized():
                self.reply(200)

        def do_POST(self):
            if not self.authorized():
                return
            message = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            method = message["method"]
            if "id" not in message:
                self.reply(202)
                return
            if method == "initialize":
                result = {"protocolVersion": message["params"]["protocolVersion"],
                          "capabilities": {"tools": {}},
                          "serverInfo": {"name": "fixture", "version": "1"}}
            elif method == "tools/list":
                result = {"tools": [{"name": "echo", "description": "Fixture echo",
                          "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}]}
            elif method == "tools/call":
                value = message["params"]["arguments"]["text"]
                result = {"content": [{"type": "text", "text": value}], "isError": value == "failure"}
            else:
                self.reply(400)
                return
            self.reply(200, {"jsonrpc": "2.0", "id": message["id"], "result": result})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = mcp_client.MCPClient(MCPServerConfig(
        name="fixture", transport=MCPTransport.HTTP,
        url=f"http://127.0.0.1:{server.server_port}/mcp",
        headers={"Authorization": "Bearer fixture-only"}, timeout=5,
    ))

    async def run():
        try:
            assert await client.connect(), client.get_status().error
            assert len(client.tools) == 1
            assert client.tools[0].input_schema["required"] == ["text"]
            success = await client.call_tool("echo", {"text": "success"})
            assert success.content == "success" and not success.is_error
            failure = await client.call_tool("echo", {"text": "failure"})
            assert failure.content == "failure" and failure.is_error
        finally:
            await client.disconnect()
        assert client._session is None
        assert client._sse_client is None
        assert not client.is_connected

    try:
        asyncio.run(run())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert not thread.is_alive()
    assert requests and all(auth == "Bearer fixture-only" for _, auth in requests)
    assert any(method == "DELETE" for method, _ in requests)
