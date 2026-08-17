# SPDX-License-Identifier: Apache-2.0
"""
MCP client for connecting to individual MCP servers.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from .types import (
    MCPServerConfig,
    MCPServerState,
    MCPServerStatus,
    MCPTool,
    MCPToolResult,
    MCPTransport,
)

logger = logging.getLogger(__name__)


# --- MCP SDK 1.x / 2.x compatibility -----------------------------------------
#
# The SDK renamed its public surface to snake_case in 2.0.0. This module was
# written against 1.x, and the field reported it: the app bundles 2.0.0 while
# the dev venv had 1.26.0, so every http-transport server failed for users
# while every test passed locally. `mcp>=1.0.0` in pyproject has no upper
# bound, so a user's environment can legitimately hold either major.
#
# Support BOTH rather than pinning users to an old SDK. Each shim tries the
# 2.x name first (that is what ships) and falls back to the 1.x name.

_STREAMABLE_HTTP_FACTORY_NAMES = ("streamable_http_client", "streamablehttp_client")


def _resolve_streamable_http_client():
    """Return the streamable-HTTP client factory from either SDK major."""
    import mcp.client.streamable_http as _mod

    for name in _STREAMABLE_HTTP_FACTORY_NAMES:
        factory = getattr(_mod, name, None)
        if factory is not None:
            return factory
    raise ImportError(
        f"none of {_STREAMABLE_HTTP_FACTORY_NAMES} exist in "
        f"mcp.client.streamable_http"
    )


def _sdk_attr(obj, *names, default=None):
    """Read the first attribute that exists, 2.x name first.

    Renamed in 2.0.0 and all four of these are load-bearing:
      protocolVersion -> protocol_version   (handshake logging)
      serverInfo      -> server_info        (handshake logging)
      inputSchema     -> input_schema       (TOOL DISCOVERY - a miss here
                                             reports 0 tools from a healthy
                                             server, which is exactly what
                                             users saw)
      isError         -> is_error           (tool-result error handling - a
                                             miss here silently treats a
                                             failed call as success)
    """
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


class MCPClient:
    """
    Client for connecting to a single MCP server.

    Supports both stdio and SSE transports.
    """

    def __init__(self, config: MCPServerConfig):
        """
        Initialize MCP client.

        Args:
            config: Server configuration
        """
        self.config = config
        self._session = None
        self._read = None
        self._write = None
        self._tools: List[MCPTool] = []
        self._state = MCPServerState.DISCONNECTED
        self._error: Optional[str] = None
        self._last_connected: Optional[float] = None
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        """Get server name."""
        return self.config.name

    @property
    def state(self) -> MCPServerState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self._state == MCPServerState.CONNECTED

    @property
    def tools(self) -> List[MCPTool]:
        """Get discovered tools."""
        return self._tools

    def get_status(self) -> MCPServerStatus:
        """Get server status."""
        return MCPServerStatus(
            name=self.name,
            state=self._state,
            transport=self.config.transport,
            tools_count=len(self._tools),
            error=self._error,
            last_connected=self._last_connected,
        )

    async def connect(self) -> bool:
        """
        Connect to the MCP server.

        Returns:
            True if connection successful, False otherwise
        """
        async with self._lock:
            if self._state == MCPServerState.CONNECTED:
                return True

            if not self.config.enabled:
                logger.info(f"MCP server '{self.name}' is disabled")
                return False

            self._state = MCPServerState.CONNECTING
            self._error = None

            try:
                # 30s timeout prevents indefinite hang if MCP server never responds
                async def _do_connect():
                    if self.config.transport == MCPTransport.STDIO:
                        await self._connect_stdio()
                    elif self.config.transport == MCPTransport.SSE:
                        await self._connect_sse()
                    elif self.config.transport == MCPTransport.HTTP:
                        await self._connect_http()
                    else:
                        raise ValueError(f"Unknown transport: {self.config.transport}")

                    # Initialize session
                    await self._initialize_session()

                    # Discover tools
                    await self._discover_tools()

                await asyncio.wait_for(_do_connect(), timeout=30)

                self._state = MCPServerState.CONNECTED
                self._last_connected = time.time()
                logger.info(
                    f"Connected to MCP server '{self.name}' "
                    f"({len(self._tools)} tools available)"
                )
                return True

            except Exception as e:
                self._state = MCPServerState.ERROR
                self._error = str(e)
                logger.error(f"Failed to connect to MCP server '{self.name}': {e}")
                # Clean up any partially-initialized resources
                try:
                    if self._session:
                        await self._session.__aexit__(None, None, None)
                        self._session = None
                    if hasattr(self, "_stdio_client") and self._stdio_client:
                        await self._stdio_client.__aexit__(None, None, None)
                        self._stdio_client = None
                    if hasattr(self, "_sse_client") and self._sse_client:
                        await self._sse_client.__aexit__(None, None, None)
                        self._sse_client = None
                except Exception as cleanup_err:
                    logger.warning(f"Error during cleanup of '{self.name}': {cleanup_err}")
                self._read = None
                self._write = None
                return False

    async def _connect_stdio(self):
        """Connect via stdio transport."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            raise ImportError(
                "MCP SDK required for MCP support. Install with: pip install mcp"
            )

        # Security: Log the command being executed for audit trail
        logger.info(
            f"MCP SECURITY AUDIT: Server '{self.name}' executing command: "
            f"{self.config.command} {' '.join(self.config.args or [])}"
        )

        server_params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args or [],
            env=self.config.env,
        )

        # Create stdio client context
        self._stdio_client = stdio_client(server_params)
        self._read, self._write = await self._stdio_client.__aenter__()

        # Create session — clean up stdio client if this fails
        try:
            self._session = ClientSession(self._read, self._write)
            await self._session.__aenter__()
        except Exception:
            await self._stdio_client.__aexit__(None, None, None)
            self._stdio_client = None
            self._read = None
            self._write = None
            raise

    async def _connect_sse(self):
        """Connect via SSE transport."""
        try:
            from mcp import ClientSession
            from mcp.client.sse import sse_client
        except ImportError:
            raise ImportError(
                "MCP SDK required for MCP support. Install with: pip install mcp"
            )

        # Forward auth headers (e.g., Bearer tokens for Exa / GitHub remote MCP).
        # mcp SDK's sse_client accepts a `headers` kwarg in 1.x.
        _kwargs: Dict[str, Any] = {}
        if self.config.headers:
            _kwargs["headers"] = dict(self.config.headers)
        self._sse_client = sse_client(self.config.url, **_kwargs)
        self._read, self._write = await self._sse_client.__aenter__()

        # Create session — clean up SSE client if this fails
        try:
            self._session = ClientSession(self._read, self._write)
            await self._session.__aenter__()
        except Exception:
            await self._sse_client.__aexit__(None, None, None)
            self._sse_client = None
            self._read = None
            self._write = None
            raise

    async def _connect_http(self):
        """Connect via Streamable HTTP transport (modern remote MCP servers)."""
        try:
            from mcp import ClientSession

            streamablehttp_client = _resolve_streamable_http_client()
        except ImportError as exc:
            # The old message here said "Upgrade with: pip install -U mcp",
            # which pointed users the WRONG WAY: the observed failure in the
            # field was an SDK that was too NEW (2.0.0 renamed the factory),
            # so upgrading could not have helped. Report what was actually
            # searched for instead of guessing at the cause.
            raise ImportError(
                "MCP streamable-HTTP transport not found in the installed MCP "
                f"SDK (tried {', '.join(_STREAMABLE_HTTP_FACTORY_NAMES)} in "
                f"mcp.client.streamable_http): {exc}"
            ) from exc

        _kwargs: Dict[str, Any] = {}
        if self.config.headers:
            _kwargs["headers"] = dict(self.config.headers)
        self._sse_client = streamablehttp_client(self.config.url, **_kwargs)
        # streamablehttp_client returns (read, write, get_session_id) — we
        # only consume the read/write pair; session-id callback is optional.
        _ctx_result = await self._sse_client.__aenter__()
        if isinstance(_ctx_result, tuple) and len(_ctx_result) >= 2:
            self._read, self._write = _ctx_result[0], _ctx_result[1]
        else:
            self._read, self._write = _ctx_result

        try:
            self._session = ClientSession(self._read, self._write)
            await self._session.__aenter__()
        except Exception:
            await self._sse_client.__aexit__(None, None, None)
            self._sse_client = None
            self._read = None
            self._write = None
            raise

    async def _initialize_session(self):
        """Initialize the MCP session."""
        if self._session is None:
            raise RuntimeError("Session not created")

        # Initialize with capabilities
        result = await self._session.initialize()
        logger.debug(
            f"MCP server '{self.name}' initialized: "
            f"protocol={_sdk_attr(result, 'protocol_version', 'protocolVersion')}, "
            f"server={getattr(_sdk_attr(result, 'server_info', 'serverInfo'), 'name', 'unknown')}"
        )

    async def _discover_tools(self):
        """Discover available tools from the server."""
        if self._session is None:
            raise RuntimeError("Session not initialized")

        try:
            result = await self._session.list_tools()
            self._tools = []

            for tool in result.tools:
                mcp_tool = MCPTool(
                    server_name=self.name,
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=(
                        _sdk_attr(tool, "input_schema", "inputSchema", default={})
                    ),
                )
                self._tools.append(mcp_tool)
                logger.debug(f"Discovered tool: {mcp_tool.full_name}")

        except Exception as e:
            logger.warning(f"Failed to discover tools from '{self.name}': {e}")
            self._tools = []

    async def disconnect(self):
        """Disconnect from the MCP server."""
        async with self._lock:
            if self._state == MCPServerState.DISCONNECTED:
                return

            try:
                if self._session:
                    await self._session.__aexit__(None, None, None)
                    self._session = None

                if hasattr(self, "_stdio_client") and self._stdio_client:
                    await self._stdio_client.__aexit__(None, None, None)
                    self._stdio_client = None

                if hasattr(self, "_sse_client") and self._sse_client:
                    await self._sse_client.__aexit__(None, None, None)
                    self._sse_client = None

            except Exception as e:
                logger.warning(f"Error disconnecting from '{self.name}': {e}")

            finally:
                self._state = MCPServerState.DISCONNECTED
                self._tools = []
                logger.info(f"Disconnected from MCP server '{self.name}'")

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> MCPToolResult:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool (without server prefix)
            arguments: Tool arguments
            timeout: Optional timeout in seconds

        Returns:
            MCPToolResult with the result or error
        """
        if not self.is_connected:
            return MCPToolResult(
                tool_name=tool_name,
                content=None,
                is_error=True,
                error_message=f"Not connected to server '{self.name}'",
            )

        if self._session is None:
            return MCPToolResult(
                tool_name=tool_name,
                content=None,
                is_error=True,
                error_message="Session not initialized",
            )

        try:
            # Call with timeout
            timeout = timeout if timeout is not None else self.config.timeout

            result = await asyncio.wait_for(
                self._session.call_tool(tool_name, arguments),
                timeout=timeout,
            )

            # Extract content from result
            content = self._extract_content(result)

            return MCPToolResult(
                tool_name=tool_name,
                content=content,
                is_error=bool(
                    _sdk_attr(result, "is_error", "isError", default=False)
                ),
            )

        except asyncio.TimeoutError:
            return MCPToolResult(
                tool_name=tool_name,
                content=None,
                is_error=True,
                error_message=f"Tool call timed out after {timeout}s",
            )
        except Exception as e:
            return MCPToolResult(
                tool_name=tool_name,
                content=None,
                is_error=True,
                error_message=str(e),
            )

    # Max total content size from MCP tool responses (10 MB)
    _MAX_CONTENT_SIZE = 10 * 1024 * 1024

    def _extract_content(self, result) -> Any:
        """Extract content from MCP tool result.

        Always returns strings to ensure JSON serialization in to_message() never crashes.
        """
        if not hasattr(result, "content") or not result.content:
            return None

        # Handle list of content items (with size limit to prevent OOM)
        contents: list[str] = []
        total_size = 0
        for item in result.content:
            if hasattr(item, "text"):
                text = str(item.text) if not isinstance(item.text, str) else item.text
            elif hasattr(item, "data"):
                text = str(item.data) if not isinstance(item.data, str) else item.data
            else:
                text = str(item)
            total_size += len(text)
            if total_size > self._MAX_CONTENT_SIZE:
                contents.append("[Content truncated — MCP tool response exceeded 10 MB]")
                break
            contents.append(text)

        # Return single string or joined string (not a list — avoids json.dumps in to_message)
        if len(contents) == 1:
            return contents[0]
        return "\n".join(contents)

    async def refresh_tools(self):
        """Refresh the list of available tools."""
        if not self.is_connected:
            return

        await self._discover_tools()
