#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end MCP probe: a REAL stdio server driven by vMLX's own client.

Why this exists. A user reported (2026-08-16) that MCP servers connected and
then the model could not see their tools. Root cause: the app bundles mcp
2.0.0, which renamed its public surface to snake_case, while the dev venv held
1.26.0 — so the unit suite was green while every user's http-transport server
failed and stdio servers reported 0 tools. Unit shims are necessary but they
cannot catch that class; only driving the real client against a real server on
the SHIPPING SDK can.

It checks the three things the renames actually broke:

  1. the server reaches CONNECTED (not ERROR)
  2. tools arrive WITH a non-empty input schema — the shipped bug produced
     tools whose schema was `{}`, which is precisely why a model "couldn't
     see them", and it did NOT raise because the read was behind a hasattr
     guard with a default
  3. a failing tool reports is_error=True — the shipped bug reported
     failures as SUCCESS, same guard, same silence

Run against the interpreter you care about; the bundled one is what users get:

  ./panel/bundled-python/python/bin/python3.12 \
      tests/cross_matrix/run_mcp_stdio_e2e_probe.py

Exit code 0 = pass. `--out FILE` writes a JSON artifact.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# A server exposing one succeeding tool with real parameters and one that
# always raises. Written to a temp file so the probe is self-contained.
SERVER_SOURCE = '''
from mcp.server import MCPServer

server = MCPServer(name="vmlx-e2e", version="1.0.0")


@server.add_tool
def echo(text: str, times: int = 1) -> str:
    """Echo text back, optionally repeated."""
    return " ".join([text] * max(1, times))


@server.add_tool
def always_fails(reason: str = "by design") -> str:
    """Always raises, to exercise the error path."""
    raise RuntimeError(f"tool failed: {reason}")


if __name__ == "__main__":
    server.run()
'''


async def probe(server_path: str) -> dict:
    sys.path.insert(0, str(REPO_ROOT))
    import importlib.metadata as meta

    from vmlx_engine.mcp.client import MCPClient
    from vmlx_engine.mcp.types import MCPServerConfig, MCPTransport

    report: dict = {
        "sdk_version": meta.version("mcp"),
        "interpreter": sys.executable,
        "repo_root": str(REPO_ROOT),
    }

    client = MCPClient(
        MCPServerConfig(
            name="e2e",
            transport=MCPTransport.STDIO,
            command=sys.executable,
            args=[server_path],
            enabled=True,
        )
    )

    connected = await client.connect()
    report["connected"] = bool(connected)
    report["state"] = str(client.state)
    if not connected:
        report["error"] = client.get_status().error
        report["checks"] = {"connected": False}
        return report

    try:
        names = sorted(t.name for t in client.tools)
        report["tools"] = names

        schemas = {}
        for tool in client.tools:
            schema = (
                getattr(tool, "input_schema", None)
                or getattr(tool, "parameters", None)
                or {}
            )
            schemas[tool.name] = sorted((schema or {}).get("properties") or {})
        report["tool_schema_properties"] = schemas

        good = await client.call_tool("echo", {"text": "vmlx", "times": 2})
        bad = await client.call_tool("always_fails", {"reason": "e2e"})
        report["echo_is_error"] = bool(good.is_error)
        report["failing_is_error"] = bool(bad.is_error)

        report["checks"] = {
            "connected": True,
            "both_tools_discovered": names == ["always_fails", "echo"],
            # An empty schema is the shipped failure: discovered but uncallable.
            "echo_schema_non_empty": schemas.get("echo") == ["text", "times"],
            "echo_call_succeeded": good.is_error is False,
            # Silently reporting a failure as success is the worse half.
            "failing_call_reports_error": bad.is_error is True,
        }
    finally:
        try:
            await client.disconnect()
        except Exception:
            pass
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="write a JSON artifact here")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        server_path = os.path.join(tmp, "e2e_mcp_server.py")
        Path(server_path).write_text(SERVER_SOURCE)
        report = asyncio.run(probe(server_path))

    checks = report.get("checks") or {}
    report["status"] = "pass" if checks and all(checks.values()) else "fail"

    print(f"SDK under test : mcp {report.get('sdk_version')}")
    print(f"connected      : {report.get('connected')} ({report.get('state')})")
    if report.get("error"):
        print(f"error          : {report['error']}")
    print(f"tools          : {report.get('tools')}")
    print(f"schemas        : {report.get('tool_schema_properties')}")
    for name, ok in checks.items():
        print(f"  {name:<28} {'PASS' if ok else 'FAIL'}")
    print(f"VERDICT: {report['status'].upper()}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"artifact -> {args.out}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
