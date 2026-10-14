#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""API admission contract for advisory projected Metal output estimates.

This harness uses FastAPI's in-process TestClient with a minimal loaded-engine
shape that records requested limits and raises at the forward boundary.
It proves admission/parameter transport, not successful generation or model quality.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient


REQUESTS: dict[str, tuple[str, dict[str, Any]]] = {
    "chat_completions": (
        "/v1/chat/completions",
        {
            "model": "fake",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8192,
        },
    ),
    "chat_completions_stream": (
        "/v1/chat/completions",
        {
            "model": "fake",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8192,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ),
    "responses": (
        "/v1/responses",
        {
            "model": "fake",
            "input": "hi",
            "max_output_tokens": 8192,
        },
    ),
    "responses_stream": (
        "/v1/responses",
        {
            "model": "fake",
            "input": "hi",
            "max_output_tokens": 8192,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ),
    "anthropic_messages": (
        "/v1/messages",
        {
            "model": "fake",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8192,
        },
    ),
    "anthropic_messages_stream": (
        "/v1/messages",
        {
            "model": "fake",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8192,
            "stream": True,
        },
    ),
    "ollama_chat": (
        "/api/chat",
        {
            "model": "fake",
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"num_predict": 8192},
        },
    ),
    "ollama_chat_stream": (
        "/api/chat",
        {
            "model": "fake",
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"num_predict": 8192},
            "stream": True,
        },
    ),
    "ollama_generate": (
        "/api/generate",
        {
            "model": "fake",
            "prompt": "hi",
            "options": {"num_predict": 8192},
        },
    ),
    "ollama_generate_stream": (
        "/api/generate",
        {
            "model": "fake",
            "prompt": "hi",
            "options": {"num_predict": 8192},
            "stream": True,
        },
    ),
}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return repr(value)


class _ForwardBoundaryProbe(SimpleNamespace):
    """Never load/run a model; record the limit that reaches the engine."""
    def _record(self, method, kwargs):
        self.forward_calls.append({"method": method, "max_tokens": kwargs.get("max_tokens")})
        raise RuntimeError("projected-headroom-forward-boundary-probe")

    async def chat(self, *args, **kwargs):
        self._record("chat", kwargs)

    async def generate(self, *args, **kwargs):
        self._record("generate", kwargs)

    async def stream_chat(self, *args, **kwargs):
        self._record("stream_chat", kwargs)
        yield  # Makes this an async iterator; never reached.

    async def stream_generate(self, *args, **kwargs):
        self._record("stream_generate", kwargs)
        yield


def _install_fake_loaded_engine(
    server: Any,
    *,
    model_name: str = "fake-metal-headroom-model",
    max_tokens: int = 8192,
    max_tokens_explicit: bool = True,
) -> None:
    cfg = SimpleNamespace(
        num_hidden_layers=2,
        num_key_value_heads=4,
        head_dim=8,
        torch_dtype="bfloat16",
    )
    server._api_key = None
    server._engine = _ForwardBoundaryProbe(
        forward_calls=[],
        tokenizer=None,
        model=SimpleNamespace(config=cfg),
        is_mllm=False,
        preserve_native_tool_format=False,
    )
    server._model_path = model_name
    server._model_name = model_name
    server._served_model_name = None
    server._default_max_tokens = max_tokens
    server._default_max_tokens_explicit = bool(max_tokens_explicit)
    server._metal_projected_output_token_cap = lambda model_name="": 1


def _resolve_default_clamp(server: Any) -> dict[str, Any]:
    """CLI/session defaults must survive the projected estimate unchanged."""
    try:
        value = server._resolve_max_tokens(None, "fake-metal-headroom-model")
    except Exception as exc:
        return {
            "ok": False,
            "resolved": None,
            "status_code": getattr(exc, "status_code", None),
            "body_excerpt": str(getattr(exc, "detail", exc))[:500],
        }
    return {
        "ok": value == 8192,
        "resolved": value,
        "status_code": None,
        "body_excerpt": "",
    }


def _run_server_main_cli_contract() -> dict[str, Any]:
    from vmlx_engine import server

    observed: dict[str, Any] = {}

    def fake_load_model(model_name: str, **kwargs: Any) -> None:
        observed["load_model"] = _json_safe({"model_name": model_name, **kwargs})
        _install_fake_loaded_engine(
            server,
            model_name=model_name,
            max_tokens=int(kwargs.get("max_tokens") or 0),
            max_tokens_explicit=bool(kwargs.get("max_tokens_explicit")),
        )

    def fake_uvicorn_run(*args: Any, **kwargs: Any) -> None:
        observed["uvicorn_run"] = _json_safe(
            {"args": [str(a) for a in args], "kwargs": kwargs}
        )

    argv = [
        "python",
        "--model",
        "fake-metal-headroom-model",
        "--host",
        "127.0.0.1",
        "--port",
        "8010",
        "--max-tokens",
        "8192",
    ]
    with (
        patch.object(sys, "argv", argv),
        patch.object(server, "load_model", fake_load_model),
        patch.object(server.uvicorn, "run", fake_uvicorn_run),
    ):
        server.main()

    route = _resolve_default_clamp(server)
    load = observed.get("load_model") or {}
    ok = (
        route["ok"]
        and load.get("max_tokens") == 8192
        and load.get("max_tokens_explicit") is True
    )
    return {
        "entrypoint": "python -m vmlx_engine.server",
        "argv": argv,
        "load_model": load,
        "route": route,
        "ok": ok,
    }


def _run_vmlx_engine_serve_cli_contract() -> dict[str, Any]:
    from vmlx_engine import cli, server

    observed: dict[str, Any] = {}

    def fake_load_model(model_name: str, **kwargs: Any) -> None:
        observed["load_model"] = _json_safe({"model_name": model_name, **kwargs})
        _install_fake_loaded_engine(
            server,
            model_name=model_name,
            max_tokens=int(kwargs.get("max_tokens") or 0),
            max_tokens_explicit=bool(kwargs.get("max_tokens_explicit")),
        )

    def fake_uvicorn_run(*args: Any, **kwargs: Any) -> None:
        observed["uvicorn_run"] = _json_safe(
            {"args": [str(a) for a in args], "kwargs": kwargs}
        )

    argv = [
        "vmlx-engine",
        "serve",
        "fake-metal-headroom-model",
        "--host",
        "127.0.0.1",
        "--port",
        "8011",
        "--max-tokens",
        "8192",
    ]
    with (
        patch.object(sys, "argv", argv),
        patch.object(server, "load_model", fake_load_model),
        patch("uvicorn.run", fake_uvicorn_run),
    ):
        cli.main()

    route = _resolve_default_clamp(server)
    load = observed.get("load_model") or {}
    ok = (
        route["ok"]
        and load.get("max_tokens") == 8192
        and load.get("max_tokens_explicit") is True
    )
    return {
        "entrypoint": "vmlx-engine serve",
        "argv": argv,
        "load_model": load,
        "route": route,
        "ok": ok,
    }


def run_contract() -> dict[str, Any]:
    from vmlx_engine import server

    cli_rows = {
        "cli_server_main_explicit_max_tokens": _run_server_main_cli_contract(),
        "cli_vmlx_engine_serve_explicit_max_tokens": _run_vmlx_engine_serve_cli_contract(),
    }

    _install_fake_loaded_engine(
        server,
        model_name="fake-metal-headroom-model",
        max_tokens=8192,
        max_tokens_explicit=True,
    )

    client = TestClient(server.app, raise_server_exceptions=False)
    rows: dict[str, Any] = {}
    failures: list[str] = []
    for name, (path, body) in REQUESTS.items():
        before = len(server._engine.forward_calls)
        response = client.post(path, json=body)
        text = response.text
        forwarded = server._engine.forward_calls[before:]
        ok = response.status_code != 413 and bool(forwarded) and all(
            item["max_tokens"] == 8192 for item in forwarded
        )
        rows[name] = {
            "path": path,
            "status_code": response.status_code,
            "ok": ok,
            "body_excerpt": text[:500],
            "forwarded": forwarded,
        }
        if not ok:
            failures.append(name)

    for name, row in cli_rows.items():
        rows[name] = row
        if not row.get("ok"):
            failures.append(name)

    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "contract": "metal_headroom_projected_output_advisory_preserves_forward_limit",
        "estimated_cap": 1,
        "requested_tokens": 8192,
        "surfaces": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="build/current-metal-headroom-guard-contract.json",
    )
    args = parser.parse_args()
    result = run_contract()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "out": str(out),
                "failures": result["failures"],
            }
        )
    )
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
