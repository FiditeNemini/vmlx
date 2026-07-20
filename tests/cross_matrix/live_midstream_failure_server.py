#!/usr/bin/env python3
"""Live proof server for progressive-output failure and immediate recovery.

This intentionally uses the production ``stream_chat_completion`` and
``stream_responses_api`` generators.  Only model inference is replaced by a
deterministic engine that raises after two visible deltas when the prompt
contains ``FAIL``.  The harness therefore exercises the real server terminal
error/usage contract without loading a multi-gigabyte model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

import vmlx_engine.server as server
from vmlx_engine.api.models import ChatCompletionRequest, ResponsesRequest
from vmlx_engine.engine.base import GenerationOutput


MODEL_ID = "midstream-live-proof"


def _message_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(value or "")


def _latest_user_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if hasattr(message, "model_dump"):
            message = message.model_dump(exclude_none=True)
        if isinstance(message, dict) and message.get("role") == "user":
            return _message_text(message.get("content"))
    return ""


class FailureProofEngine:
    is_mllm = False
    tokenizer = SimpleNamespace(has_thinking=False)

    def __init__(self) -> None:
        self.aborted: list[str] = []

    async def stream_chat(self, *, messages: list[Any], **kwargs: Any):
        prompt = _latest_user_text(messages)
        is_chat = "UI-CHAT" in prompt or "RAW-CHAT" in prompt
        prefix = "CHAT-" if is_chat else "RESP-"
        if "FAIL" in prompt:
            chunks = (f"{prefix}PARTIAL-", "VISIBLE")
            text = ""
            for index, delta in enumerate(chunks, start=1):
                text += delta
                yield GenerationOutput(
                    text=text,
                    new_text=delta,
                    tokens=[index],
                    prompt_tokens=5,
                    completion_tokens=index,
                    finished=False,
                    finish_reason=None,
                )
                await asyncio.sleep(0.35)
            await asyncio.sleep(0.45)
            raise RuntimeError(f"{prefix}MIDSTREAM-PROBE-FAILURE")

        chunks = (f"{prefix}RECOVERY-", "OK")
        text = ""
        for index, delta in enumerate(chunks, start=1):
            text += delta
            yield GenerationOutput(
                text=text,
                new_text=delta,
                tokens=[index],
                prompt_tokens=6,
                completion_tokens=index,
                finished=index == len(chunks),
                finish_reason="stop" if index == len(chunks) else None,
            )
            await asyncio.sleep(0.25)

    async def abort_request(self, request_id: str) -> bool:
        self.aborted.append(request_id)
        return True


def build_app(request_log: Path) -> FastAPI:
    app = FastAPI()
    engine = FailureProofEngine()

    server._model_name = MODEL_ID
    server._model_path = None
    server._reasoning_parser = None
    server._tool_call_parser = None
    server._tool_call_parser_disabled_explicitly = True
    server._default_timeout = 10.0

    def record(endpoint: str, body: dict[str, Any]) -> None:
        request_log.parent.mkdir(parents=True, exist_ok=True)
        with request_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"endpoint": endpoint, "body": body}) + "\n")

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_name": MODEL_ID,
            "engine_type": "live-proof-production-streamers",
            "aborted_requests": len(engine.aborted),
        }

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [{"id": MODEL_ID, "object": "model", "owned_by": "vmlx-proof"}],
        }

    @app.post("/v1/responses")
    async def responses(request: Request):
        body = await request.json()
        record("/v1/responses", body)
        parsed = ResponsesRequest(**body)
        messages = server._responses_input_to_messages(
            parsed.input,
            parsed.instructions,
            preserve_multimodal=False,
        )
        if not parsed.stream:
            return JSONResponse(status_code=400, content={"error": "stream=true required"})
        return StreamingResponse(
            server.stream_responses_api(
                engine,
                messages,
                parsed,
                fastapi_request=request,
                history_messages=messages,
            ),
            media_type="text/event-stream",
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        record("/v1/chat/completions", body)
        parsed = ChatCompletionRequest(**body)
        messages = [
            item.model_dump(exclude_none=True)
            if hasattr(item, "model_dump")
            else item
            for item in parsed.messages
        ]
        if not parsed.stream:
            return JSONResponse(status_code=400, content={"error": "stream=true required"})
        return StreamingResponse(
            server.stream_chat_completion(
                engine,
                messages,
                parsed,
                fastapi_request=request,
            ),
            media_type="text/event-stream",
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--request-log", required=True, type=Path)
    args = parser.parse_args()
    uvicorn.run(
        build_app(args.request_log),
        host="127.0.0.1",
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
