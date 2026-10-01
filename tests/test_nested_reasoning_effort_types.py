"""Malformed template effort must fail before native template execution."""
import pytest
from pydantic import ValidationError

from vmlx_engine.api.models import ChatCompletionRequest, ResponsesRequest


def request(kind, **kwargs):
    payload = ({"messages": [{"role": "user", "content": "Hello"}]}
               if kind is ChatCompletionRequest else {"input": "Hello"})
    return kind(model="native-model", **payload, **kwargs)


@pytest.mark.parametrize("kind", [ChatCompletionRequest, ResponsesRequest])
@pytest.mark.parametrize("effort", [True, False, 7, 0, 1.5, [], {}, ["low"]])
def test_nested_effort_rejects_non_string(kind, effort):
    with pytest.raises(ValidationError, match="chat_template_kwargs.reasoning_effort"):
        request(kind, chat_template_kwargs={"reasoning_effort": effort})


@pytest.mark.parametrize("kind", [ChatCompletionRequest, ResponsesRequest])
@pytest.mark.parametrize("effort", [None, "", " LOW ", "high", "max", "xhigh"])
def test_native_effort_and_null_are_not_rewritten(kind, effort):
    kwargs = {"reasoning_effort": effort, "clear_thinking": True}
    result = request(kind, reasoning_effort="high", chat_template_kwargs=kwargs)
    assert result.reasoning_effort == "high"
    assert result.chat_template_kwargs == kwargs


@pytest.mark.parametrize("kind", [ChatCompletionRequest, ResponsesRequest])
def test_omitted_effort_does_not_invent_default(kind):
    result = request(kind, chat_template_kwargs={"clear_thinking": True})
    assert result.reasoning_effort is None
    assert result.enable_thinking is None
    assert result.chat_template_kwargs == {"clear_thinking": True}


@pytest.mark.parametrize("path,fields", [
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello"}]}),
    ("/v1/responses", {"input": "Hello"}),
    ("/v1/messages", {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32}),
    ("/api/chat", {"messages": [{"role": "user", "content": "Hello"}]}),
    ("/api/generate", {"prompt": "Hello"}),
])
@pytest.mark.parametrize("effort", [False, 7])
def test_malformed_effort_rejected_before_engine(monkeypatch, path, fields, effort):
    from fastapi.testclient import TestClient
    from vmlx_engine import server

    class UnusedEngine:
        def __getattr__(self, name):
            pytest.fail(f"Invalid request reached engine: {name}")

    monkeypatch.setattr(server, "_engine", UnusedEngine())
    monkeypatch.setattr(server, "_resolve_model_name", lambda: "native-model")
    overrides = dict(server.app.dependency_overrides)
    for dependency in (server.verify_api_key, server.check_rate_limit,
                       server.check_memory_pressure, server.check_metal_working_set_pressure):
        server.app.dependency_overrides[dependency] = lambda: None
    try:
        response = TestClient(server.app).post(path, json={
            "model": "native-model", **fields,
            "chat_template_kwargs": {"reasoning_effort": effort},
        })
    finally:
        server.app.dependency_overrides.clear()
        server.app.dependency_overrides.update(overrides)
    assert response.status_code in (400, 422), response.text
    assert "reasoning_effort" in response.text
