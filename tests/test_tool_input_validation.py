import json

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from vmlx_engine.api.models import ChatCompletionRequest, ResponsesRequest
from vmlx_engine.api.validation_errors import request_validation_error_response


app = FastAPI()
app.add_exception_handler(RequestValidationError, request_validation_error_response)


@app.post("/chat")
def chat(request: ChatCompletionRequest):
    return request.model_dump(exclude_none=True)


@app.post("/responses")
def responses(request: ResponsesRequest):
    return request.model_dump(exclude_none=True)


client = TestClient(app)


def body(path, function):
    if path == "/chat":
        return {"model": "test", "messages": [{"role": "user", "content": "Hello"}],
                "tools": [{"type": "function", "function": function}]}
    return {"model": "test", "input": "Hello", "tools": [{"type": "function", **function}]}


@pytest.mark.parametrize("path", ["/chat", "/responses"])
@pytest.mark.parametrize("parameters", [[], {"type": "object", "properties": []}, {"required": "label"}])
def test_malformed_schema_is_a_client_error(path, parameters):
    result = client.post(path, json=body(path, {"name": "lookup", "parameters": parameters}))
    assert result.status_code == 422
    assert "parameters" in result.text


@pytest.mark.parametrize("path", ["/chat", "/responses"])
@pytest.mark.parametrize("arguments", [
    '{"label":', '["alpha"]', '"alpha"',
    '{"value": NaN}', '{"value": Infinity}', '{"value": -Infinity}', '{"value": 1e999}',
])
def test_malformed_prior_function_arguments_are_rejected(path, arguments):
    call = {"id": "call_a", "type": "function", "function": {"name": "lookup", "arguments": arguments}}
    request = body(path, {"name": "lookup"})
    if path == "/chat":
        request['messages'].append({"role": "assistant", "tool_calls": [call]})
    else:
        request['input'] = [{"type": "function_call", "call_id": "call_a", "name": "lookup", "arguments": arguments}]
    result = client.post(path, json=request)
    assert result.status_code == 422
    assert "arguments" in result.text


@pytest.mark.parametrize("path", ["/chat", "/responses"])
def test_valid_nested_schema_and_escaped_history_are_preserved(path):
    schema = {"type": "object", "properties": {"options": {"type": "object", "properties": {"note": {"type": "string"}}, "additionalProperties": False}}, "$defs": {"flag": {"type": "boolean"}}}
    request = body(path, {"name": "lookup", "parameters": schema})
    arguments = json.dumps({"options": {"note": 'quote " slash \\ snow 雪'}})
    call = {"type": "function_call", "name": "old_tool", "call_id": "call_a", "arguments": arguments}
    if path == "/chat":
        request['messages'].append({"role": "assistant", "tool_calls": [{"id": "call_a", "type": "function", "function": {"name": "old_tool", "arguments": arguments}}]})
    else:
        request['input'] = [call, {"type": "function_call_output", "call_id": "call_a", "output": "not JSON, intentionally"}]
    result = client.post(path, json=request)
    assert result.status_code == 200
    assert result.json()['tools'] == request['tools']
    history_key = 'messages' if path == '/chat' else 'input'
    assert result.json()[history_key] == request[history_key]


def test_responses_builtin_tools_and_plain_text_are_unaffected():
    result = client.post('/responses', json={'model': 'test', 'input': 'hello', 'tools': [{'type': 'web_search'}]})
    assert result.status_code == 200


@pytest.mark.parametrize("path", ["/chat", "/responses"])
def test_declared_older_schema_draft_is_preserved(path):
    schema = {"$schema": "http://json-schema.org/draft-04/schema#",
              "type": "object", "properties": {
                  "count": {"type": "number", "minimum": 0, "exclusiveMinimum": True}}}
    request = body(path, {"name": "lookup", "parameters": schema})
    result = client.post(path, json=request)
    assert result.status_code == 200
    assert result.json()['tools'] == request['tools']
    # The same boolean is invalid when the caller declares draft 2020-12.
    schema['$schema'] = "https://json-schema.org/draft/2020-12/schema"
    assert client.post(path, json=body(path, {"name": "lookup", "parameters": schema})).status_code == 422


@pytest.mark.parametrize("path", ["/chat", "/responses"])
def test_decoded_history_object_rejects_nonfinite_values(path):
    request = body(path, {"name": "lookup"})
    arguments = {"value": float("nan")}
    if path == "/chat":
        request['messages'].append({"role": "assistant", "tool_calls": [
            {"type": "function", "function": {"name": "lookup", "arguments": arguments}}]})
    else:
        request['input'] = [{"type": "function_call", "call_id": "call_a",
                             "name": "lookup", "arguments": arguments}]
    # Raw body exercises parsers which accept non-standard JSON constants;
    # the request validator must still reject them before model inference.
    result = client.post(path, content=json.dumps(request), headers={"Content-Type": "application/json"})
    assert result.status_code == 422


@pytest.mark.parametrize("path", ["/chat", "/responses"])
def test_real_server_rejects_nonfinite_tool_history_without_loading_model(path):
    from vmlx_engine.server import app as server_app

    request = body(path, {"name": "lookup"})
    function = {"name": "lookup", "arguments": {"value": float("inf")}}
    if path == "/chat":
        request['messages'].append({"role": "assistant", "tool_calls": [
            {"id": "call_a", "type": "function", "function": function}]})
        endpoint = "/v1/chat/completions"
    else:
        request['input'] = [{"type": "function_call", "call_id": "call_a", **function}]
        endpoint = "/v1/responses"
    # No context manager: lifespan/model loading is intentionally not started.
    result = TestClient(server_app).post(
        endpoint, content=json.dumps(request), headers={"Content-Type": "application/json"})
    assert result.status_code == 422
    assert result.json()['detail'][0]['type'] == 'value_error'


@pytest.mark.parametrize("path", ["/chat", "/responses"])
@pytest.mark.parametrize("schema", [
    {"$schema": "https://example.invalid/unknown-draft", "type": "object"},
    {"$schema": 123, "type": "object"},
    {"type": "object", "properties": {"value": {"minimum": float("nan")}}},
])
def test_unsupported_drafts_and_nonfinite_schemas_are_client_errors(path, schema):
    request = body(path, {"name": "lookup", "parameters": schema})
    result = client.post(path, content=json.dumps(request), headers={"Content-Type": "application/json"})
    assert result.status_code == 422
    assert "parameters" in result.text
