"""CPU-only request schema and actual generation-kwarg forwarding contracts."""

import ast
from pathlib import Path

import pytest

from vmlx_engine.api.models import ChatCompletionRequest, ResponsesRequest


def _forwarder():
    # Execute the production pure helper without importing the GPU server.
    path = Path(__file__).resolve().parents[1] / "vmlx_engine/server.py"
    tree = ast.parse(path.read_text())
    nodes = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "_forward_reasoning_effort_kwargs")
        or (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_EFFORT_THINKING_BUDGET"
            for target in node.targets
        ))
    ]
    assert len(nodes) == 2
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_forward_reasoning_effort_kwargs"]


@pytest.fixture(scope="module")
def forward():
    return _forwarder()


@pytest.mark.parametrize("request_type", [ChatCompletionRequest, ResponsesRequest])
@pytest.mark.parametrize("fields,expected_enabled,expected_effort", [
    ({}, None, None),
    ({"enable_thinking": True, "chat_template_kwargs": {"enable_thinking": True}}, True, None),
    ({"enable_thinking": False, "thinking_mode": "instruct"}, False, None),
    # Public callers explicitly using the legacy alias retain its existing meaning.
    ({"thinking_mode": "reasoning"}, True, "medium"),
    *[({"enable_thinking": True, "reasoning_effort": effort,
        "thinking_mode": "max" if effort == "max" else "reasoning"}, True, effort)
      for effort in ("low", "medium", "high", "xhigh", "max")],
])
def test_schema_and_generation_forwarding(request_type, fields, expected_enabled, expected_effort, forward):
    payload = {"model": "bundle", **fields}
    payload["input" if request_type is ResponsesRequest else "messages"] = [{"role": "user", "content": "hello"}]
    request = request_type.model_validate(payload)
    assert request.enable_thinking is expected_enabled
    assert request.reasoning_effort == expected_effort
    generation = {}
    template = dict(request.chat_template_kwargs or {})
    forward(generation, template, request.reasoning_effort)
    assert generation.get("reasoning_effort") == expected_effort
    assert template.get("reasoning_effort") == expected_effort
    if expected_effort is None:
        assert "thinking_budget" not in template
