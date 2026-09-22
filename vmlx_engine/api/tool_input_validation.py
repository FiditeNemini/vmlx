"""Validate client tool schemas/history before rendering a model prompt."""
import json

from jsonschema import Draft202012Validator, SchemaError


def _mapping(value):
    return value.model_dump(exclude_none=True) if hasattr(value, "model_dump") else value


def validate_tool_schemas(tools):
    for index, value in enumerate(tools or []):
        tool = _mapping(value)
        if not isinstance(tool, dict) or tool.get("type", "function") != "function":
            continue  # Built-in/custom tool types have their own contracts.
        function = tool.get("function", tool)
        if not isinstance(function, dict):
            raise ValueError(f"tools[{index}].function must be an object")
        parameters = function.get("parameters")
        if parameters is None:
            continue
        if not isinstance(parameters, dict):
            raise ValueError(f"tools[{index}].parameters must be a JSON Schema object")
        try:
            Draft202012Validator.check_schema(parameters)
        except SchemaError as exc:
            # A malformed client schema is a request error, not an unconstrained
            # generation request. Do not resolve references or rewrite schemas.
            path = "/".join(str(part) for part in exc.path) or "<root>"
            raise ValueError(f"tools[{index}].parameters is invalid at {path}: {exc.message}") from exc


def _validate_arguments(arguments, location):
    # Preserve legacy no-argument and already-decoded object histories.
    if arguments is None or arguments == "" or isinstance(arguments, dict):
        return
    if not isinstance(arguments, str):
        raise ValueError(f"{location} must be a JSON object or its JSON string")
    try:
        decoded = json.loads(arguments)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{location} contains malformed JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{location} must encode a JSON object")


def validate_tool_history(items):
    if not isinstance(items, list):
        return
    for index, value in enumerate(items):
        item = _mapping(value)
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call":
            _validate_arguments(item.get("arguments"), f"input[{index}].arguments")
        for call_index, call in enumerate(item.get("tool_calls") or []):
            if not isinstance(call, dict) or call.get("type", "function") != "function":
                continue
            function = call.get("function")
            if isinstance(function, dict):
                _validate_arguments(function.get("arguments"), f"messages[{index}].tool_calls[{call_index}].function.arguments")
