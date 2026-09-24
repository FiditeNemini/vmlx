# SPDX-License-Identifier: Apache-2.0
"""XML values follow the declared schema before JSON decoding destroys bytes."""
import json

import pytest

from vmlx_engine.tool_parsers.qwen_tool_parser import QwenToolParser


def request_for(prop, *, flat=False):
    fn = {"name": "write_file", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "content": prop,
    }, "required": ["path", "content"]}}
    return {"tools": [{"type": "function", **fn} if flat else {"type": "function", "function": fn}]}


def xml_call(value, *, wrapper="normal"):
    body = "<parameter=path>\n/tmp/fixture.json\n</parameter>\n" + (
        f"<parameter=content>\n{value}\n</parameter>"
    )
    if wrapper == "orphan":
        return f"<tool_call>\nwrite_file\n{body}\n</function>\n</tool_call>"
    if wrapper == "doubled":
        return f"<tool_call><function=tool_call><function=write_file>\n{body}\n</function></function></tool_call>"
    return f"<tool_call><function=write_file>\n{body}\n</function></tool_call>"


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("wrapper", ["normal", "orphan", "doubled"])
@pytest.mark.parametrize("value", ['{"code":"RC64-4S","n":17}\n', '[1,2]', '17', 'true', 'null', '"quoted"', r'line\npath\t', '    x = 17\n\n'])
def test_declared_string_keeps_native_xml_bytes(flat, wrapper, value):
    parser = QwenToolParser()
    request = request_for({"type": "string"}, flat=flat)
    text = xml_call(value, wrapper=wrapper)
    result = parser.extract_tool_calls(text, request=request)
    assert result.tools_called and len(result.tool_calls) == 1
    assert json.loads(result.tool_calls[0]["arguments"]) == {"path": "/tmp/fixture.json", "content": value}
    streamed = parser.extract_tool_calls_streaming("", text, text, request=request)
    assert streamed and len(streamed["tool_calls"]) == 1
    assert json.loads(streamed["tool_calls"][0]["function"]["arguments"])["content"] == value


@pytest.mark.parametrize("prop,value,expected", [
    ({"type": "object"}, '{"n":17}', {"n": 17}),
    ({"type": "array"}, '[1,2]', [1, 2]),
    ({"type": "integer"}, '17', 17),
    ({"type": "boolean"}, 'false', False),
    ({"type": ["string", "object"]}, '{"n":17}', {"n": 17}),
    ({"anyOf": [{"type": "string"}, {"type": "object"}]}, '{"n":17}', {"n": 17}),
    ({"type": ["string", "null"]}, 'null', None),
    ({"type": "string", "nullable": True}, 'None', None),
    ({"oneOf": [{"type": "string"}, {"type": "null"}]}, '{"n":17}', '{"n":17}'),
    ({}, '{"n":17}', {"n": 17}),
])
def test_nonstring_union_nullable_and_untyped_policy(prop, value, expected):
    result = QwenToolParser().extract_tool_calls(xml_call(value), request=request_for(prop))
    assert json.loads(result.tool_calls[0]["arguments"])["content"] == expected


def test_json_native_object_is_not_fabricated_into_a_string():
    raw = '<tool_call>{"name":"write_file","arguments":{"path":"x","content":{"n":17}}}</tool_call>'
    result = QwenToolParser().extract_tool_calls(raw, request=request_for({"type": "string"}))
    assert json.loads(result.tool_calls[0]["arguments"])["content"] == {"n": 17}


@pytest.mark.parametrize("value,expected", [("False", False), ("TRUE", True), (" false ", False)])
@pytest.mark.parametrize("flat", [False, True])
def test_referenced_xml_boolean_spelling_is_typed_on_both_parser_paths(value, expected, flat):
    request = request_for({"$ref": "#/$defs/flag"}, flat=flat)
    fn = request["tools"][0] if flat else request["tools"][0]["function"]
    fn["parameters"]["$defs"] = {"flag": {"type": "boolean"}}
    text = xml_call(value)
    parser = QwenToolParser()
    result = parser.extract_tool_calls(text, request=request)
    assert json.loads(result.tool_calls[0]["arguments"])["content"] is expected
    streamed = parser.extract_tool_calls_streaming("", text, text, request=request)
    assert json.loads(streamed["tool_calls"][0]["function"]["arguments"])["content"] is expected


@pytest.mark.parametrize("prop", [{"type": "string"}, {"type": ["boolean", "string"]}, {}])
def test_xml_boolean_spelling_does_not_override_string_or_ambiguous_schema(prop):
    result = QwenToolParser().extract_tool_calls(xml_call("False"), request=request_for(prop))
    assert json.loads(result.tool_calls[0]["arguments"])["content"] == "False"


def test_json_native_quoted_boolean_is_not_retyped():
    raw = '<tool_call>{"name":"write_file","arguments":{"path":"x","content":"False"}}</tool_call>'
    result = QwenToolParser().extract_tool_calls(raw, request=request_for({"type": "boolean"}))
    assert json.loads(result.tool_calls[0]["arguments"])["content"] == "False"
