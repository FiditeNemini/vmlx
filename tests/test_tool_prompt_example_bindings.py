"""Fallback examples must preserve explicitly requested scalar values."""

import re

import pytest

from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools


class PlainTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(message.get("content") or "" for message in messages)


def _examples(request, properties):
    tools = [{"type": "function", "function": {
        "name": "record_payload",
        "parameters": {"type": "object", "properties": properties,
                       "required": list(properties)},
    }}]
    prompt = check_and_inject_fallback_tools(
        "<|im_start|>system\n<tools></tools>\n<function=example_function_name>",
        [{"role": "user", "content": request}],
        tools, PlainTokenizer(), {"tools": tools, "tool_choice": "required"},
        tool_parser_id="qwen",
    )
    # Inspect only the injected example, not the echoed user request.
    example = prompt.split("<function=record_payload>", 1)[1].split("</function>", 1)[0]
    return {name: body.strip() for name, body in re.findall(
        r"<parameter=([^>]+)>\s*([\s\S]*?)\s*</parameter>", example
    )}


def test_live_typed_payload_example_preserves_requested_values():
    request = ('Call record_payload. Set content to the literal string {"n":1}, '
               'label to the string 123, flag to boolean false, '
               'nil_text to the literal string null, and false_text to the literal string false.')
    expected = {"content": '{"n":1}', "label": "123", "flag": "false",
                "nil_text": "null", "false_text": "false"}
    properties = {name: {"type": "boolean" if name == "flag" else "string"}
                  for name in expected}
    assert _examples(request, properties) == expected


@pytest.mark.parametrize("assignment,expected", [
    ("value to false", "false"), ("value is 123", "123"),
    ("value: -2.5", "-2.5"), ("value=0", "0"),
    ('value to the literal string "to"', "to"),
    ('value argument must be the literal string "is"', "is"),
    ('value to the string "hello world"', "hello world"),
    ('value to the literal string {"a":[1,2],"b":false}', '{"a":[1,2],"b":false}'),
    ("value to boolean false", "false"),
    ("value to integer -12", "-12"),
    ("value to [1, 2]", "[1, 2]"),
    ("value to `panel/package.json`", "panel/package.json"),
    ("value toad", "toad"), ("value island", "island"),
])
def test_explicit_bindings_precede_bare_word_fallback(assignment, expected):
    assert _examples("Call record_payload with " + assignment + ".",
                     {"value": {"type": "string"}}) == {"value": expected}


@pytest.mark.parametrize("assignment", ["path to panel/package.json", "path is panel/package.json"])
def test_path_binding_does_not_capture_assignment_word(assignment):
    assert _examples("Call record_payload with " + assignment + ".",
                     {"path": {"type": "string"}}) == {"path": "panel/package.json"}


def test_comma_separates_explicit_bindings_without_whitespace():
    assert _examples("Call record_payload with flag to false,label to 123.",
                     {"flag": {"type": "boolean"}, "label": {"type": "string"}}) == {
                         "flag": "false", "label": "123"}


@pytest.mark.parametrize("value", ['{"n":', '[1,', '"unfinished', '`unfinished'])
def test_incomplete_explicit_value_does_not_become_assignment_word(value):
    assert _examples("Call record_payload with content to " + value,
                     {"content": {"type": "string"}}) == {}
