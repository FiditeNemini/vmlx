"""Literal native XML values versus newline-framed tool dialects."""
import json
from types import SimpleNamespace

import pytest

from vmlx_engine.tool_parsers.xml_function_tool_parser import XMLFunctionToolParser

LITERAL_TEMPLATE = """The value enclosed between parameter tags is preserved exactly as-is, including newlines and spaces.
{{- '<parameter=' + args_name + '>' }}{{- args_value }}{{- '</parameter>' }}"""

@pytest.mark.parametrize('payload', ['line\n', '\nline', '\n  line\n\n', '\r\nline\r\n', '{"x":1}\n'])
@pytest.mark.parametrize('flat', [False, True], ids=['chat', 'responses'])
def test_literal_template_preserves_string_bytes(payload, flat):
    parser = XMLFunctionToolParser(SimpleNamespace(chat_template=LITERAL_TEMPLATE))
    fn = {'name': 'write_file', 'parameters': {'type': 'object', 'properties': {'content': {'type': 'string'}}}}
    request = {'tools': [{'type': 'function', **fn} if flat else {'type': 'function', 'function': fn}]}
    text = f'<tool_call><function=write_file><parameter=content>{payload}</parameter></function></tool_call>'
    result = parser.extract_tool_calls(text, request)
    assert json.loads(result.tool_calls[0]['arguments'])['content'] == payload
    streamed = parser.extract_tool_calls_streaming('', text, '</tool_call>', request=request)
    assert json.loads(streamed['tool_calls'][0]['function']['arguments'])['content'] == payload


def test_literal_contract_does_not_change_framed_parser_instances():
    literal = XMLFunctionToolParser(SimpleNamespace(chat_template=LITERAL_TEMPLATE))
    framed = XMLFunctionToolParser(None)
    body = '<tool_call><function=f><parameter=x>\n value\n</parameter></function></tool_call>'
    request = {'tools': [{'type':'function','function':{'name':'f','parameters':{'type':'object','properties':{'x':{'type':'string'}}}}}]}
    assert json.loads(literal.extract_tool_calls(body,request).tool_calls[0]['arguments'])['x'] == '\n value\n'
    assert json.loads(framed.extract_tool_calls(body,request).tool_calls[0]['arguments'])['x'] == ' value'
