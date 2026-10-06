"""Incomplete trailing native envelopes are not assistant prose or new calls."""
import pytest
from vmlx_engine.tool_parsers.xml_function_tool_parser import XMLFunctionToolParser
from vmlx_engine.request_diagnostics import begin_capture, take

CALL = '<tool_call>\n<function=multiply>\n<parameter=a>17</parameter>\n<parameter=b>23</parameter>\n</function>\n</tool_call>'
REQUEST = {'tools':[{'type':'function','function':{'name':'multiply','parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}}}}}]}


def test_completed_parallel_calls_keep_identity_but_not_partial_tail():
    begin_capture()
    result = XMLFunctionToolParser().extract_tool_calls(CALL + CALL + '<tool_call>\n<function=multiply>\n<parameter=a>\n1', REQUEST)
    assert result.tools_called and len(result.tool_calls) == 2
    assert not result.content
    assert len({c['id'] for c in result.tool_calls}) == 2
    assert any('incomplete trailing tool call' in w for w in take())


@pytest.mark.parametrize('tail', ['The calculation is ready.', 'Example: <tool_call>\n<function=multiply>', '```xml\n<tool_call>\n<function=multiply>\n```'])
def test_plain_or_quoted_content_is_not_removed(tail):
    result = XMLFunctionToolParser().extract_tool_calls(CALL + tail, REQUEST)
    assert result.content == tail
