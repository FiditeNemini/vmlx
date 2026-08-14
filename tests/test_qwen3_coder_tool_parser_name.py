"""The `qwen3_coder` tool-parser name must resolve, and to the RIGHT parser.

Qwen3.6-27B D-series bundles are stamped `tool parser qwen3_coder`. Before the
name existed the engine exited at startup ("Tool parser 'qwen3_coder' not
found") — the whole session died for every stamped bundle, found live the
moment the first D-series bundle was served.

The name follows the FORMAT, not the model family: the D-series chat template
emits `<tool_call>\n<function=NAME>\n<parameter=KEY>` — the XML-function shape,
NOT the plain `<tool_call>{json}` the `qwen`/`qwen3` parser reads. Aliasing it
to the qwen parser would have started the server and then failed on every
actual tool call, which is strictly worse than failing at startup.
"""

from vmlx_engine.tool_parsers.abstract_tool_parser import ToolParserManager
from vmlx_engine.tool_parsers.xml_function_tool_parser import XMLFunctionToolParser


def test_qwen3_coder_resolves():
    assert ToolParserManager.get_tool_parser("qwen3_coder") is XMLFunctionToolParser


def test_qwen3_coder_is_the_xml_format_not_the_json_one():
    from vmlx_engine.tool_parsers.qwen_tool_parser import QwenToolParser

    p = ToolParserManager.get_tool_parser("qwen3_coder")
    assert p is not QwenToolParser, (
        "qwen3_coder emits <function=NAME><parameter=KEY> XML, not qwen JSON; "
        "binding it to the qwen parser starts the server and then breaks every "
        "tool call"
    )


def test_qwen3_coder_parses_the_d_series_template_shape():
    # The exact shape the bundle's chat_template.jinja emits.
    text = (
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nSeoul\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    parser = XMLFunctionToolParser.__new__(XMLFunctionToolParser)
    m = XMLFunctionToolParser.TOOL_CALL_PATTERN.search(text)
    assert m is not None
    f = XMLFunctionToolParser.FUNCTION_PATTERN.search(m.group(1))
    assert f is not None and f.group(1) == "get_weather"
    a = XMLFunctionToolParser.PARAM_PATTERN.search(f.group(2))
    assert a is not None and a.group(1) == "city" and a.group(2).strip() == "Seoul"
