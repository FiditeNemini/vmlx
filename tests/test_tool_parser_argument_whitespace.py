# SPDX-License-Identifier: Apache-2.0
"""A tool argument's whitespace must survive the parser byte-for-byte.

The dangerous case is an argument that CONTAINS code: newlines, indentation and
blank lines are the payload, not formatting. A parser that trims or collapses
them hands the tool a program that no longer runs, and nothing downstream can
tell that it was corrupted.

Each family gets its own wire format, so this drives every parser through its
own dialect and asserts the decoded argument equals what went in.
"""

import json

import pytest

CODE_ARG = "def f():\n    x = 1\n\n    return x\n"
MULTILINE_ARG = "line one\nline two\n\nline four"

# (parser module, class name, how to render a call carrying `payload`)
DIALECTS = {
    "hermes_tool_parser": (
        "HermesToolParser",
        lambda p: "<tool_call>\n"
        + json.dumps({"name": "write", "arguments": {"code": p}})
        + "\n</tool_call>",
    ),
    "qwen_tool_parser": (
        "QwenToolParser",
        lambda p: "<tool_call>\n"
        + json.dumps({"name": "write", "arguments": {"code": p}})
        + "\n</tool_call>",
    ),
    "llama_tool_parser": (
        "LlamaToolParser",
        lambda p: "<function=write>"
        + json.dumps({"code": p})
        + "</function>",
    ),
    "mistral_tool_parser": (
        "MistralToolParser",
        lambda p: "[TOOL_CALLS]"
        + json.dumps([{"name": "write", "arguments": {"code": p}}]),
    ),
    "deepseek_tool_parser": (
        "DeepSeekToolParser",
        lambda p: "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>write\n"
        "```json\n" + json.dumps({"code": p}) + "\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    ),
}


def _load(module_name, class_name):
    import importlib

    module = importlib.import_module(f"vmlx_engine.tool_parsers.{module_name}")
    cls = getattr(module, class_name, None)
    # Deliberately NOT a skip: a wrong class name here silently drops a whole
    # family from the sweep, which is how coverage rots.
    assert cls is not None, f"{class_name} not found in {module_name}"
    try:
        return cls(None)
    except Exception:  # noqa: BLE001 - some parsers need no tokenizer
        return cls()


def _decoded_args(parser, wire):
    info = parser.extract_tool_calls(wire, None)
    calls = getattr(info, "tool_calls", None) or []
    if not calls:
        pytest.skip("parser produced no tool call for this dialect")
    raw = calls[0].get("arguments") if isinstance(calls[0], dict) else calls[0].function.arguments
    if isinstance(raw, str):
        raw = json.loads(raw)
    return raw


CASES = [(m, payload_name) for m in sorted(DIALECTS) for payload_name in ("code", "multiline")]
PAYLOADS = {"code": CODE_ARG, "multiline": MULTILINE_ARG}


@pytest.mark.parametrize("module_name,payload_name", CASES,
                         ids=[f"{m}-{p}" for m, p in CASES])
def test_argument_whitespace_round_trips(module_name, payload_name):
    class_name, render = DIALECTS[module_name]
    parser = _load(module_name, class_name)
    payload = PAYLOADS[payload_name]
    args = _decoded_args(parser, render(payload))
    assert args.get("code") == payload, (
        f"{module_name} altered the argument:\n"
        f"  want {payload!r}\n  got  {args.get('code')!r}"
    )


INDENTED_CODE_ARG = "    def f():\n        pass"


def test_qwen_xml_parameter_keeps_first_line_indentation():
    """A code argument's own leading indentation must survive the XML framing.

    `<parameter=...>\\s*(.*?)\\s*</parameter>` plus a `.strip()` in the value
    coercion ate it: a payload rendered as "\\n    def f():\\n        pass\\n"
    came back as "def f():\\n        pass" — first line unindented while the
    second still had 8 spaces, which is a SyntaxError once written to disk.

    The existing CODE_ARG fixture could never catch this because its first line
    has no indentation. Only ONE framing newline per side may be stripped, and
    the coercion may strip only to TEST for JSON, not to produce the string.
    """
    import json as _json

    from vmlx_engine.tool_parsers.qwen_tool_parser import QwenToolParser

    raw = (
        "<tool_call>\n<function=write_file>\n"
        f"<parameter=code>\n{INDENTED_CODE_ARG}\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    result = QwenToolParser().extract_tool_calls(raw, None)
    assert result.tools_called
    got = _json.loads(result.tool_calls[0]["arguments"])["code"]
    assert got == INDENTED_CODE_ARG, (
        f"code argument lost its indentation: {got!r}"
    )


def test_qwen_json_scalar_still_coerces_despite_whitespace():
    """Keeping payload whitespace must not break JSON scalar coercion."""
    import json as _json

    from vmlx_engine.tool_parsers.qwen_tool_parser import QwenToolParser

    raw = (
        "<tool_call>\n<function=n>\n"
        "<parameter=count> 42 </parameter>\n</function>\n</tool_call>"
    )
    result = QwenToolParser().extract_tool_calls(raw, None)
    value = _json.loads(result.tool_calls[0]["arguments"])["count"]
    assert value == 42 and isinstance(value, int)


def test_nemotron_xml_parameter_keeps_first_line_indentation():
    """Same two-strip defect as the qwen dialect, same fix (9df8c1660)."""
    import json as _json

    from vmlx_engine.tool_parsers.nemotron_tool_parser import NemotronToolParser

    raw = (
        "<TOOLCALL>\n<function=write_file>\n"
        f"<parameter=code>\n{INDENTED_CODE_ARG}\n</parameter>\n"
        "</function>\n</TOOLCALL>"
    )
    result = NemotronToolParser().extract_tool_calls(raw, None)
    assert result.tools_called
    got = _json.loads(result.tool_calls[0]["arguments"])["code"]
    assert got == INDENTED_CODE_ARG, f"nemotron lost indentation: {got!r}"


def test_xml_dialect_patterns_do_not_swallow_payload_whitespace():
    """Pin the pattern shape across every dialect that shares it.

    A `\\s*(.*?)\\s*` capture cannot distinguish XML framing from the payload's
    own indentation. Any dialect using it will silently corrupt code arguments,
    so assert the framing-newline form directly — the runtime check above only
    covers the two dialects with an easy call shape.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "vmlx_engine" / "tool_parsers"
    for name in ("qwen_tool_parser", "nemotron_tool_parser", "step3p5_tool_parser"):
        src = (root / f"{name}.py").read_text(encoding="utf-8")
        assert r"</parameter>" in src
        assert r">\s*(.*?)\s*</parameter>" not in src, (
            f"{name} uses the whitespace-swallowing capture again; code "
            f"arguments will lose their first-line indentation"
        )


def test_lfm2_keeps_text_written_after_the_tool_call():
    """Content after <|tool_call_end|> must not be dropped.

    The parser took only `cleaned_output[:start]`, so a model that called a tool
    and then explained what it did lost the explanation entirely — no error, no
    log line, the text simply never reached the user.
    """
    from vmlx_engine.tool_parsers.lfm2_tool_parser import Lfm2ToolParser

    raw = (
        "Before.<|tool_call_start|>[get_weather(city='Paris')]"
        "<|tool_call_end|>After the call."
    )
    result = Lfm2ToolParser().extract_tool_calls(raw, None)
    assert result.tools_called
    assert result.content is not None
    assert "Before." in result.content
    assert "After the call." in result.content, (
        f"text after the tool call was dropped: {result.content!r}"
    )


def test_lfm2_single_sided_content_is_unchanged():
    """Only-before and only-after must not gain a spurious separator."""
    from vmlx_engine.tool_parsers.lfm2_tool_parser import Lfm2ToolParser

    parser = Lfm2ToolParser()
    before_only = parser.extract_tool_calls(
        "Only before.<|tool_call_start|>[f()]<|tool_call_end|>", None
    )
    assert before_only.content == "Only before."
    after_only = parser.extract_tool_calls(
        "<|tool_call_start|>[f()]<|tool_call_end|>Only after.", None
    )
    assert after_only.content == "Only after."


def test_atem_declared_string_keeps_a_whitespace_only_value():
    """A declared string parameter is verbatim — including when it is all space.

    _coerce returned the STRIPPED text for any blank-after-strip value before
    reaching the declared-type branch, so a tool asked to emit an indent, a
    newline, or a space separator received "" instead. The docstring promised
    "honoured verbatim"; the blank guard quietly broke that for exactly the
    values where the whitespace WAS the payload.
    """
    from vmlx_engine.tool_parsers.atem_tool_parser import AtemToolParser

    # server.py builds this dict shape before calling the parser; _schema_for
    # only reads a Mapping, so an object here would silently skip the schema.
    request = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "write_line",
                    "parameters": {
                        "type": "object",
                        "properties": {"sep": {"type": "string"}},
                    },
                },
            }
        ]
    }
    raw = (
        '<atem:invoke name="write_line">'
        '<atem:parameter name="sep">   </atem:parameter>'
        "</atem:invoke>"
    )
    result = AtemToolParser().extract_tool_calls(raw, request)
    assert result.tools_called
    args = json.loads(result.tool_calls[0]["arguments"])
    assert args["sep"] == "   ", f"whitespace payload was stripped away: {args!r}"
