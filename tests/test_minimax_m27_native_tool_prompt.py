"""MiniMax-M2.7 templates render tool schemas natively (<tools> block + one
generic <invoke> example). The fallback probe must accept that as the native
contract instead of injecting a second schema (user report 2026-09-04: every
tool call logged "needs fallback tool schema injection" and parsed calls were
then dropped for empty required arguments)."""

import json
import logging
from pathlib import Path

import jinja2
from jinja2 import sandbox

from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools
from vmlx_engine.tool_parsers.minimax_tool_parser import MiniMaxToolParser

FIXTURE = Path(__file__).parent / "fixtures" / "minimax_m27_chat_template.jinja"

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "terminal", "description": "Run a command",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]


class _M27Tokenizer:
    """Renders the real bundle template the way transformers does."""

    def __init__(self) -> None:
        env = sandbox.ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
        )
        env.filters["tojson"] = lambda v, ensure_ascii=False, **kw: json.dumps(v, ensure_ascii=ensure_ascii, **kw)
        env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(jinja2.TemplateError(m))
        self._template = env.from_string(FIXTURE.read_text())

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, tokenize=False, **kwargs):
        return self._template.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt, **kwargs)


def test_m27_template_is_accepted_as_native(caplog):
    tok = _M27Tokenizer()
    messages = [{"role": "user", "content": "List the repo root."}]
    prompt = tok.apply_chat_template(messages, tools=TOOLS)
    assert "<tools>" in prompt and "read_file" in prompt and "terminal" in prompt
    assert '<invoke name="tool-name-1">' in prompt  # generic example, not per-tool
    with caplog.at_level(logging.WARNING, logger="vmlx_engine.api.tool_calling"):
        out = check_and_inject_fallback_tools(
            prompt, messages, TOOLS, tok, {"tools": TOOLS}, tool_parser_id="minimax"
        )
    assert out == prompt, "native M2.7 tool prompt must not be re-rendered with an injected schema"
    assert "needs fallback tool schema injection" not in caplog.text


def test_m27_template_without_tools_block_still_falls_back():
    # A template that silently drops the tools kwarg must still get the injection.
    class _Bare:
        def apply_chat_template(self, messages, **_kw):
            return "".join(f"[{m['role']}]{m.get('content','')}" for m in messages) + "[assistant]"
    messages = [{"role": "user", "content": "hi"}]
    prompt = _Bare().apply_chat_template(messages)
    out = check_and_inject_fallback_tools(prompt, messages, TOOLS, _Bare(), {"tools": TOOLS}, tool_parser_id="minimax")
    assert out != prompt and "read_file" in out


def test_m27_parser_keeps_required_arguments():
    text = (
        "<minimax:tool_call>\n<invoke name=\"terminal\">\n<parameter name=\"command\">ls -la\n"
        "src/</parameter>\n</invoke>\n<invoke name=\"read_file\">\n<parameter name=\"path\">README.md</parameter>\n"
        "</invoke>\n</minimax:tool_call>"
    )
    parser = MiniMaxToolParser(tokenizer=None)
    result = parser.extract_tool_calls(text, request=None)
    calls = result.tool_calls

    def _fn(c):
        # The parser yields plain dicts: {"id", "name", "arguments"}.
        name = c["name"] if isinstance(c, dict) else c.function.name
        args = c["arguments"] if isinstance(c, dict) else c.function.arguments
        return name, (json.loads(args) if isinstance(args, str) else args)

    parsed = [_fn(c) for c in calls]
    assert [n for n, _ in parsed] == ["terminal", "read_file"]
    assert parsed[0][1]["command"] == "ls -la\nsrc/"
    assert parsed[1][1]["path"] == "README.md"


class _FixtureTokenizer(_M27Tokenizer):
    def __init__(self, name: str) -> None:
        env = sandbox.ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
        )
        env.filters["tojson"] = lambda v, ensure_ascii=False, **kw: json.dumps(v, ensure_ascii=ensure_ascii, **kw)
        env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(jinja2.TemplateError(m))
        self._template = env.from_string((Path(__file__).parent / "fixtures" / name).read_text())


def test_qwen38_27b_and_nanbeige_json_schema_templates_are_native(caplog):
    # Both render tools as JSON entries inside <tools> plus a
    # <function=example_function_name> exemplar; the xml_function rule used to
    # demand <name>tool</name> XML entries and sent them through the fallback.
    for fixture, parser in (
        ("qwen38_27b_d_chat_template.jinja", "qwen3_coder"),
        ("nanbeige42_chat_template.jinja", "xml_function"),
    ):
        tok = _FixtureTokenizer(fixture)
        messages = [{"role": "user", "content": "List the repo root."}]
        prompt = tok.apply_chat_template(messages, tools=TOOLS)
        assert "<tools>" in prompt and '"name": "read_file"' in prompt
        with caplog.at_level(logging.WARNING, logger="vmlx_engine.api.tool_calling"):
            out = check_and_inject_fallback_tools(
                prompt, messages, TOOLS, tok, {"tools": TOOLS}, tool_parser_id=parser
            )
        assert out == prompt, f"{fixture} must be accepted as native"
        assert "needs fallback tool schema injection" not in caplog.text
