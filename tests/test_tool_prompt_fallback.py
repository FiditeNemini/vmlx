# SPDX-License-Identifier: Apache-2.0
"""Tool prompt fallback contracts.

These tests pin the native fallback examples that are injected when a model's
chat template drops tool schemas. The examples must not invent fake parameters:
models copy those examples, and a fake arg on a zero-argument tool corrupts
mixed built-in/MCP tool calls on DSV4.
"""

import json

from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools
from vmlx_engine.loaders.dsv4_chat_encoder import select_tools_for_explicit_request


class DSV4LikeTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        rendered = []
        for message in messages:
            role = message.get("role")
            content = message.get("content") or ""
            if role == "system":
                rendered.append(content)
            elif role == "user":
                rendered.append(f"<｜User｜>{content}")
            elif role == "assistant":
                rendered.append(f"<｜Assistant｜>{content}")
        rendered.append("<｜Assistant｜>")
        return "\n".join(rendered)


class PlainTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        return "\n".join((message.get("content") or "") for message in messages)


class OpenPanguLikeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        tools = kwargs.get("tools") or []
        rendered = ["<|pangu_text_start|><|message_start|>system"]
        if tools:
            rendered.append("<tools>")
            rendered.append(json.dumps(tools, ensure_ascii=False))
            rendered.append("</tools>")
            rendered.append(
                "<|tool_call_start|>\n"
                '[{"name": "<函数名1>", "arguments": <args1 json对象>}]\n'
                "<|tool_call_end|>"
            )
        rendered.append("<|message_end|>")
        for message in messages:
            role = message.get("role")
            content = message.get("content") or ""
            if role == "system":
                rendered.append(f"<|message_start|>system\n{content}<|message_end|>")
            elif role == "user":
                rendered.append(f"<|message_start|>user\n{content}<|message_end|>")
            elif role == "assistant":
                rendered.append(f"<|message_start|>assistant\n{content}<|message_end|>")
            elif role == "tool":
                rendered.append(f"<|message_start|>tool\n{content}<|message_end|>")
        rendered.append("<|message_start|>assistant\n")
        return "\n".join(rendered)


def _qwen_prompt() -> str:
    return """<|im_start|>system
# Tools
<tools>
{"type":"function","function":{"name":"run_command"}}
{"type":"function","function":{"name":"read_file"}}
</tools>
<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
</function>
</tool_call>
<|im_end|>
<|im_start|>user
Use a tool.<|im_end|>
<|im_start|>assistant
""".strip()


def _qwen_test_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a shell command.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
    ]


def _file_info_tool() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "file_info",
                "description": "Return information for a filesystem path.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]


def test_openpangu_explicit_file_info_uses_native_json_list_with_bound_path():
    tools = _file_info_tool()
    user_request = (
        "Call the built-in file_info tool exactly once with path "
        "panel/package.json. After the tool result, reply exactly DONE."
    )
    prompt = OpenPanguLikeTokenizer().apply_chat_template(
        [{"role": "user", "content": user_request}],
        tools=tools,
    )

    injected = check_and_inject_fallback_tools(
        prompt,
        [{"role": "user", "content": user_request}],
        tools,
        OpenPanguLikeTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="openpangu",
    )

    assert "native openPangu JSON-list shape" in injected
    assert "<tool_call>\n" not in injected
    assert '"name": "file_info"' in injected
    assert '"path": "panel/package.json"' in injected
    assert '"path": "VALUE_HERE"' not in injected


def test_openpangu_tool_result_continuation_finishes_visible_answer():
    tools = _file_info_tool()
    messages = [
        {"role": "user", "content": "Call file_info with path panel/package.json."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "file_info",
                        "arguments": '{"path":"panel/package.json"}',
                    },
                }
            ],
        },
        {"role": "tool", "content": "Path: panel/package.json\nType: file"},
    ]
    prompt = OpenPanguLikeTokenizer().apply_chat_template(messages, tools=tools)

    injected = check_and_inject_fallback_tools(
        prompt,
        messages,
        tools,
        OpenPanguLikeTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="openpangu",
    )

    assert "Native openPangu tool-result continuation" in injected
    assert "Do not emit another <|tool_call_start|> block" in injected
    assert "<|message_start|>tool\nPath: panel/package.json" in injected


def test_qwen_explicit_run_exactly_binds_command_and_narrows_schema():
    tools = _qwen_test_tools()
    user_request = (
        "Use the run_command tool exactly once. Run exactly: "
        "printf B1TOOL > bonsai_native.txt . "
        "After the tool result, reply only: B1TOOL-OK"
    )

    injected = check_and_inject_fallback_tools(
        _qwen_prompt(),
        [{"role": "user", "content": user_request}],
        tools,
        PlainTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="qwen",
    )

    assert "run_command" in injected
    assert "Tool: read_file" not in injected
    assert "<function=read_file>" not in injected
    assert (
        "<parameter=command>\nprintf B1TOOL > bonsai_native.txt\n</parameter>"
        in injected
    )
    command = injected.split("<parameter=command>", 1)[1].split("</parameter>", 1)[0]
    assert "After the tool result" not in command


def test_qwen_explicit_to_run_binds_command_and_narrows_schema():
    """The live Electron prompt commonly says ``to run:`` rather than
    ``run exactly:``; keep that command in the canonical Qwen example instead
    of teaching the model an empty required argument.
    """
    tools = _qwen_test_tools()
    user_request = (
        "Use the run_command tool exactly once to run: printf Q36_TOOL_811. "
        "After seeing the tool result, reply exactly Q36-TOOL=OK."
    )

    injected = check_and_inject_fallback_tools(
        _qwen_prompt(),
        [{"role": "user", "content": user_request}],
        tools,
        PlainTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="qwen",
    )

    assert "Tool: read_file" not in injected
    assert "<function=read_file>" not in injected
    assert "<parameter=command>\nprintf Q36_TOOL_811\n</parameter>" in injected
    command = injected.split("<parameter=command>", 1)[1].split("</parameter>", 1)[0]
    assert "After seeing the tool result" not in command


def test_qwen_tool_result_continuation_prevents_duplicate_execution():
    tools = _qwen_test_tools()
    command = "printf B1TOOL > bonsai_native.txt"
    messages = [
        {
            "role": "user",
            "content": (
                "Use the run_command tool exactly once. Run exactly: "
                f"{command} . After the tool result, reply only: B1TOOL-OK"
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "run_command", "arguments": {"command": command}}}
            ],
        },
        {"role": "tool", "content": "B1TOOL"},
    ]

    injected = check_and_inject_fallback_tools(
        _qwen_prompt(),
        messages,
        tools,
        PlainTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="qwen",
    )

    assert "run_command" in injected
    assert "already ran" in injected
    assert "Do not emit another <tool_call>" in injected
    assert "<function=run_command>" not in injected
    assert f"<parameter=command>\n{command}\n</parameter>" not in injected
    assert "Tool: read_file" not in injected


def test_dsv4_encoder_prompt_tools_narrow_to_explicit_latest_user_tool():
    tools = [
        {"type": "function", "function": {"name": "read_file"}},
        {"type": "function", "function": {"name": "run_command"}},
    ]
    messages = [
        {"role": "user", "content": "Use the run_command tool exactly once."},
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "tool", "content": "prior result"},
    ]

    selected = select_tools_for_explicit_request(messages, tools)

    assert selected == [tools[1]]


def test_dsv4_encoder_prompt_tools_keep_all_when_no_registered_name_is_mentioned():
    tools = [
        {"type": "function", "function": {"name": "read_file"}},
        {"type": "function", "function": {"name": "run_command"}},
    ]

    selected = select_tools_for_explicit_request(
        [{"role": "user", "content": "Use the built-in shell tool."}],
        tools,
    )

    assert selected == tools


def test_dsv4_encoder_prompt_tools_preserve_recent_tool_schema_on_continuation():
    tools = [
        {"type": "function", "function": {"name": "read_file"}},
        {"type": "function", "function": {"name": "run_command"}},
    ]
    messages = [
        {
            "role": "user",
            "content": (
                "Use the run_command tool exactly once to create a file named "
                "z.txt. Write the text Z7 into that file."
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "run_command", "arguments": {"command": "printf OK42"}}}
            ],
        },
        {"role": "tool", "content": "$ printf OK42\n\nOK42"},
        {"role": "assistant", "content": "OK42"},
        {"role": "user", "content": "Add one to the previous number."},
    ]

    selected = select_tools_for_explicit_request(messages, tools)

    assert selected == [tools[1]]


def test_dsv4_fallback_does_not_invent_arg1_for_zero_arg_tool():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_datetime",
                "description": "Get the current date and time.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "smoke__echo",
                "description": "Return the provided text.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        },
    ]
    prompt = "<｜User｜>Use the tools.<｜Assistant｜>"

    injected = check_and_inject_fallback_tools(
        prompt,
        [{"role": "user", "content": "Use the tools."}],
        tools,
        DSV4LikeTokenizer(),
        {"tokenize": False, "add_generation_prompt": True},
        tool_parser_id="dsml",
    )

    assert '<｜DSML｜invoke name="get_current_datetime">' in injected
    assert '<｜DSML｜parameter name="arg1"' not in injected
    datetime_block = injected.split('<｜DSML｜invoke name="get_current_datetime">', 1)[1]
    datetime_block = datetime_block.split("</｜DSML｜invoke>", 1)[0]
    assert "<｜DSML｜parameter" not in datetime_block
    assert '<｜DSML｜invoke name="smoke__echo">' in injected
    echo_block = injected.split('<｜DSML｜invoke name="smoke__echo">', 1)[1]
    echo_block = echo_block.split("</｜DSML｜invoke>", 1)[0]
    assert "<｜DSML｜parameter" not in echo_block
    assert "VALUE HERE" not in injected
    assert "- text (string, required)" in injected


def test_dsv4_schema_only_prompt_gets_concrete_per_tool_examples():
    """DSV4's bundled encoder renders JSON schemas plus generic DSML syntax.

    Live DSV4 JANGTQ-K mixed a zero-argument built-in tool with a one-argument
    MCP tool under that schema-only prompt. The fallback must require concrete
    examples for each actual tool name, not accept the generic DSML block.
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_datetime",
                "description": "Get the current date and time.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "smoke__echo",
                "description": "Return the provided text.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        },
    ]
    schema_only = """
<｜begin▁of▁sentence｜>system
<｜DSML｜tool_calls>
<｜DSML｜invoke name="$TOOL_NAME">
<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>
{"name": "get_current_datetime", "parameters": {"type": "object", "properties": {}, "required": []}}
{"name": "smoke__echo", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}
<｜User｜>Use the tools.
<｜Assistant｜>
""".strip()

    injected = check_and_inject_fallback_tools(
        schema_only,
        [{"role": "user", "content": "Use the tools."}],
        tools,
        DSV4LikeTokenizer(),
        {"tokenize": False, "add_generation_prompt": True},
        tool_parser_id="dsml",
    )

    assert injected != schema_only
    assert '<｜DSML｜invoke name="get_current_datetime">' in injected
    assert '<｜DSML｜invoke name="smoke__echo">' in injected


def test_dsv4_explicit_run_command_binds_exact_request_value_without_placeholder():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a shell command.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
    ]
    user_request = (
        "Use the run_command tool exactly once to create a file named "
        "dsv4_ui_tool_probe.txt. Write the text DSV4_TOOL_OK into that file."
    )

    injected = check_and_inject_fallback_tools(
        f"<｜User｜>{user_request}<｜Assistant｜>",
        [{"role": "user", "content": user_request}],
        tools,
        DSV4LikeTokenizer(),
        {"tokenize": False, "add_generation_prompt": True},
        tool_parser_id="dsml",
    )

    assert "The current user explicitly named an available tool." in injected
    assert (
        '<｜DSML｜parameter name="command" string="true">'
        "printf %s DSV4_TOOL_OK > dsv4_ui_tool_probe.txt"
        "</｜DSML｜parameter>"
    ) in injected
    assert "Tool: read_file" not in injected
    assert '<｜DSML｜invoke name="read_file">' not in injected
    assert "VALUE HERE" not in injected


def test_dsv4_explicit_direct_command_binds_only_text_before_followup_sentence():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    user_request = (
        "Use run_command to run exactly: printf DSV4_TOOL_OK . "
        "After the tool returns, reply only: DSV4_TOOL_OK"
    )

    injected = check_and_inject_fallback_tools(
        f"<｜User｜>{user_request}<｜Assistant｜>",
        [{"role": "user", "content": user_request}],
        tools,
        DSV4LikeTokenizer(),
        {"tokenize": False, "add_generation_prompt": True},
        tool_parser_id="dsml",
    )

    assert (
        '<｜DSML｜parameter name="command" string="true">'
        "printf DSV4_TOOL_OK</｜DSML｜parameter>"
    ) in injected
    assert "After the tool returns" not in injected.split(
        '<｜DSML｜parameter name="command" string="true">', 1
    )[1].split("</｜DSML｜parameter>", 1)[0]


def test_dsv4_fallback_preserves_recent_tool_schema_on_later_user_turn():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
    ]
    messages = [
        {
            "role": "user",
            "content": (
                "Use the run_command tool exactly once to create a file named "
                "z.txt. Write the text Z7 into that file."
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "run_command",
                        "arguments": {"command": "printf \"%s\" \"Z7\" > z.txt"},
                    }
                }
            ],
        },
        {"role": "tool", "content": "$ printf Z7\n\nZ7"},
        {"role": "assistant", "content": "Z7"},
        {"role": "user", "content": "Add one to the previous number."},
    ]

    injected = check_and_inject_fallback_tools(
        "<｜User｜>Add one to the previous number.<｜Assistant｜>",
        messages,
        tools,
        DSV4LikeTokenizer(),
        {"tokenize": False, "add_generation_prompt": True},
        tool_parser_id="dsml",
    )

    assert "Tool: run_command" in injected
    assert "The current user explicitly named an available tool." in injected
    assert '<｜DSML｜invoke name="run_command">' in injected
    assert (
        '<｜DSML｜parameter name="command" string="true">'
        "printf %s Z7 > z.txt</｜DSML｜parameter>"
    ) in injected
    assert "Tool: read_file" not in injected
    assert '<｜DSML｜invoke name="read_file">' not in injected


def test_lfm2_fallback_for_file_request_forbids_content_only_pseudo_call():
    """LFM2 must not treat a JSON content blob as a tool call substitute.

    A live Responses UI run on LFM2.5 produced ``{"content": "..."}`` prose
    instead of the native Python-call-list shape for a natural file-create
    request. The fallback prompt must explicitly forbid that failure mode while
    still deriving the exact shell command from the user request.
    """

    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a shell command.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    prompt = "<|im_start|>user\nCreate a file.<|im_end|>\n<|im_start|>assistant\n"
    user_request = (
        "Use the run_command tool exactly once to create a file named "
        "real_ui_tool_probe_1.txt in the configured working directory. "
        "Write the text REAL_UI_LIVE_TOOL_ONE into that file."
    )

    injected = check_and_inject_fallback_tools(
        prompt,
        [{"role": "user", "content": user_request}],
        tools,
        PlainTokenizer(),
        {"tokenize": False, "add_generation_prompt": True},
        tool_parser_id="lfm2",
    )

    assert "<|tool_call_start|>[run_command(command=" in injected
    assert "printf %s REAL_UI_LIVE_TOOL_ONE > real_ui_tool_probe_1.txt" in injected
    assert 'Do not emit JSON such as {"content": "..."}' in injected


def test_lfm2_direct_run_exactly_binds_command_without_placeholder():
    """Direct shell requests must produce a concrete Liquid call example.

    The real Electron LFM2.5 turn copied the old ``VALUE_HERE`` example and
    executed ``:`` instead of the user's command.
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a shell command.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    user_request = (
        "Use run_command exactly once to run exactly: printf LFM_TOOL_519 . "
        "After the tool result, reply exactly LFM-TOOL=OK."
    )

    injected = check_and_inject_fallback_tools(
        "user: run a command\nassistant:",
        [{"role": "user", "content": user_request}],
        tools,
        PlainTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="lfm2",
    )

    assert (
        "<|tool_call_start|>[run_command(command='printf LFM_TOOL_519')]"
        "<|tool_call_end|>"
    ) in injected
    assert "run_command(command='VALUE_HERE')" not in injected
    assert "After the tool result" not in injected.split(
        "<|tool_call_start|>", 1
    )[1].split("<|tool_call_end|>", 1)[0]


def test_lfm2_direct_file_info_binds_explicit_path_without_placeholder():
    """Named scalar parameters must not leave VALUE_HERE for LFM2 to copy."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "file_info",
                "description": "Return information for a filesystem path.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]
    user_request = (
        "Call the built-in file_info tool exactly once with path "
        "panel/package.json. After the tool result, reply exactly DONE."
    )

    injected = check_and_inject_fallback_tools(
        (
            "<|im_start|>system\n"
            "<|tool_call_start|>[file_info(path='VALUE_HERE')]"
            "<|tool_call_end|><|im_end|>\n"
            "<|im_start|>user\ninspect a file<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        [{"role": "user", "content": user_request}],
        tools,
        PlainTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="lfm2",
    )

    assert (
        "<|tool_call_start|>[file_info(path='panel/package.json')]"
        "<|tool_call_end|>"
    ) in injected
    assert "For this request, file_info.path must be exactly: panel/package.json" in injected
    assert "file_info(path='VALUE_HERE')" not in injected


def test_lfm2_tool_result_continuation_removes_schema_and_requires_final_answer():
    class CaptureTokenizer:
        last_kwargs = None
        last_messages = None

        def apply_chat_template(self, messages, **kwargs):
            self.last_kwargs = kwargs
            self.last_messages = messages
            contents = "\n".join(
                str(message.get("content", "")) for message in messages
            )
            return (
                "<|im_start|>assistant\n"
                "<|tool_call_start|>[run_command(command='printf LFM_TOOL_519')]"
                "<|tool_call_end|><|im_end|>\n"
                "<|im_start|>tool\n"
                f"{contents}<|im_end|>\n<|im_start|>assistant\n"
            )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    messages = [
        {
            "role": "user",
            "content": (
                "Use run_command exactly once to run exactly: printf LFM_TOOL_519 . "
                "After the tool result, reply exactly LFM-TOOL=OK."
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_lfm",
                    "function": {
                        "name": "run_command",
                        "arguments": {"command": "printf LFM_TOOL_519"},
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_lfm", "content": "LFM_TOOL_519"},
    ]
    tokenizer = CaptureTokenizer()

    injected = check_and_inject_fallback_tools(
        "user: prior tool result\nassistant:",
        messages,
        tools,
        tokenizer,
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="lfm2",
    )

    assert '[{"result": "LFM_TOOL_519"}]' in injected
    assert injected.count("<|tool_call_start|>[run_command") == 1
    assert tokenizer.last_kwargs["tools"] == tools
    assert tokenizer.last_messages[-1]["content"] == '[{"result": "LFM_TOOL_519"}]'


def test_lfm2_shell_tool_result_continuation_uses_structured_json():
    class CaptureTokenizer:
        last_messages = None

        def apply_chat_template(self, messages, **kwargs):
            self.last_messages = messages
            contents = "\n".join(
                str(message.get("content", "")) for message in messages
            )
            return (
                "<|im_start|>assistant\n"
                "<|tool_call_start|>[run_command(command='pwd')]"
                "<|tool_call_end|><|im_end|>\n"
                "<|im_start|>tool\n"
                f"{contents}<|im_end|>\n<|im_start|>assistant\n"
            )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    messages = [
        {"role": "user", "content": "Use run_command with command `pwd`."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_lfm",
                    "function": {
                        "name": "run_command",
                        "arguments": {"command": "pwd"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_lfm",
            "content": "$ pwd\n\nExit code: 0\n\n/Users/example/mlx/vllm-mlx\n",
        },
    ]
    tokenizer = CaptureTokenizer()

    rendered = check_and_inject_fallback_tools(
        "user: prior tool result\nassistant:",
        messages,
        tools,
        tokenizer,
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="lfm2",
    )

    expected = (
        '[{"exit_code": 0, "stdout": "/Users/example/mlx/vllm-mlx"}]'
    )
    assert expected in rendered
    assert tokenizer.last_messages[-1]["content"] == expected
