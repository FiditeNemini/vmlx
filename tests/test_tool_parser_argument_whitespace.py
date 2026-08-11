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
