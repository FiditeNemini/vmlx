# SPDX-License-Identifier: Apache-2.0
"""Generic XML function-call parser.

Parses templates that emit:

<tool_call>
<function=name>
<parameter=arg>value</parameter>
</function>
</tool_call>
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
    generate_tool_id,
)


@ToolParserManager.register_module(["xml_function", "mimo_xml_function", "qwen3_coder"])
class XMLFunctionToolParser(ToolParser):
    """Parse XML function calls used by MiMo-style chat templates.

    ``qwen3_coder`` is registered here rather than on the Qwen parser because
    the formats differ and the name has to follow the FORMAT: Qwen3.6-27B
    D-series bundles are stamped ``tool parser qwen3_coder`` and their chat
    template emits exactly this parser's shape —
    ``<tool_call>\\n<function=NAME>\\n<parameter=KEY>\\nVALUE\\n</parameter>`` —
    not the plain ``<tool_call>{json}`` the ``qwen``/``qwen3`` parser reads.
    Verified against the bundle's chat_template.jinja, not inferred from the
    name. Before this name existed the engine exited at startup
    ("Tool parser 'qwen3_coder' not found"), which took the whole session down
    for every stamped D-series bundle.
    """

    SUPPORTS_NATIVE_TOOL_FORMAT = True

    TOOL_CALL_PATTERN = re.compile(
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        re.DOTALL,
    )
    FUNCTION_PATTERN = re.compile(
        r"<function=([^>]+)>\s*(.*?)\s*</function>",
        re.DOTALL,
    )
    PARAM_PATTERN = re.compile(
        r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
        re.DOTALL,
    )
    # Ornith-1.0 (qwen3.5 + Gemma vocab frankenmerge) emits a malformed
    # variant of the template-specified `<parameter=K>V</parameter>` format:
    #   <arg_key>K</arg_key>
    #   <value>V</value>
    # sometimes followed by a spurious extra `</arg_key>`. Accept this pair
    # as a fallback when the standard pattern matches nothing in the body.
    # Engine tolerance for a model-quality merge artifact.
    ORNITH_ARG_KEY_VALUE_PATTERN = re.compile(
        r"<arg_key>\s*(.*?)\s*</arg_key>\s*<value>\s*(.*?)\s*</value>",
        re.DOTALL,
    )
    INVOKE_PATTERN = re.compile(
        r"<invoke>\s*(.*?)\s*</invoke>",
        re.DOTALL,
    )
    TOOL_NAME_PATTERN = re.compile(
        r"<tool_name>\s*(.*?)\s*</tool_name>",
        re.DOTALL,
    )
    ARGUMENTS_PATTERN = re.compile(
        r"<arguments>\s*(.*?)\s*</arguments>",
        re.DOTALL,
    )
    SIMPLE_XML_ARG_PATTERN = re.compile(
        r"<([A-Za-z_][A-Za-z0-9_]*)>\s*(.*?)\s*</\1>",
        re.DOTALL,
    )
    VALUE_WRAPPER_PATTERN = re.compile(
        r"^<value>\s*(.*?)\s*</value>$",
        re.DOTALL,
    )

    @classmethod
    def _coerce_value(cls, value: str) -> Any:
        value = value.strip()
        wrapped = cls.VALUE_WRAPPER_PATTERN.match(value)
        if wrapped:
            value = wrapped.group(1).strip()
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value

    @staticmethod
    def _request_tool_names(request: dict[str, Any] | None) -> set[str]:
        if not isinstance(request, dict):
            return set()
        tools = request.get("tools")
        if not isinstance(tools, list):
            return set()
        names: set[str] = set()
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function")
            if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                names.add(fn["name"])
        return names

    # Relaxed function pattern: Ornith-1.0 may emit `<function=name>` without
    # the matching `</function>` close. Body extends to the next `</tool_call>`
    # or end-of-string. Only used as fallback when the strict pattern misses.
    RELAXED_FUNCTION_PATTERN = re.compile(
        r"<function=([^>]+)>\s*(.*?)\s*(?=</function>|</tool_call>|$)",
        re.DOTALL,
    )

    # Qwen3.6-35B at temp 0 emits a malformed doubled-wrapper nesting
    # (observed verbatim in the 2026-08-15 smoke, and the cause of the
    # family's empty visible turns once the markup was suppressed):
    #   <tool_call>
    #   <function=function>
    #   <function=record_fact>
    #   <function=value>
    #   blue-cat
    #   </parameter>
    #   </function>
    #   </tool_call>
    # The real function name and every parameter are unambiguous when the
    # request's tool names are known: the only opener matching a requested
    # tool is the call, and `<function=KEY>V</parameter>` inside it is a
    # miskeyed `<parameter=KEY>V</parameter>`.
    MISKEYED_PARAM_PATTERN = re.compile(
        r"<function=([A-Za-z_][A-Za-z0-9_]*)>\s*(.*?)\s*</parameter>",
        re.DOTALL,
    )

    @classmethod
    def _recover_doubled_wrapper_calls(
        cls, text: str, *, allowed_names: set[str] | None
    ) -> list[dict[str, Any]]:
        if not allowed_names:
            return []
        tool_calls: list[dict[str, Any]] = []
        for match in re.finditer(r"<function=([^>]+)>", text):
            name = match.group(1).strip()
            if name not in allowed_names:
                continue
            body = text[match.end():]
            next_real = None
            for later in re.finditer(r"<function=([^>]+)>", body):
                if later.group(1).strip() in allowed_names:
                    next_real = later.start()
                    break
            if next_real is not None:
                body = body[:next_real]
            arguments = cls._extract_arguments_from_body(body)
            if not arguments:
                for key, value in cls.MISKEYED_PARAM_PATTERN.findall(body):
                    key = key.strip()
                    if key in allowed_names:
                        continue
                    arguments[key] = cls._coerce_value(value)
            tool_calls.append(
                {
                    "id": generate_tool_id(),
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                }
            )
        return tool_calls

    @classmethod
    def _extract_arguments_from_body(cls, body: str) -> dict[str, Any]:
        """Extract args trying strict `<parameter=K>V</parameter>` first, then
        the Ornith `<arg_key>K</arg_key><value>V</value>` fallback."""
        arguments: dict[str, Any] = {}
        for param_name, param_value in cls.PARAM_PATTERN.findall(body):
            arguments[param_name.strip()] = cls._coerce_value(param_value)
        if not arguments:
            for k, v in cls.ORNITH_ARG_KEY_VALUE_PATTERN.findall(body):
                arguments[k.strip()] = cls._coerce_value(v)
        return arguments

    @classmethod
    def _parse_functions(
        cls, text: str, *, allowed_names: set[str] | None = None
    ) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        matches = list(cls.FUNCTION_PATTERN.findall(text))
        if not matches:
            # Fallback: Ornith-1.0 may emit `<function=name>` without `</function>`.
            matches = list(cls.RELAXED_FUNCTION_PATTERN.findall(text))
        for func_name, body in matches:
            name = func_name.strip()
            if allowed_names is not None and name not in allowed_names:
                continue
            arguments = cls._extract_arguments_from_body(body)
            tool_calls.append(
                {
                    "id": generate_tool_id(),
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                }
            )
        return tool_calls

    @classmethod
    def _parse_nested_invoke_functions(
        cls, text: str, *, allowed_names: set[str] | None = None
    ) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        for body in cls.INVOKE_PATTERN.findall(text):
            name_match = cls.TOOL_NAME_PATTERN.search(body)
            if not name_match:
                continue
            name = name_match.group(1).strip()
            if allowed_names is not None and name not in allowed_names:
                continue
            arguments: dict[str, Any] = {}
            args_match = cls.ARGUMENTS_PATTERN.search(body)
            if args_match:
                args_body = args_match.group(1)
                for arg_name, arg_value in cls.SIMPLE_XML_ARG_PATTERN.findall(args_body):
                    arguments[arg_name.strip()] = cls._coerce_value(arg_value)
            tool_calls.append(
                {
                    "id": generate_tool_id(),
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                }
            )
        return tool_calls

    @classmethod
    def _strip_repaired_function_blocks(cls, text: str) -> str:
        cleaned = cls.FUNCTION_PATTERN.sub("", text)
        cleaned = cls.INVOKE_PATTERN.sub("", cleaned)
        cleaned = re.sub(r"</?function_call>", "", cleaned)
        cleaned = cleaned.replace("<tool_call>", "")
        cleaned = cleaned.replace("</tool_call>", "")
        cleaned = re.sub(r"```(?:xml|XML)?", "", cleaned)
        cleaned = cleaned.replace("```", "")
        return cleaned.strip()

    def extract_tool_calls(
        self,
        model_output: str,
        request: dict[str, Any] | None = None,
    ) -> ExtractedToolCallInformation:
        if "<tool_call>" not in model_output:
            allowed_names = self._request_tool_names(request)
            if (
                allowed_names
                and "</tool_call>" in model_output
                and "<function=" in model_output
            ):
                repaired_calls = self._parse_functions(
                    model_output, allowed_names=allowed_names
                )
                if repaired_calls:
                    cleaned_text = self._strip_repaired_function_blocks(model_output)
                    return ExtractedToolCallInformation(
                        tools_called=True,
                        tool_calls=repaired_calls,
                        content=cleaned_text if cleaned_text else None,
                    )
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        tool_calls: list[dict[str, Any]] = []
        allowed_names_for_recovery = self._request_tool_names(request)
        for block in self.TOOL_CALL_PATTERN.findall(model_output):
            parsed = self._parse_functions(block)
            # A doubled `<function=function>` wrapper parses as one bogus call
            # named "function" with no arguments; recover the real nested call
            # when the request's tool names disambiguate it.
            if allowed_names_for_recovery and parsed and all(
                call.get("name") not in allowed_names_for_recovery
                for call in parsed
            ):
                recovered = self._recover_doubled_wrapper_calls(
                    block, allowed_names=allowed_names_for_recovery
                )
                if recovered:
                    parsed = recovered
            tool_calls.extend(parsed)
            if not tool_calls:
                tool_calls.extend(self._parse_nested_invoke_functions(block))

        cleaned_text = self.TOOL_CALL_PATTERN.sub("", model_output).strip()
        if not tool_calls:
            allowed_names = self._request_tool_names(request)
            if allowed_names and "<function=" in model_output:
                repaired_calls = self._parse_functions(
                    model_output,
                    allowed_names=allowed_names,
                )
                if repaired_calls:
                    cleaned_text = self._strip_repaired_function_blocks(model_output)
                    return ExtractedToolCallInformation(
                        tools_called=True,
                        tool_calls=repaired_calls,
                        content=cleaned_text if cleaned_text else None,
                    )
            if allowed_names and "<tool_name>" in model_output:
                repaired_calls = self._parse_nested_invoke_functions(
                    model_output,
                    allowed_names=allowed_names,
                )
                if repaired_calls:
                    cleaned_text = self._strip_repaired_function_blocks(model_output)
                    return ExtractedToolCallInformation(
                        tools_called=True,
                        tool_calls=repaired_calls,
                        content=cleaned_text if cleaned_text else None,
                    )
        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=cleaned_text if cleaned_text else None,
            )
        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=model_output,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int] | None = None,
        current_token_ids: Sequence[int] | None = None,
        delta_token_ids: Sequence[int] | None = None,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if "<tool_call>" not in current_text:
            return {"content": delta_text}
        if "</tool_call>" in delta_text:
            result = self.extract_tool_calls(current_text, request=request)
            if result.tools_called:
                return {
                    "tool_calls": [
                        {
                            "index": i,
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for i, tc in enumerate(result.tool_calls)
                    ]
                }
        return None
