# SPDX-License-Identifier: Apache-2.0
"""
StepFun Step-3.5 tool call parser for vmlx-engine.

Handles StepFun's XML-based tool calling format:
- <tool_call><function=name><parameter=p>value</parameter></function></tool_call>

This is the same XML format used by Nemotron models, with additional
type coercion for parameter values (string → int/float/bool) based
on the tool schema provided in the request.
"""

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


@ToolParserManager.register_module(["step3p5", "stepfun"])
class Step3p5ToolParser(ToolParser):
    """
    Tool call parser for StepFun Step-3.5 models.

    Supports Step-3.5's XML tool call format:
    <tool_call>
    <function=get_weather>
    <parameter=city>Paris</parameter>
    <parameter=units>celsius</parameter>
    </function>
    </tool_call>

    Also supports JSON arguments inside function tags:
    <tool_call><function=get_weather>{"city": "Paris"}</function></tool_call>

    Features:
    - Parameter type coercion: converts string values to int/float/bool
      based on the tool schema in the request (matching vLLM step3p5 behavior)
    - Handles multi-line parameter values

    Used when --enable-auto-tool-choice --tool-call-parser step3p5 are set.
    """

    NATIVE_MARKERS = ("<tool_call>",)

    # Pattern for XML-style with <function=name>...</function> inside <tool_call>
    TOOL_CALL_PATTERN = re.compile(
        r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>",
        re.DOTALL,
    )

    # Pattern to extract <parameter=name>value</parameter>
    PARAM_PATTERN = re.compile(
        # Strip at most ONE framing newline per side. `\s*` ate the payload's
        # own leading indentation, so a code argument came back with its first
        # line unindented and later lines intact — a SyntaxError once written
        # to disk. Same defect as the qwen dialect (9df8c1660).
        r"<parameter=([^>]+)>[ \t]*\n?(.*?)\n?[ \t]*</parameter>",
        re.DOTALL,
    )

    # Step-3.7 variant: arguments wrapped in <args><name>value</name>...</args>
    # (instead of <parameter=name>value</parameter>). Tolerated additively so a
    # single parser handles both Step-3.5 and Step-3.7 tool-call formats.
    ARGS_BLOCK_PATTERN = re.compile(r"<args>(.*?)</args>", re.DOTALL)
    ARGS_KV_PATTERN = re.compile(r"<([A-Za-z_][\w-]*)>\s*(.*?)\s*</\1>", re.DOTALL)

    def _get_param_schema(
        self, func_name: str, param_name: str, request: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Look up parameter schema from request tools for type coercion."""
        if not request:
            return None
        tools = request.get("tools")
        if not tools:
            return None
        for tool in tools:
            func = tool.get("function", {})
            if func.get("name") == func_name:
                props = func.get("parameters", {}).get("properties", {})
                return props.get(param_name)
        return None

    def _coerce_value(
        self, value: str, schema: dict[str, Any] | None
    ) -> Any:
        """
        Coerce a string parameter value to the correct type based on schema.

        Matches vLLM step3p5 parser behavior: converts string values to
        int, float, or bool when the schema specifies those types.
        """
        if not schema:
            return value

        param_type = schema.get("type", "string")

        if param_type == "integer":
            try:
                return int(value)
            except (ValueError, TypeError):
                return value
        elif param_type == "number":
            try:
                return float(value)
            except (ValueError, TypeError):
                return value
        elif param_type == "boolean":
            lower = value.lower().strip()
            if lower in ("true", "1", "yes"):
                return True
            elif lower in ("false", "0", "no"):
                return False
            return value
        elif param_type == "array" or param_type == "object":
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value

        return value

    def _parse_args_block(
        self, func_name: str, content: str, request: dict[str, Any] | None
    ) -> str | None:
        """Parse the Step-3.7 ``<args><name>value</name>...</args>`` variant.

        Returns a JSON-encoded arguments string, or ``None`` if the content
        contains no recognizable ``<args>`` block (so callers can fall back).
        """
        block = self.ARGS_BLOCK_PATTERN.search(content)
        if not block:
            return None
        pairs = self.ARGS_KV_PATTERN.findall(block.group(1))
        if not pairs:
            return None
        arguments: dict[str, Any] = {}
        for name, value in pairs:
            name = name.strip()
            schema = self._get_param_schema(func_name, name, request)
            arguments[name] = self._coerce_value(value.strip(), schema)
        return json.dumps(arguments, ensure_ascii=False)

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from StepFun Step-3.5 model output.
        """
        # Strip think tags first (handles models with reasoning + tool calls)
        cleaned_output = self.strip_think_tags(model_output)

        if "<tool_call>" not in cleaned_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=cleaned_output
            )

        tool_calls = []
        cleaned_text = cleaned_output

        matches = self.TOOL_CALL_PATTERN.findall(cleaned_output)
        for func_name, content in matches:
            func_name = func_name.strip()

            # Try to parse content as JSON first
            content = content.strip()
            if content.startswith("{"):
                try:
                    json.loads(content)
                    tool_calls.append(
                        {
                            "id": generate_tool_id(),
                            "name": func_name,
                            "arguments": content,
                        }
                    )
                    continue
                except json.JSONDecodeError:
                    pass

            # Parse parameter tags
            params = self.PARAM_PATTERN.findall(content)
            if params:
                arguments = {}
                for param_name, param_value in params:
                    param_name = param_name.strip()
                    param_value = param_value.strip()

                    # Type coercion based on schema
                    schema = self._get_param_schema(func_name, param_name, request)
                    arguments[param_name] = self._coerce_value(param_value, schema)

                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": func_name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    }
                )
            elif (args_json := self._parse_args_block(func_name, content, request)) is not None:
                # Step-3.7 <args><name>value</name>...</args> variant
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": func_name,
                        "arguments": args_json,
                    }
                )
            else:
                # Raw content without parameter tags, or empty content
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": func_name,
                        "arguments": content if content else "{}",
                    }
                )

        # Clean the text
        if matches:
            cleaned_text = self.TOOL_CALL_PATTERN.sub("", cleaned_text).strip()

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=cleaned_text if cleaned_text else None,
            )
        else:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
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
        """
        Extract tool calls from streaming StepFun model output.
        """
        if "<tool_call>" not in current_text:
            return {"content": delta_text}

        if "</tool_call>" in delta_text:
            result = self.extract_tool_calls(current_text, request)
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
