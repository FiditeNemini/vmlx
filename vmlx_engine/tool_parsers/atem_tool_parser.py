# SPDX-License-Identifier: Apache-2.0
"""ATEM tool call parser (Muse Glimmer).

Handles the namespaced invoke/parameter dialect Muse Glimmer emits:

    <atem:function_calls>
      <atem:invoke name="get_weather">
        <atem:parameter name="location">San Francisco</atem:parameter>
        <atem:parameter name="unit">celsius</atem:parameter>
      </atem:invoke>
    </atem:function_calls>

Deliberately regex-based rather than XML-based. The bundle's own chat template
says so outright: "The output is not expected to be valid XML and is parsed with
regular expressions." Feeding this to an XML parser would reject perfectly
ordinary generations — unescaped ``&`` and ``<`` inside a parameter value, a
truncated tail when the model hits max_tokens, or an omitted closing tag.

Value encoding, per the same template: booleans render as ``true``/``false``,
``None`` as ``null``, mappings and non-string iterables via ``tojson``, and
every other scalar raw. Inverting that is ambiguous — a literal string "true"
is indistinguishable from the boolean — so the request's JSONSchema decides
whenever it is available, and JSON-shaped decoding is only a fallback.
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


def _extract_name(name_str: str) -> str:
    """Extract a name from ``name="x"``, ``name='x'`` or ``name=x``."""
    name_str = name_str.strip()
    if (name_str.startswith('"') and name_str.endswith('"')) or (
        name_str.startswith("'") and name_str.endswith("'")
    ):
        return name_str[1:-1]
    return name_str


def _schema_for(request: Any, fn_name: str) -> dict[str, Any]:
    """Return the JSONSchema ``properties`` map advertised for ``fn_name``."""
    if not isinstance(request, dict):
        return {}
    for tool in request.get("tools") or ():
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        if not isinstance(fn, dict) or fn.get("name") != fn_name:
            continue
        params = fn.get("parameters")
        props = params.get("properties") if isinstance(params, dict) else None
        return props if isinstance(props, dict) else {}
    return {}


def _coerce(value: str, declared_type: str | None) -> Any:
    """Decode one parameter value, preferring the declared JSONSchema type.

    ``declared_type`` of ``string`` is honoured verbatim: a tool that declares a
    string parameter must receive "true" as the two-character string, not a
    boolean. Only when the schema is silent do we fall back to JSON shape.

    The declared-string branch runs BEFORE the blank check, because "verbatim"
    has to include whitespace. A newline or a run of spaces is a legitimate
    value for a string parameter — a tool writing an indent or a separator
    passes exactly that — and returning the stripped text handed it "" instead.
    """
    if declared_type == "string":
        return value

    text = value.strip()
    if not text:
        return text
    if declared_type in {"object", "array"}:
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return value
    if declared_type == "boolean":
        lowered = text.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        return value
    if declared_type in {"integer", "number"}:
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return value
        return parsed if isinstance(parsed, (int, float)) and not isinstance(parsed, bool) else value

    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return value


@ToolParserManager.register_module(["atem", "muse_glimmer", "muse"])
class AtemToolParser(ToolParser):
    """Tool call parser for the Muse Glimmer ATEM dialect."""

    SUPPORTS_NATIVE_TOOL_FORMAT = True

    # Whole <atem:function_calls>...</atem:function_calls> blocks.
    TOOL_CALL_PATTERN = re.compile(
        r"<atem:function_calls>(.*?)</atem:function_calls>",
        re.DOTALL,
    )

    INVOKE_PATTERN = re.compile(
        r"<atem:invoke name=([^>]+)>(.*?)</atem:invoke>",
        re.DOTALL,
    )

    PARAM_PATTERN = re.compile(
        r"<atem:parameter name=([^>]+)>(.*?)</atem:parameter>",
        re.DOTALL,
    )

    # Truncation salvage: a generation cut off by max_tokens mid-invoke still
    # carries a complete, actionable call up to the cut. Scoped to a real
    # ``<atem:invoke name=`` opener so visible prose is never promoted.
    LENIENT_INVOKE_PATTERN = re.compile(
        r"<atem:invoke name=([^>]+)>(.*?)(?=(?:</atem:invoke>|<atem:invoke name=|$))",
        re.DOTALL,
    )
    LENIENT_PARAM_PATTERN = re.compile(
        r"<atem:parameter name=([^>]+)>(.*?)"
        r"(?=(?:</atem:parameter>|<atem:parameter name=|</atem:invoke>|$))",
        re.DOTALL,
    )

    def _parse_invocations(
        self,
        block: str,
        request: Any,
        *,
        lenient: bool,
    ) -> list[dict[str, Any]]:
        invoke_pat = self.LENIENT_INVOKE_PATTERN if lenient else self.INVOKE_PATTERN
        param_pat = self.LENIENT_PARAM_PATTERN if lenient else self.PARAM_PATTERN

        calls: list[dict[str, Any]] = []
        for raw_name, body in invoke_pat.findall(block):
            fn_name = _extract_name(raw_name)
            if not fn_name:
                continue
            props = _schema_for(request, fn_name)
            args: dict[str, Any] = {}
            for raw_key, raw_value in param_pat.findall(body):
                key = _extract_name(raw_key)
                if not key:
                    continue
                declared = props.get(key)
                declared_type = (
                    declared.get("type") if isinstance(declared, dict) else None
                )
                args[key] = _coerce(raw_value, declared_type)
            # FLAT shape. The engine reads tc["name"]/tc["arguments"] directly
            # and wraps them into the OpenAI envelope itself; returning a
            # pre-nested {"function": {...}} raises KeyError inside the
            # dispatcher, which swallows the exception and passes the raw
            # <atem:...> markup through to the user as visible prose.
            calls.append(
                {
                    "id": generate_tool_id(),
                    "name": fn_name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                }
            )
        return calls

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """Extract ATEM tool calls, preserving any visible prose before them."""
        cleaned_text = self.strip_think_tags(model_output)

        if "<atem:invoke" not in cleaned_text:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        tool_calls: list[dict[str, Any]] = []
        blocks = self.TOOL_CALL_PATTERN.findall(cleaned_text)
        for block in blocks:
            tool_calls.extend(self._parse_invocations(block, request, lenient=False))

        # No complete block, or a block whose invokes never closed: salvage from
        # the first opener onward. A run truncated by max_tokens lands here.
        if not tool_calls:
            start = cleaned_text.find("<atem:function_calls>")
            if start < 0:
                start = cleaned_text.find("<atem:invoke")
            if start >= 0:
                tool_calls = self._parse_invocations(
                    cleaned_text[start:], request, lenient=True
                )

        if not tool_calls:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        marker = cleaned_text.find("<atem:function_calls>")
        if marker < 0:
            marker = cleaned_text.find("<atem:invoke")
        content = cleaned_text[:marker].strip() if marker > 0 else None

        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=content or None,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: dict[str, Any] | None = None,
    ) -> Any:
        """Buffer until the call is complete, then emit it once.

        ATEM parameter values are only decodable once their closing tag lands,
        so emitting partial arguments would surface malformed JSON to clients.
        """
        if "<atem:invoke" not in current_text:
            return None
        if "</atem:function_calls>" not in current_text:
            return None
        if "</atem:function_calls>" in previous_text:
            return None
        return self.extract_tool_calls(current_text, request)
