# SPDX-License-Identifier: Apache-2.0
"""
Qwen tool call parser for vmlx-engine.

Handles Qwen's tool calling formats:
- XML style: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
- Bracket style: [Calling tool: func_name({"arg": "value"})]
"""

import json
import re
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    is_valid_tool_name,
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
    generate_tool_id,
)


@ToolParserManager.register_module(["qwen", "qwen3"])
class QwenToolParser(ToolParser):
    """
    Tool call parser for Qwen models.

    Supports multiple Qwen tool call formats:
    - XML: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
    - Bracket: [Calling tool: func_name({"arg": "value"})]

    Used when --enable-auto-tool-choice --tool-call-parser qwen are set.
    """

    # Pattern for XML-style: <tool_call>{"json"}</tool_call>
    XML_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

    # Pattern for bracket-style: [Calling tool: func_name({...})]
    BRACKET_PATTERN = re.compile(r"\[Calling tool:\s*(\w+)\((\{.*?\})\)\]", re.DOTALL)

    # Qwen3-Coder / Qwen3.6 XML function-parameter format (issue #192):
    #   canonical (per chat template): <function=name><parameter=arg>value</parameter></function>
    # Tolerate the attribute variant the model sometimes emits off-format
    # (observed live on Qwen3.6-27B-MXFP4): <function name="name">, <parameter name="arg">.
    # Without this the equals-only regex misses the attribute form, the call is
    # dropped for "missing required argument", and the raw XML leaks into content.
    # The colon separator (`<function:name>`) is the same dialect as the colon
    # parameter form below; tolerated defensively so a mixed-dialect bundle
    # can't drop the whole call.
    FUNCTION_PATTERN = re.compile(
        r"<function(?:=|:|\s+name=[\"'])([^>\"':]+)[\"']?>\s*(.*?)\s*</function>",
        re.DOTALL,
    )
    # Third tolerated variant (observed live on Holo3-35B-A3B-mxfp4, sweep
    # 2026-07-05): the model writes `>` instead of `=` after "parameter" —
    # `<parameter>city>\nParis\n</parameter>`. The equals/attribute-only
    # regex missed it, the call was dropped, and the raw XML leaked into
    # visible content. The `>NAME>` form requires a whitespace-free name
    # token followed by a second `>` before the value, so a genuine unnamed
    # `<parameter>value</parameter>` body cannot false-match.
    # Fourth tolerated variant (observed live 2026-07-08 on Qwen3.6-27B-JANG_4M
    # under reasoning-on + multi-parameter tools): the model writes a COLON name
    # separator — `<parameter:city>Tokyo</parameter>`. The equals/attribute/`>`
    # only regex missed it, so the parameter dropped, the call arrived with empty
    # required args, the server dropped it "missing required argument", and the
    # whole `<tool_call><function=...><parameter:...>` XML leaked into content
    # (reason_off/tool was unaffected because that path emitted JSON args).
    # `\s*` around the capture ate the argument's own leading indentation:
    # a value rendered as "\n    def f():\n        pass\n" came back as
    # "def f():\n        pass" — first line unindented, second still at 8
    # spaces, i.e. broken Python once written to disk. Strip at most ONE
    # framing newline per side (plus spaces/tabs on the marker's own line), so
    # the XML framing goes and the payload's relative indentation survives.
    PARAMETER_PATTERN = re.compile(
        r"<parameter(?:=|:|\s+name=[\"']|>)([^>\"'\s]+)[\"']?>"
        r"[ \t]*\n?(.*?)\n?[ \t]*</parameter>",
        re.DOTALL,
    )
    BARE_ARG_PATTERN = re.compile(
        r"<([A-Za-z_][A-Za-z0-9_]*)>[ \t]*\n?(.*?)\n?[ \t]*</\1>", re.DOTALL
    )
    TOOL_BLOCK_PATTERN = re.compile(
        r"<tool_call>\s*.*?\s*</tool_call>",
        re.DOTALL,
    )
    # Off-template but observed live on Qwen3.6-35B-JANGTQ with a required
    # file_info turn: the model wrote a markdown-looking call header, then
    # continued into a fake result because the parser never stopped generation:
    #
    #   # Calling file_info
    #   # path=panel/package.json
    #
    # Treat only a schema-valid header+single required string parameter as a
    # tool call. Never parse or trust any following "# Tool result" block.
    MARKDOWN_CALL_PATTERN = re.compile(
        r"(?m)^\s*#\s*Calling\s+([A-Za-z_][A-Za-z0-9_-]*)\s*$"
        r"\s*^\s*#\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$"
    )
    # Second off-template shape observed on the same live Qwen3.6 JANGTQ
    # required-tool prompt after the markdown-call fix was loaded:
    #
    #   [Q36-JT-UI-TOOL2]
    #   file_info
    #   path: panel/package.json
    #
    # This is still the model choosing a tool and providing an argument, but
    # with a leading prompt label. Scan for a schema-valid tool-name line
    # followed by one required string parameter line. Gated to the active
    # request schema; normal prose cannot invent unadvertised tools.
    LABELED_LINE_CALL_PATTERN = re.compile(
        r"(?m)^\s*([A-Za-z_][A-Za-z0-9_-]*)\s*$"
        r"\s*^\s*([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(.*?)\s*$"
    )
    EMPTY_MARKDOWN_FENCE_PATTERN = re.compile(
        r"^\s*```(?:text|xml)?\s*```\s*$",
        re.DOTALL | re.IGNORECASE,
    )

    # Residual tool-scaffolding strip. Some off-format / quant-degraded models
    # (observed live on Holo3-35B-A3B-mxfp4, 2026-07-08) emit ORPHAN scaffolding
    # AFTER the parsed call: a premature </function> then a second
    # <parameter=unit>celsius</parameter> outside any <function>, extra
    # </function> tags, a hallucinated <result>{...}</result> fake tool output,
    # or an <argument>/<value> dialect. FUNCTION_PATTERN only removes a matched
    # <function=..>..</function> block, so these orphans leak into visible
    # content. In a CONFIRMED tool-call turn (tool_calls already extracted) none
    # of these tags are user-facing prose, so strip them so nothing leaks. Gated
    # to only-when-tool_calls-found (below) — normal text turns are untouched.
    RESIDUAL_TOOL_MARKUP = re.compile(
        r"<parameter\b[^>]*>.*?</parameter>"
        r"|</?parameter\b[^>]*>"
        r"|</?function\b[^>]*>"
        r"|</?tool_call\b[^>]*>"
        r"|<result\b[^>]*>.*?</result>"
        r"|<argument\b[^>]*>.*?</argument>"
        r"|<value\b[^>]*>.*?</value>",
        re.DOTALL,
    )

    @classmethod
    def _plain_tool_line_call(
        cls, text: str, request: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        lines = [line.strip() for line in text.strip().splitlines()]
        lines = [line for line in lines if line]
        if len(lines) < 2:
            return None
        tool_name = lines[0]
        schema = cls._function_schema_for_tool(request, tool_name)
        if not isinstance(schema, dict):
            return None
        properties = schema.get("properties")
        required = schema.get("required")
        if not isinstance(properties, dict) or not isinstance(required, list):
            return None
        required_names = [name for name in required if isinstance(name, str)]
        if len(required_names) != 1:
            return None
        param_name = required_names[0]
        param_schema = properties.get(param_name)
        if not isinstance(param_schema, dict):
            return None
        param_type = param_schema.get("type")
        if param_type not in (None, "string"):
            return None
        value = "\n".join(lines[1:]).strip()
        if not value:
            return None
        return {
            "id": generate_tool_id(),
            "name": tool_name,
            "arguments": json.dumps({param_name: value}, ensure_ascii=False),
        }

    @classmethod
    def _markdown_tool_call(
        cls, text: str, request: dict[str, Any] | None
    ) -> tuple[dict[str, Any], int] | None:
        match = cls.MARKDOWN_CALL_PATTERN.search(text)
        return cls._schema_single_string_call_from_match(match, request)

    @classmethod
    def _labeled_line_tool_call(
        cls, text: str, request: dict[str, Any] | None
    ) -> tuple[dict[str, Any], int] | None:
        if not request:
            return None
        for match in cls.LABELED_LINE_CALL_PATTERN.finditer(text):
            parsed = cls._schema_single_string_call_from_match(match, request)
            if parsed is not None:
                return parsed
        return None

    @classmethod
    def _orphan_function_wrapper_tool_call(
        cls, text: str, request: dict[str, Any] | None
    ) -> tuple[dict[str, Any], re.Match[str]] | None:
        """Parse a closed Qwen tool block whose function opener was omitted.

        Observed live in the Electron full-tool-catalog path on
        Qwen3.6-35B-A3B-JANGTQ-CRACK::

            <tool_call>
            file_info
            <parameter=path>panel/package.json</parameter>
            </function>
            </tool_call>

        The call is accepted only when the bare first line names an advertised
        tool and every emitted parameter belongs to that tool's request schema.
        This keeps ordinary XML/prose from being promoted into executable work.
        """
        if not request:
            return None
        for match in cls.TOOL_BLOCK_PATTERN.finditer(text):
            block = match.group(0)
            if "<function" in block:
                continue
            inner = re.sub(r"^\s*<tool_call>\s*", "", block, count=1)
            inner = re.sub(r"\s*</tool_call>\s*$", "", inner, count=1)
            lines = [line.strip() for line in inner.splitlines() if line.strip()]
            if not lines:
                continue
            params = cls.PARAMETER_PATTERN.findall(block)
            tool_name = lines[0]
            # A second live degraded form replaced the missing function opener
            # with an empty parameter named after the function:
            #   <parameter=file_info></parameter>
            #   <parameter=path>panel/package.json</parameter>
            # Accept it only when that empty first tag names an advertised tool.
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", tool_name):
                if not params or params[0][1].strip():
                    continue
                candidate_name = params[0][0].strip()
                if cls._function_schema_for_tool(request, candidate_name) is None:
                    continue
                tool_name = candidate_name
                params = params[1:]
            schema = cls._function_schema_for_tool(request, tool_name)
            if not isinstance(schema, dict):
                continue
            properties = schema.get("properties")
            required = schema.get("required") or []
            if not isinstance(properties, dict) or not isinstance(required, list):
                continue
            if not params:
                continue
            arguments: dict[str, Any] = {}
            invalid = False
            for param_name, raw_value in params:
                name = param_name.strip()
                if name not in properties or name in arguments:
                    invalid = True
                    break
                arguments[name] = cls._coerce_arg_value(raw_value)
            if invalid or any(
                name not in arguments
                or arguments[name] is None
                or (isinstance(arguments[name], str) and not arguments[name].strip())
                for name in required
                if isinstance(name, str)
            ):
                continue
            return (
                {
                    "id": generate_tool_id(),
                    "name": tool_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
                match,
            )
        return None

    @classmethod
    def _schema_single_string_call_from_match(
        cls, match: re.Match[str] | None, request: dict[str, Any] | None
    ) -> tuple[dict[str, Any], int] | None:
        if not match:
            return None
        tool_name = match.group(1).strip()
        param_name = match.group(2).strip()
        value = match.group(3).strip()
        if not value:
            return None
        if (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ):
            value = value[1:-1].strip()
        if not value:
            return None

        schema = cls._function_schema_for_tool(request, tool_name)
        if not isinstance(schema, dict):
            return None
        properties = schema.get("properties")
        required = schema.get("required")
        if not isinstance(properties, dict) or not isinstance(required, list):
            return None
        required_names = [name for name in required if isinstance(name, str)]
        if required_names != [param_name]:
            return None
        param_schema = properties.get(param_name)
        if not isinstance(param_schema, dict):
            return None
        param_type = param_schema.get("type")
        if param_type not in (None, "string"):
            return None

        return (
            {
                "id": generate_tool_id(),
                "name": tool_name,
                "arguments": json.dumps({param_name: value}, ensure_ascii=False),
            },
            match.end(),
        )

    @staticmethod
    def _coerce_arg_value(value: str) -> Any:
        # Strip only to TEST for JSON, never to produce the string result.
        # `value.strip()` before the fallback removed the payload's own leading
        # indentation: a code argument rendered as
        #     "\n    def f():\n        pass\n"
        # came back as "def f():\n        pass" — first line unindented while
        # the second kept 8 spaces, which is a syntax error once written to
        # disk. JSON scalars still coerce (" 42 " -> 42) because the parse runs
        # on the stripped copy.
        candidate = value.strip()
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            return value

    @classmethod
    def _parse_function_blocks(cls, text: str) -> list[dict[str, Any]]:
        """Parse Qwen3-Coder XML tool format (issue #192).

        <function=name><parameter=p>v</parameter></function>; falls back to
        bare <p>v</p> params when no <parameter=> tags are present.
        """
        calls: list[dict[str, Any]] = []
        for func_name, body in cls.FUNCTION_PATTERN.findall(text):
            name = func_name.strip()
            if not name:
                continue
            arguments: dict[str, Any] = {}
            params = cls.PARAMETER_PATTERN.findall(body)
            if params:
                for pn, pv in params:
                    arguments[pn.strip()] = cls._coerce_arg_value(pv)
            else:
                for pn, pv in cls.BARE_ARG_PATTERN.findall(body):
                    arguments[pn.strip()] = cls._coerce_arg_value(pv)
            calls.append(
                {
                    "id": generate_tool_id(),
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                }
            )
        return calls

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete Qwen model response.
        """
        tool_calls = []

        # Strip <think> tags first (fallback when no reasoning parser)
        cleaned_text = self.strip_think_tags(model_output)

        # Try bracket pattern first (Qwen3 style)
        bracket_matches = self.BRACKET_PATTERN.findall(cleaned_text)
        for name, args_str in bracket_matches:
            try:
                arguments = json.loads(args_str)
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": name.strip(),
                        "arguments": (
                            json.dumps(arguments, ensure_ascii=False)
                            if isinstance(arguments, dict)
                            else str(arguments)
                        ),
                    }
                )
            except json.JSONDecodeError:
                continue

        if bracket_matches:
            cleaned_text = self.BRACKET_PATTERN.sub("", cleaned_text).strip()

        # Try XML pattern (traditional Qwen style)
        xml_matches = self.XML_PATTERN.findall(cleaned_text)
        for match in xml_matches:
            try:
                data = json.loads(match)
                name = data.get("name", "")
                arguments = data.get("arguments", {})
                if is_valid_tool_name(name):
                    tool_calls.append(
                        {
                            "id": generate_tool_id(),
                            "name": name,
                            "arguments": self._serialize_tool_arguments(
                                name, arguments, request
                            ),
                        }
                    )
            except json.JSONDecodeError:
                continue

        if xml_matches:
            cleaned_text = self.XML_PATTERN.sub("", cleaned_text).strip()

        # Qwen3-Coder / Qwen3.6 XML function-parameter format (issue #192)
        if not tool_calls and "<function" in cleaned_text:
            func_calls = self._parse_function_blocks(cleaned_text)
            if func_calls:
                tool_calls.extend(func_calls)
                cleaned_text = self.FUNCTION_PATTERN.sub("", cleaned_text)
                cleaned_text = (
                    cleaned_text.replace("<tool_call>", "")
                    .replace("</tool_call>", "")
                    .strip()
                )

        if tool_calls:
            if cleaned_text:
                # Strip orphan tool scaffolding a degraded model may emit after
                # the parsed call so it never leaks into user-visible content.
                cleaned_text = self.RESIDUAL_TOOL_MARKUP.sub("", cleaned_text).strip()
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=cleaned_text if cleaned_text else None,
            )
        orphan_wrapper_call = self._orphan_function_wrapper_tool_call(
            cleaned_text, request
        )
        if orphan_wrapper_call is not None:
            call, match = orphan_wrapper_call
            remaining = (
                cleaned_text[: match.start()] + cleaned_text[match.end() :]
            ).strip()
            if self.EMPTY_MARKDOWN_FENCE_PATTERN.fullmatch(remaining):
                remaining = ""
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=[call],
                content=remaining or None,
            )
        markdown_call = self._markdown_tool_call(cleaned_text, request)
        if markdown_call is not None:
            call, _ = markdown_call
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=[call],
                content=None,
            )
        labeled_call = self._labeled_line_tool_call(cleaned_text, request)
        if labeled_call is not None:
            call, _ = labeled_call
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=[call],
                content=None,
            )
        plain_call = self._plain_tool_line_call(cleaned_text, request)
        if plain_call is not None:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=[plain_call],
                content=None,
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
        Extract tool calls from streaming Qwen model output.
        """
        # Check for tool call markers
        has_tool_marker = (
            "<tool_call>" in current_text
            or "[Calling tool:" in current_text
            or "<function=" in current_text  # issue #192 (canonical)
            or "<function name=" in current_text  # off-format attribute variant
            or "<function:" in current_text  # colon dialect (2026-07-08)
            or re.search(r"(?m)^\s*#\s*Calling\s+", current_text) is not None
            or self._labeled_line_tool_call(current_text, request) is not None
        )

        if not has_tool_marker:
            return {"content": delta_text}

        # If we're in a tool call, accumulate and parse at the end
        # For simplicity, return None during accumulation
        if (
            "</tool_call>" in delta_text
            or ")]" in delta_text
            or "</function>" in delta_text  # issue #192
            or self.MARKDOWN_CALL_PATTERN.search(current_text) is not None
            or self._labeled_line_tool_call(current_text, request) is not None
        ):
            # Tool call complete, parse the whole thing
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

    def _stream_stop_candidate_valid(self, candidate: str) -> bool:
        """Return whether a closed candidate is executable for this request.

        Qwen remains globally non-early-stop because ordinary Qwen turns may
        contain multiple/interleaved calls. The server consults these helpers
        only for an actual user request that explicitly requires one tool
        exactly once. Keep the candidate schema-aware so an empty/malformed
        first attempt cannot terminate generation before the model repairs it.
        """
        request = getattr(self, "_stream_stop_request", None)
        result = self.extract_tool_calls(candidate, request=request)
        if not result.tools_called or len(result.tool_calls) != 1:
            return False

        tool_call = result.tool_calls[0]
        name = tool_call.get("name")
        if not isinstance(name, str) or not name:
            return False
        schema = self._function_schema_for_tool(request, name)
        if request and schema is None:
            return False
        if not isinstance(schema, dict):
            return True

        try:
            arguments = json.loads(tool_call.get("arguments") or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        if not isinstance(arguments, dict):
            return False
        required = schema.get("required") or []
        for key in required:
            value = arguments.get(key)
            if value is None or (isinstance(value, str) and not value.strip()):
                return False
        return True

    def _first_complete_stream_call_end(self, buffered_text: str) -> int | None:
        candidates: list[tuple[int, str]] = []
        candidates.extend(
            (match.end(), match.group(0))
            for match in self.TOOL_BLOCK_PATTERN.finditer(buffered_text)
        )
        candidates.extend(
            (match.end(), match.group(0))
            for match in self.BRACKET_PATTERN.finditer(buffered_text)
        )
        candidates.extend(
            (match.end(), match.group(0))
            for match in self.MARKDOWN_CALL_PATTERN.finditer(buffered_text)
        )
        candidates.extend(
            (match.end(), match.group(0))
            for match in self.LABELED_LINE_CALL_PATTERN.finditer(buffered_text)
        )
        for end, candidate in sorted(candidates, key=lambda item: item[0]):
            if self._stream_stop_candidate_valid(candidate):
                return end
        return None

    def stream_tool_calls_complete(self, buffered_text: str) -> bool:
        """Detect the first closed, schema-valid call for exact-once requests.

        ``STREAM_STOPS_AFTER_COMPLETE_CALL`` deliberately remains ``False``.
        The server opts into this method per request only when the user names a
        single available tool and explicitly says to call it exactly once.
        """
        return self._first_complete_stream_call_end(buffered_text) is not None

    def stream_tool_call_stop_truncate(self, buffered_text: str) -> str:
        """Drop Qwen's post-call reasoning after the first exact-once call."""
        end = self._first_complete_stream_call_end(buffered_text)
        return buffered_text if end is None else buffered_text[:end].rstrip()
