# SPDX-License-Identifier: Apache-2.0
"""
openPangu tool call parser for vmlx-engine.

Handles openPangu-2.0 (openpangu_v2) tool calling format: a JSON LIST of
{"name": ..., "arguments": ...} objects wrapped in dedicated special-token
tags (token ids 148903 / 148904):

    <|tool_call_start|>
    [{"name": "get_weather", "arguments": {"location": "Paris"}},
     {"name": "get_time", "arguments": {"timezone": "CET"}}]
    <|tool_call_end|>

The chat template documents exactly this shape ("多个调用组成一个列表" —
multiple calls form one list) and re-renders assistant tool_calls in the same
format, so this parser is the round-trip counterpart of the template.

NOTE: the stamped tool_parser in current JANG bundles is "qwen"
(<tool_call>{...}</tool_call>), which never matches this format — live-proven
tool_calls=None on openPangu-2.0-Flash-JANG_2L. model_config_registry
neutralizes that stale stamp so this parser is used instead.
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


@ToolParserManager.register_module(["openpangu", "openpangu_v2"])
class OpenPanguToolParser(ToolParser):
    """
    Tool call parser for openPangu-2.0 (openpangu_v2) models.

    Format: <|tool_call_start|>\\n[{"name": ..., "arguments": {...}}, ...]\\n<|tool_call_end|>

    - The payload is a JSON LIST; each element is one tool call.
    - "arguments" is normally a JSON object (per template docs) and must be
      serialized to a JSON string for the OpenAI tool_calls schema.
    - Tool calls can appear mid-reasoning (inside the model-owned <think>
      rail) or after </think> — extraction scans the RAW output so a call
      emitted inside the think block is not lost to think-tag stripping.
    - Content outside the tag pair (minus think tags) remains content.

    Used when --enable-auto-tool-choice --tool-call-parser openpangu are set.
    """

    # The template handles role="tool" messages and assistant tool_calls
    # natively (renders <|message_start|>tool ... and re-encodes tool_calls
    # into the <|tool_call_start|> list), so no text-format conversion needed.
    SUPPORTS_NATIVE_TOOL_FORMAT = True

    START_TAG = "<|tool_call_start|>"
    END_TAG = "<|tool_call_end|>"

    # Complete <|tool_call_start|>...<|tool_call_end|> blocks. Whitespace /
    # newlines around the JSON list are part of the canonical format.
    TOOL_CALL_PATTERN = re.compile(
        r"<\|tool_call_start\|>\s*(.*?)\s*<\|tool_call_end\|>",
        re.DOTALL,
    )

    @staticmethod
    def _serialize_arguments(arguments: Any) -> str:
        """Serialize a tool call's arguments to a JSON string (OpenAI schema).

        The openPangu format carries arguments as a JSON object; the template
        also accepts a pre-serialized string ("arguments is string" branch),
        so pass strings through verbatim.
        """
        if isinstance(arguments, str):
            return arguments
        return json.dumps(arguments, ensure_ascii=False)

    @classmethod
    def _tool_calls_from_payload(cls, payload: str) -> list[dict[str, Any]]:
        """Parse one tag-pair payload (a JSON list) into tool call dicts."""
        payload = payload.strip()
        if not payload:
            return []
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return []

        # Canonical shape is a list; tolerate a bare single object.
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return []

        tool_calls: list[dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            arguments = item.get("arguments", item.get("parameters", {}))
            tool_calls.append(
                {
                    "id": generate_tool_id(),
                    "name": name.strip(),
                    "arguments": cls._serialize_arguments(arguments),
                }
            )
        return tool_calls

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete openPangu model response.
        """
        if self.START_TAG not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=self.strip_think_tags(model_output),
            )

        # Scan the RAW output for tool call blocks BEFORE think stripping:
        # openPangu can emit a call mid-reasoning, and the implicit-think
        # strip (open <think> rail in the prompt) would otherwise delete it.
        tool_calls: list[dict[str, Any]] = []
        for match in self.TOOL_CALL_PATTERN.finditer(model_output):
            tool_calls.extend(self._tool_calls_from_payload(match.group(1)))

        # Unterminated block (max_tokens hit after the JSON list closed but
        # before <|tool_call_end|>): only a COMPLETE JSON payload parses, so
        # this cannot promote truncated garbage to a tool call.
        if not tool_calls:
            start = model_output.rfind(self.START_TAG)
            tail = model_output[start + len(self.START_TAG):]
            if self.END_TAG not in tail:
                tool_calls.extend(self._tool_calls_from_payload(tail))

        # Content = think-stripped text with tool call blocks removed.
        content_text = self.strip_think_tags(model_output)
        content_text = self.TOOL_CALL_PATTERN.sub("", content_text).strip()
        if tool_calls and self.START_TAG in content_text:
            # Residue of an unterminated block — never user-visible content.
            content_text = content_text.split(self.START_TAG, 1)[0].strip()

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content_text if content_text else None,
            )
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=content_text
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
        Extract tool calls from streaming openPangu model output.
        """
        # No tool call marker yet — pass through as content
        if self.START_TAG not in current_text:
            return {"content": delta_text}

        # Tool call block just completed — parse the full accumulated text.
        # The ids come from extract_tool_calls, one stable id per call: the
        # START delta and any later argument deltas for the same call must
        # share the same id (#219 contract).
        if self.END_TAG in delta_text:
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

        # Still accumulating tool call content — suppress output
        return None
