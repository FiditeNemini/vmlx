# SPDX-License-Identifier: Apache-2.0
"""openPangu (openpangu_v2) tool parser tests.

openPangu-2.0 emits tool calls as a JSON LIST wrapped in dedicated
special-token tags (token ids 148903 / 148904):

    <|tool_call_start|>
    [{"name": "fn", "arguments": {...}}, ...]
    <|tool_call_end|>

The previously stamped qwen parser (<tool_call>{...}</tool_call>) never
matches this format — live-proven tool_calls=None on
openPangu-2.0-Flash-JANG_2L. This file pins the dedicated parser's contract:
JSON-list extraction, multi-call lists, arguments-object serialization,
content/tool separation, mid-reasoning and post-</think> positions, and the
streaming buffer/emit convention with stable per-call ids (#219 contract).
"""

import json

import pytest

from vmlx_engine.tool_parsers.openpangu_tool_parser import OpenPanguToolParser


@pytest.fixture
def parser():
    return OpenPanguToolParser(tokenizer=None)


class TestOpenPanguToolParser:
    def test_no_tool_calls_returns_content(self, parser):
        out = parser.extract_tool_calls("Paris is the capital of France.")
        assert out.tools_called is False
        assert out.tool_calls == []
        assert out.content == "Paris is the capital of France."

    def test_single_call_in_list(self, parser):
        text = (
            '<|tool_call_start|>\n'
            '[{"name": "get_weather", "arguments": {"location": "Paris"}}]\n'
            '<|tool_call_end|>'
        )
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        assert len(out.tool_calls) == 1
        tc = out.tool_calls[0]
        assert tc["name"] == "get_weather"
        assert json.loads(tc["arguments"]) == {"location": "Paris"}
        assert tc["id"].startswith("call_")
        assert out.content is None

    def test_multiple_calls_in_one_list(self, parser):
        text = (
            '<|tool_call_start|>\n'
            '[{"name": "get_weather", "arguments": {"location": "Paris"}}, '
            '{"name": "get_time", "arguments": {"timezone": "CET"}}]\n'
            '<|tool_call_end|>'
        )
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        assert [tc["name"] for tc in out.tool_calls] == ["get_weather", "get_time"]
        assert json.loads(out.tool_calls[1]["arguments"]) == {"timezone": "CET"}
        # Each call gets its own unique id.
        ids = [tc["id"] for tc in out.tool_calls]
        assert len(set(ids)) == 2

    def test_arguments_object_serialized_to_json_string(self, parser):
        """Nested arguments objects must round-trip through a JSON STRING
        (OpenAI schema), preserving non-ASCII text."""
        args = {"query": "天气", "options": {"units": "metric", "days": 3}}
        text = (
            '<|tool_call_start|>['
            + json.dumps({"name": "search", "arguments": args}, ensure_ascii=False)
            + ']<|tool_call_end|>'
        )
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        serialized = out.tool_calls[0]["arguments"]
        assert isinstance(serialized, str)
        assert json.loads(serialized) == args
        assert "天气" in serialized  # ensure_ascii=False

    def test_whitespace_tolerance_around_list(self, parser):
        text = (
            '<|tool_call_start|>  \n\n  '
            '[ {"name": "fn", "arguments": {}} ]  \n  <|tool_call_end|>'
        )
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        assert out.tool_calls[0]["name"] == "fn"
        assert json.loads(out.tool_calls[0]["arguments"]) == {}

    def test_content_outside_tags_preserved(self, parser):
        text = (
            'Let me check the weather.\n'
            '<|tool_call_start|>\n'
            '[{"name": "get_weather", "arguments": {"location": "Paris"}}]\n'
            '<|tool_call_end|>'
        )
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        assert out.content == "Let me check the weather."
        assert "<|tool_call_start|>" not in (out.content or "")

    def test_tool_call_after_reasoning_text(self, parser):
        """think_in_template=True: the prompt opens the <think> rail, so the
        output carries only the closing tag. The call after </think> must
        parse and the reasoning must not leak into content."""
        text = (
            'The user wants weather data, I should call the tool.</think>'
            'Checking now.\n'
            '<|tool_call_start|>\n'
            '[{"name": "get_weather", "arguments": {"location": "Paris"}}]\n'
            '<|tool_call_end|>'
        )
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        assert out.tool_calls[0]["name"] == "get_weather"
        assert out.content == "Checking now."
        assert "should call the tool" not in (out.content or "")

    def test_tool_call_mid_reasoning_not_lost_to_think_strip(self, parser):
        """A call emitted INSIDE the think block must still be extracted —
        extraction scans the raw output before think stripping."""
        text = (
            '<think>I need data first.\n'
            '<|tool_call_start|>\n'
            '[{"name": "get_weather", "arguments": {"location": "Paris"}}]\n'
            '<|tool_call_end|>\n'
            'Waiting for the result.</think>'
        )
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        assert out.tool_calls[0]["name"] == "get_weather"
        # The think text itself is not content.
        assert out.content is None

    def test_bare_single_object_tolerated(self, parser):
        """Canonical payload is a list, but a bare object is unambiguous."""
        text = (
            '<|tool_call_start|>'
            '{"name": "get_time", "arguments": {"timezone": "CET"}}'
            '<|tool_call_end|>'
        )
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        assert out.tool_calls[0]["name"] == "get_time"

    def test_malformed_json_is_not_promoted(self, parser):
        text = '<|tool_call_start|>\n[{"name": "fn", "arguments": {broken]\n<|tool_call_end|>'
        out = parser.extract_tool_calls(text)
        assert out.tools_called is False
        assert out.tool_calls == []

    def test_unterminated_block_with_complete_json_parses(self, parser):
        """max_tokens after the list closed but before <|tool_call_end|>."""
        text = (
            'Answer coming.\n<|tool_call_start|>\n'
            '[{"name": "get_weather", "arguments": {"location": "Paris"}}]'
        )
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        assert out.tool_calls[0]["name"] == "get_weather"
        # Unterminated block residue must never leak as content.
        assert out.content == "Answer coming."

    def test_missing_name_skipped(self, parser):
        text = (
            '<|tool_call_start|>'
            '[{"arguments": {"x": 1}}, {"name": "ok_fn", "arguments": {"x": 2}}]'
            '<|tool_call_end|>'
        )
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        assert len(out.tool_calls) == 1
        assert out.tool_calls[0]["name"] == "ok_fn"

    def test_registry_aliases_resolve(self):
        from vmlx_engine.tool_parsers.abstract_tool_parser import ToolParserManager

        for alias in ("openpangu", "openpangu_v2"):
            cls = ToolParserManager.get_tool_parser(alias)
            assert cls is OpenPanguToolParser, (
                f"alias {alias!r} should resolve to OpenPanguToolParser, got {cls}"
            )


class TestOpenPanguToolParserStreaming:
    def test_streaming_passthrough_then_buffer_then_emit(self, parser):
        # Plain text before any marker — pass through as content
        prev = ""
        cur = "Let me check."
        delta = "Let me check."
        assert parser.extract_tool_calls_streaming(prev, cur, delta) == {
            "content": "Let me check."
        }

        # Marker opened — buffer (suppress output)
        prev = cur
        cur = 'Let me check.<|tool_call_start|>\n[{"name": "get_weather",'
        delta = '<|tool_call_start|>\n[{"name": "get_weather",'
        assert parser.extract_tool_calls_streaming(prev, cur, delta) is None

        # Mid-JSON delta — still buffering
        prev = cur
        cur = prev + ' "arguments": {"location": "Paris"}}]\n'
        delta = ' "arguments": {"location": "Paris"}}]\n'
        assert parser.extract_tool_calls_streaming(prev, cur, delta) is None

        # Close tag arrives — emit the tool_calls payload
        prev = cur
        cur = prev + "<|tool_call_end|>"
        delta = "<|tool_call_end|>"
        result = parser.extract_tool_calls_streaming(prev, cur, delta)
        assert result is not None
        assert "tool_calls" in result
        tc = result["tool_calls"][0]
        assert tc["index"] == 0
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"location": "Paris"}

    def test_streaming_ids_stable_and_unique_per_call(self, parser):
        """#219 contract: every streaming delta for one tool_call must carry
        ONE stable id (START and argument deltas are reconciled by id in
        OpenAI SDKs). The parser emits the complete call in a single delta,
        so each entry must have a well-formed unique id, distinct per call."""
        body = (
            '[{"name": "get_weather", "arguments": {"location": "Paris"}}, '
            '{"name": "get_time", "arguments": {"timezone": "CET"}}]'
        )
        cur = f"<|tool_call_start|>\n{body}\n<|tool_call_end|>"
        result = parser.extract_tool_calls_streaming(
            previous_text=f"<|tool_call_start|>\n{body}\n",
            current_text=cur,
            delta_text="<|tool_call_end|>",
        )
        assert result is not None
        calls = result["tool_calls"]
        assert [c["index"] for c in calls] == [0, 1]
        ids = [c["id"] for c in calls]
        assert all(i.startswith("call_") for i in ids)
        assert len(set(ids)) == len(ids), "ids must be unique per call"
        # Within one emitted delta, id and function stay paired: re-parsing
        # the same buffered text must not reshuffle names across indexes.
        assert calls[0]["function"]["name"] == "get_weather"
        assert calls[1]["function"]["name"] == "get_time"

    def test_streaming_server_marker_list_covers_openpangu_tags(self):
        """The server's buffer-then-parse trigger list must contain the
        openPangu special-token tags, or streaming would leak the raw JSON
        list as visible content before the parser ever runs."""
        from vmlx_engine import server

        assert "<|tool_call_start|>" in server._TOOL_CALL_MARKERS
        assert "<|tool_call_end|>" in server._TOOL_CALL_MARKERS
