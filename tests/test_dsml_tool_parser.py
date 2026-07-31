# SPDX-License-Identifier: Apache-2.0
"""DSV4 DSML tool parser tests — no-leak coverage.

Issue: per `docs/internal/AUDIT_2026_05_10_PARSER_NO_LEAK_COVERAGE.md` Finding T1,
the DSML tool parser (used by deepseek_v4 / DSV4 Flash) had no dedicated test
file. Tool XML markup (`<｜DSML｜invoke>`, `<｜DSML｜parameter>`) could silently
leak into the user-visible `content` if any extraction path regressed.

Behavior: extract_tool_calls() must produce a clean ExtractedToolCallInformation
with all DSML markup stripped from `content` whenever tools_called=True.

Fix: this test file pins the contract per parser API — no source patch needed
(parser already strips correctly via `_INVOKE_RE.sub("", model_output).strip()`).

Format reference (DSML / DeepSeek Markup Language):
    <｜DSML｜invoke name="search_web">
    <｜DSML｜parameter name="query" string="true">weather in LA</｜DSML｜parameter>
    <｜DSML｜parameter name="limit" string="false">5</｜DSML｜parameter>
    </｜DSML｜invoke>

DSML delimiter is fullwidth vertical bar `｜` (U+FF5C), same character class as
DeepSeek's other special tokens (`<｜begin▁of▁sentence｜>`, `<｜User｜>`,
`<｜Assistant｜>`).
"""

import json

import pytest

from vmlx_engine.tool_parsers.dsml_tool_parser import DSML_PREFIX, DSMLToolParser


@pytest.fixture
def parser():
    return DSMLToolParser(tokenizer=None)


class TestDSMLToolParser:
    def test_no_tool_calls_returns_content_unchanged(self, parser):
        """Plain text without DSML markup must pass through verbatim."""
        out = parser.extract_tool_calls("Hello world, no tools here.")
        assert out.tools_called is False
        assert out.tool_calls == []
        assert out.content == "Hello world, no tools here."

    def test_single_invoke_with_string_param(self, parser):
        """Single invoke with one string="true" parameter."""
        text = (
            f'<{DSML_PREFIX}invoke name="get_weather">\n'
            f'<{DSML_PREFIX}parameter name="city" string="true">Paris</{DSML_PREFIX}parameter>\n'
            f'</{DSML_PREFIX}invoke>'
        )
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        assert len(out.tool_calls) == 1
        assert out.tool_calls[0]["name"] == "get_weather"
        args = json.loads(out.tool_calls[0]["arguments"])
        assert args == {"city": "Paris"}
        # No DSML markup may leak into content.
        assert out.content is None or DSML_PREFIX not in out.content
        assert out.content is None or "<" + DSML_PREFIX not in out.content

    def test_string_param_preserves_exact_threejs_code_argument(self, parser):
        """Exact code fidelity must be owned by generation, not changed by DSML parsing."""
        expected_code = (
            "const scene = new THREE.Scene();\n"
            "const renderer = new THREE.WebGLRenderer();\n"
            "const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);\n"
            "const cube = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial());\n"
            "scene.add(cube);\n"
            "renderer.render(scene, camera);"
        )
        text = (
            f'<{DSML_PREFIX}invoke name="write_file">\n'
            f'<{DSML_PREFIX}parameter name="path" string="true">landing-p/scene.js</{DSML_PREFIX}parameter>\n'
            f'<{DSML_PREFIX}parameter name="content" string="true">{expected_code}</{DSML_PREFIX}parameter>\n'
            f'</{DSML_PREFIX}invoke>'
        )

        out = parser.extract_tool_calls(
            text,
            request={
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "content": {"type": "string"},
                                },
                                "required": ["path", "content"],
                            },
                        },
                    }
                ]
            },
        )

        assert out.tools_called is True
        args = json.loads(out.tool_calls[0]["arguments"])
        assert args["path"] == "landing-p/scene.js"
        assert args["content"] == expected_code
        assert "THREE.WebGLRenderer" in args["content"]
        assert args["content"].endswith("renderer.render(scene, camera);")

    def test_rejects_schema_keyed_noncanonical_dsml_parameter(self, parser):
        """A schema-looking attribute must not become an executable argument."""
        generated_code = (
            "const scene = new THREE.Scene();\n"
            "const renderer = new THREE.WebWebGLRenderer();\n"
            "const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);\n"
            "const cube = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MMeshBasicMaterial());\n"
            "scene.add(cube);\n"
            "renderer.render(scene, camera);"
        )
        text = (
            f"<{DSML_PREFIX}tool_calls>\n"
            f'<{DSML_PREFIX}invoke name="write_file">\n'
            f'<{DSML_PREFIX}parameter name="path" string="true">landing-p/scene.js</{DSML_PREFIX}parameter>\n'
            f'<{DSML_PREFIX}parameter content string="true">{generated_code}</{DSML_PREFIX}parameter>\n'
            f"</{DSML_PREFIX}inv>\n"
            f"</{DSML_PREFIX}tool_cs>"
        )
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["path", "content"],
                        },
                    },
                }
            ]
        }

        out = parser.extract_tool_calls(text, request=request)

        assert out.tools_called is False
        assert out.tool_calls == []
        assert out.content is None

    def test_invoke_with_typed_params_string_false_decodes_json(self, parser):
        """string="false" parameters parse as JSON (numbers, bools, arrays)."""
        text = (
            f'<{DSML_PREFIX}invoke name="set_temperature">\n'
            f'<{DSML_PREFIX}parameter name="celsius" string="false">22.5</{DSML_PREFIX}parameter>\n'
            f'<{DSML_PREFIX}parameter name="auto_adjust" string="false">true</{DSML_PREFIX}parameter>\n'
            f'<{DSML_PREFIX}parameter name="label" string="true">kitchen</{DSML_PREFIX}parameter>\n'
            f'</{DSML_PREFIX}invoke>'
        )
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        args = json.loads(out.tool_calls[0]["arguments"])
        assert args == {"celsius": 22.5, "auto_adjust": True, "label": "kitchen"}

    def test_multiple_invokes_in_one_completion(self, parser):
        """Two invoke blocks back-to-back must each become their own tool call."""
        text = (
            f'<{DSML_PREFIX}invoke name="fn_a">\n'
            f'<{DSML_PREFIX}parameter name="x" string="false">1</{DSML_PREFIX}parameter>\n'
            f'</{DSML_PREFIX}invoke>\n'
            f'<{DSML_PREFIX}invoke name="fn_b">\n'
            f'<{DSML_PREFIX}parameter name="y" string="false">2</{DSML_PREFIX}parameter>\n'
            f'</{DSML_PREFIX}invoke>'
        )
        out = parser.extract_tool_calls(text)
        assert len(out.tool_calls) == 2
        assert out.tool_calls[0]["name"] == "fn_a"
        assert out.tool_calls[1]["name"] == "fn_b"
        assert json.loads(out.tool_calls[0]["arguments"]) == {"x": 1}
        assert json.loads(out.tool_calls[1]["arguments"]) == {"y": 2}

    def test_visible_text_around_invoke_preserved_no_dsml_leak(self, parser):
        """Surrounding visible text stays in content; DSML markup must not leak."""
        text = (
            "Sure, calling the function now.\n"
            f'<{DSML_PREFIX}invoke name="ping">\n'
            f'<{DSML_PREFIX}parameter name="host" string="true">example.com</{DSML_PREFIX}parameter>\n'
            f'</{DSML_PREFIX}invoke>\n'
            "Done."
        )
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        assert out.tool_calls[0]["name"] == "ping"
        assert out.content is not None
        assert "Sure, calling the function now." in out.content
        assert "Done." in out.content
        # Critical no-leak assertions — every DSML token must be stripped.
        assert "<" + DSML_PREFIX + "invoke" not in out.content
        assert "</" + DSML_PREFIX + "invoke>" not in out.content
        assert "<" + DSML_PREFIX + "parameter" not in out.content
        assert "</" + DSML_PREFIX + "parameter>" not in out.content
        assert DSML_PREFIX not in out.content

    def test_dsml_only_completion_returns_none_content(self, parser):
        """When the whole completion is DSML markup, content should be None."""
        text = (
            f'<{DSML_PREFIX}invoke name="fn">\n'
            f'<{DSML_PREFIX}parameter name="k" string="true">v</{DSML_PREFIX}parameter>\n'
            f'</{DSML_PREFIX}invoke>'
        )
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        # All text was tool envelope — content must be None or empty/whitespace.
        assert out.content is None or out.content.strip() == ""

    def test_no_dsml_markers_short_circuit_returns_content(self, parser):
        """Short-circuit: text without DSML prefix or `<invoke_` returns content."""
        # Common prose-only chat with non-DSML angle brackets (e.g. inequalities).
        text = "If x < 5 and y > 3 then..."
        out = parser.extract_tool_calls(text)
        assert out.tools_called is False
        assert out.content == text

    def test_registry_aliases_resolve(self):
        """DSMLToolParser must register under both `dsml` and `deepseek_v4`."""
        from vmlx_engine.tool_parsers.abstract_tool_parser import ToolParserManager

        for alias in ("dsml", "deepseek_v4"):
            cls = ToolParserManager.get_tool_parser(alias)
            assert cls is DSMLToolParser, (
                f"alias {alias!r} should resolve to DSMLToolParser, got {cls}"
            )

    def test_supports_native_tool_format(self):
        """DSMLToolParser exposes supports_native_format()=True so server.py
        keeps the raw template's tool affordances rather than injecting the
        generic JSON-tools instructions block."""
        assert DSMLToolParser.supports_native_format() is True
        assert DSMLToolParser.STRICT_NATIVE_TOOL_FORMAT is True
        assert DSMLToolParser.SUPPRESS_INVALID_NATIVE_MARKUP is True

    def test_complete_dsml_wrapper_opts_into_stream_early_stop(self, parser):
        parser._stream_stop_request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "file_info",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ]
        }
        call = (
            f"<{DSML_PREFIX}tool_calls>\n"
            f'<{DSML_PREFIX}invoke name="file_info">\n'
            f'<{DSML_PREFIX}parameter name="path" string="true">README.md'
            f"</{DSML_PREFIX}parameter>\n"
            f"</{DSML_PREFIX}invoke>\n"
            f"</{DSML_PREFIX}tool_calls>"
        )
        buffered = "reasoning before\n" + call + "\npost-call rambling " * 20

        assert DSMLToolParser.STREAM_STOPS_AFTER_COMPLETE_CALL is True
        assert parser.stream_tool_calls_complete(buffered) is True
        assert parser.stream_tool_call_stop_truncate(buffered) == (
            "reasoning before\n" + call
        )

    def test_stream_early_stop_waits_for_wrapper_close(self, parser):
        parser._stream_stop_request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "file_info",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ]
        }
        incomplete = (
            f"<{DSML_PREFIX}tool_calls>\n"
            f'<{DSML_PREFIX}invoke name="file_info">\n'
            f'<{DSML_PREFIX}parameter name="path" string="true">README.md'
            f"</{DSML_PREFIX}parameter>\n"
            f"</{DSML_PREFIX}invoke>"
        )

        assert parser.stream_tool_calls_complete(incomplete) is False
        assert parser.stream_tool_call_stop_truncate(incomplete) == incomplete

    def test_stream_early_stop_rejects_missing_required_argument(self, parser):
        parser._stream_stop_request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "file_info",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ]
        }
        malformed = (
            f"<{DSML_PREFIX}tool_calls>\n"
            f'<{DSML_PREFIX}invoke name="file_info">\n'
            f"</{DSML_PREFIX}invoke>\n"
            f"</{DSML_PREFIX}tool_calls>"
        )

        assert parser.stream_tool_calls_complete(malformed) is False
        assert parser.stream_tool_call_stop_truncate(malformed) == malformed

    def test_stream_early_stop_allows_bare_invoke_without_canonical_encoder(
        self, parser
    ):
        """Older-bundle compatibility remains explicit without model_path."""
        parser._stream_stop_request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "file_info",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ]
        }
        call = (
            f'<{DSML_PREFIX}invoke name="file_info">\n'
            f'<{DSML_PREFIX}parameter name="path" string="true">README.md'
            f"</{DSML_PREFIX}parameter>\n"
            f"</{DSML_PREFIX}invoke>"
        )

        assert parser.stream_tool_calls_complete(call) is True
        assert parser.stream_tool_call_stop_truncate(call + " trailing") == call

    def test_canonical_encoder_rejection_blocks_regex_only_bare_invoke(
        self, parser, monkeypatch
    ):
        """A selected 0731 encoder owns rejection of non-wrapper DSML."""
        from vmlx_engine.loaders import dsv4_chat_encoder

        class RejectingCanonicalEncoder:
            eos_token = "<DSV4_EOS>"

            @staticmethod
            def parse_message_from_completion_text(_text, thinking_mode):
                assert thinking_mode == "chat"
                raise ValueError("official parser requires tool_calls wrapper")

        monkeypatch.setattr(
            dsv4_chat_encoder,
            "_load_encoding_dsv4_module",
            lambda model_path=None: RejectingCanonicalEncoder,
        )
        request = {
            "model_path": "/models/dsv4-0731",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "file_info",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ],
        }
        call = (
            f'<{DSML_PREFIX}invoke name="file_info">\n'
            f'<{DSML_PREFIX}parameter name="path" string="true">README.md'
            f"</{DSML_PREFIX}parameter>\n"
            f"</{DSML_PREFIX}invoke>"
        )

        complete = parser.extract_tool_calls(call, request=request)
        parser._stream_stop_request = request

        assert complete.tools_called is False
        assert complete.tool_calls == []
        assert complete.content is None
        assert parser.stream_tool_calls_complete(call) is False
        assert parser.stream_tool_call_stop_truncate(call) == call

    def test_tools_called_implies_no_dsml_in_content(self, parser):
        """Cross-cutting invariant: whenever tools_called=True the visible
        content field must be free of every DSML token form. Regression guard
        per AUDIT_2026_05_10_PARSER_NO_LEAK_COVERAGE.md Finding T1."""
        for text in [
            f'pre<{DSML_PREFIX}invoke name="a"><{DSML_PREFIX}parameter name="k" string="true">v</{DSML_PREFIX}parameter></{DSML_PREFIX}invoke>post',
            f'<{DSML_PREFIX}invoke name="a"><{DSML_PREFIX}parameter name="k" string="false">42</{DSML_PREFIX}parameter></{DSML_PREFIX}invoke>',
            f'lead\n<{DSML_PREFIX}invoke name="a">\n<{DSML_PREFIX}parameter name="k" string="true">x</{DSML_PREFIX}parameter>\n</{DSML_PREFIX}invoke>\ntail',
        ]:
            out = parser.extract_tool_calls(text)
            if out.tools_called:
                content = out.content or ""
                assert DSML_PREFIX not in content, (
                    f"DSML token leaked into content for input: {text!r}\n"
                    f"content: {content!r}"
                )
                assert "<" + DSML_PREFIX not in content
                assert "</" + DSML_PREFIX not in content

    def test_rejects_dsml_invoke_with_plain_param_tags(self, parser):
        """Plain HTML params are not canonical DSML and cannot execute."""
        text = (
            f'<{DSML_PREFIX}invoke name="read_file">\n'
            '    <param name="path">docs/vendor_memo.md</param>\n'
            f'</{DSML_PREFIX}invoke>'
        )
        request = {
            "tools": [
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
                }
            ]
        }

        out = parser.extract_tool_calls(text, request=request)

        assert out.tools_called is False
        assert out.tool_calls == []
        assert out.content is None

    def test_rejects_self_closing_dsml_parameter_with_value_in_string_attr(
        self, parser
    ):
        """A malformed string attribute must not be reinterpreted as a value."""
        text = (
            f'<{DSML_PREFIX}tool_calls>\n'
            f'<{DSML_PREFIX}invoke name="list_directory">\n'
            f'<{DSML_PREFIX}parameter name="path" string="." />\n'
            f'</{DSML_PREFIX}{DSML_PREFIX}parameter>\n'
            f'</{DSML_PREFIX}invoke>\n'
            f'</{DSML_PREFIX}tool_calls>'
        )
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "list_directory",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ]
        }

        out = parser.extract_tool_calls(text, request=request)

        assert out.tools_called is False
        assert out.tool_calls == []
        assert out.content is None

    def test_rejects_dsml_invuse_typo_with_complete_parameters(self, parser):
        """A typoed invoke name is malformed protocol, not a tool call."""
        text = (
            f'<{DSML_PREFIX}tool_calls>\n'
            f'<{DSML_PREFIX}invuse name="write_file">\n'
            f'<{DSML_PREFIX}parameter name="path" string="true">landing-p/scene.js</{DSML_PREFIX}parameter>\n'
            f'<{DSML_PREFIX}parameter name="content" string="true">const camera = new THREE.PersPerspectiveCamera();</{DSML_PREFIX}parameter>\n'
            f'</{DSML_PREFIX}invoke>\n'
            f'</{DSML_PREFIX}tool_calls>'
        )
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["path", "content"],
                        },
                    },
                }
            ]
        }

        out = parser.extract_tool_calls(text, request=request)

        assert out.tools_called is False
        assert out.tool_calls == []
        assert out.content is None

    def test_rejects_dsml_invue_typo_with_complete_parameters(self, parser):
        """A degraded invoke close must fail closed."""
        text = (
            f'<{DSML_PREFIX}tool_calls>\n'
            f'<{DSML_PREFIX}invue name="write_file">\n'
            f'<{DSML_PREFIX}parameter name="path" string="true">landing-p/scene.js</{DSML_PREFIX}parameter>\n'
            f'<{DSML_PREFIX}parameter name="content" string="true">const renderer = new THREE.WebGLRenderer();</{DSML_PREFIX}parameter>\n'
            f'</{DSML_PREFIX}invue>\n'
            f'</{DSML_PREFIX}tool_calls>'
        )
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["path", "content"],
                        },
                    },
                }
            ]
        }

        out = parser.extract_tool_calls(text, request=request)

        assert out.tools_called is False
        assert out.tool_calls == []
        assert out.content is None

    def test_rejects_canonical_string_equals_args_without_raw_repair(
        self, parser, monkeypatch
    ):
        """Bogus canonical output must not trigger a second repair parser."""
        from vmlx_engine.loaders import dsv4_chat_encoder

        text = (
            f'<{DSML_PREFIX}tool_calls>\n'
            f'<{DSML_PREFIX}invuse name="write_file">\n'
            f'<{DSML_PREFIX}parameter name="path" string="true">landing-p/scene.js</{DSML_PREFIX}parameter>\n'
            f'<{DSML_PREFIX}parameter name="content" string="true">const camera = new THREE.PersPerspectiveCamera();</{DSML_PREFIX}parameter>\n'
            f'</{DSML_PREFIX}invoke>\n'
            f'</{DSML_PREFIX}tool_calls>'
        )

        class FakeEncoding:
            @staticmethod
            def parse_message_from_completion_text(_text):
                return {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "write_file",
                                "arguments": {
                                    "path": " string=",
                                    "content": " string=",
                                },
                            }
                        }
                    ]
                }

        monkeypatch.setattr(
            dsv4_chat_encoder,
            "_load_encoding_dsv4_module",
            lambda model_path=None: FakeEncoding,
        )
        request = {
            "model_path": "/models/dsv4",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["path", "content"],
                        },
                    },
                }
            ],
        }

        out = parser.extract_tool_calls(text, request=request)

        assert out.tools_called is False
        assert out.tool_calls == []
        assert out.content is None

    def test_rejects_dsml_tool_calls_wrapper_with_truncated_plain_param_invokes(
        self, parser
    ):
        """Truncated mixed-dialect invokes must never execute."""
        text = (
            f'<{DSML_PREFIX}tool_calls>\n'
            f'<{DSML_PREFIX}invoke name="read_file">\n'
            '    <param>\n'
            '    <param name="path">docs/vendor_memo.md</param>\n'
            '</inv>\n\n'
            '<invoke name="read_file">\n'
            '    <param name="path">docs/release_excerpt.md</param>\n'
            '</inv>'
        )
        request = {
            "tools": [
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
                }
            ]
        }

        out = parser.extract_tool_calls(text, request=request)

        assert out.tools_called is False
        assert out.tool_calls == []
        assert out.content is None

    def test_canonical_encoder_empty_required_args_does_not_trigger_repair(
        self, parser, monkeypatch
    ):
        """Missing canonical arguments remain non-executable."""
        from vmlx_engine.loaders import dsv4_chat_encoder

        class FakeEncoding:
            @staticmethod
            def parse_message_from_completion_text(_text):
                return {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": {},
                            }
                        }
                    ],
                }

        monkeypatch.setattr(
            dsv4_chat_encoder,
            "_load_encoding_dsv4_module",
            lambda **_kwargs: FakeEncoding,
        )

        text = (
            f'<{DSML_PREFIX}tool_calls>\n'
            f'<{DSML_PREFIX}invoke name="read_file">\n'
            '    <param name="path">docs/vendor_memo.md</param>\n'
            '</inv>'
        )
        request = {
            "tools": [
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
                }
            ]
        }

        out = parser.extract_tool_calls(text, request=request)

        assert out.tools_called is False
        assert out.tool_calls == []
        assert out.content is None

    def test_strict_extraction_does_not_synthesize_encoder_arguments(
        self, parser, monkeypatch, tmp_path
    ):
        """An encoder cannot invent required args absent from emitted DSML."""
        from vmlx_engine.loaders import dsv4_chat_encoder

        captured = {}

        class FakeEncoding:
            @staticmethod
            def parse_message_from_completion_text(_text):
                return {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": {"path": "docs/vendor_memo.md"},
                            }
                        }
                    ],
                }

        def fake_loader(**kwargs):
            captured.update(kwargs)
            return FakeEncoding

        monkeypatch.setattr(
            dsv4_chat_encoder,
            "_load_encoding_dsv4_module",
            fake_loader,
        )

        request = {
            "model_path": str(tmp_path),
            "tools": [
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
                }
            ],
        }

        out = parser.extract_tool_calls(
            f'<{DSML_PREFIX}invoke name="read_file"></{DSML_PREFIX}invoke>',
            request=request,
        )

        assert captured == {}
        assert out.tools_called is False
        assert out.tool_calls == []
        assert out.content is None

    def test_malformed_wrapper_cannot_be_overridden_by_encoder(self, parser, monkeypatch):
        """Malformed wrapper residue must remain non-executable."""
        from vmlx_engine.loaders import dsv4_chat_encoder

        class FakeEncoding:
            @staticmethod
            def parse_message_from_completion_text(_text):
                return {
                    "content": f"<{DSML_PREFIX}tool_calls>\n\n</{DSML_PREFIX}tool_c>",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "Tokyo"},
                            }
                        }
                    ],
                }

        monkeypatch.setattr(
            dsv4_chat_encoder,
            "_load_encoding_dsv4_module",
            lambda **_kwargs: FakeEncoding,
        )
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ]
        }

        out = parser.extract_tool_calls(
            f"<{DSML_PREFIX}tool_calls>...</{DSML_PREFIX}tool_c>",
            request=request,
        )

        assert out.tools_called is False
        assert out.tool_calls == []
        assert out.content is None

    def test_streaming_buffers_dsml_tool_calls_wrapper_before_invoke(self, parser):
        """DSV4 may stream <｜DSML｜tool_calls> before the inner invoke.

        The streaming parser must not surface that wrapper as content while
        waiting for the complete invoke block.
        """
        first = f"\n\n<{DSML_PREFIX}tool_calls>\n<{DSML_PREFIX}tool_c"
        out = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=first,
            delta_text=first,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
        )
        assert out is None

    def test_streaming_and_nonstreaming_share_strict_completed_call(
        self, parser
    ):
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "file_info",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ]
        }
        text = (
            f"<{DSML_PREFIX}tool_calls>\n"
            f'<{DSML_PREFIX}invoke name="file_info">\n'
            f'<{DSML_PREFIX}parameter name="path" string="true">README.md'
            f"</{DSML_PREFIX}parameter>\n"
            f"</{DSML_PREFIX}invoke>\n"
            f"</{DSML_PREFIX}tool_calls>"
        )

        complete = parser.extract_tool_calls(text, request=request)
        stream = parser.extract_tool_calls_streaming(
            previous_text=text[:-1],
            current_text=text,
            delta_text=text[-1:],
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )

        assert complete.tools_called is True
        assert complete.content is None
        assert complete.tool_calls[0]["name"] == "file_info"
        assert json.loads(complete.tool_calls[0]["arguments"]) == {
            "path": "README.md"
        }
        assert stream is not None
        assert "file_info" in str(stream)
        assert "README.md" in str(stream)

    @pytest.mark.parametrize(
        ("parameters", "parameter"),
        [
            (
                {"type": "object", "properties": {}, "required": []},
                (
                    f'<{DSML_PREFIX}parameter name="unexpected" string="true">'
                    f"owned</{DSML_PREFIX}parameter>"
                ),
            ),
            (
                {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                (
                    f'<{DSML_PREFIX}parameter name="path" string="false">'
                    f"123</{DSML_PREFIX}parameter>"
                ),
            ),
        ],
    )
    def test_schema_invalid_arguments_fail_closed_in_every_dsml_path(
        self, parser, parameters, parameter
    ):
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "file_info",
                        "parameters": parameters,
                    },
                }
            ]
        }
        text = (
            f'<{DSML_PREFIX}invoke name="file_info">\n'
            f"{parameter}\n"
            f"</{DSML_PREFIX}invoke>"
        )

        complete = parser.extract_tool_calls(text, request=request)
        stream = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=text,
            delta_text=text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        parser._stream_stop_request = request

        assert complete.tools_called is False
        assert complete.tool_calls == []
        assert complete.content is None
        assert stream is None
        assert parser.stream_tool_calls_complete(text) is False

    @pytest.mark.parametrize("partial", ["<｜DSML", "</｜DSML"])
    def test_terminal_split_dsml_prefix_never_streams_or_leaks(
        self, parser, partial
    ):
        text = "Safe visible prefix.\n" + partial

        complete = parser.extract_tool_calls(text, request={"tools": []})
        stream = parser.extract_tool_calls_streaming(
            previous_text="Safe visible prefix.\n",
            current_text=text,
            delta_text=partial,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request={"tools": []},
        )

        assert complete.tools_called is False
        assert complete.tool_calls == []
        assert complete.content == "Safe visible prefix."
        assert stream is None

    @pytest.mark.parametrize(
        "malformed",
        [
            (
                f'<{DSML_PREFIX}invoke name="file_info">\n'
                f'<{DSML_PREFIX}parameter name="path" string="true">README.md'
                f"</{DSML_PREFIX}parameter>\n</"
            ),
            '<invoke_file_info><param name="path">README.md</param></inv',
            (
                f'<{DSML_PREFIX}invuse name="file_info">\n'
                f'<{DSML_PREFIX}parameter name="path" string="true">README.md'
                f"</{DSML_PREFIX}parameter>\n</{DSML_PREFIX}invuse>"
            ),
            (
                f'<{DSML_PREFIX}invoke name="file_info">\n'
                f'<{DSML_PREFIX}parameter name="path" string="false">not-json'
                f"</{DSML_PREFIX}parameter>\n</{DSML_PREFIX}invoke>"
            ),
            (
                f'<{DSML_PREFIX}invoke name="file_info">\n'
                f"</{DSML_PREFIX}invoke>"
            ),
            (
                f'<{DSML_PREFIX}invoke name="file_info">\n'
                f'<{DSML_PREFIX}parameter name="path" string="true">README.md'
                f"</{DSML_PREFIX}parameter>\n"
                f'<{DSML_PREFIX}parameter name="path" string="true">other.md'
                f"</{DSML_PREFIX}parameter>\n</{DSML_PREFIX}invoke>"
            ),
            (
                f'<{DSML_PREFIX}invoke name="run_command">\n'
                f'<{DSML_PREFIX}parameter name="command" string="true">pwd'
                f"</{DSML_PREFIX}parameter>\n</{DSML_PREFIX}invoke>"
            ),
        ],
    )
    def test_malformed_native_protocol_never_executes_streams_or_leaks(
        self, parser, malformed
    ):
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "file_info",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ]
        }
        text = "Safe visible prefix.\n" + malformed

        complete = parser.extract_tool_calls(text, request=request)
        stream = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=text,
            delta_text=text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )

        assert complete.tools_called is False
        assert complete.tool_calls == []
        assert complete.content == "Safe visible prefix."
        assert DSML_PREFIX not in complete.content
        assert "<invoke_" not in complete.content
        assert stream is None

    def test_plain_tool_result_continuation_remains_visible(self, parser):
        text = "Tool result received. The README is 5.2 KB."

        complete = parser.extract_tool_calls(text, request={"tools": []})
        stream = parser.extract_tool_calls_streaming(
            previous_text="Tool result received. ",
            current_text=text,
            delta_text="The README is 5.2 KB.",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request={"tools": []},
        )

        assert complete.tools_called is False
        assert complete.content == text
        assert stream is not None
        assert "The README is 5.2 KB." in str(stream)

    def test_server_strict_dsml_path_never_restores_malformed_markup(
        self, monkeypatch
    ):
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest

        monkeypatch.setattr(server, "_tool_call_parser", "dsml")
        monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", False)
        monkeypatch.setattr(server, "_engine", None)
        monkeypatch.setattr(server, "_model_path", "/models/dsv4")
        request = ResponsesRequest(
            model="dsv4",
            input="Call file_info for README.md.",
            tools=[
                {
                    "type": "function",
                    "name": "file_info",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ],
        )
        malformed = (
            "Safe visible prefix.\n"
            f'<{DSML_PREFIX}invoke name="file_info">\n'
            f'<{DSML_PREFIX}parameter name="path" string="true">README.md'
            f"</{DSML_PREFIX}parameter>\n</"
        )

        cleaned, calls = server._parse_tool_calls_with_parser(malformed, request)

        assert cleaned == "Safe visible prefix."
        assert calls is None
        assert DSML_PREFIX not in cleaned

    def test_server_strict_dsml_rejects_calls_without_effective_tools(
        self, monkeypatch
    ):
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest

        monkeypatch.setattr(server, "_tool_call_parser", "dsml")
        monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", False)
        monkeypatch.setattr(server, "_engine", None)
        monkeypatch.setattr(server, "_model_path", "/models/dsv4")
        monkeypatch.setattr(
            server,
            "_effective_tools_for_tool_parsing",
            lambda _request: [],
        )
        request = ResponsesRequest(model="dsv4", input="Answer plainly.")
        canonical = (
            "Safe visible prefix.\n"
            f'<{DSML_PREFIX}invoke name="delete_everything">\n'
            f'<{DSML_PREFIX}parameter name="target" string="true">all'
            f"</{DSML_PREFIX}parameter>\n"
            f"</{DSML_PREFIX}invoke>"
        )

        direct = DSMLToolParser(None)
        direct_result = direct.extract_tool_calls(
            canonical,
            request={"tools": []},
        )
        cleaned, calls = server._parse_tool_calls_with_parser(canonical, request)
        plain, plain_calls = server._parse_tool_calls_with_parser(
            "Ordinary visible answer.", request
        )

        assert direct_result.tools_called is False
        assert direct_result.tool_calls == []
        assert direct_result.content == "Safe visible prefix."
        assert cleaned == "Safe visible prefix."
        assert calls is None
        assert plain == "Ordinary visible answer."
        assert plain_calls is None

    def test_server_strict_dsml_suppresses_terminal_split_namespace(
        self, monkeypatch
    ):
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest

        monkeypatch.setattr(server, "_tool_call_parser", "dsml")
        monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", False)
        monkeypatch.setattr(server, "_engine", None)
        monkeypatch.setattr(server, "_model_path", "/models/dsv4")
        request = ResponsesRequest(
            model="dsv4",
            input="Call file_info.",
            tools=[
                {
                    "type": "function",
                    "name": "file_info",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ],
        )

        cleaned, calls = server._parse_tool_calls_with_parser(
            "Safe visible prefix.\n<｜DSML", request
        )

        assert cleaned == "Safe visible prefix."
        assert calls is None

    def test_server_strict_dsml_init_failure_never_uses_generic_repair(
        self, monkeypatch
    ):
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest

        class BrokenStrictDSMLParser:
            STRICT_NATIVE_TOOL_FORMAT = True
            SUPPRESS_INVALID_NATIVE_MARKUP = True

            def __init__(self, _tokenizer):
                raise RuntimeError("strict parser init failed")

        monkeypatch.setattr(server, "_tool_call_parser", "dsml")
        monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", False)
        monkeypatch.setattr(server, "_engine", None)
        monkeypatch.setattr(
            server.ToolParserManager,
            "get_tool_parser",
            lambda _name: BrokenStrictDSMLParser,
        )

        def generic_repair_must_not_run(_text):
            pytest.fail("strict DSML initialization failure reached generic repair")

        monkeypatch.setattr(server, "parse_tool_calls", generic_repair_must_not_run)
        request = ResponsesRequest(
            model="dsv4",
            input="Call file_info.",
            tools=[
                {
                    "type": "function",
                    "name": "file_info",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ],
        )
        native = (
            "Safe visible prefix.\n"
            f'<{DSML_PREFIX}invoke name="file_info">\n'
            f'<{DSML_PREFIX}parameter name="path" string="true">README.md'
            f"</{DSML_PREFIX}parameter>\n"
            f"</{DSML_PREFIX}invoke>"
        )

        cleaned, calls = server._parse_tool_calls_with_parser(native, request)
        plain, plain_calls = server._parse_tool_calls_with_parser(
            "Ordinary visible answer.", request
        )

        assert cleaned == "Safe visible prefix."
        assert calls is None
        assert DSML_PREFIX not in cleaned
        assert plain == "Ordinary visible answer."
        assert plain_calls is None

    def test_server_buffers_dsml_wrapper_marker_before_invoke(self):
        """Server marker list must catch DSV4 wrapper chunks before invoke."""
        from vmlx_engine.server import _TOOL_CALL_MARKERS

        assert "<｜DSML｜tool" in _TOOL_CALL_MARKERS
        assert "<｜DSML｜tool_c" in _TOOL_CALL_MARKERS
