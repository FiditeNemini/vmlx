"""Required-argument validation for parsed tool calls (model-free)."""
from vmlx_engine.server import missing_required_tool_args, tool_schema_allows_null

SCHEMA = {
    "name": "record_measurement",
    "parameters": {
        "type": "object",
        "required": ["label", "value", "tags", "meta", "verified", "parent", "maybe", "either"],
        "properties": {
            "label": {"type": "string"},
            "value": {"type": "number"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "meta": {"type": "object"},
            "verified": {"type": "boolean"},
            "parent": {"type": ["string", "null"]},
            "maybe": {"type": "string", "nullable": True},
            "either": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        },
    },
}
FULL = {"label": "x", "value": 0, "tags": [], "meta": {}, "verified": False, "parent": None, "maybe": None, "either": None}


def test_explicit_null_is_a_value_for_nullable_required_properties():
    assert missing_required_tool_args(SCHEMA, FULL) == []
    assert tool_schema_allows_null(SCHEMA, "parent")
    assert tool_schema_allows_null(SCHEMA, "maybe")
    assert tool_schema_allows_null(SCHEMA, "either")
    assert not tool_schema_allows_null(SCHEMA, "label")


def test_null_for_a_non_nullable_required_property_is_missing():
    assert missing_required_tool_args(SCHEMA, {**FULL, "label": None}) == ["label"]


def test_absent_keys_are_missing_but_falsy_and_blank_values_are_present():
    args = dict(FULL); del args["tags"]
    assert missing_required_tool_args(SCHEMA, args) == ["tags"]
    # presence only: a blank string is a present value (its acceptability is the schema's minLength)
    assert missing_required_tool_args(SCHEMA, {**FULL, "label": ""}) == []
    # False / 0 / [] / {} are values
    assert missing_required_tool_args(SCHEMA, FULL) == []


def test_schemas_without_required_or_malformed_args_never_drop():
    assert missing_required_tool_args({"name": "t", "parameters": {"type": "object"}}, {}) == []
    assert missing_required_tool_args(SCHEMA, "not-a-dict") == [k for k in SCHEMA["parameters"]["required"]]
    assert missing_required_tool_args(None, {}) == []


# ---- presence vs schema validation (separate layers) ----------------------------------
from vmlx_engine.server import validate_tool_args_against_schema, tool_args_schema_mode

STRICT = {
    "name": "set_mode",
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "required": ["mode", "level", "label", "meta"],
        "properties": {
            "mode": {"type": "string", "enum": ["eco", "balanced", "turbo"]},
            "level": {"type": "integer", "minimum": 1, "maximum": 5},
            "label": {"type": "string", "minLength": 1},
            "meta": {"type": "object", "required": ["unit"], "properties": {"unit": {"type": "string"}}},
            "note": {"type": ["string", "null"]},
        },
    },
}


def test_blank_string_is_present_for_the_presence_rule_and_invalid_only_by_schema():
    args = {"mode": "eco", "level": 1, "label": "", "meta": {"unit": "kg"}}
    assert missing_required_tool_args(STRICT, args) == []
    status, problems = validate_tool_args_against_schema(STRICT, args)
    assert status == "invalid" and any(p.startswith("label:") for p in problems)
    # JSON Schema counts characters: three spaces satisfy minLength 1 (no trimming rule exists)
    assert validate_tool_args_against_schema(STRICT, {**args, "label": "   "}) == ("valid", [])


def test_schema_validation_is_strict_about_types_enums_bounds_and_nesting():
    ok = {"mode": "turbo", "level": 5, "label": "x", "meta": {"unit": "kg"}, "note": None}
    assert validate_tool_args_against_schema(STRICT, ok) == ("valid", [])
    cases = {
        "enum": {**ok, "mode": "hyperspeed"},
        "maximum": {**ok, "level": 9},
        "string-for-integer": {**ok, "level": "5"},
        "boolean-for-integer": {**ok, "level": True},
        "nested-required": {**ok, "meta": {}},
        "additional-property": {**ok, "extra": 1},
    }
    for label, args in cases.items():
        status, problems = validate_tool_args_against_schema(STRICT, args)
        assert status == "invalid" and problems, label


def test_malformed_or_absent_schemas_are_unconstrained_never_drops():
    assert validate_tool_args_against_schema({"name": "t"}, {"a": 1}) == ("unconstrained", [])
    status, problems = validate_tool_args_against_schema({"name": "t", "parameters": {"type": "object", "properties": {"a": {"type": "not-a-type"}}}}, {"a": 1})
    assert status == "unconstrained" and problems and "malformed input schema" in problems[0]
    status, _ = validate_tool_args_against_schema({"name": "t", "parameters": {"type": "array"}}, {"a": 1})
    assert status == "unconstrained"
    status, problems = validate_tool_args_against_schema(STRICT, "not-a-dict")
    assert status == "invalid" and problems == ["arguments are not a JSON object"]


def test_schema_mode_switch_defaults_to_warn(monkeypatch):
    monkeypatch.delenv("VMLX_TOOL_ARGS_SCHEMA_VALIDATION", raising=False)
    monkeypatch.delenv("VMLINUX_TOOL_ARGS_SCHEMA_VALIDATION", raising=False)
    assert tool_args_schema_mode() == "warn"
    monkeypatch.setenv("VMLX_TOOL_ARGS_SCHEMA_VALIDATION", "enforce")
    assert tool_args_schema_mode() == "enforce"
    monkeypatch.setenv("VMLX_TOOL_ARGS_SCHEMA_VALIDATION", "bogus")
    assert tool_args_schema_mode() == "warn"


# ---- warnings reach the caller on every lane -------------------------------------------
import inspect
import json

from vmlx_engine import server as _server
from vmlx_engine.server import (
    ChatCompletionRequest,
    _begin_tool_call_drop_capture,
    _parse_tool_calls_with_parser,
    _record_tool_call_drop,
    _take_tool_call_drop_diagnostics,
)


def test_drop_diagnostic_capture_round_trip_and_noop_without_capture():
    _server._TOOL_CALL_DROP_DIAGNOSTICS.set(None)
    _record_tool_call_drop("ignored: capture not started")
    assert _take_tool_call_drop_diagnostics() == []
    _begin_tool_call_drop_capture()
    _record_tool_call_drop("first")
    _record_tool_call_drop("second")
    assert _take_tool_call_drop_diagnostics() == ["first", "second"]
    assert _take_tool_call_drop_diagnostics() == []


def _request_with_strict_tool() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "set the mode"}],
        tools=[{"type": "function", "function": STRICT}],
    )


def _set_mode_call(args: dict) -> str:
    return "<tool_call>" + json.dumps({"name": "set_mode", "arguments": args}) + "</tool_call>"


def test_warn_mode_delivers_the_call_and_records_the_schema_problem(monkeypatch):
    monkeypatch.delenv("VMLX_TOOL_ARGS_SCHEMA_VALIDATION", raising=False)
    monkeypatch.setattr(_server, "_tool_call_parser_disabled_explicitly", False, raising=False)
    _begin_tool_call_drop_capture()
    _, calls = _parse_tool_calls_with_parser(
        _set_mode_call({"mode": "nope", "level": 1, "label": "ok", "meta": {"unit": "kg"}}),
        _request_with_strict_tool(),
    )
    diagnostics = _take_tool_call_drop_diagnostics()
    assert calls and len(calls) == 1 and calls[0].function.name == "set_mode"
    assert diagnostics and "set_mode" in diagnostics[0] and "mode" in diagnostics[0]


def test_enforce_mode_drops_the_call_and_records_why(monkeypatch):
    monkeypatch.setenv("VMLX_TOOL_ARGS_SCHEMA_VALIDATION", "enforce")
    monkeypatch.setattr(_server, "_tool_call_parser_disabled_explicitly", False, raising=False)
    _begin_tool_call_drop_capture()
    _, calls = _parse_tool_calls_with_parser(
        _set_mode_call({"mode": "nope", "level": 1, "label": "ok", "meta": {"unit": "kg"}}),
        _request_with_strict_tool(),
    )
    diagnostics = _take_tool_call_drop_diagnostics()
    assert not calls
    assert diagnostics and "set_mode" in diagnostics[0]


def test_every_lane_surfaces_dropped_call_diagnostics_in_warnings():
    """Non-streaming Chat and Responses start capture before parsing and merge the
    diagnostics into `warnings`; the chat stream delivers them even when a call was
    emitted (warn mode delivers the call together with its problems)."""
    for fn in (_server.create_chat_completion, _server.create_response):
        src = inspect.getsource(fn)
        assert src.index("_begin_tool_call_drop_capture()") < src.index("_parse_tool_calls_with_parser(")
        assert "_take_tool_call_drop_diagnostics() or None," in src
    for fn in (_server.stream_chat_completion, _server.stream_responses_api):
        src = inspect.getsource(fn)
        assert "_begin_tool_call_drop_capture()" in src
        assert "_take_tool_call_drop_diagnostics()" in src
    chat_stream = inspect.getsource(_server.stream_chat_completion)
    assert "_dropped_tc_diagnostics and not tool_calls_emitted" not in chat_stream
    assert "if _dropped_tc_diagnostics:" in chat_stream


def test_enforce_drop_returns_cleaned_text_not_raw_markup(monkeypatch):
    """When every parsed call was dropped for invalid arguments the caller gets the
    cleaned text (no native markup) plus the diagnostic — the markup is not an answer."""
    monkeypatch.setenv("VMLX_TOOL_ARGS_SCHEMA_VALIDATION", "enforce")
    monkeypatch.setattr(_server, "_tool_call_parser_disabled_explicitly", False, raising=False)
    _begin_tool_call_drop_capture()
    text, calls = _parse_tool_calls_with_parser(
        _set_mode_call({"mode": "nope", "level": 1, "label": "ok", "meta": {"unit": "kg"}}),
        _request_with_strict_tool(),
    )
    assert not calls
    assert "<tool_call>" not in text and "set_mode" not in text
    assert _take_tool_call_drop_diagnostics()


def test_unavailable_name_drop_keeps_the_text_as_plain_answer(monkeypatch):
    monkeypatch.delenv("VMLX_TOOL_ARGS_SCHEMA_VALIDATION", raising=False)
    monkeypatch.setattr(_server, "_tool_call_parser_disabled_explicitly", False, raising=False)
    _begin_tool_call_drop_capture()
    raw = "<tool_call>" + json.dumps({"name": "not_a_tool", "arguments": {"x": 1}}) + "</tool_call>"
    text, calls = _parse_tool_calls_with_parser(raw, _request_with_strict_tool())
    assert not calls
    assert text == raw
    assert any("not_a_tool" in d for d in _take_tool_call_drop_diagnostics())


def test_chat_stream_tool_call_path_delivers_diagnostics_before_done():
    src = inspect.getsource(_server.stream_chat_completion)
    tc_path = src.index("Skip normal end-of-stream handling")
    assert src.rfind("_tc_diagnostics = _take_tool_call_drop_diagnostics()", 0, tc_path) != -1
    assert src.index("_tc_warning_chunk = ChatCompletionChunk(") < tc_path


def test_shared_filter_covers_both_parser_branches_and_every_dialect():
    """Point 6 of the release review: the presence + schema filter is applied on
    the named-parser branch AND the generic branch of the parser wrapper, and the
    Anthropic and Ollama dialects reach it through the chat lane (they do not
    parse tool calls on their own)."""
    wrapper = inspect.getsource(_server._parse_tool_calls_with_parser)
    assert wrapper.count("_filter_to_request_tools(") >= 3  # definition + parser branch + generic branch
    anthropic = inspect.getsource(_server.create_anthropic_message)
    assert "stream_chat_completion(" in anthropic and "dispatch_omni_chat_completion(" in anthropic
    ollama = inspect.getsource(_server.ollama_chat)
    assert "create_chat_completion(" in ollama
    for name, src in (("anthropic", anthropic), ("ollama", ollama)):
        assert "parse_tool_calls(" not in src and "_filter_to_request_tools" not in src, name
    responses = inspect.getsource(_server.create_response)
    assert "_parse_tool_calls_with_parser(" in responses
