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
