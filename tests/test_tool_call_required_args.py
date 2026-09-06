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


def test_absent_keys_and_blank_strings_are_missing_but_falsy_values_are_not():
    args = dict(FULL); del args["tags"]
    assert missing_required_tool_args(SCHEMA, args) == ["tags"]
    assert missing_required_tool_args(SCHEMA, {**FULL, "label": "   "}) == ["label"]
    # False / 0 / [] / {} are values
    assert missing_required_tool_args(SCHEMA, FULL) == []


def test_schemas_without_required_or_malformed_args_never_drop():
    assert missing_required_tool_args({"name": "t", "parameters": {"type": "object"}}, {}) == []
    assert missing_required_tool_args(SCHEMA, "not-a-dict") == [k for k in SCHEMA["parameters"]["required"]]
    assert missing_required_tool_args(None, {}) == []
