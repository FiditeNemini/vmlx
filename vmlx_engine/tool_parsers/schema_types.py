"""Conservative type hints for native XML scalars, without changing schemas.

JSON-native tool calls keep their generated types. XML string bodies need a
type hint before json.loads, or numeric filenames and JSON source become data.
References resolve only against resources embedded in the supplied schema.
"""
from typing import Any

from referencing import Registry, Resource
from referencing.exceptions import Unresolvable
from referencing.jsonschema import DRAFT202012, specification_with

_ALL_TYPES = frozenset({"string", "boolean", "null", "integer", "number", "array", "object"})
_LEGACY_REF = {"draft-03", "draft-04", "draft-06", "draft-07"}


def xml_parameter_type_hints(schema: dict[str, Any]) -> dict[str, Any]:
    """Return decoding hints, never a replacement validation/prompt schema.

    Unknown references, recursion and exhausted traversal budgets contribute no
    type restriction. The default Registry has no remote retrieval callback.
    allOf intersects restrictions; anyOf/oneOf take their conservative union.
    Constraints such as patterns still belong to the full argument validator.
    """
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return {}
    try:
        spec = specification_with(schema.get("$schema", ""), default=DRAFT202012)
        resource = Resource.from_contents(schema, default_specification=spec)
        resolver = Registry().resolver_with_root(resource)
    except (TypeError, ValueError):
        return properties

    def types(node, scope, dialect, seen, budget, enter_scope=True):
        if node is False:
            return frozenset()
        if not isinstance(node, dict) or id(node) in seen or budget[0] <= 0:
            return _ALL_TYPES
        budget[0] -= 1
        seen = seen | {id(node)}
        dialect = specification_with(node.get("$schema", ""), default=dialect)
        if enter_scope:
            scope = scope.in_subresource(Resource.from_contents(node, default_specification=dialect))
        allowed = _ALL_TYPES
        ref = node.get("$ref")
        if isinstance(ref, str):
            try:
                resolved = scope.lookup(ref)
                allowed = types(resolved.contents, resolved.resolver, dialect, seen, budget, False)
            except Unresolvable:
                pass
            # Older drafts ignore siblings of $ref; modern drafts intersect.
            if dialect.name in _LEGACY_REF:
                return allowed
        declared = node.get("type")
        if isinstance(declared, str):
            allowed = allowed & {declared}
        elif isinstance(declared, list) and all(isinstance(t, str) for t in declared):
            allowed = allowed & set(declared)
        for keyword in ("allOf", "anyOf", "oneOf"):
            branches = node.get(keyword)
            if not isinstance(branches, list) or not branches:
                continue
            combined = _ALL_TYPES if keyword == "allOf" else frozenset()
            for branch in branches:
                branch_types = types(branch, scope, dialect, seen, budget)
                combined = combined & branch_types if keyword == "allOf" else combined | branch_types
            allowed = allowed & combined
        return allowed

    hints = {}
    for name, prop in properties.items():
        try:
            allowed = types(prop, resolver, spec, frozenset(), [128])
            # Preserve the existing non-JSON-Schema nullable extension.
            hints[name] = {"type": sorted(allowed)}
            if isinstance(prop, dict) and prop.get("nullable") is True:
                hints[name]["nullable"] = True
        except (TypeError, ValueError, RecursionError):
            hints[name] = {}  # Malformed/unresolvable hints never guess a type.
    return hints
