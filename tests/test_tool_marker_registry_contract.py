"""Native tool-marker registry contract.

Every registered tool parser declares its dialect's stream-visible opening
markers as ``NATIVE_MARKERS``. The server's ``_TOOL_CALL_MARKERS`` visibility
guard is what keeps tool-control payload out of rendered text; a dialect whose
opener is absent from that list leaks raw markup to users (Muse's ``<atem:``
blocks reached the transcript exactly this way — the list was the ONE place
its dialect was missing).

This contract makes that drift a test failure instead of a live-UI leak:
- every registered parser must declare at least one native marker unless it
  is an explicitly listed markerless dialect (plain-JSON parsers), and
- every declared marker must be covered by ``_TOOL_CALL_MARKERS`` via prefix
  match (the server list holds prefixes such as ``<tool_call`` precisely so
  split deltas still trip the guard).
"""

from __future__ import annotations

import importlib
import pkgutil

import vmlx_engine.tool_parsers as tool_parsers_pkg
from vmlx_engine.server import _TOOL_CALL_MARKERS
from vmlx_engine.tool_parsers.abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
)

# Dialects that genuinely have no stream-visible markup of their own:
# - auto / generic: the delegating aggregator (registered under both names)
MARKERLESS_OK = {"auto", "generic"}


def _registered_parsers() -> dict[str, type]:
    for module in pkgutil.iter_modules(tool_parsers_pkg.__path__):
        importlib.import_module(f"{tool_parsers_pkg.__name__}.{module.name}")
    registry = dict(ToolParserManager.tool_parsers)
    assert registry, "tool parser registry is empty — import wiring broke"
    return registry


def test_every_registered_parser_declares_native_markers():
    missing = []
    for name, cls in _registered_parsers().items():
        if name in MARKERLESS_OK:
            continue
        markers = tuple(getattr(cls, "NATIVE_MARKERS", ()) or ())
        if not markers:
            missing.append(name)
    assert not missing, (
        "parsers with no NATIVE_MARKERS declaration (declare the dialect's "
        f"stream-visible openers or list as markerless): {sorted(set(missing))}"
    )


def test_every_native_marker_is_covered_by_server_visibility_guard():
    uncovered = []
    for name, cls in _registered_parsers().items():
        for marker in tuple(getattr(cls, "NATIVE_MARKERS", ()) or ()):
            if not any(marker.startswith(guard) for guard in _TOOL_CALL_MARKERS):
                uncovered.append((name, marker))
    assert not uncovered, (
        "native markers NOT covered by server._TOOL_CALL_MARKERS — this is "
        f"the atem-leak class; add coverage: {uncovered}"
    )


def test_base_class_declares_empty_default():
    assert getattr(ToolParser, "NATIVE_MARKERS", None) == ()
