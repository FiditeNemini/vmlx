# SPDX-License-Identifier: Apache-2.0
"""`/api/show` must not advertise "tools" for a model with no tool parser.

Ollama's `capabilities` list is what GitHub Copilot and Open WebUI gate on. The
endpoint appended "tools" unconditionally, justified in-source as a "permissive
default — most models support tools", directly beneath a line that fetched
`globals().get("_tool_parser")` and never read the result.

That name has never existed anywhere in server.py — the real global is
`_tool_call_parser`. So the lookup was always None and the gate it was clearly
meant to feed never ran. Worth stating plainly, because the obvious "fix" of
wiring `if _tp is not None` would have removed "tools" from EVERY model and
dropped vMLX out of Copilot's picker entirely. The typo is the only reason the
endpoint worked at all.

Advertising tools without a parser is not harmless: the client sends tool
schemas, the model emits its native tool markup, nothing extracts it, and the
raw markup is delivered to the user as the visible answer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _show_source(*, code_only: bool = False) -> str:
    src = (ROOT / "vmlx_engine" / "server.py").read_text(encoding="utf-8")
    start = src.index("async def ollama_show(")
    body = src[start : src.index("\n@app.", start + 1)]
    if not code_only:
        return body
    # The comment in this handler QUOTES the dead lookup in order to explain it,
    # so a "must not appear" assertion has to look at code lines only.
    return "\n".join(
        line for line in body.split("\n") if not line.lstrip().startswith("#")
    )


def test_tools_capability_is_gated_on_a_resolvable_parser():
    body = _show_source()
    assert 'capabilities.append("tools")' in body
    assert "_tool_call_parser_disabled_explicitly" in body, (
        "/api/show ignores an explicit --tool-call-parser none"
    )
    assert "get_tool_parser" in body, (
        "/api/show no longer asks the registry whether this bundle has a parser"
    )
    code = _show_source(code_only=True)
    assert 'globals().get("_tool_parser")' not in code, (
        "the dead misspelled lookup is back"
    )
    assert "permissive default" not in code, (
        "the unconditional-tools justification is back"
    )


def test_the_misspelled_global_still_does_not_exist():
    """Guards the premise, so this test starts failing if someone defines it.

    If a `_tool_parser` global is ever introduced, the reasoning above stops
    holding and this file needs revisiting rather than silently passing.
    """
    from vmlx_engine import server as srv

    assert not hasattr(srv, "_tool_parser"), (
        "server now has a _tool_parser global; /api/show's history assumed it "
        "never existed — re-check the capability gate"
    )
    assert hasattr(srv, "_tool_call_parser"), "the real global was renamed"


@pytest.mark.parametrize(
    "cli_parser, disabled, registry_parser, expect_tools",
    [
        ("qwen", False, None, True),        # explicit CLI parser wins
        (None, False, "gemma4", True),      # registry detects one for the bundle
        (None, False, None, False),         # nothing resolves -> do not advertise
        ("qwen", True, "gemma4", False),    # explicit disable beats both
        (None, False, "no_such_parser", False),  # unresolvable name is the same lie
    ],
)
def test_capability_matches_what_the_request_path_would_do(
    monkeypatch, cli_parser, disabled, registry_parser, expect_tools
):
    from fastapi.testclient import TestClient

    from vmlx_engine import server as srv

    monkeypatch.setattr(srv, "_tool_call_parser", cli_parser, raising=False)
    monkeypatch.setattr(
        srv, "_tool_call_parser_disabled_explicitly", disabled, raising=False
    )

    class _Registry:
        def get_tool_parser(self, _name):
            return registry_parser

    import vmlx_engine.model_config_registry as reg

    monkeypatch.setattr(reg, "get_model_config_registry", lambda: _Registry())

    resp = TestClient(srv.app).post("/api/show", json={"name": "default"})
    assert resp.status_code == 200, resp.text[:200]
    caps = resp.json().get("capabilities") or []
    assert ("tools" in caps) is expect_tools, (
        f"cli={cli_parser!r} disabled={disabled} registry={registry_parser!r} "
        f"-> capabilities={caps}"
    )
    assert "completion" in caps, "the base capability must never be dropped"
