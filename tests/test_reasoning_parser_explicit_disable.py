"""`--reasoning-parser none` must survive to the streaming request.

A deliberate opt-out and a failed detection both leave the module global
``_reasoning_parser`` as None. ``_new_request_reasoning_parser`` was written to
recover from the *accidental* case (stale sidecar, launch race) by re-reading
the registry stamp -- which meant it silently resurrected the parser the user
had just turned off, on every streaming request, for both `/v1/chat/completions`
and `/v1/responses`. Non-streaming keyed on the global and honoured the opt-out,
so the two surfaces disagreed.

The tool-call side has carried ``_tool_call_parser_disabled_explicitly`` for
exactly this reason. These tests pin the reasoning counterpart, and pin that the
accidental-absence recovery still works when the user did NOT opt out.
"""

from types import SimpleNamespace

import pytest

from vmlx_engine import server


@pytest.fixture
def dsv4_config():
    """A registry stamp that names a parser, as DSV4's bundle does."""
    return SimpleNamespace(
        reasoning_parser="deepseek_r1",
        family_name="deepseek_v4_v7",
        think_in_template=False,
    )


@pytest.fixture(autouse=True)
def _restore_parser_globals():
    saved = (server._reasoning_parser, server._reasoning_parser_disabled_explicitly)
    yield
    server._reasoning_parser, server._reasoning_parser_disabled_explicitly = saved


def _request_parser(model_config, *, surface="responses", think_in_template=False):
    return server._new_request_reasoning_parser(
        configured_parser=server._reasoning_parser,
        model_config=model_config,
        effective_think_in_template=think_in_template,
        harmony_active=False,
        stream_surface=surface,
    )


@pytest.mark.parametrize("surface", ["responses", "chat"])
def test_explicit_none_is_honoured_on_every_streaming_surface(dsv4_config, surface):
    # The app talks to /v1/responses; /v1/chat/completions is the plain-API
    # path. Both called the fallback unconditionally.
    server._set_active_reasoning_parser(None, None)
    server._reasoning_parser_disabled_explicitly = True

    assert _request_parser(dsv4_config, surface=surface) is None


def test_explicit_none_beats_the_think_in_template_fallback(dsv4_config):
    # The second fallback installs deepseek_r1 whenever the rendered prompt
    # opens inside a <think> rail. An opt-out must outrank that too.
    server._set_active_reasoning_parser(None, None)
    server._reasoning_parser_disabled_explicitly = True

    assert _request_parser(dsv4_config, think_in_template=True) is None


def test_accidental_absence_still_recovers_from_the_registry(dsv4_config):
    # The control: without the opt-out the recovery path must still fire, or
    # this fix would have traded one silent failure for another.
    server._set_active_reasoning_parser(None, None)
    server._reasoning_parser_disabled_explicitly = False

    parser = _request_parser(dsv4_config)
    assert parser is not None
    assert type(parser).__name__ == "DeepSeekR1ReasoningParser"


def test_an_installed_parser_is_unaffected_by_a_stale_disable_flag(dsv4_config):
    # configured_parser wins outright: the flag only gates the fallbacks.
    from vmlx_engine.reasoning import get_parser

    server._set_active_reasoning_parser(get_parser("deepseek_r1")(), "deepseek_r1")
    server._reasoning_parser_disabled_explicitly = True

    parser = _request_parser(dsv4_config)
    assert type(parser).__name__ == "DeepSeekR1ReasoningParser"


def test_flag_defaults_to_false_so_untouched_deployments_keep_recovering():
    # Guards against the flag being introduced as True-by-accident, which would
    # disable reasoning everywhere.
    import importlib

    fresh = importlib.import_module("vmlx_engine.server")
    assert fresh._reasoning_parser_disabled_explicitly in (True, False)
    # The module-level literal, read from source, is the real default.
    import inspect
    import re

    src = inspect.getsource(fresh)
    match = re.search(
        r"^_reasoning_parser_disabled_explicitly:\s*bool\s*=\s*(\w+)",
        src,
        re.MULTILINE,
    )
    assert match, "module-level default declaration not found"
    assert match.group(1) == "False"


class TestToolParserDisableSymmetry:
    """The module entry point is a SECOND LAUNCHER, and it under-recorded.

    `server.main()` cleared `_tool_call_parser` for `--tool-call-parser none`
    but never published `_tool_call_parser_disabled_explicitly`, which is what
    the four per-request sites actually consult. Those sites re-detect a parser
    from the registry whenever a request carries tools -- with or without
    `--enable-auto-tool-choice` -- so the opt-out was inert on this path in the
    same way the reasoning opt-out was inert on the streaming path.

    `_delegate_module_main_to_cli` routes `python -m vmlx_engine.server` to the
    CLI, but `main()` still runs for a malformed `--model` and for anything that
    imports and calls `server.main()` directly, which that code path documents
    as deliberately keeping the old behaviour.
    """

    def test_disable_is_published_even_without_enable_auto_tool_choice(self):
        import inspect

        src = inspect.getsource(server.main)
        # The publish must not be nested under the enable_auto_tool_choice
        # branch, or passing `none` alone leaves the flag False.
        publish = src.index("_tool_call_parser_disabled_explicitly = True")
        branch = src.index("if args.enable_auto_tool_choice:")
        assert publish < branch, (
            "the explicit-disable publish must precede (not sit inside) the "
            "enable_auto_tool_choice branch"
        )

    def test_main_declares_the_flag_global(self):
        import inspect

        src = inspect.getsource(server.main)
        assert "global _tool_call_parser_disabled_explicitly" in src, (
            "without the global declaration the assignment binds a local and "
            "the module flag silently stays False"
        )

    def test_the_flag_actually_suppresses_parsing_when_a_request_carries_tools(self):
        # The behaviour the two source assertions above exist to protect. The
        # registry fallback fires on `request.tools`, so a request carrying
        # tools is the case where an unpublished disable leaks a parser back in.
        saved = server._tool_call_parser_disabled_explicitly
        try:
            request = SimpleNamespace(
                tools=[{"type": "function", "function": {"name": "get_weather"}}],
                model="m",
            )
            text = "<tool_call>{\"name\": \"get_weather\"}</tool_call>"

            server._tool_call_parser_disabled_explicitly = True
            cleaned, calls = server._parse_tool_calls_with_parser(text, request)
            assert calls is None, "an explicit disable must suppress tool parsing"
            assert cleaned == text, "disabled parsing must not rewrite the text"
        finally:
            server._tool_call_parser_disabled_explicitly = saved
