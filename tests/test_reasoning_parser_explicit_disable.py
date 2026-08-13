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
