"""The contentless-assistant drop must exist on every dialect's request prep.

cf627b3cd fixed chat + /v1/responses; the Anthropic and streaming-Ollama
preps build their own message lists and were instances 3 and 4 of the
fix-one-of-N class. All four now share _drop_contentless_assistant_turns.
"""

from vmlx_engine.server import _drop_contentless_assistant_turns


def test_drop_removes_assistant_turns_with_no_content_and_no_tool_calls():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": ""},
        {"role": "assistant"},
        {"role": "user", "content": "again"},
    ]

    assert _drop_contentless_assistant_turns(messages) == [
        {"role": "user", "content": "hi"},
        {"role": "user", "content": "again"},
    ]


def test_drop_keeps_tool_call_only_and_normal_turns():
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "assistant", "content": "fine"},
        {"role": "tool", "content": ""},
    ]

    assert _drop_contentless_assistant_turns(messages) == messages


def test_every_dialect_request_prep_calls_the_shared_drop():
    import inspect
    import vmlx_engine.server as server

    src = inspect.getsource(server)
    # def + chat + responses + anthropic + ollama-stream
    assert src.count("_drop_contentless_assistant_turns(") >= 5


def test_server_default_thinking_off_degrades_to_auto_on_no_instruct_families(monkeypatch):
    """A server-resolved enable_thinking=false default must not make the
    engine 400 its own configuration on families without an instruct mode
    (lfm2/step37 rejected every text request while a global default was set);
    it degrades to Auto. A CALLER-requested false still 400s."""
    import vmlx_engine.server as server

    class _MC:
        family_name = "lfm2"
        model_type = "lfm2"
        supports_instruct_mode = False
        supports_thinking = True
        reasoning_parser = "qwen3"
        think_in_template = False
        architecture_hints = {}

    class _Registry:
        def lookup(self, key):
            return _MC()

    monkeypatch.setattr(
        "vmlx_engine.model_config_registry.get_model_config_registry",
        lambda: _Registry(),
    )
    monkeypatch.setattr(server, "_default_enable_thinking", False)

    resolved = server._resolve_enable_thinking(
        None, {}, False, "lfm2-test", engine=None
    )
    assert resolved is not False, "server default false must degrade, not force off"

    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        server._resolve_enable_thinking(False, {}, False, "lfm2-test", engine=None)
