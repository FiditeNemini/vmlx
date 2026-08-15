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


def test_responses_chain_replay_drops_the_stored_reasoning_only_turn():
    """dialect F2: the store deliberately keeps a reasoning-only assistant turn
    (content "", reasoning_content set) so the chain warning and introspection
    work — but replaying it through previous_response_id handed strict
    templates the exact contentless turn every dialect drops from client
    input, reintroducing the fixed 500. The getter is the replay boundary."""
    import vmlx_engine.server as server

    stored = [
        {"role": "user", "content": "think about it"},
        {"role": "assistant", "content": "", "reasoning_content": "private chain"},
        {"role": "user", "content": "follow-up"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "f", "arguments": "{}"}}
            ],
        },
    ]
    server._responses_store_history("resp_test_f2", stored, reasoning_only=True)
    try:
        replayed = server._responses_get_history("resp_test_f2")
        roles_contents = [
            (m.get("role"), bool(m.get("content")), bool(m.get("tool_calls")))
            for m in replayed
        ]
        assert ("assistant", False, False) not in roles_contents, replayed
        assert any(r == "assistant" and tc for r, _, tc in roles_contents), (
            "tool-call-only assistant turns are valid and must survive replay"
        )
        with server._responses_history_lock:
            in_store = server._responses_history["resp_test_f2"]
        assert any(
            m.get("role") == "assistant" and m.get("reasoning_content")
            and not m.get("content") and not m.get("tool_calls")
            for m in in_store
        ), "the STORE must keep the reasoning-only turn for the chain warning"
    finally:
        with server._responses_history_lock:
            server._responses_history.pop("resp_test_f2", None)
            server._responses_was_reasoning_only.discard("resp_test_f2")
