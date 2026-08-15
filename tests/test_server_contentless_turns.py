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
