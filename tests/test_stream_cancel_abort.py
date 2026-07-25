"""Streaming client cancellation must abort scheduler work."""

import inspect

import vmlx_engine.server as server_mod


def test_chat_stream_cancelled_error_aborts_scheduler_request():
    source = inspect.getsource(server_mod.stream_chat_completion)

    assert "except asyncio.CancelledError:" in source
    assert 'logger.info("Chat Completions stream cancelled, aborting %s", response_id)' in source
    assert "await engine.abort_request(response_id)" in source


def test_chat_visible_answer_cancel_aborts_both_request_ids():
    source = inspect.getsource(server_mod.stream_chat_completion)

    assert "Chat Completions visible answer pass cancelled" in source
    assert 'await engine.abort_request(f"{response_id}:visible-answer")' in source
    assert "await engine.abort_request(response_id)" in source


def test_responses_stream_cancelled_error_aborts_scheduler_request():
    source = inspect.getsource(server_mod.stream_responses_api)

    assert "except asyncio.CancelledError:" in source
    assert 'logger.info("Responses stream cancelled, aborting %s", response_id)' in source
    assert "await engine.abort_request(response_id)" in source


def test_responses_visible_answer_cancel_aborts_both_request_ids():
    source = inspect.getsource(server_mod.stream_responses_api)

    assert "Responses visible answer pass cancelled" in source
    assert 'await engine.abort_request(f"{response_id}:visible-answer")' in source
    assert "await engine.abort_request(response_id)" in source
