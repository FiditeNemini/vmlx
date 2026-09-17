"""Translated Ollama input validation must remain a client error."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import pytest
from fastapi.testclient import TestClient

@pytest.mark.parametrize('path,extra', [('/api/chat', {'messages': [{'role':'user','content':'hi'}]}),
    ('/api/generate', {'prompt':'hi'}), ('/api/generate', {'prompt':'hi','raw':True})])
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('value', ['not-a-number', 12.5, 'Infinity'])
def test_invalid_num_predict_is_400_before_generation(monkeypatch, path, extra, stream, value):
    from vmlx_engine import server
    monkeypatch.setattr(server, '_engine', SimpleNamespace(is_mllm=False))
    monkeypatch.setattr(server, '_api_key', None)
    monkeypatch.setattr(server, '_standby_state', None)
    generation = Mock(side_effect=AssertionError('invalid request reached inference'))
    monkeypatch.setattr(server, 'create_chat_completion', generation)
    monkeypatch.setattr(server, 'create_completion', generation)
    client = TestClient(server.app, raise_server_exceptions=False)
    result = client.post(path, json={'model':'test','stream':stream,'options':{'num_predict':value},**extra})
    client.close()
    assert result.status_code == 400, result.text
    assert result.json()['code'] == 'invalid_request_error'
    assert 'num_predict' in result.json()['error']
    generation.assert_not_called()


@pytest.mark.parametrize('path,extra', [
    ('/api/chat', {'messages': [{'role': 'user', 'content': 'hi'}]}),
    ('/api/generate', {'prompt': 'hi'}),
])
@pytest.mark.parametrize('kind', ['prompt_limit', 'media_controls'])
def test_nonstream_returned_chat_error_preserves_status(monkeypatch, path, extra, kind):
    """Exercise the real route, not a converter handed an already valid answer."""
    from vmlx_engine import server
    from vmlx_engine.errors import MediaControlsUnmeetableError, PromptTooLongError
    monkeypatch.setattr(server, '_engine', SimpleNamespace(is_mllm=False))
    monkeypatch.setattr(server, '_api_key', None)
    monkeypatch.setattr(server, '_standby_state', None)
    if kind == 'prompt_limit':
        rejection = server._prompt_too_long_response_from_error(
            PromptTooLongError(1025, 1024)
        )
    else:
        rejection = server._media_controls_unmeetable_response_from_error(
            MediaControlsUnmeetableError('Requested size is below the processor floor')
        )
    upstream_error = json.loads(rejection.body)['error']
    generation = AsyncMock(return_value=rejection)
    monkeypatch.setattr(server, 'create_chat_completion', generation)
    with TestClient(server.app, raise_server_exceptions=False) as client:
        result = client.post(path, json={'model': 'test', 'stream': False, **extra})

    assert result.status_code == rejection.status_code, result.text
    assert result.json() == {
        'error': f"{upstream_error['code']}: {upstream_error['message']}"
    }
    generation.assert_awaited_once()


def test_nonstream_generate_success_keeps_content_reasoning_and_usage(monkeypatch):
    from vmlx_engine import server
    from starlette.responses import JSONResponse

    monkeypatch.setattr(server, '_engine', SimpleNamespace(is_mllm=False))
    monkeypatch.setattr(server, '_api_key', None)
    monkeypatch.setattr(server, '_standby_state', None)
    generation = AsyncMock(return_value=JSONResponse(content={
        'choices': [{'message': {'role': 'assistant', 'content': 'Answer',
                                'reasoning_content': 'Reasoning'},
                     'finish_reason': 'length'}],
        'usage': {'prompt_tokens': 12, 'completion_tokens': 7},
    }))
    monkeypatch.setattr(server, 'create_chat_completion', generation)
    with TestClient(server.app, raise_server_exceptions=False) as client:
        result = client.post('/api/generate', json={
            'model': 'test', 'prompt': 'hi', 'stream': False,
        })

    assert result.status_code == 200, result.text
    row = result.json()
    assert row['model'] == 'test'
    assert row['response'] == 'Answer'
    assert row['thinking'] == 'Reasoning'
    assert row['done'] is True
    assert row['done_reason'] == 'length'
    assert row['prompt_eval_count'] == 12
    assert row['eval_count'] == 7
    assert row['total_duration'] > 0
    assert 'error' not in row
    generation.assert_awaited_once()

@pytest.mark.parametrize('path,extra', [('/api/chat', {'messages': [{'role':'user','content':'hi'}]}),
    ('/api/generate', {'prompt':'hi'})])
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('nested', [False, True])
def test_invalid_media_control_is_400_before_generation(monkeypatch, path, extra, stream, nested):
    from vmlx_engine import server
    monkeypatch.setattr(server, '_engine', SimpleNamespace(is_mllm=False))
    monkeypatch.setattr(server, '_api_key', None)
    monkeypatch.setattr(server, '_standby_state', None)
    generation = Mock(side_effect=AssertionError('invalid request reached inference'))
    monkeypatch.setattr(server, 'create_chat_completion', generation)
    controls = {'image_max_pixels': 0, 'media_controls_strict': True}
    body = {'model':'test', 'stream':stream, **extra,
            **({'options':controls} if nested else controls)}
    with TestClient(server.app, raise_server_exceptions=False) as client:
        result = client.post(path, json=body)
    assert result.status_code == 400, result.text
    assert 'image_max_pixels' in result.json()['error']
    generation.assert_not_called()
