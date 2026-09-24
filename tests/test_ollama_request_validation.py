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


@pytest.mark.parametrize('is_mllm', [False, True])
@pytest.mark.parametrize('rejected', [False, True])
def test_raw_nonstream_completion_returned_status(monkeypatch, is_mllm, rejected):
    """Keep the real completion handler's exception-to-response conversion."""
    from vmlx_engine import server
    from vmlx_engine.errors import PromptTooLongError

    generation = AsyncMock(
        side_effect=PromptTooLongError(33, 32) if rejected else None,
        return_value=SimpleNamespace(text='Raw result', finish_reason='length',
                                     prompt_tokens=4, completion_tokens=2),
    )
    engine = SimpleNamespace(is_mllm=is_mllm, generate=generation,
                             chat=generation, stop=AsyncMock())
    monkeypatch.setattr(server, '_engine', engine)
    monkeypatch.setattr(server, '_api_key', None)
    monkeypatch.setattr(server, '_standby_state', None)
    monkeypatch.setattr(server, '_max_prompt_tokens', 0)
    monkeypatch.setattr(server, '_model_path', None)
    monkeypatch.setattr(server, '_model_name', 'test')
    with TestClient(server.app, raise_server_exceptions=False) as client:
        result = client.post('/api/generate', json={
            'model': 'test', 'prompt': 'hi', 'stream': False, 'raw': True,
            'options': {'num_ctx': 32, 'num_predict': 2, 'temperature': 0},
        })

    generation.assert_awaited_once()
    assert generation.await_args.kwargs['max_prompt_tokens'] == 32
    assert generation.await_args.kwargs['max_tokens'] == 2
    if rejected:
        assert result.status_code == 413, result.text
        assert set(result.json()) == {'error'}
        assert result.json()['error'].startswith('prompt_too_long: ')
        assert '~33 tokens' in result.json()['error']
        assert '~32 tokens' in result.json()['error']
    else:
        assert result.status_code == 200, result.text
        row = result.json()
        assert row['response'] == 'Raw result'
        assert row['done'] is True
        assert row['done_reason'] == 'length'
        assert row['prompt_eval_count'] == 4
        assert row['eval_count'] == 2
        assert row['total_duration'] > 0
        assert row['load_duration'] == 0
        assert 'prompt_eval_duration' not in row
        assert 'eval_duration' not in row
        assert 'error' not in row


@pytest.mark.asyncio
@pytest.mark.parametrize('usage', [
    {'prompt_tokens': 4, 'completion_tokens': 2},
    {'prompt_tokens': 0, 'completion_tokens': 0},
    {},
])
async def test_raw_nonstream_measures_total_without_inventing_split_or_counts(monkeypatch, usage):
    from vmlx_engine import server

    generation = AsyncMock(return_value={
        'choices': [{'text': 'Raw result', 'finish_reason': 'stop'}],
        'usage': usage,
    })
    monkeypatch.setattr(server, 'create_completion', generation)
    monkeypatch.setattr(server, '_engine', SimpleNamespace(is_mllm=False))
    monkeypatch.setattr(server, '_max_prompt_tokens', 0)
    clock = Mock(side_effect=[100, 160])
    monkeypatch.setattr(server.time, 'perf_counter_ns', clock)
    request = SimpleNamespace(
        json=AsyncMock(return_value={'model': 'test', 'prompt': 'hi', 'raw': True, 'stream': False}),
        state=SimpleNamespace(vmlx_wake_ns=30),
    )
    row = await server.ollama_generate(request)
    assert row['response'] == 'Raw result'
    assert row['done_reason'] == 'stop'
    assert row['total_duration'] == 90
    assert row['load_duration'] == 30
    assert 'eval_duration' not in row
    assert 'prompt_eval_duration' not in row
    for source, target in [('prompt_tokens', 'prompt_eval_count'), ('completion_tokens', 'eval_count')]:
        if source in usage:
            assert row[target] == usage[source]
        else:
            assert target not in row
    generation.assert_awaited_once()
    assert clock.call_count == 2

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


@pytest.mark.parametrize('path,extra,handler', [
    ('/api/chat', {'messages': [{'role': 'user', 'content': 'hi'}]}, 'create_chat_completion'),
    ('/api/generate', {'prompt': 'hi'}, 'create_chat_completion'),
    ('/api/generate', {'prompt': 'hi', 'raw': True}, 'create_completion'),
])
@pytest.mark.parametrize('controls', [
    {'skip_prefix_cache': True}, {'cache_salt': 'fresh-request'},
    {'skip_prefix_cache': False, 'cache_salt': ''},
])
def test_cache_controls_reach_canonical_request(monkeypatch, path, extra, handler, controls):
    from vmlx_engine import server
    monkeypatch.setattr(server, '_engine', SimpleNamespace(is_mllm=False))
    monkeypatch.setattr(server, '_api_key', None)
    monkeypatch.setattr(server, '_standby_state', None)
    generation = AsyncMock(return_value={
        'choices': [{'text': 'ok', 'message': {'role': 'assistant', 'content': 'ok'},
                     'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 1, 'completion_tokens': 1},
    })
    monkeypatch.setattr(server, handler, generation)
    with TestClient(server.app, raise_server_exceptions=False) as client:
        result = client.post(path, json={'model': 'test', 'stream': False, **extra, **controls})
    assert result.status_code == 200, result.text
    generation.assert_awaited_once()
    request = generation.await_args.args[0]
    for field, value in controls.items():
        assert getattr(request, field) == value
    assert server._compute_bypass_prefix_cache(request) is bool(
        controls.get('skip_prefix_cache') or controls.get('cache_salt')
    )
