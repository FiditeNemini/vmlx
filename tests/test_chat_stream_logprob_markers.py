"""Token metadata must survive reasoning parser display suppression."""
import json
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize('parser_name', ['think_xml', None])
async def test_hidden_markers_keep_every_logprob_once(monkeypatch, parser_name):
    from vmlx_engine import server
    from vmlx_engine.engine.base import GenerationOutput
    from vmlx_engine.api.models import ChatCompletionRequest
    from vmlx_engine.reasoning import get_parser

    pieces = ['<think>', '</think>', '\n\n', 'Answer']

    class Engine:
        tokenizer = SimpleNamespace(has_thinking=False, decode=lambda ids: pieces[ids[0]])

        async def stream_chat(self, **kwargs):
            records = []
            text = ''
            for i, piece in enumerate(pieces):
                text += piece
                records.append({'token_id': i, 'logprob': -0.25, 'top_logprobs': [(i, -0.25)]})
                yield GenerationOutput(text=text, raw_text=text, new_text=piece,
                    prompt_tokens=3, completion_tokens=i + 1,
                    logprobs=list(records), finished=False)
            yield GenerationOutput(text=text, raw_text=text, new_text='',
                prompt_tokens=3, completion_tokens=len(pieces),
                logprobs=list(records), finished=True, finish_reason='stop')

    monkeypatch.setattr(server, '_default_timeout', 5.0)
    monkeypatch.setattr(server, '_model_name', 'metadata-test')
    monkeypatch.setattr(server, '_model_path', None)
    monkeypatch.setattr(server, '_reasoning_parser', get_parser(parser_name)() if parser_name else None)
    monkeypatch.setattr(server, '_tool_call_parser', None)
    messages = [{'role': 'user', 'content': 'Answer briefly.'}]
    request = ChatCompletionRequest(model='metadata-test', messages=messages,
        stream=True, logprobs=True, top_logprobs=1)
    payloads = []
    async for frame in server.stream_chat_completion(Engine(), messages, request):
        for line in frame.splitlines():
            if line.startswith('data: ') and line != 'data: [DONE]':
                payloads.append(json.loads(line[6:]))
    choices = [choice for payload in payloads for choice in payload.get('choices', [])]
    records = [entry for choice in choices for entry in (choice.get('logprobs') or {}).get('content', [])]
    assert [entry['token'] for entry in records] == pieces
    assert all(entry['logprob'] == -0.25 for entry in records)
    assert sum(choice.get('finish_reason') == 'stop' for choice in choices) == 1
    if parser_name:
        visible = ''.join(choice.get('delta', {}).get('content') or '' for choice in choices)
        assert visible.strip() == 'Answer'
        assert '<think>' not in visible
