"""Strict Responses schema failures must not claim successful completion."""
import json
from types import SimpleNamespace
import pytest
from vmlx_engine import server
from vmlx_engine.engine.base import GenerationOutput

@pytest.mark.asyncio
@pytest.mark.parametrize('text,expected', [('not JSON', 'response.failed'), ('{"total":"wrong type"}', 'response.failed'), ('{"total":3973}', 'response.completed')])
async def test_strict_schema_terminal(monkeypatch, text, expected):
    class Engine:
        tokenizer = SimpleNamespace(has_thinking=False)
        is_mllm = False
        async def stream_chat(self, **kwargs):
            yield GenerationOutput(text=text, new_text=text, tokens=[], prompt_tokens=10, completion_tokens=5, finished=True, finish_reason='stop')
    engine = Engine()
    for name, value in [('_engine',engine), ('_model_name','schema-unit'), ('_served_model_name','schema-unit'), ('_model_path',None), ('_reasoning_parser',None), ('_tool_call_parser',None)]:
        monkeypatch.setattr(server,name,value)
    request = server.ResponsesRequest(model='schema-unit', input='calculate', stream=True, text={'format':{'type':'json_schema','name':'total','strict':True,'schema':{'type':'object','properties':{'total':{'type':'integer'}},'required':['total'],'additionalProperties':False}}})
    events = []
    async for chunk in server.stream_responses_api(engine,[{'role':'user','content':'calculate'}],request,fastapi_request=None):
        for line in chunk.splitlines():
            if line.startswith('data: ') and line != 'data: [DONE]':
                events.append(json.loads(line[6:]))
    terminal, = [e for e in events if e['type'] in ('response.completed','response.failed','response.incomplete')]
    assert terminal['type'] == expected
    if expected == 'response.failed':
        assert terminal['response']['error']['code'] == 'json_validation_failed'
