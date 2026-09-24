"""Legacy unique-suffix recovery must decode values with the recovered schema."""
import copy
import json

import pytest

from vmlx_engine.tool_parsers.qwen_tool_parser import QwenToolParser


def request(flat=False, second_name=None):
    schema = {
        'type': 'object', '$defs': {'text': {'type': 'string'}, 'flag': {'type': 'boolean'}},
        'properties': {
            'content': {'$ref': '#/$defs/text'},
            'label': {'allOf': [{'$ref': '#/$defs/text'}, {'minLength': 1}]},
            'flag': {'$ref': '#/$defs/flag'},
            'nil_text': {'$ref': '#/$defs/text'},
            'false_text': {'$ref': '#/$defs/text'},
        },
        'required': ['content', 'label', 'flag', 'nil_text', 'false_text'],
        'additionalProperties': False,
    }
    tools = []
    for name in ['record_payload', *([second_name] if second_name else [])]:
        fn = {'name': name, 'parameters': schema}
        tools.append({'type': 'function', **fn} if flat else {'type': 'function', 'function': fn})
    return {'tools': tools}


def block(name='_payload', content='{"n":1}', flag='false'):
    values = {'content': content, 'label': '123', 'flag': flag, 'nil_text': 'null', 'false_text': 'false'}
    return ('<tool_call>\n<function=' + name + '>\n'
            + ''.join(f'<parameter={key}>\n{value}\n</parameter>\n' for key, value in values.items())
            + '</function>\n</tool_call>')


@pytest.mark.parametrize('flat', [False, True], ids=['chat', 'responses'])
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('content,flag', [('{"n":1}', 'false'), ('  日本語 "quotes"  ', 'False')])
def test_recovered_suffix_retains_schema_strings(flat, stream, content, flag):
    req = request(flat); original = copy.deepcopy(req)
    parser = QwenToolParser.__new__(QwenToolParser)
    text = block(content=content, flag=flag)
    if stream:
        result = parser.extract_tool_calls_streaming('', text, '</tool_call>', request=req)
        call = result['tool_calls'][0]['function']
    else:
        call = parser.extract_tool_calls(text, req).tool_calls[0]
    assert call['name'] == 'record_payload'
    args = json.loads(call['arguments'])
    assert args == {'content': content, 'label': '123', 'flag': False, 'nil_text': 'null', 'false_text': 'false'}
    assert type(args['flag']) is bool
    assert all(type(args[k]) is str for k in ('content', 'label', 'nil_text', 'false_text'))
    assert req == original


def test_duplicate_suffix_blocks_keep_their_own_original_values():
    parser = QwenToolParser.__new__(QwenToolParser)
    result = parser.extract_tool_calls(block(content='123') + block(content='false'), request())
    assert [json.loads(c['arguments'])['content'] for c in result.tool_calls] == ['123', 'false']


@pytest.mark.parametrize('name,second_name', [('_payload', 'send_payload'), ('unrelated', None), ('oad', None)])
def test_value_fix_does_not_expand_legacy_name_recovery(name, second_name):
    parser = QwenToolParser.__new__(QwenToolParser)
    result = parser.extract_tool_calls(block(name=name), request(second_name=second_name))
    assert all(c['name'] == name for c in result.tool_calls)


def test_invalid_boolean_is_not_invented_during_recovery():
    parser = QwenToolParser.__new__(QwenToolParser)
    result = parser.extract_tool_calls(block(flag='maybe'), request())
    assert json.loads(result.tool_calls[0]['arguments'])['flag'] == 'maybe'


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('required', [False, True])
async def test_recovered_values_survive_chat_http_validation(monkeypatch, stream, required):
    import httpx
    from types import SimpleNamespace
    from vmlx_engine import server
    from vmlx_engine.engine.base import GenerationOutput

    text = block()
    class Engine:
        tokenizer = SimpleNamespace(has_thinking=False)
        is_mllm = True
        preserve_native_tool_format = True

        async def chat(self, **kwargs):
            return GenerationOutput(text=text, raw_text=text, tokens=[], prompt_tokens=10,
                                    completion_tokens=90, finished=True, finish_reason='stop')

        async def stream_chat(self, **kwargs):
            for start in range(0, len(text), 7):
                end = min(start + 7, len(text)); final = end == len(text)
                yield GenerationOutput(text=text[:end], new_text=text[start:end], tokens=[],
                                       prompt_tokens=10, completion_tokens=90 if final else 1,
                                       finished=final, finish_reason='stop' if final else None)

    for name, value in {
        '_engine': Engine(), '_model_name': 'qwen-suffix-schema-test',
        '_served_model_name': 'qwen-suffix-schema-test', '_model_path': None,
        '_reasoning_parser': None, '_tool_call_parser': 'qwen',
        '_tool_call_parser_disabled_explicitly': False, '_api_key': None,
        '_default_enable_thinking': None, '_mcp_manager': None,
    }.items():
        monkeypatch.setattr(server, name, value)
    body = {'model': 'qwen-suffix-schema-test', 'stream': stream, 'max_tokens': 256,
            'enable_thinking': False, 'tools': request()['tools'],
            'tool_choice': 'required' if required else 'auto',
            'messages': [{'role': 'user', 'content': 'Record the supplied literal values.'}]}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
        response = await client.post('/v1/chat/completions', json=body)
    assert response.status_code == 200, response.text
    if stream:
        events = [json.loads(line[6:]) for line in response.text.splitlines()
                  if line.startswith('data: ') and line != 'data: [DONE]']
        assert not any(e.get('error') or e.get('warnings') for e in events)
        by_index = {}
        for event in events:
            for choice in event.get('choices', []):
                for delta in choice.get('delta', {}).get('tool_calls', []):
                    call = by_index.setdefault(delta['index'], {'name': '', 'arguments': ''})
                    for key in call:
                        call[key] += delta.get('function', {}).get(key) or ''
        calls = list(by_index.values())
        assert [c['finish_reason'] for e in events for c in e.get('choices', []) if c.get('finish_reason')] == ['tool_calls']
    else:
        out = response.json(); assert not out.get('warnings')
        choice = out['choices'][0]; assert choice['finish_reason'] == 'tool_calls'
        calls = [call['function'] for call in choice['message']['tool_calls']]
    assert len(calls) == 1 and calls[0]['name'] == 'record_payload'
    assert json.loads(calls[0]['arguments']) == {
        'content': '{"n":1}', 'label': '123', 'flag': False, 'nil_text': 'null', 'false_text': 'false'}
