"""Preserve parallel tool identities through MiMo's positional native template.

Fixture: XiaomiMiMo/MiMo-V2.6-Flash-RL, revision5711b268169967567844e1e560e8a3966da959b1.
"""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from jinja2 import sandbox
from vmlx_engine.models.mimo_v26_contract import canonicalize_mimo_v26_tool_results as canonicalize

FIXTURE = Path(__file__).parent / 'fixtures/mimo_v26_chat_template.jinja'


class NativeTokenizer:
    def __init__(self):
        template = FIXTURE.read_text()
        assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == '853650bee57bf95020373e4c928bd5a4b41b9915adf964a77711d2b49a291887'
        env = sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.filters['tojson'] = lambda value, **kw: json.dumps(value, **kw)
        self.template = env.from_string(template)
        self.chat_template = template

    def apply_chat_template(self, messages, **kwargs):
        return self.template.render(messages=messages, add_generation_prompt=True, **kwargs)


@pytest.fixture
def history():
    return [{'role':'user','content':'Look up P17 and P29.'},
            {'role':'assistant','content':'','reasoning_content':'I will check both records.', 'tool_calls':[
                {'id':'a','type':'function','function':{'name':'lookup','arguments':{'id':'P17'}}},
                {'id':'b','type':'function','function':{'name':'lookup','arguments':{'id':'P29'}}}]},
            {'role':'tool','tool_call_id':'a','content':'READY'},
            {'role':'tool','tool_call_id':'b','content':'HELD'}]


@pytest.mark.parametrize('thinking', [False, True])
def test_equivalent_result_permutations_have_identical_native_prompt(history, thinking):
    tokenizer = NativeTokenizer()
    reversed_results = history[:2] + history[2:][::-1]
    original = copy.deepcopy(reversed_results)
    expected = tokenizer.apply_chat_template(history, enable_thinking=thinking)
    actual = tokenizer.apply_chat_template(canonicalize(reversed_results), enable_thinking=thinking)
    assert actual == expected
    assert '<think>I will check both records.</think>' in actual
    assert canonicalize(canonicalize(reversed_results)) == canonicalize(reversed_results)
    assert reversed_results == original


def test_different_id_associations_no_longer_collide(history):
    tokenizer = NativeTokenizer()
    swapped = copy.deepcopy(history)
    swapped[-2]['tool_call_id'] = 'b';swapped[-1]['tool_call_id'] = 'a'
    assert tokenizer.apply_chat_template(swapped) == tokenizer.apply_chat_template(history)
    assert tokenizer.apply_chat_template(canonicalize(swapped)) != tokenizer.apply_chat_template(canonicalize(history))


@pytest.mark.parametrize('invalid', ['partial','unknown','duplicate_result','duplicate_call','missing_id','empty_id','broken_anchor'])
def test_ambiguous_id_batches_fail_closed(history, invalid):
    if invalid == 'partial':history.pop()
    elif invalid == 'unknown':history[-1]['tool_call_id'] = 'missing'
    elif invalid == 'duplicate_result':history[-1]['tool_call_id'] = 'a'
    elif invalid == 'duplicate_call':history[1]['tool_calls'][1]['id'] = 'a'
    elif invalid == 'missing_id':history[-1].pop('tool_call_id')
    elif invalid == 'empty_id':history[-1]['tool_call_id'] = ''
    elif invalid == 'broken_anchor':history.insert(2, {'role':'user','content':'New turn'})
    with pytest.raises(ValueError, match='complete, unique tool-result batch'):
        canonicalize(history)


def test_native_idless_history_keeps_its_positional_order(history):
    for result in history[2:]:result.pop('tool_call_id')
    assert canonicalize(history) == history


def test_independent_tool_rounds_do_not_mix(history):
    second = copy.deepcopy(history[1:])
    second[-2]['content'] = 'ROUND2_READY'
    second[-1]['content'] = 'ROUND2_HELD'
    reversed_rounds = history[:2]+history[2:][::-1]+second[:1]+second[1:][::-1]
    assert canonicalize(reversed_rounds) == history + second


def test_server_maps_invalid_mimo_history_to_422_only_for_its_bundle(history, tmp_path, monkeypatch):
    from fastapi import HTTPException
    from vmlx_engine import server
    (tmp_path/'config.json').write_text('{"model_type":"mimo_v2"}')
    (tmp_path/'jang_config.json').write_text('{"weight_format":"mixed_affine_mxfp4"}')
    monkeypatch.setattr(server, '_model_path', str(tmp_path))
    with pytest.raises(HTTPException) as error:
        server._canonicalize_mimo_v26_tool_history(history[:-1])
    assert error.value.status_code == 422
    assert server._canonicalize_mimo_v26_tool_history(history[:2]+history[2:][::-1]) == history
    (tmp_path/'config.json').write_text('{"model_type":"another_family"}')
    partial = history[:-1]
    assert server._canonicalize_mimo_v26_tool_history(partial) is partial


def test_direct_processor_keeps_the_vendor_prompt(history):
    from vmlx_engine.models.mimo_v26 import MiMoV26Processor
    tokenizer = NativeTokenizer()
    processor = MiMoV26Processor(tokenizer, SimpleNamespace())
    assert processor.apply_chat_template(history[:2]+history[2:][::-1]) == tokenizer.apply_chat_template(history)


def test_mllm_rejects_before_generic_last_user_fallback(history, monkeypatch):
    from vmlx_engine.models.mllm import MLXMultimodalLM
    import mlx_vlm.prompt_utils as prompt_utils
    def unexpected(*args, **kwargs):pytest.fail('Invalid history reached template fallback')
    monkeypatch.setattr(prompt_utils, 'get_chat_template', unexpected)
    model = object.__new__(MLXMultimodalLM)
    model.config = {'model_type':'mimo_v2'}
    model.processor = SimpleNamespace(_mimo_v26_runtime=True)
    with pytest.raises(ValueError, match='complete, unique tool-result batch'):
        model._apply_chat_template(history[:-1])


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.asyncio
async def test_responses_rejects_unknown_ids_before_orphan_fallback(tmp_path, monkeypatch, stream):
    from fastapi import HTTPException, Request
    from vmlx_engine import server
    from vmlx_engine.api.models import ResponsesRequest
    (tmp_path/'config.json').write_text('{"model_type":"mimo_v2"}')
    (tmp_path/'jang_config.json').write_text('{"weight_format":"mixed_affine_mxfp4"}')
    monkeypatch.setattr(server, '_model_path', str(tmp_path))
    monkeypatch.setattr(server, '_resolve_model_name', lambda: 'test')
    monkeypatch.setattr(server, 'get_engine', lambda: SimpleNamespace(is_mllm=True))
    def unexpected(messages):pytest.fail('Malformed IDs reached orphan coercion')
    monkeypatch.setattr(server, '_coerce_orphan_tool_messages_for_template', unexpected)
    request = ResponsesRequest(model='test', stream=stream, input=[
        {'type':'function_call','call_id':'a','name':'lookup','arguments':'{}'},
        {'type':'function_call_output','call_id':'wrong','output':'value'}])
    with pytest.raises(HTTPException) as error:
        await server.create_response(request, Request({'type':'http','headers':[]}))
    assert error.value.status_code == 422


@pytest.mark.asyncio
async def test_responses_restored_history_and_instructions_keep_associations(history, tmp_path, monkeypatch):
    from fastapi import Request
    from vmlx_engine import server
    from vmlx_engine.api.models import ResponsesRequest
    (tmp_path/'config.json').write_text('{"model_type":"mimo_v2"}')
    (tmp_path/'jang_config.json').write_text('{"weight_format":"mixed_affine_mxfp4"}')
    monkeypatch.setattr(server, '_model_path', str(tmp_path))
    monkeypatch.setattr(server, '_resolve_model_name', lambda: 'test')
    monkeypatch.setattr(server, 'get_engine', lambda: SimpleNamespace(is_mllm=True))
    monkeypatch.setattr(server, '_responses_get_history', lambda response_id: history[:2])
    class Prepared(Exception):pass
    def capture(messages):
        assert [m['tool_call_id'] for m in messages if m['role']=='tool'] == ['a','b']
        assert next(m for m in messages if m['role']=='assistant')['reasoning_content'] == history[1]['reasoning_content']
        assert any(m['role']=='system' and m['content']=='Use prior results.' for m in messages)
        raise Prepared
    monkeypatch.setattr(server, '_coerce_orphan_tool_messages_for_template', capture)
    request = ResponsesRequest(model='test', previous_response_id='prior', instructions='Use prior results.', input=[
        {'type':'function_call_output','call_id':'b','output':'HELD'},
        {'type':'function_call_output','call_id':'a','output':'READY'}])
    with pytest.raises(Prepared):
        await server.create_response(request, Request({'type':'http','headers':[]}))
