"""Terminal classification must not change native generation or truncation."""
import pytest
from vmlx_engine.reasoning.qwen3_parser import Qwen3ReasoningParser
from vmlx_engine.reasoning.deepseek_r1_parser import DeepSeekR1ReasoningParser


@pytest.mark.parametrize('parser_type', [Qwen3ReasoningParser, DeepSeekR1ReasoningParser])
@pytest.mark.parametrize('finish', ['stop', 'length', 'cancelled', 'error', None, 'tool_calls'])
def test_terminal_cause(parser_type, finish):
    parser = parser_type()
    parser.reset_state(think_in_prompt=True)
    text = '\nORCHID\n'
    assert parser.native_eos_answer(text, finish_reason=finish) == (text if finish == 'stop' else None)
    # Incremental extraction still owns the reasoning rail until termination.
    assert parser.extract_reasoning(text)[1] is None


@pytest.mark.parametrize('text', ['', '  ', '<think>planning', 'planning</think>answer', '<thinking>planning'])
def test_explicit_markers_and_empty_are_not_reclassified(text):
    parser = Qwen3ReasoningParser()
    parser.reset_state(think_in_prompt=True)
    assert parser.native_eos_answer(text, finish_reason='stop') is None


@pytest.mark.parametrize('kwargs', [{'visible_content': 'answer'}, {'has_tool_calls': True}, {'has_user_stop': True}])
def test_other_output_or_stop_contract_wins(kwargs):
    parser = Qwen3ReasoningParser()
    parser.reset_state(think_in_prompt=True)
    assert parser.native_eos_answer('ORCHID', finish_reason='stop', **kwargs) is None


def test_no_verified_template_prefix():
    parser = Qwen3ReasoningParser()
    assert parser.native_eos_answer('ORCHID', finish_reason='stop') is None
    parser.reset_state(think_in_prompt=False)
    assert parser.native_eos_answer('ORCHID', finish_reason='stop') is None


@pytest.mark.asyncio
@pytest.mark.parametrize('surface', ['chat', 'responses'])
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('finish,text,expected', [('stop', 'ORCHID', 'ORCHID'), ('length', 'ORCHID', ''), ('stop', 'plan</think>ORCHID', 'ORCHID')])
async def test_stream_terminal_classification(monkeypatch, surface, stream, finish, text, expected):
    import json
    from types import SimpleNamespace
    from vmlx_engine import server
    from vmlx_engine.engine.base import GenerationOutput

    class Engine:
        tokenizer = SimpleNamespace(has_thinking=True)
        is_mllm = False
        preserve_native_tool_format = True
        calls = 0
        async def chat(self, **kwargs):
            self.calls += 1
            return GenerationOutput(text=text, raw_text=text, tokens=[], prompt_tokens=10,
                                    completion_tokens=4, finished=True, finish_reason=finish)
        async def stream_chat(self, **kwargs):
            self.calls += 1
            for i in range(0, len(text), 2):
                end = min(i + 2, len(text))
                yield GenerationOutput(text=text[:end], new_text=text[i:end], tokens=[], prompt_tokens=10,
                                       completion_tokens=4, finished=end == len(text),
                                       finish_reason=finish if end == len(text) else None)
    engine = Engine()
    server._begin_tool_call_drop_capture()
    monkeypatch.setattr(server, '_engine', engine)
    monkeypatch.setattr(server, '_model_name', 'MiniCPM-V-4.7-35B')
    monkeypatch.setattr(server, '_model_path', None)
    monkeypatch.setattr(server, '_reasoning_parser', Qwen3ReasoningParser())
    monkeypatch.setattr(server, '_tool_call_parser', None)
    monkeypatch.setattr(server, '_engine_prompt_starts_in_reasoning', lambda *a, **k: True)
    monkeypatch.setattr(server, '_template_completes_thinking', lambda *a, **k: False)
    messages = [{'role': 'user', 'content': 'Reply ORCHID'}]
    if surface == 'chat':
        request = server.ChatCompletionRequest(model='MiniCPM-V-4.7-35B', messages=messages, stream=True, enable_thinking=True)
        iterator = server.stream_chat_completion(engine, messages, request, fastapi_request=None, max_tokens=128)
    else:
        request = server.ResponsesRequest(model='MiniCPM-V-4.7-35B', input='Reply ORCHID', stream=True, enable_thinking=True)
        iterator = server.stream_responses_api(engine, messages, request, fastapi_request=None, max_tokens=128)
    if not stream:
        request.stream = False
        if surface == 'chat':
            response = await server.create_chat_completion(request, fastapi_request=None)
            content = response.choices[0].message.content or ''
        else:
            response = await server.create_response(request, fastapi_request=None)
            content = ''.join(part.text for item in response.output if item.type == 'message' for part in item.content)
        assert content == expected
        assert engine.calls == 1
        assert ('reasoning_closed_by_eos' in response.model_dump_json(exclude_none=True)) == (text == 'ORCHID' and finish == 'stop')
        return
    if surface == 'chat':
        iterator = server._terminal_finish_guard(iterator)
    chunks = [chunk async for chunk in iterator]
    events = [json.loads(line[6:]) for chunk in chunks for line in chunk.splitlines() if line.startswith('data: ') and line != 'data: [DONE]']
    if surface == 'chat':
        content = ''.join(c.get('delta', {}).get('content') or '' for e in events for c in e.get('choices', []))
        terminal_indices = [i for i, e in enumerate(events) if any(c.get('finish_reason') for c in e.get('choices', []))]
        assert len(terminal_indices) == 1
        assert all(i <= terminal_indices[0] for i, e in enumerate(events) if any(c.get('delta', {}).get('content') for c in e.get('choices', [])))
    else:
        content = ''.join(e.get('delta', '') for e in events if e.get('type') == 'response.output_text.delta')
        terminal = [e for e in events if e.get('type') in {'response.completed', 'response.incomplete'}]
        assert len(terminal) == 1
        assert terminal[0]['type'] == ('response.incomplete' if finish == 'length' else 'response.completed')
    assert content == expected
    assert engine.calls == 1
    assert ('reasoning_closed_by_eos' in json.dumps(events)) == (text == 'ORCHID' and finish == 'stop')


def test_classification_history_does_not_duplicate_tokens():
    from vmlx_engine import server
    items = [{'type': 'reasoning', 'summary': [{'type': 'summary_text', 'text': 'ORCHID'}]},
             {'type': 'message', 'content': [{'type': 'output_text', 'text': 'ORCHID'}]}]
    assert server._responses_output_to_assistant_messages(items, reasoning_closed_by_eos=True) == [
        {'role': 'assistant', 'content': 'ORCHID'}]
    # Without terminal classification preserve the ordinary native history rail.
    assert server._responses_output_to_assistant_messages(items)[0]['reasoning_content'] == 'ORCHID'


def test_suppressed_reasoning_is_not_promoted():
    from types import SimpleNamespace
    from vmlx_engine import server
    parser = Qwen3ReasoningParser()
    parser.reset_state(think_in_prompt=True)
    server._begin_tool_call_drop_capture()
    assert server._native_eos_visible_answer(
        parser, SimpleNamespace(enable_thinking=False), SimpleNamespace(finish_reason='stop'),
        'ORCHID', None, None
    ) is None


@pytest.mark.parametrize('ids', [[42, 10, 11, 99], [9, 42, 99]])
def test_token_delimiters_hidden_by_detokenizer_are_not_marker_free(ids):
    class Tokenizer:
        def encode(self, text, **kwargs):
            return {'<think>': [9], '</think>': [10, 11], '<thinking>': [12], '</thinking>': [13]}[text]
    parser = Qwen3ReasoningParser()
    parser.reset_state(think_in_prompt=True)
    assert parser.native_eos_answer(
        'planning', finish_reason='stop', generated_token_ids=ids, tokenizer=Tokenizer()
    ) is None
