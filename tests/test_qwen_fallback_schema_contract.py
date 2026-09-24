"""Fallback guidance must retain the schema that the validator will enforce."""
import copy
import json
import pytest
from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools


@pytest.mark.parametrize('flat', [False, True])
@pytest.mark.parametrize('property_schema,label', [
    ({'$ref': '#/$defs/flag'}, 'boolean'),
    ({'allOf': [{'$ref': '#/$defs/flag'}]}, 'boolean'),
    ({'anyOf': [{'type': 'boolean'}, {'type': 'null'}]}, 'boolean or null'),
    ({'$ref': '#/$defs/text'}, 'string'),
    ({'type': 'integer'}, 'integer'),
    ({'$ref': 'https://invalid.example/absent'}, 'any JSON value'),
])
def test_qwen_required_fallback_retains_full_schema_and_correct_type(flat, property_schema, label):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            self.kwargs = kwargs
            return '\n'.join(m.get('content', '') for m in messages)
    parameters = {'type': 'object', '$defs': {'flag': {'type': 'boolean'}, 'text': {'type': 'string'}},
                  'properties': {'flag': property_schema, 'nested': {'type': 'object', 'properties': {'count': {'type': 'integer', 'minimum': 2}}, 'required': ['count']}},
                  'required': ['flag'], 'additionalProperties': False}
    fn = {'name': 'record_payload', 'parameters': parameters}
    tools = [{'type': 'function', **fn} if flat else {'type': 'function', 'function': fn}]
    original = copy.deepcopy(tools)
    prompt = '<|im_start|>system\n# Tools\n<tools>\n' + json.dumps(tools) + '\n</tools>\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n</function>\n</tool_call>\n<|im_end|>\n<|im_start|>assistant\n'
    tokenizer = Tokenizer()
    rendered = check_and_inject_fallback_tools(prompt, [{'role': 'user', 'content': 'Call record_payload with flag false.'}], tools, tokenizer,
        {'tokenize': False, 'add_generation_prompt': True, 'tools': tools, 'tool_choice': 'required'}, tool_parser_id='qwen')
    assert f'- flag ({label}, required)' in rendered
    assert 'Parameter JSON Schema: ' + json.dumps(parameters, ensure_ascii=False) in rendered
    assert tools == original
    assert 'tools' not in tokenizer.kwargs
