"""CPU-only execution of production sampler factory; no MLX or model imports."""
import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


def logprobs(x):
    x = np.asarray(x)
    maximum = np.max(x, axis=-1, keepdims=True)
    return x - maximum - np.log(np.exp(x - maximum).sum(axis=-1, keepdims=True))


def factory(monkeypatch, temperature=0.7, top_p=0.8, top_k=3, min_p=0.1, bias=None):
    # The fixture distribution intentionally has all three filters so losing
    # any sampler contract changes acceptance probabilities.
    def distribution(lp):
        p = np.exp(logprobs(lp))
        order = np.argsort(-p, axis=-1)
        sorted_p = np.take_along_axis(p, order, axis=-1)
        keep = np.cumsum(sorted_p, axis=-1) - sorted_p < top_p
        mask = np.zeros_like(keep)
        np.put_along_axis(mask, order, keep, axis=-1)
        p = np.where(mask & (p >= p.max(axis=-1, keepdims=True) * min_p), p, 0)
        if top_k:
            rank = np.argsort(np.argsort(-p, axis=-1), axis=-1)
            p = np.where(rank < top_k, p, 0)
        with np.errstate(divide='ignore'):
            return logprobs(np.log(p) / (temperature or 1))
    def base(values):
        return np.argmax(values, axis=-1)
    base._vmlx_is_greedy = temperature == 0
    base._vmlx_accepts_logits = temperature == 0
    base._vmlx_acceptance_logprobs = distribution
    base._vmlx_random_uniform = lambda: .5
    base._vmlx_categorical = lambda x: np.argmax(x, axis=-1)
    base.temp, base.top_p, base.top_k, base.min_p = temperature, top_p, top_k, min_p
    sampling = ModuleType('vmlx_engine.sampling')
    sampling.make_sampler = lambda **kwargs: base
    processor_module = ModuleType('vmlx_engine.utils.token_logits_processors')
    def make_processor(**kwargs):
        if not kwargs['logit_bias']:
            return None
        def process(tokens, values):
            out = values.copy()
            for token, delta in kwargs['logit_bias'].items():
                out[..., int(token)] += delta
            return out
        return process
    processor_module.make_openai_token_penalty_processor = make_processor
    acceptance = ModuleType('vmlx_engine.native_mtp_acceptance')
    acceptance.accept_lp_for = lambda sampler, lp: sampler._vmlx_acceptance_logprobs(lp)
    for module in (sampling, processor_module, acceptance):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    path = Path(__file__).parents[1] / 'vmlx_engine/mllm_batch_generator.py'
    tree = ast.parse(path.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_make_request_sampler')
    namespace = dict(__package__='vmlx_engine', mx=SimpleNamespace(array=np.array, eval=lambda *args: None),
                     _native_mtp_logprobs=logprobs, _native_mtp_ensure_uint32=lambda x: np.asarray(x, dtype=np.uint32),
                     _native_mtp_sampler_accepts_logits=lambda s: getattr(s, '_vmlx_accepts_logits', False),
                     _native_mtp_sampler_is_greedy=lambda s: getattr(s, '_vmlx_is_greedy', False))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method], type_ignores=[])), str(path), 'exec'), namespace)
    for name in ('_native_mtp_sample_one', '_native_mtp_sample_rows'):
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])), str(path), 'exec'), namespace)
    request = SimpleNamespace(temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p,
                              logit_bias=bias, repetition_penalty=1, enable_thinking=True,
                              _original_token_ids=[0, 1], output_tokens=[])
    sampler = namespace['_make_request_sampler'](SimpleNamespace(_model_type='qwen4_exp'), request)
    request._test_helpers = namespace
    return sampler, base, request


def test_no_bias_returns_same_base_object(monkeypatch):
    sampler, base, _ = factory(monkeypatch)
    assert sampler is base


def test_greedy_bias_preserves_identity_verification(monkeypatch):
    sampler, _, _ = factory(monkeypatch, temperature=0, bias={3: 20})
    assert sampler._vmlx_is_greedy
    assert sampler._vmlx_accepts_logits
    assert sampler(np.array([[5., 4., 3., 0.]])).tolist() == [3]


@pytest.mark.parametrize('temp,top_p,top_k,min_p', [(0.7, .8, 3, .1), (1.2, 1., 2, .3), (.5, .7, 0, 0.)])
def test_biased_acceptance_matches_base_distribution_for_draft_and_verify(monkeypatch, temp, top_p, top_k, min_p):
    bias = {0: -3., 3: 2.}
    sampler, base, request = factory(monkeypatch, temp, top_p, top_k, min_p, bias)
    assert sampler.temp == temp
    assert sampler.top_p == top_p
    assert sampler.top_k == top_k
    assert sampler.min_p == min_p
    rows = np.array([[4., 3., 2., 1.], [1., 2., 3., 4.]])
    biased = rows + np.array([-3., 0., 0., 2.])
    expected = base._vmlx_acceptance_logprobs(logprobs(biased))
    _, lps, _ = request._test_helpers["_native_mtp_sample_rows"](rows, sampler)
    actual = np.stack([sampler._vmlx_acceptance_logprobs(lp) for lp in lps])
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    # A draft singleton and the same verify row obey the same static transform.
    for i in range(2):
        _, lp = request._test_helpers["_native_mtp_sample_one"](rows[i:i+1], sampler)
        np.testing.assert_allclose(sampler._vmlx_acceptance_logprobs(lp), actual[i], atol=1e-12)


def test_fp16_large_offset_bias_is_applied_before_normalizing(monkeypatch):
    sampler, base, request = factory(monkeypatch, temperature=.7, top_p=.7, top_k=3, min_p=.1, bias={0: .2, 1: .4})
    rows = np.array([[1000, 999, 998, 997]], dtype=np.float16)
    biased = rows.copy()
    biased[..., 0] += .2
    biased[..., 1] += .4
    expected = base._vmlx_acceptance_logprobs(logprobs(biased))
    _, lps, _ = request._test_helpers['_native_mtp_sample_rows'](rows, sampler)
    actual = sampler._vmlx_acceptance_logprobs(lps[0])
    np.testing.assert_array_equal(actual, expected[0])
    _, draft_lp = request._test_helpers['_native_mtp_sample_one'](rows, sampler)
    np.testing.assert_array_equal(lps[0], draft_lp)


def test_biased_wrapper_keeps_request_rng_hooks(monkeypatch):
    sampler, base, _ = factory(monkeypatch, bias={3: 2.})
    assert sampler._vmlx_random_uniform is base._vmlx_random_uniform
    assert sampler._vmlx_categorical is base._vmlx_categorical
