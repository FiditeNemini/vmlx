"""H1: MLLM token-zero and decode sampling share one input-space contract."""

import inspect
from types import SimpleNamespace

import mlx.core as mx
import pytest

from vmlx_engine.mllm_batch_generator import (
    MLLMBatchGenerator,
    _sample_mllm_prefill_logits,
)
from vmlx_engine.sampling import make_sampler


def _sample_ids(sampler, logits, seeds=None):
    if seeds is None:
        seeds = range(12)
    ids = []
    for seed in seeds:
        mx.random.seed(seed)
        token, _ = _sample_mllm_prefill_logits(logits, sampler)
        mx.eval(token)
        ids.append(int(token.item()))
    return ids


@pytest.mark.parametrize(
    "top_p,min_p",
    [
        (0.80, 0.0),
        (1.0, 0.20),
    ],
)
def test_top_p_and_min_p_receive_logprobabilities_at_token_zero(top_p, min_p):
    logits = mx.array([[10.0, 9.0, 8.0, 0.0]])
    sampler = make_sampler(temp=0.9, top_p=top_p, top_k=0, min_p=min_p)
    normalized = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    actual = _sample_ids(sampler, logits)
    expected = []
    for seed in range(12):
        mx.random.seed(seed)
        token = sampler(normalized)
        mx.eval(token)
        expected.append(int(token.item()))
    assert actual == expected


@pytest.mark.parametrize("output_tokens", [[], [1, 0, 1]])
def test_repetition_sampler_contract_matches_at_token_zero_and_later(
    output_tokens, monkeypatch
):
    import vmlx_engine.sampling as sampling

    observed = []

    def fake_generic_sampler(**_kwargs):
        def sample(values):
            observed.append(values)
            return mx.argmax(values, axis=-1)

        return sample

    monkeypatch.setattr(sampling, "_mlx_make_sampler", fake_generic_sampler)
    request = SimpleNamespace(
        temperature=0.9,
        top_p=0.9,
        top_k=0,
        min_p=0.05,
        repetition_penalty=1.25,
        enable_thinking=False,
        _original_token_ids=[0, 2],
        input_ids=mx.array([[0, 2]], dtype=mx.int32),
        output_tokens=list(output_tokens),
    )
    generator = SimpleNamespace(_model_type="qwen3_5")
    sampler = MLLMBatchGenerator._make_request_sampler(generator, request)
    logits = mx.array([[4.0, 3.0, 2.0, 1.0]])
    normalized = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    token, _ = _sample_mllm_prefill_logits(logits, sampler)
    mx.eval(token)

    from mlx_lm.sample_utils import make_logits_processors

    expected = normalized
    context = mx.array([0, 2] + list(output_tokens))
    for processor in make_logits_processors(repetition_penalty=1.25):
        expected = processor(context, expected)
    assert len(observed) == 1
    assert bool(mx.allclose(observed[0], expected))


def test_decode_step_uses_the_same_normalizing_helper_as_prefill():
    source = inspect.getsource(MLLMBatchGenerator._step)
    assert source.count("_sample_mllm_prefill_logits(") == 3
    assert "sampled = shared_sampler(logits)" not in source
    assert "req_sampler(logits[i:i+1])" not in source
