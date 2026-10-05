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
    token, _ = _sample_mllm_prefill_logits(logits, sampler)
    mx.eval(token)

    from mlx_lm.sample_utils import make_logits_processors

    # Rebuild the original values because mlx-lm's processor mutates its logits
    # array in place inside the sampler wrapper.
    expected = mx.array([[4.0, 3.0, 2.0, 1.0]])
    context = mx.array([0, 2] + list(output_tokens))
    for processor in make_logits_processors(repetition_penalty=1.25):
        expected = processor(context, expected)
    expected = expected - mx.logsumexp(expected, axis=-1, keepdims=True)
    assert len(observed) == 1
    assert bool(mx.allclose(observed[0], expected))


def test_greedy_processor_path_is_exact_argmax_of_processed_logits():
    request = SimpleNamespace(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        repetition_penalty=2.0,
        enable_thinking=False,
        _original_token_ids=[0],
        input_ids=mx.array([[0]], dtype=mx.int32),
        output_tokens=[],
    )
    generator = SimpleNamespace(_model_type="qwen3_5")
    sampler = MLLMBatchGenerator._make_request_sampler(generator, request)
    # Raw argmax is token 0. Repetition penalty changes token 0 from 4 -> 2,
    # making token 1 the exact processed-logit argmax.
    token, _ = _sample_mllm_prefill_logits(
        mx.array([[4.0, 3.0, 2.0, 1.0]]), sampler
    )
    mx.eval(token)
    assert int(token.item()) == 1


def test_decode_step_uses_the_same_normalizing_helper_as_prefill():
    source = inspect.getsource(MLLMBatchGenerator._step)
    assert source.count("_sample_mllm_prefill_logits(") == 3
    assert "sampled = shared_sampler(logits)" not in source
    assert "req_sampler(logits[i:i+1])" not in source


def _pending_decode_generator(requests, logits):
    generator = object.__new__(MLLMBatchGenerator)
    generator._model_type = "glm5_next"
    generator._decode_trace = False
    generator.language_model = lambda tokens, **kwargs: mx.array(logits)[:, None, :]
    generator.active_batch = SimpleNamespace(requests=requests)
    return generator


def _pending_decode_request(**kwargs):
    from vmlx_engine.mllm_batch_generator import MLLMBatchRequest

    request = MLLMBatchRequest(
        uid=0, request_id="pending", prompt="", temperature=0.0, top_p=1.0,
        **kwargs,
    )
    request._original_token_ids = [0]
    return request


def test_decode_repetition_penalty_includes_consumed_pending_token():
    request = _pending_decode_request(repetition_penalty=2.0)
    generator = _pending_decode_generator([request], [[1.0, 3.0, 4.0]])

    token, _ = generator._step(mx.array([[2]]), [])
    mx.eval(token)

    # Token 2 is already in the model context, although _next has not emitted
    # it into output_tokens yet. Its score must be 4/2, so token 1 wins.
    assert int(token.item()) == 1
    assert request.output_tokens == []
    assert not hasattr(request, "_sampler_pending_token_ids")
    # A subsequent non-decode caller must not inherit the pending token.
    token, _ = _sample_mllm_prefill_logits(
        mx.array([[1.0, 3.0, 4.0]]), request._cached_sampler
    )
    mx.eval(token)
    assert int(token.item()) == 2


def test_decode_frequency_penalty_counts_identical_pending_token_again():
    request = _pending_decode_request(frequency_penalty=1.0)
    request.output_tokens = [2]
    generator = _pending_decode_generator([request], [[0.0, 2.5, 4.0]])

    token, _ = generator._step(mx.array([[2]]), [])
    mx.eval(token)

    # The last emitted token and the pending token both happen to be 2.
    # Deduplicating them leaves score 3 and picks the wrong next token.
    assert int(token.item()) == 1
    assert request.output_tokens == [2]


def test_decode_pending_penalty_context_is_per_request():
    requests = [_pending_decode_request(repetition_penalty=2.0) for _ in range(2)]
    generator = _pending_decode_generator(requests, [[1.0, 3.0, 4.0]] * 2)

    tokens, _ = generator._step(mx.array([[2], [1]]), [])
    mx.eval(tokens)

    assert tokens.tolist() == [1, 2]
    assert all(not hasattr(r, "_sampler_pending_token_ids") for r in requests)


def test_decode_pending_context_is_cleared_when_sampling_raises(monkeypatch):
    request = _pending_decode_request(repetition_penalty=2.0)
    generator = _pending_decode_generator([request], [[1.0, 3.0, 4.0]])

    def fail(_request):
        raise RuntimeError("sampling failed")

    monkeypatch.setattr(generator, "_make_request_sampler", fail)
    with pytest.raises(RuntimeError, match="sampling failed"):
        generator._step(mx.array([[2]]), [])
    assert not hasattr(request, "_sampler_pending_token_ids")
    assert request.output_tokens == []


@pytest.mark.parametrize("control", ["frequency_penalty", "presence_penalty", "logit_bias"])
def test_decode_token_controls_do_not_share_the_first_request_history(control):
    options = [{control: 1.0}, {control: 1.0}]
    if control == "logit_bias":
        options = [{"logit_bias": {"2": -2.0}}, {"logit_bias": {"1": -2.0}}]
    requests = [_pending_decode_request(**option) for option in options]
    generator = _pending_decode_generator(requests, [[0.0, 2.5, 3.0]] * 2)

    tokens, _ = generator._step(mx.array([[2], [1]]), [])
    mx.eval(tokens)

    # Each row has its own consumed token and controls. Sharing row zero's
    # processor penalizes token 2 in BOTH rows, incorrectly returning [1, 1].
    assert tokens.tolist() == [1, 2]
    assert all(not hasattr(r, "_sampler_pending_token_ids") for r in requests)


def test_decode_second_request_token_control_is_not_dropped():
    requests = [_pending_decode_request(), _pending_decode_request(frequency_penalty=1.0)]
    generator = _pending_decode_generator(requests, [[0.0, 2.5, 3.0]] * 2)
    tokens, _ = generator._step(mx.array([[2], [2]]), [])
    mx.eval(tokens)
    assert tokens.tolist() == [2, 1]


@pytest.mark.parametrize('temperature', [0.0, 0.7])
def test_static_bias_native_mtp_contract_matches_processed_distribution(temperature):
    from vmlx_engine.native_mtp_acceptance import accept_lp_for
    from vmlx_engine.mllm_batch_generator import _native_mtp_sample_one, _native_mtp_sample_rows
    from vmlx_engine.utils.token_logits_processors import make_openai_token_penalty_processor

    request = SimpleNamespace(
        temperature=temperature, top_p=0.8, top_k=3, min_p=0.1,
        repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0,
        logit_bias={0: -3.0, 3: 2.0}, enable_thinking=True,
        _original_token_ids=[0, 1], output_tokens=[],
    )
    sampler = MLLMBatchGenerator._make_request_sampler(SimpleNamespace(_model_type='qwen4_exp'), request)
    logits = mx.array([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
    _, target_lps, target_ids = _native_mtp_sample_rows(logits, sampler)
    if temperature == 0:
        assert sampler._vmlx_is_greedy
        assert target_lps == [None, None]
        assert target_ids == [1, 3]
        _, draft_lp = _native_mtp_sample_one(logits[:1], sampler)
        assert draft_lp is None
    else:
        base = make_sampler(temp=temperature, top_p=.8, top_k=3, min_p=.1)
        processor = make_openai_token_penalty_processor(logit_bias=request.logit_bias)
        biased = processor([], logits)
        expected = accept_lp_for(base, biased - mx.logsumexp(biased, axis=-1, keepdims=True))
        actual = mx.stack([accept_lp_for(sampler, row) for row in target_lps])
        assert bool(mx.allclose(mx.exp(actual), mx.exp(expected), atol=1e-6))
        _, draft_lp = _native_mtp_sample_one(logits[:1], sampler)
        assert bool(mx.allclose(mx.exp(accept_lp_for(sampler, draft_lp)), mx.exp(actual[0]), atol=1e-6))


@pytest.mark.parametrize('dtype', [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize('offset', [8.0, 1000.0])
@pytest.mark.parametrize('top_p,top_k,min_p', [(.7, 3, 0.), (.8, 0, .3)])
def test_native_mtp_bias_preserves_finite_precision_draw_inputs(dtype, offset, top_p, top_k, min_p):
    """Normalize-then-bias loses raw-logit rounding and can change filter support."""
    from vmlx_engine.native_mtp_acceptance import accept_lp_for
    from vmlx_engine.mllm_batch_generator import _native_mtp_sample_one, _native_mtp_sample_rows
    from vmlx_engine.utils.token_logits_processors import make_openai_token_penalty_processor

    request = SimpleNamespace(
        temperature=.7, top_p=top_p, top_k=top_k, min_p=min_p,
        repetition_penalty=1., frequency_penalty=0., presence_penalty=0.,
        logit_bias={0: .2, 1: .4}, enable_thinking=True,
        _original_token_ids=[0, 1], output_tokens=[],
    )
    wrapper = MLLMBatchGenerator._make_request_sampler(SimpleNamespace(_model_type='qwen4_exp'), request)
    base = make_sampler(temp=.7, top_p=top_p, top_k=top_k, min_p=min_p)
    logits = mx.array([[offset, offset - 1, offset - 2, offset - 3]], dtype=dtype)
    processor = make_openai_token_penalty_processor(logit_bias=request.logit_bias)
    actual_biased_logits = processor([], logits)
    _, expected_lps, _ = _native_mtp_sample_rows(actual_biased_logits, base)
    _, actual_lps, _ = _native_mtp_sample_rows(logits, wrapper)
    _, draft_lp = _native_mtp_sample_one(logits, wrapper)
    assert bool(mx.array_equal(actual_lps[0], expected_lps[0]))
    assert bool(mx.array_equal(draft_lp, expected_lps[0]))
    expected = accept_lp_for(base, expected_lps[0])
    actual = accept_lp_for(wrapper, actual_lps[0])
    assert bool(mx.array_equal(mx.isfinite(actual), mx.isfinite(expected)))
    assert bool(mx.array_equal(actual, expected))
