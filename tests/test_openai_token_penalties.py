import math

import mlx.core as mx
import pytest
from pydantic import ValidationError

from vmlx_engine.api.models import ChatCompletionRequest, CompletionRequest, Message
from vmlx_engine.engine.base import GenerationOutput
from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchRequest
from vmlx_engine.request import Request, SamplingParams
from vmlx_engine.scheduler import Scheduler
from vmlx_engine.server import _forward_openai_token_controls
from vmlx_engine.utils.token_logits_processors import (
    make_openai_token_penalty_processor,
)


def _chat_request(**overrides):
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload.update(overrides)
    return ChatCompletionRequest(**payload)


def test_openai_token_controls_are_declared_on_chat_and_completions():
    chat = _chat_request(
        logit_bias={"7": 4.5},
        frequency_penalty=0.75,
        presence_penalty=-0.25,
    )
    completion = CompletionRequest(
        model="test-model",
        prompt="hello",
        logit_bias={"8": -3},
        frequency_penalty=-0.5,
        presence_penalty=1.25,
    )

    assert chat.logit_bias == {"7": 4.5}
    assert chat.frequency_penalty == 0.75
    assert chat.presence_penalty == -0.25
    assert completion.logit_bias == {"8": -3.0}
    assert completion.frequency_penalty == -0.5
    assert completion.presence_penalty == 1.25


@pytest.mark.parametrize(
    "payload",
    [
        {"logit_bias": {"not-a-token": 1}},
        {"logit_bias": {"-1": 1}},
        {"logit_bias": {"1": 100.01}},
        {"logit_bias": {"1": math.nan}},
        {"frequency_penalty": 2.01},
        {"presence_penalty": -2.01},
    ],
)
def test_openai_token_controls_fail_closed_at_request_validation(payload):
    with pytest.raises(ValidationError):
        _chat_request(**payload)


def test_sampling_params_preserve_openai_token_controls():
    params = SamplingParams(
        logit_bias={3: 8.0},
        frequency_penalty=0.5,
        presence_penalty=-0.25,
    )

    assert params.logit_bias == {3: 8.0}
    assert params.frequency_penalty == 0.5
    assert params.presence_penalty == -0.25


def test_openai_token_processor_combines_bias_frequency_and_presence():
    processor = make_openai_token_penalty_processor(
        logit_bias={4: 7.0},
        frequency_penalty=0.5,
        presence_penalty=1.25,
    )
    assert processor is not None

    logits = mx.zeros((1, 8), dtype=mx.float32)
    processed = processor([1, 1, 2], logits)
    mx.eval(processed)

    values = processed.tolist()[0]
    assert values[1] == pytest.approx(-2.25)
    assert values[2] == pytest.approx(-1.75)
    assert values[4] == pytest.approx(7.0)
    assert values[0] == 0.0


def test_openai_token_processor_updates_counts_incrementally():
    processor = make_openai_token_penalty_processor(
        frequency_penalty=0.5,
        presence_penalty=1.0,
    )
    logits = mx.zeros((1, 6), dtype=mx.float32)

    first = processor([2], logits)
    mx.eval(first)
    second = processor([2, 2, 3], logits)
    mx.eval(second)

    assert first.tolist()[0][2] == pytest.approx(-1.5)
    assert second.tolist()[0][2] == pytest.approx(-2.0)
    assert second.tolist()[0][3] == pytest.approx(-1.5)


def test_openai_token_processor_rejects_out_of_vocab_bias():
    processor = make_openai_token_penalty_processor(logit_bias={9: 1.0})

    with pytest.raises(ValueError, match="outside vocabulary"):
        processor([], mx.zeros((1, 8), dtype=mx.float32))


def test_openai_token_processor_none_when_controls_are_inert():
    assert make_openai_token_penalty_processor() is None
    assert (
        make_openai_token_penalty_processor(
            logit_bias={"2": 0},
            frequency_penalty=0,
            presence_penalty=0,
        )
        is None
    )


def test_server_forwards_explicit_token_controls_without_token_ids_in_logs():
    request = _chat_request(
        logit_bias={"7": 4.5},
        frequency_penalty=0.75,
        presence_penalty=-0.25,
    )
    kwargs = {}

    _forward_openai_token_controls(kwargs, request)

    assert kwargs == {
        "frequency_penalty": 0.75,
        "presence_penalty": -0.25,
        "logit_bias": {"7": 4.5},
    }


@pytest.mark.asyncio
async def test_chat_endpoint_delivers_token_controls_to_engine(monkeypatch):
    import vmlx_engine.server as server

    class _Engine:
        is_mllm = False
        preserve_native_tool_format = False

        def __init__(self):
            self.kwargs = None

        async def chat(self, *, messages, **kwargs):
            self.kwargs = kwargs
            return GenerationOutput(
                text="ok",
                prompt_tokens=2,
                completion_tokens=1,
                finish_reason="stop",
            )

    engine = _Engine()
    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setattr(server, "_served_model_name", "loaded-model")
    monkeypatch.setattr(server, "_model_name", "loaded-model")
    monkeypatch.setattr(server, "_model_path", None)
    monkeypatch.setattr(server, "_model_type", "llm")
    monkeypatch.setattr(server, "_reasoning_parser", None)
    monkeypatch.setattr(server, "_mcp_manager", None)

    await server.create_chat_completion(
        ChatCompletionRequest(
            model="loaded-model",
            messages=[Message(role="user", content="hello")],
            max_tokens=1,
            frequency_penalty=0.5,
            presence_penalty=-0.25,
            logit_bias={"7": 4.5},
        ),
        fastapi_request=None,
    )

    assert engine.kwargs["frequency_penalty"] == 0.5
    assert engine.kwargs["presence_penalty"] == -0.25
    assert engine.kwargs["logit_bias"] == {"7": 4.5}


def test_text_scheduler_builds_request_local_openai_processor():
    scheduler = object.__new__(Scheduler)
    scheduler._long_repetition_context = False
    request = Request(
        request_id="token-controls",
        prompt=[1],
        sampling_params=SamplingParams(logit_bias={5: 9.0}),
    )

    processors = scheduler._request_logits_processors(request, [1])

    assert processors is not None and len(processors) == 1
    processed = processors[0]([1], mx.zeros((1, 8), dtype=mx.float32))
    mx.eval(processed)
    assert processed.tolist()[0][5] == pytest.approx(9.0)


def test_mllm_sampler_applies_openai_token_controls_before_greedy_sample():
    generator = object.__new__(MLLMBatchGenerator)
    generator._model_type = "test"
    request = MLLMBatchRequest(
        uid=1,
        request_id="mllm-token-controls",
        prompt="hello",
        temperature=0.0,
        top_p=1.0,
        logit_bias={6: 100.0},
    )
    request.input_ids = mx.array([[1]], dtype=mx.int32)

    sampler = generator._make_request_sampler(request)
    sampled = sampler(mx.zeros((1, 8), dtype=mx.float32))
    mx.eval(sampled)

    assert sampled.item() == 6
