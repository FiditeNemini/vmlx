from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

# The managed test runner can deny Metal even for import-time ``mx.compile``
# decorators.  These tests exercise Python cache ownership/accounting only, so
# keep decorators inert before importing the engine modules; no MLX operations
# are mocked or executed below.
import mlx.core as mx


def _passthrough_compile(function=None, **_kwargs):
    if function is None:
        return lambda inner: inner
    return function


mx.compile = _passthrough_compile

from vmlx_engine.mllm_batch_generator import (  # noqa: E402
    MLLMBatchGenerator,
    _hybrid_cache_layout,
    _warn_if_hybrid_detection_disabled,
)
from vmlx_engine.mllm_scheduler import MLLMScheduler  # noqa: E402
from vmlx_engine.prefix_cache import BlockAwarePrefixCache  # noqa: E402
from vmlx_engine.request import RequestOutput  # noqa: E402


class TurboQuantKVCache:
    """Class-name-compatible attention cache stand-in for structural tests."""


class ArraysCache:
    """Non-KV cumulative cache stand-in."""


def test_hybrid_layout_walks_through_jang_affine_language_wrapper():
    class InnerTextModel:
        def make_cache(self):
            return [TurboQuantKVCache(), ArraysCache(), ArraysCache()]

    class JangAffineLanguageWrapper:
        def __init__(self):
            self.model = InnerTextModel()

    language_model = JangAffineLanguageWrapper()
    outer_model = SimpleNamespace(language_model=language_model)

    owner, owner_path, template, positions, error = _hybrid_cache_layout(
        outer_model, language_model
    )

    assert owner is language_model.model
    assert owner_path == "language_model.model"
    assert [type(cache).__name__ for cache in template] == [
        "TurboQuantKVCache",
        "ArraysCache",
        "ArraysCache",
    ]
    assert positions == [0]
    assert error is None

    previous_stream = MLLMBatchGenerator._stream
    MLLMBatchGenerator._stream = object()
    try:
        generator = MLLMBatchGenerator(
            model=outer_model,
            processor=SimpleNamespace(tokenizer=SimpleNamespace()),
            enable_prefix_cache=False,
            enable_vision_cache=False,
        )
    finally:
        MLLMBatchGenerator._stream = previous_stream

    assert generator._cache_model is language_model.model
    assert generator._cache_model_path == "language_model.model"
    assert generator._hybrid_kv_positions == [0]
    assert generator._hybrid_num_layers == 3
    assert generator._is_hybrid is True


def test_declared_hybrid_without_cache_template_logs_warning(caplog):
    model = SimpleNamespace(
        config={
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "layer_types": ["linear_attention", "full_attention"],
            },
        }
    )

    with caplog.at_level(logging.WARNING, logger="vmlx_engine.mllm_batch_generator"):
        _warn_if_hybrid_detection_disabled(
            model=model,
            language_model=model,
            is_hybrid=False,
            owner_path=None,
            template=None,
            error="no callable make_cache owner",
        )

    assert "Hybrid-family model resolved _is_hybrid=False" in caplog.text
    assert "SSM companion lookup/store is disabled" in caplog.text
    assert "no callable make_cache owner" in caplog.text


def _accounting_cache(*, credited_tokens: int = 128) -> BlockAwarePrefixCache:
    cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
    cache._hits = 1
    cache._misses = 0
    cache._tokens_saved = credited_tokens
    cache._hit_credits = {"request-1": credited_tokens}
    cache._request_tables = {}
    cache._entries_by_type = {
        "assistant": {},
        "user": {},
        "system": {},
    }
    cache.paged_cache = SimpleNamespace(get_memory_usage=lambda: {})
    return cache


def test_missing_ssm_companion_rolls_back_kv_hit_accounting():
    cache = _accounting_cache()

    assert cache.adjust_cache_hit_credit("request-1", accepted_tokens=0) is True

    stats = cache.get_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 1
    assert stats["tokens_saved"] == 0
    assert cache._hit_credits == {}


def test_shorter_ssm_checkpoint_keeps_only_consumed_token_credit():
    cache = _accounting_cache()

    assert cache.adjust_cache_hit_credit("request-1", accepted_tokens=64) is True

    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 0
    assert stats["tokens_saved"] == 64
    assert cache._hit_credits == {"request-1": 64}

    cache.finalize_cache_hit_credit("request-1")
    assert cache._hit_credits == {}
    assert cache.get_stats()["tokens_saved"] == 64


def test_nonstream_generate_stamps_request_cache_metadata_on_final_output():
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    request_id = "request-1"
    request = SimpleNamespace()
    scheduler.requests = {request_id: request}
    scheduler._cache_hit_requests = 0
    scheduler._cache_hit_tokens = 0
    scheduler._cache_hit_tokens_by_detail = {}
    scheduler._record_cache_hit(
        SimpleNamespace(cached_tokens=192, cache_detail="paged+ssm+disk"),
        request,
    )

    async def add_request_async(**_kwargs):
        return request_id

    async def stream_outputs(_request_id):
        assert _request_id == request_id
        # Match the real deferred-cleanup race: terminal dispatch wakes the
        # consumer, while cleanup can remove the scheduler dictionary entry.
        scheduler.requests.pop(request_id)
        yield RequestOutput(
            request_id=request_id,
            output_text="done",
            finished=True,
            finish_reason="stop",
            cached_tokens=0,
            cache_detail="",
        )

    scheduler.add_request_async = add_request_async
    scheduler.stream_outputs = stream_outputs

    output = asyncio.run(scheduler.generate(prompt="hello"))

    assert output.cached_tokens == 192
    assert output.cache_detail == "paged+ssm+disk"
    assert request_id not in scheduler.requests
