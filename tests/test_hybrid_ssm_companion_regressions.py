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
from vmlx_engine.scheduler import Scheduler  # noqa: E402


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


def test_text_scheduler_rejected_block_hit_rolls_back_accounting_and_refs():
    cache = _accounting_cache()
    released = []
    detached = []
    cache.paged_cache = SimpleNamespace(
        release_request_refs=released.append,
        get_memory_usage=lambda: {},
    )
    cache.detach_request = detached.append
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.block_aware_cache = cache
    table = SimpleNamespace(block_ids=[1, 2], num_tokens=128)
    request = SimpleNamespace(
        request_id="request-1",
        block_table=table,
        shared_prefix_blocks=2,
    )

    scheduler._release_unusable_paged_hit(request)

    assert cache.get_stats()["hits"] == 0
    assert cache.get_stats()["misses"] == 1
    assert cache.get_stats()["tokens_saved"] == 0
    assert released == [table]
    assert detached == ["request-1"]
    assert request.block_table is None
    assert request.shared_prefix_blocks == 0


def test_text_scheduler_partial_block_hit_keeps_only_accepted_credit():
    cache = _accounting_cache()
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.block_aware_cache = cache
    request = SimpleNamespace(request_id="request-1")

    scheduler._accept_paged_hit_credit(request, 64)

    assert cache.get_stats()["hits"] == 1
    assert cache.get_stats()["misses"] == 0
    assert cache.get_stats()["tokens_saved"] == 64
    assert cache._hit_credits == {"request-1": 64}


def test_text_scheduler_surfaces_ssm_rederive_as_idle_task_not_request():
    # vmlx#245: the queued re-derive must NOT keep step() on the response
    # path (has_requests stays False); it is surfaced via has_idle_tasks()
    # and drained by the engine loop's idle branch after responses finalize.
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.waiting = []
    scheduler.running = {}
    scheduler._pending_aborts = set()
    scheduler._is_hybrid = True
    scheduler._ssm_rederive_queue = [([1, 2, 3], 3, "request-1")]

    scheduler._ensure_ssm_rederive_idle_task()

    assert scheduler.has_requests() is False
    assert scheduler.has_idle_tasks() is True

    # Registration is once-per-queue-lifetime (ensure is idempotent).
    scheduler._ensure_ssm_rederive_idle_task()
    assert len(scheduler._idle_tasks) == 1


def test_text_scheduler_retargets_ssm_rederive_to_truncated_kv_boundary():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._is_hybrid = True
    scheduler._ssm_rederive_queue = [
        (list(range(706)), 706, "request-1"),
        ([9, 8], 2, "other"),
    ]

    scheduler._retarget_ssm_rederive_to_paged_boundary(
        "request-1",
        list(range(706)),
        SimpleNamespace(num_tokens=576),
    )

    assert scheduler._ssm_rederive_queue == [
        ([9, 8], 2, "other"),
        (list(range(576)), 576, "request-1"),
    ]


def test_mllm_scheduler_surfaces_ssm_rederive_as_idle_task_not_request():
    # vmlx#245: same contract as the text scheduler — the queue never keeps
    # step() on the response path; _process_loop's idle branch drains it.
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.waiting = []
    scheduler.running = {}
    scheduler._is_hybrid = True
    scheduler.batch_generator = SimpleNamespace(
        _ssm_rederive_queue=[([1, 2, 3], 3, "request-1")]
    )

    assert scheduler.has_requests() is False
    assert scheduler.has_idle_tasks() is True

    scheduler.batch_generator._ssm_rederive_queue.clear()
    assert scheduler.has_idle_tasks() is False


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


# ── vmlx#245: post-response idle-task hook regressions ───────────────────────


def _idle_ready_scheduler():
    """Text scheduler skeleton with an idle-drainable rederive queue."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.waiting = []
    scheduler.running = {}
    scheduler._pending_aborts = set()
    scheduler._is_hybrid = True
    scheduler._uses_zaya_cache = False
    scheduler.config = SimpleNamespace(enable_prefix_cache=True)
    scheduler._hybrid_kv_positions = [0]
    scheduler._ssm_rederive_queue = []
    return scheduler


def test_run_one_idle_task_processes_single_entry_and_stores(monkeypatch):
    from types import MethodType

    import vmlx_engine.scheduler as scheduler_mod

    monkeypatch.setattr(
        scheduler_mod, "clear_mlx_memory_cache", lambda **_kwargs: None
    )

    scheduler = _idle_ready_scheduler()
    stored = []
    scheduler._ssm_state_cache = SimpleNamespace(
        store=lambda tokens, plen, layers: stored.append(
            (tokens, plen, layers)
        )
    )
    kv_layer = SimpleNamespace()
    ssm_layer = SimpleNamespace()

    def _fake_prefill(self, tokens, should_stop=None):
        return [kv_layer, ssm_layer]

    scheduler._prefill_for_prompt_only_cache = MethodType(
        _fake_prefill, scheduler
    )
    scheduler._ssm_rederive_queue.append(([1, 2, 3], 3, "req-1"))
    scheduler._ensure_ssm_rederive_idle_task()

    assert scheduler.run_one_idle_task() is True

    # KV layer (position 0) excluded; SSM layer stored under the prompt key.
    assert stored == [([1, 2, 3], 3, [ssm_layer])]
    assert scheduler._ssm_rederive_queue == []
    # DONE contract: task dequeued, registration flag reset for next queue.
    assert scheduler.has_idle_tasks() is False
    assert scheduler._ssm_rederive_task_queued is False


def test_idle_rederive_parks_before_pop_when_foreground_pending():
    scheduler = _idle_ready_scheduler()
    scheduler._ssm_state_cache = SimpleNamespace(store=lambda *a: None)
    scheduler._ssm_rederive_queue.append(([1, 2, 3], 3, "req-1"))
    scheduler._ensure_ssm_rederive_idle_task()
    scheduler.running = {"req-live": object()}

    assert scheduler.run_one_idle_task() is True

    # PARKED contract: entry untouched, task re-queued at the front.
    assert scheduler._ssm_rederive_queue == [([1, 2, 3], 3, "req-1")]
    assert scheduler.has_idle_tasks() is True


def test_idle_rederive_parks_mid_prefill_and_requeues_entry_at_front():
    from types import MethodType

    scheduler = _idle_ready_scheduler()
    scheduler._ssm_state_cache = SimpleNamespace(store=lambda *a: None)
    first = ([1, 2, 3], 3, "req-a")
    second = ([9, 8], 2, "req-b")
    scheduler._ssm_rederive_queue.extend([first, second])
    scheduler._ensure_ssm_rederive_idle_task()

    def _fake_prefill(self, tokens, should_stop=None):
        # Foreground request arrives between prefill chunks: the park poll
        # must abort the prefill so the engine can serve it immediately.
        self.waiting.append(object())
        assert should_stop is not None and should_stop() is True
        return None

    scheduler._prefill_for_prompt_only_cache = MethodType(
        _fake_prefill, scheduler
    )

    assert scheduler.run_one_idle_task() is True

    # The popped entry is re-queued at the FRONT (no loss, no reorder).
    assert scheduler._ssm_rederive_queue == [first, second]
    assert scheduler.has_idle_tasks() is True


def test_idle_rederive_foreground_race_yields_before_first_model_chunk(
    monkeypatch,
):
    """A request arriving after dequeue must not pay one maintenance chunk."""
    import vmlx_engine.scheduler as scheduler_mod

    monkeypatch.setattr(
        scheduler_mod, "clear_mlx_memory_cache", lambda **_kwargs: None
    )

    scheduler = _idle_ready_scheduler()
    stored = []
    model_calls = []
    scheduler._ssm_state_cache = SimpleNamespace(
        store=lambda *args: stored.append(args)
    )
    first = ([1, 2, 3], 3, "req-raced")
    second = ([9, 8], 2, "req-next")
    scheduler._ssm_rederive_queue.extend([first, second])
    scheduler._ensure_ssm_rederive_idle_task()
    scheduler._uses_dsv4_cache = False
    kv_layer = SimpleNamespace()
    ssm_layer = SimpleNamespace()

    class _Model:
        def make_cache(self):
            # The idle drain already passed its initial foreground check and
            # popped `first`; foreground arrives just before chunk zero.
            scheduler.waiting.append(object())
            return [kv_layer, ssm_layer]

        def __call__(self, input_ids, cache=None):
            model_calls.append(int(input_ids.shape[-1]))
            return mx.zeros((1, int(input_ids.shape[-1]), 1))

    scheduler.model = _Model()

    assert scheduler.run_one_idle_task() is True

    assert model_calls == []
    assert stored == []
    assert scheduler._ssm_rederive_queue == [first, second]
    assert scheduler.has_idle_tasks() is True


def test_idle_rederive_clears_unservable_queue():
    scheduler = _idle_ready_scheduler()
    scheduler._is_hybrid = False  # queue can never be consumed
    scheduler._ssm_state_cache = None
    scheduler._ssm_rederive_queue.append(([1], 1, "req-stale"))
    scheduler._ensure_ssm_rederive_idle_task()

    assert scheduler.run_one_idle_task() is True

    assert scheduler._ssm_rederive_queue == []
    assert scheduler.has_idle_tasks() is False
    assert scheduler._ssm_rederive_task_queued is False


def test_idle_task_exception_drops_task():
    scheduler = Scheduler.__new__(Scheduler)

    def _boom():
        raise RuntimeError("idle task failure")

    scheduler.register_idle_task(_boom, name="boom")

    assert scheduler.run_one_idle_task() is True
    assert scheduler.has_idle_tasks() is False


def test_mllm_run_one_idle_task_yields_to_foreground_and_holds_batch_lock():
    import threading

    calls = []
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.waiting = [object()]
    scheduler.running = {}
    scheduler._is_hybrid = True
    scheduler._batch_lock = threading.Lock()
    scheduler.batch_generator = SimpleNamespace(
        _ssm_rederive_queue=[([1], 1, "req-1")],
        run_idle_rederive=lambda: calls.append("ran") or True,
    )

    # Foreground pending → must NOT touch the generator.
    assert scheduler.run_one_idle_task() is False
    assert calls == []

    scheduler.waiting = []
    assert scheduler.run_one_idle_task() is True
    assert calls == ["ran"]


def test_scheduler_shutdown_drops_idle_tasks_and_rederive_queue():
    """vmlx#245 shutdown contract: no forward pass after teardown begins."""
    from pathlib import Path

    import vmlx_engine.scheduler as scheduler_mod

    src = Path(scheduler_mod.__file__).read_text()
    shutdown_idx = src.index("def shutdown")
    block = src[shutdown_idx : src.index("Flush prompt-level", shutdown_idx)]
    assert "_idle_tasks" in block and ".clear()" in block
    assert "_ssm_rederive_queue" in block
    assert "_ssm_rederive_task_queued = False" in block


def test_mllm_stop_drops_rederive_queue_before_generator_close():
    """vmlx#245 shutdown contract on the MLLM path."""
    from pathlib import Path

    import vmlx_engine.mllm_scheduler as mllm_mod

    src = Path(mllm_mod.__file__).read_text()
    stop_idx = src.index("async def stop")
    stop_block = src[stop_idx : src.index("MLLM Scheduler stopped", stop_idx)]
    clear_idx = stop_block.index("rederive_queue.clear()")
    close_idx = stop_block.index("self.batch_generator.close()")
    assert clear_idx < close_idx


def test_engine_loop_drains_idle_tasks_off_the_response_path():
    """vmlx#245 engine contract: idle branch drains at most one task on the
    step executor; the sync generate() path drains bounded post-loop so the
    LFM disk-only-restore fix (3376b1dc6) keeps working without keeping
    step() awake."""
    from pathlib import Path

    import vmlx_engine.engine_core as engine_mod

    src = Path(engine_mod.__file__).read_text()
    assert "has_idle_tasks" in src
    assert "run_one_idle_task" in src
    # Idle branch dispatches on the step executor (Metal stream affinity).
    idle_idx = src.index("run_one_idle_task")
    assert "_step_executor" in src[max(0, idle_idx - 800) : idle_idx + 800]


def test_mllm_process_loop_idle_branch_drains_one_task():
    """vmlx#245: _process_loop's idle arm must drain via run_one_idle_task
    on the step executor instead of sleeping past queued maintenance."""
    from pathlib import Path

    import vmlx_engine.mllm_scheduler as mllm_mod

    src = Path(mllm_mod.__file__).read_text()
    loop_idx = src.index("async def _process_loop")
    loop_block = src[loop_idx : src.index("\n    async def ", loop_idx + 10)]
    assert "self.has_idle_tasks()" in loop_block
    assert "self.run_one_idle_task" in loop_block
    assert "_step_executor" in loop_block
