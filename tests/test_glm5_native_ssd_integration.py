"""Opt-in GLM native SSD admission, prefill ownership and idle maintenance.

Tiny native states exercise wiring, not real-model numerical equivalence.
"""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import pytest

from tests.test_glm5_companion_disk_codec import native_facade, native_state
from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchRequest
from vmlx_engine.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig
from vmlx_engine.models.glm5_next.glm5_next import Glm5KDACache, Glm5MLACache
from vmlx_engine.persistence_outcome import LEDGER
from vmlx_engine.utils.glm5_native_prefix_cache import Glm5NativePrefixCache
from vmlx_engine.utils.ssm_companion_disk_store import SSMCompanionDiskStore


class TinyLanguageModel:
    config = SimpleNamespace(model_type="glm5_next_text", num_hidden_layers=2)

    def __init__(self):
        self.calls = []

    def make_cache(self):
        return [Glm5KDACache(), Glm5MLACache(4, absorbed=True)]

    def __call__(self, inputs, *, cache, return_logits=True, **kwargs):
        self.calls.append((inputs.shape[1], return_logits))
        total = cache[1].offset + inputs.shape[1]
        cache[:] = native_state(total)
        return mx.zeros((1, inputs.shape[1] if return_logits else 1, 8))


def tiny_model():
    return SimpleNamespace(language_model=TinyLanguageModel(), config=TinyLanguageModel.config)


def tiny_processor():
    return SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=0, eos_token_ids={0}))


@pytest.mark.parametrize("override,eligible", [
    ({}, True), ({"enable_prefix_cache": False}, False),
    ({"enable_block_disk_cache": False}, False),
    ({"use_paged_cache": True}, False), ({"max_num_seqs": 2}, False),
])
def test_scheduler_admits_only_explicit_ssd_single_native_layout(tmp_path, monkeypatch, override, eligible):
    monkeypatch.setenv("VMLX_GLM5_NATIVE_SSD", "1")
    config = dict(enable_prefix_cache=True, enable_block_disk_cache=True,
                  use_paged_cache=False, use_memory_aware_cache=True, max_num_seqs=1,
                  enable_disk_cache=True, disk_cache_dir=str(tmp_path / "unused-legacy"),
                  block_disk_cache_dir=str(tmp_path / "ssd"), block_disk_cache_max_gb=0.01,
                  model_path=str(tmp_path / "tiny-native-test"))
    config.update(override)
    scheduler = MLLMScheduler(tiny_model(), tiny_processor(), MLLMSchedulerConfig(**config))
    try:
        assert scheduler._native_glm_ssd_selected
        assert (scheduler.native_glm_cache is not None) is eligible
        assert scheduler.config.enable_prefix_cache is eligible
        assert scheduler.config.use_memory_aware_cache is False
        assert scheduler.memory_aware_cache is None
        assert scheduler.prefix_cache is None
        assert scheduler.block_aware_cache is None
        assert scheduler.paged_cache_manager is None
        assert scheduler.disk_cache is None
        assert not (tmp_path / "unused-legacy").exists()
        if eligible:
            native = scheduler.native_glm_cache
            assert scheduler._ssm_companion_disk_store is native.disk
            assert native.budget.root == (tmp_path / "ssd").resolve()
            assert native.disk.budget_bytes == int(0.01 * 1024**3)
            assert native.lookup.total_nbytes == 0
            scheduler._ensure_batch_generator()
            assert scheduler.batch_generator.native_glm_cache is native
            assert scheduler.batch_generator._ssm_state_cache is None
        else:
            assert scheduler._prefix_cache_unavailable_reason
    finally:
        asyncio.run(scheduler.stop())
    assert scheduler.native_glm_cache is None
    assert scheduler._ssm_companion_disk_store is None


def test_native_default_off_does_not_instantiate_backend(tmp_path, monkeypatch):
    monkeypatch.delenv("VMLX_GLM5_NATIVE_SSD", raising=False)
    scheduler = MLLMScheduler(tiny_model(), tiny_processor(), MLLMSchedulerConfig(
        enable_prefix_cache=True, enable_block_disk_cache=False,
        use_paged_cache=False, enable_disk_cache=True,
        disk_cache_dir=str(tmp_path / "legacy"),
    ))
    try:
        assert not scheduler._native_glm_ssd_selected
        assert scheduler.native_glm_cache is None
        assert scheduler.config.enable_prefix_cache is False
    finally:
        asyncio.run(scheduler.stop())


def test_cleanup_keeps_prefill_receipt_without_post_decode_rederive(tmp_path, monkeypatch):
    monkeypatch.setenv("VMLX_GLM5_NATIVE_SSD", "1")
    scheduler = MLLMScheduler(tiny_model(), tiny_processor(), MLLMSchedulerConfig(
        enable_prefix_cache=True, enable_block_disk_cache=True, use_paged_cache=False,
        max_num_seqs=1, block_disk_cache_dir=str(tmp_path), block_disk_cache_max_gb=0.01,
    ))
    request_id = "native-terminal-owns-no-rederive"
    extract = Mock(side_effect=AssertionError("post-decode extraction is not a causal checkpoint"))
    request = SimpleNamespace(num_output_tokens=3, _extracted_cache=extract,
                              _extracted_tokens=[1, 2, 3], _added_stop_tokens=set())
    try:
        scheduler.running[request_id] = request
        scheduler.requests[request_id] = request
        LEDGER.record(request_id, "stored", "native prefill fsynced", retained_tokens=2, durable=True)
        scheduler._cleanup_finished({request_id})
        extract.assert_not_called()
        assert LEDGER.take(request_id) == dict(outcome="stored", detail="native prefill fsynced",
                                             retained_tokens=2, durable=True)
        assert request._extracted_cache is None
    finally:
        LEDGER.take(request_id)
        asyncio.run(scheduler.stop())


@pytest.mark.parametrize("length,step", [(6, 2048), (13, 2)])
def test_prefill_publishes_exact_full_boundary_before_final_forward(tmp_path, length, step):
    native = native_facade(tmp_path)
    model = tiny_model()
    generator = MLLMBatchGenerator(model=model, processor=tiny_processor(),
                                   prefill_step_size=step, native_glm_cache=native,
                                   enable_prefix_cache=True, ssm_state_cache_size=0)
    tokens = list(range(length))
    request = MLLMBatchRequest(uid=0, request_id=f"glm-native-split-{length}",
                               prompt="", input_ids=mx.array([tokens]), temperature=0)
    request._glm_native_full_token_ids = tokens
    request._original_token_ids = tokens[:-2]  # legacy generation suffix is not our key
    request._gen_prefix_tokens = tokens[-2:]
    try:
        state = model.language_model.make_cache()
        output = generator._run_vision_encoding_inner(request, state)
        mx.eval(output)
        receipt = LEDGER.take(request.request_id)
        assert receipt["outcome"] == "stored"
        assert receipt["durable"] is True
        assert receipt["retained_tokens"] == length - 1
        assert model.language_model.calls[-1] == (1, True)
        assert sum(n for n, _ in model.language_model.calls) == length
        assert state[1].offset == length
        boundary, restored = native.fetch(tokens)
        assert boundary == restored[1].offset == length - 1
        # Exact repeat reuses the materialized checkpoint and only forwards
        # the actual final template token. Legacy usage/key fields stay intact.
        generator._restore_glm_native_prefix(request)
        assert request._cached_tokens == length - 1
        assert request._cache_detail == "native-glm+disk"
        assert request.input_ids.tolist() == [[tokens[-1]]]
        assert request._original_token_ids == tokens[:-2]
        assert request._gen_prefix_tokens == tokens[-2:]
        generator._run_vision_encoding_inner(request, request.prompt_cache)
        receipt = LEDGER.take(request.request_id)
        assert receipt["outcome"] == "already_durable"
        assert model.language_model.calls[-1] == (1, True)
        assert native.disk.stats()["stores"] == 1
    finally:
        LEDGER.take(request.request_id)
        native.close()


@pytest.mark.parametrize("kind", ["bypass", "image", "video", "historical", "audio"])
def test_unsafe_request_neither_restores_nor_publishes(kind):
    generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    generator.native_glm_cache = Mock()
    generator._tokens_contain_media_placeholders = lambda ids: 999 in ids
    req = SimpleNamespace(request_id=f"native-exclude-{kind}",
                          _glm_native_full_token_ids=[1, 2, 3])
    if kind == "bypass": req._bypass_prefix_cache = True
    elif kind == "historical": req._glm_native_full_token_ids = [1, 999, 3]
    elif kind == "image": req.images = ["not-opened"]
    elif kind == "video": req.videos = ["not-opened"]
    else: req.audio_features = object()
    generator._restore_glm_native_prefix(req)
    generator._store_glm_native_boundary(req, [])
    generator.native_glm_cache.fetch.assert_not_called()
    generator.native_glm_cache.store.assert_not_called()
    outcome = LEDGER.take(req.request_id)
    assert outcome["outcome"] == "skipped"
    assert outcome["retained_tokens"] == 0 and outcome["durable"] is False


def test_failed_publication_is_not_a_durable_boundary():
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen.native_glm_cache = SimpleNamespace(store=Mock(side_effect=OSError("disk unavailable")))
    gen._tokens_contain_media_placeholders = lambda ids: False
    req = SimpleNamespace(request_id="native-store-failure", _glm_native_full_token_ids=[1, 2])
    gen._store_glm_native_boundary(req, [])
    assert LEDGER.take(req.request_id) == dict(outcome="failed", detail="native SSD boundary failed",
                                             retained_tokens=0, durable=False)


def test_native_health_reports_real_pool_not_generic_blocks(tmp_path):
    from vmlx_engine.server import _native_cache_status
    native = native_facade(tmp_path)
    try:
        status = _native_cache_status(SimpleNamespace(native_glm_cache=native), family="glm5_next")
        assert status["experimental"] is True
        assert status["native_checkpoint_ssd"] is True
        assert status["prefix"] is True
        assert status["paged"] is status["block_disk_l2"] is status["prompt_disk_l2"] is False
        assert status["retained_ram_bytes"] == 0
        assert status["last_native_ssd_store"] is None
        assert status["cache_store_policy"]["media"] == "unsupported"
    finally:
        native.close()


@pytest.mark.parametrize("busy", [True, False, "unknown"])
def test_native_idle_maintenance_requires_quiet_and_no_inference(busy):
    cache = Glm5NativePrefixCache.__new__(Glm5NativePrefixCache)
    cache._last_activity = -100
    cache.budget = SimpleNamespace(deferred_reconcile_due=False,
        idle_reconcile_due=Mock(return_value=True), run_idle_reconcile=Mock())
    cache._activity_probe = Mock(return_value=busy) if busy != "unknown" else Mock(side_effect=RuntimeError())
    cache._idle_maintenance()
    assert cache.budget.run_idle_reconcile.call_count == (1 if busy is False else 0)


def test_companion_writer_services_idle_hook_and_survives_hook_error(tmp_path):
    fired = threading.Event()
    def maintenance():
        fired.set()
        raise RuntimeError("diagnostic maintenance error")
    disk = SSMCompanionDiskStore(directory=tmp_path, idle_maintenance=maintenance)
    try:
        assert fired.wait(2), "idle writer never serviced its maintenance owner"
        assert disk.store("f" * 64, native_state(3), True, [1, 2, 3], 3)
        assert disk.wait_for_write("f" * 64)
        assert disk.fetch("f" * 64) is not None
    finally:
        assert disk.shutdown()
