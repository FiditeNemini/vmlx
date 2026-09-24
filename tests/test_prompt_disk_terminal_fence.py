"""Terminal cleanup must publish prompt SSD writes before the next tool request."""
import threading
from types import SimpleNamespace

import pytest
import mlx.core as mx
from mlx_lm.models.cache import KVCache

from vmlx_engine.disk_cache import DiskCacheManager
from vmlx_engine.persistence_outcome import LEDGER
from vmlx_engine.request import Request, RequestStatus, SamplingParams
from vmlx_engine.scheduler import Scheduler


@pytest.mark.parametrize("memory_accepts", [False, True])
@pytest.mark.parametrize("fail_write", [False, True])
def test_memory_aware_terminal_waits_for_real_prompt_writer(tmp_path, monkeypatch, fail_write, memory_accepts):
    import vmlx_engine.scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "clear_mlx_memory_cache", lambda log=None: None)
    cache = KVCache()
    values = mx.arange(16).reshape(1, 1, 2, 8).astype(mx.bfloat16)
    cache.update_and_fetch(values, values)
    mx.eval(cache.keys, cache.values)
    request = Request(request_id="prompt-tool-fence", prompt=[10, 11, 12],
                      sampling_params=SamplingParams(max_tokens=4))
    request.prompt_token_ids = list(request.prompt)
    request.num_prompt_tokens = 3
    request.status = RequestStatus.FINISHED_STOPPED
    request._extracted_cache = [cache]
    request._cache_extra_keys = ("media-or-template-identity",)
    manager = DiskCacheManager(str(tmp_path), max_size_gb=0.1)
    scheduler = object.__new__(Scheduler)
    scheduler.running = {request.request_id: request}
    scheduler.requests = dict(scheduler.running)
    scheduler.request_id_to_uid = {}
    scheduler.uid_to_request_id = {}
    scheduler.finished_req_ids = set()
    scheduler.batch_generator = None
    scheduler.stop_tokens = set()
    scheduler.block_aware_cache = None
    scheduler.memory_aware_cache = SimpleNamespace(store=lambda *a, **kw: memory_accepts)
    scheduler.prefix_cache = None
    scheduler.disk_cache = manager
    scheduler._kv_cache_bits = 0
    scheduler._is_hybrid = False
    scheduler._uses_dsv4_cache = False
    scheduler._uses_zaya_cache = False
    scheduler._pld_pending = {}
    scheduler._pld_ngram_indices = {}
    scheduler._pick_cache_type_for_request = lambda request: "assistant"
    scheduler._cleanup_detokenizer = lambda request_id: None
    scheduler._materialize_deferred_prompt_cache = lambda *args: None
    scheduler._truncate_cache_to_prompt_length = lambda caches, length: caches
    scheduler.model = object()
    entered = threading.Event()
    release = threading.Event()
    cleanup_returned = threading.Event()
    observed = {}
    original_write = manager._write_cache

    def delayed_write(*args, **kwargs):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test did not release SSD writer")
        if fail_write:
            raise OSError("simulated SSD write failure")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(manager, "_write_cache", delayed_write)

    def observe_and_release():
        observed["writer_entered"] = entered.wait(5)
        observed["returned_before_write"] = cleanup_returned.wait(0.2)
        release.set()

    observer = threading.Thread(target=observe_and_release)
    observer.start()
    try:
        scheduler._cleanup_finished({request.request_id})
        cleanup_returned.set()
        observer.join(6)
        assert not observer.is_alive()
        assert observed["writer_entered"]
        assert not observed["returned_before_write"], "terminal cleanup returned while SSD write was blocked"
        # Use a fresh manager: the next request cannot depend on resident state.
        reader = DiskCacheManager(str(tmp_path), max_size_gb=0.1)
        try:
            restored = reader.fetch(request.prompt_token_ids, cache_extra_keys=request._cache_extra_keys)
            if fail_write:
                assert restored is None
            else:
                assert restored is not None
                assert restored[0].keys.dtype == mx.bfloat16
                assert restored[0].offset == cache.offset
                assert mx.array_equal(restored[0].keys, cache.keys[..., :cache.offset, :]).item()
        finally:
            reader.shutdown()
        outcome = LEDGER.take(request.request_id)
        assert outcome["durable"] is (not fail_write)
        assert outcome["outcome"] == ("failed" if fail_write else "stored")
    finally:
        release.set()
        observer.join(6)
        manager.shutdown()
        LEDGER.take(request.request_id)
