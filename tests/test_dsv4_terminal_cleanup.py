from types import SimpleNamespace

import pytest

from vmlx_engine.request import Request, RequestStatus, SamplingParams
from vmlx_engine.scheduler import Scheduler


class DSV4BatchGenerator:
    """Minimal name-compatible generator used by the scheduler's DSV4 gate."""

    def __init__(self, events=None):
        self.events = events if events is not None else []
        self.live_uids = {7}
        self.stop_tokens = set()

    def remove(self, uids):
        assert all(uid in self.live_uids for uid in uids)
        self.events.append(("remove", tuple(uids)))
        self.live_uids.difference_update(uids)


class BatchGenerator(DSV4BatchGenerator):
    """A non-DSV4 generator must retain the existing cleanup behavior."""


def _cleanup_scheduler(monkeypatch, *, status, generator, bypass=True):
    import vmlx_engine.scheduler as scheduler_module

    monkeypatch.setattr(
        scheduler_module,
        "clear_mlx_memory_cache",
        lambda log=None: None,
    )
    monkeypatch.setattr(
        scheduler_module,
        "unregister_generation_logprobs",
        lambda _model, _uid: None,
    )

    request = Request(
        request_id="finished-dsv4",
        prompt=[10, 11],
        sampling_params=SamplingParams(max_tokens=4),
    )
    request.prompt_token_ids = list(request.prompt)
    request.num_prompt_tokens = len(request.prompt_token_ids)
    request.status = status
    request._bypass_prefix_cache = bypass
    request._gen_prompt_len = 0

    scheduler = object.__new__(Scheduler)
    scheduler.running = {request.request_id: request}
    scheduler.requests = {request.request_id: request}
    scheduler.request_id_to_uid = {request.request_id: 7}
    scheduler.uid_to_request_id = {7: request.request_id}
    scheduler.finished_req_ids = set()
    scheduler.batch_generator = generator
    scheduler.stop_tokens = set()
    scheduler.block_aware_cache = None
    scheduler.memory_aware_cache = None
    scheduler.prefix_cache = None
    scheduler.disk_cache = None
    scheduler.config = SimpleNamespace(enable_prefix_cache=True)
    scheduler._ssm_state_cache = None
    scheduler._kv_cache_bits = 0
    scheduler._is_hybrid = False
    scheduler._uses_dsv4_cache = True
    scheduler._uses_zaya_cache = False
    scheduler._mixed_attention_cache_model = False
    scheduler._pld_pending = {}
    scheduler._pld_ngram_indices = {}
    scheduler._cleanup_detokenizer = lambda _request_id: None
    scheduler._materialize_deferred_prompt_cache = (
        lambda _request_id, _request: None
    )
    scheduler._pick_cache_type_for_request = lambda _request: "user"
    scheduler.model = object()
    return scheduler, request


@pytest.mark.parametrize(
    "status",
    [RequestStatus.FINISHED_STOPPED, RequestStatus.FINISHED_LENGTH_CAPPED],
)
def test_dsv4_terminal_cleanup_removes_eos_and_length_requests(
    monkeypatch,
    status,
):
    generator = DSV4BatchGenerator()
    scheduler, request = _cleanup_scheduler(
        monkeypatch,
        status=status,
        generator=generator,
    )

    scheduler._cleanup_finished({request.request_id})

    assert generator.events == [("remove", (7,))]
    assert generator.live_uids == set()


def test_dsv4_cleanup_stores_snapshot_before_releasing_live_generator_cache(
    monkeypatch,
):
    events = []
    generator = DSV4BatchGenerator(events)
    scheduler, request = _cleanup_scheduler(
        monkeypatch,
        status=RequestStatus.FINISHED_STOPPED,
        generator=generator,
        bypass=False,
    )
    snapshot = [SimpleNamespace()]
    request._extracted_cache = snapshot
    request._extracted_cache_from_prompt_snapshot = True

    class _MemoryCache:
        def store(self, _tokens, cache, **_kwargs):
            assert cache is snapshot
            assert generator.live_uids == {7}
            events.append(("store", cache))
            return True

    scheduler.memory_aware_cache = _MemoryCache()
    scheduler._truncate_cache_to_prompt_length = (
        lambda cache, _prompt_len: cache
    )

    scheduler._cleanup_finished({request.request_id})

    assert events == [("store", snapshot), ("remove", (7,))]


def test_non_dsv4_cleanup_does_not_call_generator_remove(monkeypatch):
    generator = BatchGenerator()
    scheduler, request = _cleanup_scheduler(
        monkeypatch,
        status=RequestStatus.FINISHED_STOPPED,
        generator=generator,
    )
    scheduler._uses_dsv4_cache = False

    scheduler._cleanup_finished({request.request_id})

    assert generator.events == []
    assert generator.live_uids == {7}


def test_request_cache_bypass_disables_dsv4_snapshot_and_delta_capture(
    monkeypatch,
):
    import mlx.core as mx

    from vmlx_engine.utils.dsv4_batch_generator import DSV4BatchGenerator

    monkeypatch.setenv("DSV4_PROMPT_SNAPSHOT_MIN_TOKENS", "0")
    export_calls = []

    class DeepseekV4Cache:
        def export_block_delta(self, *args, **kwargs):
            export_calls.append((args, kwargs))
            raise AssertionError("cache bypass invoked export_block_delta")

    cache = DeepseekV4Cache()
    snapshot_calls = []

    class _Model:
        def make_cache(self):
            return [cache]

        def __call__(self, _ids, cache=None):
            return mx.array([[[0.0, 1.0, 0.0]]], dtype=mx.float32)

    generator = DSV4BatchGenerator(
        _Model(),
        capture_prompt_snapshot=True,
    )
    generator._warmed_up = True
    monkeypatch.setattr(
        generator,
        "_snapshot_admissible_dsv4_cache",
        lambda _cache: snapshot_calls.append("snapshot"),
    )
    generator.insert(
        [list(range(300))],
        max_tokens=[1],
        capture_prompt_snapshots=[False],
    )

    prompt_responses, _ = generator.next()

    assert prompt_responses[0].prompt_cache_snapshot is None
    assert snapshot_calls == []
    assert export_calls == []
    assert not hasattr(cache, "_vmlx_dsv4_block_records")


def test_scheduler_forwards_request_cache_bypass_to_dsv4_generator():
    import inspect

    source = inspect.getsource(Scheduler._schedule_waiting)

    assert 'insert_kwargs["capture_prompt_snapshots"]' in source
    assert 'getattr(request, "_bypass_prefix_cache", False)' in source


class _SalvageGenerator(DSV4BatchGenerator):
    """DSV4-gate generator that hands out one prompt snapshot for uid 7."""

    def __init__(self, events, snapshot):
        super().__init__(events)
        self._snapshot = snapshot

    def take_prompt_snapshots(self, uids):
        self.events.append(("take_snapshots", tuple(uids)))
        if self._snapshot is None:
            return {}
        return {7: (self._snapshot, [10])}


class _SalvageBlockCache:
    def __init__(self, events):
        self.events = events
        self._request_tables = {}
        self.paged_cache = SimpleNamespace(
            delete_block_table=lambda rid: events.append(
                ("delete_table", rid)
            ),
            release_request_refs=lambda table: events.append(
                ("release_refs", table)
            ),
            detach_request=lambda rid: events.append(("detach", rid)),
            max_resident_bytes=0,
        )

    def store_cache(self, request_id, tokens, cache_data, **kwargs):
        self.events.append(
            (
                "store_cache",
                request_id,
                tuple(tokens),
                kwargs.get("cache_type"),
            )
        )
        return None


def _salvage_scheduler(monkeypatch, *, bypass, snapshot):
    events = []
    generator = _SalvageGenerator(events, snapshot)
    scheduler, request = _cleanup_scheduler(
        monkeypatch,
        status=RequestStatus.RUNNING,
        generator=generator,
        bypass=bypass,
    )
    scheduler.waiting = []
    scheduler._pending_aborts = set()
    scheduler._abort_salvage_requests = {}
    scheduler.block_aware_cache = _SalvageBlockCache(events)
    scheduler._dsv4_trace_timing = lambda *_a, **_k: None
    scheduler._extract_cache_states = lambda cache: [{"state": cache}]
    request.cached_tokens = 0
    return scheduler, request, events


def test_abort_salvages_prompt_snapshot_into_prefix_cache(monkeypatch):
    snapshot = [SimpleNamespace()]
    scheduler, request, events = _salvage_scheduler(
        monkeypatch, bypass=False, snapshot=snapshot
    )

    assert scheduler.abort_request(request.request_id) is True
    assert request.request_id in scheduler._abort_salvage_requests

    scheduler._process_pending_aborts()

    assert events == [
        ("delete_table", request.request_id),
        ("take_snapshots", (7,)),
        ("store_cache", request.request_id, (10,), "user"),
        ("release_refs", None),
        ("detach", request.request_id),
        ("remove", (7,)),
    ]
    assert scheduler._abort_salvage_requests == {}
    assert scheduler.block_aware_cache._request_tables == {}


def test_abort_salvage_skipped_for_cache_bypass_requests(monkeypatch):
    scheduler, request, events = _salvage_scheduler(
        monkeypatch, bypass=True, snapshot=[SimpleNamespace()]
    )

    assert scheduler.abort_request(request.request_id) is True
    assert scheduler._abort_salvage_requests == {}

    scheduler._process_pending_aborts()

    assert ("remove", (7,)) in events
    assert not any(e[0] == "store_cache" for e in events)
    assert not any(e[0] == "take_snapshots" for e in events)


def test_abort_salvage_skipped_when_no_snapshot_available(monkeypatch):
    scheduler, request, events = _salvage_scheduler(
        monkeypatch, bypass=False, snapshot=None
    )

    assert scheduler.abort_request(request.request_id) is True
    scheduler._process_pending_aborts()

    assert ("take_snapshots", (7,)) in events
    assert ("remove", (7,)) in events
    assert not any(e[0] == "store_cache" for e in events)


def test_abort_salvage_skipped_when_key_fully_cached(monkeypatch):
    snapshot = [SimpleNamespace()]
    scheduler, request, events = _salvage_scheduler(
        monkeypatch, bypass=False, snapshot=snapshot
    )
    request.cached_tokens = 1

    assert scheduler.abort_request(request.request_id) is True
    scheduler._process_pending_aborts()

    assert ("take_snapshots", (7,)) in events
    assert not any(e[0] == "store_cache" for e in events)
    assert ("remove", (7,)) in events
