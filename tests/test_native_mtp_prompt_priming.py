from __future__ import annotations

import threading
from collections import OrderedDict
from types import SimpleNamespace

import mlx.core as mx

from vmlx_engine.mllm_batch_generator import MLLMNativeMTPStats
from vmlx_engine.native_mtp_prompt_priming import (
    capture_prefill,
    prepare_prompt,
    prime_stats,
    take_primed,
)
from vmlx_engine.prefix_cache import BlockAwarePrefixCache


class _Cache:
    def __init__(self, offset: int = 0):
        self.offset = offset
        self.keys = mx.zeros((1, 1, offset, 1)) if offset else None
        self.values = mx.zeros((1, 1, offset, 1)) if offset else None

    def append(self, count: int) -> None:
        self.offset += count
        self.keys = mx.zeros((1, 1, self.offset, 1))
        self.values = mx.zeros((1, 1, self.offset, 1))

    def trim(self, count: int) -> int:
        count = min(max(0, int(count)), self.offset)
        self.offset -= count
        self.keys = (
            mx.zeros((1, 1, self.offset, 1)) if self.offset else None
        )
        self.values = (
            mx.zeros((1, 1, self.offset, 1)) if self.offset else None
        )
        return count


class _Host:
    def __init__(self):
        self.mtp = object()
        self.calls: list[list[int]] = []

    def make_mtp_cache(self):
        return [_Cache()]

    def mtp_forward(self, hidden, tokens, cache):
        del hidden
        ids = [int(token) for token in tokens.reshape(-1).tolist()]
        self.calls.append(ids)
        cache[0].append(len(ids))
        return mx.zeros((1, len(ids), 8))


class _SidecarStore:
    block_size = 2

    def __init__(self):
        self.snapshot = None

    def store_mtp_prefix_snapshot(self, tokens, boundary, snapshot, **kwargs):
        del tokens, kwargs
        assert boundary == snapshot.boundary_tokens
        self.snapshot = snapshot
        return True

    def restore_mtp_prefix_snapshot(self, tokens, boundary, **kwargs):
        del tokens, kwargs
        if self.snapshot is not None and self.snapshot.boundary_tokens == boundary:
            return self.snapshot
        return None


def test_cold_prompt_is_folded_and_published_at_full_block_boundary():
    host = _Host()
    sidecar = _SidecarStore()
    assert not prepare_prompt(
        host,
        request_id="cold",
        prompt_tokens=[1, 2, 3, 4],
        cached_tokens=0,
        prefix_cache=sidecar,
    )

    backbone = [_Cache(offset=4)]
    capture_prefill(
        host,
        mx.array([[1, 2, 3, 4]]),
        mx.arange(8).reshape(1, 4, 2),
        backbone,
    )
    assert host.calls == [[2, 3, 4]]
    assert prime_stats(host) == {
        "active": True,
        "folded_pairs": 3,
        "window_exceeded": False,
    }

    backbone[0].offset = 5  # the first ordinary decode/seed forward
    primed = take_primed(host, backbone, mx.array([5]))
    assert primed is not None
    mtp_cache, folded = primed
    assert folded == 4
    assert mtp_cache[0].offset == 4
    assert host.calls[-1] == [5]
    assert sidecar.snapshot is not None
    assert sidecar.snapshot.boundary_tokens == 4
    assert sidecar.snapshot.mtp_cache[0].offset == 3


def test_warm_sidecar_continues_only_the_uncached_tail():
    source = _Host()
    sidecar = _SidecarStore()
    prepare_prompt(
        source,
        request_id="source",
        prompt_tokens=[1, 2, 3, 4],
        cached_tokens=0,
        prefix_cache=sidecar,
    )
    source_backbone = [_Cache(offset=4)]
    capture_prefill(
        source,
        mx.array([[1, 2, 3, 4]]),
        mx.zeros((1, 4, 2)),
        source_backbone,
    )
    source_backbone[0].offset = 5
    assert take_primed(source, source_backbone, mx.array([5])) is not None

    warm = _Host()
    assert prepare_prompt(
        warm,
        request_id="warm",
        prompt_tokens=[1, 2, 3, 4, 6, 7],
        cached_tokens=4,
        prefix_cache=sidecar,
    )
    backbone = [_Cache(offset=6)]
    capture_prefill(
        warm,
        mx.array([[6, 7]]),
        mx.ones((1, 2, 2)),
        backbone,
    )
    assert warm.calls == [[6, 7]]
    backbone[0].offset = 7
    primed = take_primed(warm, backbone, mx.array([8]))
    assert primed is not None
    assert primed[0][0].offset == 6
    assert primed[1] == 6
    assert warm.calls[-1] == [8]


def test_warm_backbone_without_sidecar_does_not_invent_tail_only_history():
    host = _Host()
    sidecar = _SidecarStore()
    assert not prepare_prompt(
        host,
        request_id="warm-miss",
        prompt_tokens=[1, 2, 3, 4, 5, 6],
        cached_tokens=4,
        prefix_cache=sidecar,
    )
    backbone = [_Cache(offset=6)]
    capture_prefill(
        host,
        mx.array([[5, 6]]),
        mx.ones((1, 2, 2)),
        backbone,
    )
    assert host.calls == []
    backbone[0].offset = 7
    assert take_primed(host, backbone, mx.array([7])) is None


def test_cold_priming_does_not_depend_on_prefix_cache_being_enabled():
    host = _Host()
    assert not prepare_prompt(
        host,
        request_id="no-cache",
        prompt_tokens=[1, 2, 3],
        cached_tokens=0,
        prefix_cache=None,
    )
    backbone = [_Cache(offset=3)]
    capture_prefill(
        host,
        mx.array([[1, 2, 3]]),
        mx.zeros((1, 3, 2)),
        backbone,
    )
    backbone[0].offset = 4
    primed = take_primed(host, backbone, mx.array([4]))
    assert primed is not None
    assert primed[1] == 3


def test_seam_mismatch_fails_closed_instead_of_using_wrong_history():
    host = _Host()
    sidecar = _SidecarStore()
    prepare_prompt(
        host,
        request_id="rewind",
        prompt_tokens=[1, 2, 3, 4],
        cached_tokens=0,
        prefix_cache=sidecar,
    )
    backbone = [_Cache(offset=4)]
    capture_prefill(
        host,
        mx.array([[1, 2, 3, 4]]),
        mx.zeros((1, 4, 2)),
        backbone,
    )
    backbone[0].offset = 99
    assert take_primed(host, backbone, mx.array([5])) is None
    assert host.calls == [[2, 3, 4]]


def test_block_cache_sidecar_is_aligned_bounded_and_requires_live_tip():
    cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
    cache.block_size = 2
    cache._mtp_prefix_snapshots = OrderedDict()
    cache._mtp_prefix_snapshot_lock = threading.RLock()
    live = SimpleNamespace(get_block=lambda _key: object())
    cache.paged_cache = SimpleNamespace(cached_block_hash_to_block=live)

    marker = object()
    assert not cache.store_mtp_prefix_snapshot([1, 2, 3], 3, marker)
    assert cache.store_mtp_prefix_snapshot([1, 2, 3, 4], 4, marker)
    assert cache.restore_mtp_prefix_snapshot([1, 2, 3, 4], 4) is marker

    cache.paged_cache.cached_block_hash_to_block = SimpleNamespace(
        get_block=lambda _key: None
    )
    assert cache.restore_mtp_prefix_snapshot([1, 2, 3, 4], 4) is None


def test_native_mtp_stats_expose_prompt_priming_provenance():
    stats = MLLMNativeMTPStats(
        prompt_primed_pairs=127,
        prompt_prime_source="restored_prefix_and_tail",
    )
    payload = stats.to_dict(
        request_id="r",
        finish_reason="stop",
        final_depth=3,
    )
    assert payload["prompt_priming"] == {
        "source": "restored_prefix_and_tail",
        "folded_pairs": 127,
    }
