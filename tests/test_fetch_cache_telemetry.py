# SPDX-License-Identifier: Apache-2.0
"""Regression tests for BlockAwarePrefixCache.fetch_cache() match telemetry.

Task #23: cached_tokens alone cannot say which branch produced it (a chain-hash
block hit, the N-1 partial-index fallback, a DSV4 delta-checkpoint
normalization, a rotating-SWA anchor normalization, or a genuine miss). These
tests prove the telemetry recorded at every fetch_cache() exit point is
server-exact and, most importantly, that recording it never changes
fetch_cache()'s own return values or counters -- see
test_telemetry_recording_never_changes_fetch_cache_behavior below.

DSV4-delta and rotating-SWA-normalization telemetry are covered where those
branches are already exercised: test_dsv4_native_shadow_parent_extends_and_reconstructs_all_layers
(tests/test_dsv4_delta_transport.py) and
test_rotating_pending_boundary_walks_back_to_an_earlier_anchor /
test_rotating_pending_boundary_with_no_anchor_still_goes_cold
(tests/test_paged_cache.py).

Coverage gap, noted rather than faked: the "prefix_index_match" origin (the
`elif best_match:` branch, i.e. BlockAwarePrefixCache's own `_prefix_index`
dict winning over PagedCacheManager.get_computed_blocks()) is implemented and
telemetry-instrumented, but every constructible scenario tried during this
change also resolves through get_computed_blocks()'s own inline
trailing-partial-block fallback, so no isolated positive test for that
specific origin value is included here.
"""

import platform
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


def _new_cache(block_size=4, max_blocks=8):
    from vmlx_engine.paged_cache import PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    paged = PagedCacheManager(block_size=block_size, max_blocks=max_blocks)
    cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged)
    return paged, cache


def _tensor_snapshot(mx, num_tokens):
    """A real (single-layer KVCache) tensor payload, not a plain-string stub.

    Plain string/opaque payloads like ``["state"]`` (used throughout the rest
    of this test suite) never populate ``block.cache_data`` at all --
    ``_store_cache_impl`` only extracts tensor slices when the payload looks
    like ``[{"state": ..., ...}, ...]`` -- so they cannot exercise the
    memory_l1 vs disk_l2 telemetry source distinction, which reads
    ``block.cache_data``.
    """
    keys = mx.arange(num_tokens * 2, dtype=mx.float32).reshape(1, 1, num_tokens, 2)
    values = (keys + 1000).astype(mx.float32)
    return [{"class_name": "KVCache", "state": (keys, values), "meta_state": ()}]


class TestFetchCacheTelemetryTaxonomy:
    def test_chain_block_hit_records_correct_token_count(self):
        mx = pytest.importorskip("mlx.core")
        paged, cache = _new_cache()
        tokens = list(range(8))  # exactly two full blocks
        cache.store_cache("writer", tokens, _tensor_snapshot(mx, len(tokens)))

        block_table, remaining = cache.fetch_cache("reader", tokens + [99])

        assert block_table is not None
        assert remaining == [99]

        telemetry = cache.get_stats()["last_fetch_telemetry"]
        assert telemetry["request_id"] == "reader"
        assert telemetry["match_kind"] == "chain_block_hit"
        assert telemetry["origin"] == "chain_block_hit"
        assert telemetry["logical_restored_tokens"] == block_table.num_tokens == 8
        assert telemetry["native_companion_boundary"] is None
        assert telemetry["dsv4_delta_applied"] is False
        assert telemetry["rotating_swa_normalized"] is False
        assert telemetry["source"] == "memory_l1"
        assert telemetry["miss_reason"] is None

    def test_genuine_miss_records_miss_match_kind(self):
        _paged, cache = _new_cache()

        block_table, remaining = cache.fetch_cache("reader", [1, 2, 3, 4, 5])

        assert block_table is None
        assert remaining == [1, 2, 3, 4, 5]

        telemetry = cache.get_stats()["last_fetch_telemetry"]
        assert telemetry["request_id"] == "reader"
        assert telemetry["match_kind"] == "miss"
        assert telemetry["origin"] is None
        assert telemetry["logical_restored_tokens"] == 0
        assert telemetry["miss_reason"] == "no_candidate"

    def test_empty_request_records_dedicated_match_kind(self):
        _paged, cache = _new_cache()

        block_table, remaining = cache.fetch_cache("reader", [])

        assert block_table is None
        assert remaining == []
        telemetry = cache.get_stats()["last_fetch_telemetry"]
        assert telemetry["match_kind"] == "empty_request"
        assert telemetry["logical_restored_tokens"] == 0

    def test_n_minus_1_partial_index_origin_recorded_when_alt_lookup_wins(self):
        """The alt (tokens[:-1]) lookup winning is the real N-1 branch.

        Store a 1-full-block + 2-token terminal partial. Clearing
        `_partial_block_sizes` after the store removes the ONLY thing that
        would let the full-N lookup's own trailing-partial fallback also find
        that 2-token terminal (which would make both lookups tie and hide the
        branch). The N-1 lookup finds it regardless, because it always tries
        the exact remaining length first.
        """
        paged, cache = _new_cache()
        stored_tokens = [10, 11, 12, 13, 14, 15]
        cache.store_cache("writer", stored_tokens, ["state"])
        paged._partial_block_sizes.clear()

        full_prompt_tokens = stored_tokens + [16]
        block_table, remaining = cache.fetch_cache("reader", full_prompt_tokens)

        assert block_table is not None
        assert block_table.num_tokens == 6
        assert remaining == [16]

        telemetry = cache.get_stats()["last_fetch_telemetry"]
        assert telemetry["request_id"] == "reader"
        assert telemetry["origin"] == "n_minus_1_partial_index"
        assert telemetry["match_kind"] == "n_minus_1_partial_index"
        assert telemetry["logical_restored_tokens"] == 6

    def test_recent_fetch_telemetry_is_bounded_and_ordered(self):
        _paged, cache = _new_cache(block_size=4, max_blocks=64)
        for i in range(5):
            tokens = [i * 100 + j for j in range(4)]
            cache.store_cache(f"writer-{i}", tokens, ["state"])
            cache.fetch_cache(f"reader-{i}", tokens + [999])

        recent = cache.get_stats()["recent_fetch_telemetry"]
        assert [r["request_id"] for r in recent] == [
            f"reader-{i}" for i in range(5)
        ]

    def test_visible_answer_continuation_is_labeled_and_excludable(self):
        """A base request's own fetch_cache() call must stay findable as
        "last_fetch_telemetry_excluding_continuations" even after a second,
        internal ":visible-answer" retry fetch overwrites plain
        last_fetch_telemetry (task #31 -- found live: a genuinely cold base
        request's real miss/0 result was masked by a later continuation
        fetch reporting an unrelated hit, and a naive last_fetch_telemetry
        reader had no way to tell them apart without string-matching
        request_id itself).
        """
        mx = pytest.importorskip("mlx.core")
        paged, cache = _new_cache()
        tokens = list(range(8))

        # Base request: genuine miss, nothing stored yet.
        cache.fetch_cache("chatcmpl-abc", tokens)
        base_telemetry = cache.get_stats()["last_fetch_telemetry"]
        assert base_telemetry["request_id"] == "chatcmpl-abc"
        assert base_telemetry["base_request_id"] == "chatcmpl-abc"
        assert base_telemetry["is_internal_continuation"] is False
        assert base_telemetry["match_kind"] == "miss"

        # Something else stores the prefix (e.g. the base pass's own prefill),
        # then the bounded visible-answer retry re-fetches it as a genuine hit.
        cache.store_cache("writer", tokens, _tensor_snapshot(mx, len(tokens)))
        cache.fetch_cache("chatcmpl-abc:visible-answer", tokens + [99])

        stats = cache.get_stats()
        assert stats["last_fetch_telemetry"]["request_id"] == (
            "chatcmpl-abc:visible-answer"
        )
        assert stats["last_fetch_telemetry"]["is_internal_continuation"] is True
        assert stats["last_fetch_telemetry"]["base_request_id"] == "chatcmpl-abc"

        # The convenience field must skip the continuation and surface the
        # base request's own real telemetry, not silently drop it either --
        # both entries stay present in recent_fetch_telemetry.
        excluding = stats["last_fetch_telemetry_excluding_continuations"]
        assert excluding["request_id"] == "chatcmpl-abc"
        assert excluding["match_kind"] == "miss"
        assert excluding["logical_restored_tokens"] == 0
        assert {r["request_id"] for r in stats["recent_fetch_telemetry"]} == {
            "chatcmpl-abc",
            "chatcmpl-abc:visible-answer",
        }

    def test_reset_stats_clears_fetch_telemetry(self):
        _paged, cache = _new_cache()
        cache.fetch_cache("reader", [1, 2, 3])
        assert cache.get_stats()["last_fetch_telemetry"] is not None

        cache.reset_stats()

        stats = cache.get_stats()
        assert stats["last_fetch_telemetry"] is None
        assert stats["recent_fetch_telemetry"] == []


class TestFetchCacheTelemetryNeverChangesBehavior:
    """The most important test: recording telemetry must be a pure side effect.

    For each representative scenario, run fetch_cache() twice on
    independently-constructed, identically-seeded caches -- once with the real
    `_record_fetch_telemetry`, once with it monkeypatched to a no-op -- and
    assert the returned block table, remaining tokens, and hit/miss counters
    are byte-for-byte identical. If telemetry recording ever influenced
    control flow (an exception escaping, a mutated shared list, an early
    return), this is what would catch it.
    """

    @staticmethod
    def _run(monkeypatch, *, scenario, disable_telemetry):
        paged, cache = _new_cache()
        if disable_telemetry:
            monkeypatch.setattr(cache, "_record_fetch_telemetry", lambda **_kw: None)
        block_table, remaining = scenario(paged, cache)
        summary = (
            None
            if block_table is None
            else (tuple(block_table.block_ids), block_table.num_tokens)
        )
        stats = cache.get_stats()
        return summary, remaining, stats["hits"], stats["misses"], stats["tokens_saved"]

    @staticmethod
    def _scenario_hit(paged, cache):
        tokens = list(range(8))
        cache.store_cache("writer", tokens, ["state"])
        return cache.fetch_cache("reader", tokens + [99])

    @staticmethod
    def _scenario_miss(paged, cache):
        return cache.fetch_cache("reader", [1, 2, 3, 4, 5])

    @staticmethod
    def _scenario_n_minus_1(paged, cache):
        stored_tokens = [10, 11, 12, 13, 14, 15]
        cache.store_cache("writer", stored_tokens, ["state"])
        paged._partial_block_sizes.clear()
        return cache.fetch_cache("reader", stored_tokens + [16])

    @staticmethod
    def _scenario_empty(paged, cache):
        return cache.fetch_cache("reader", [])

    @pytest.mark.parametrize(
        "scenario",
        [_scenario_hit, _scenario_miss, _scenario_n_minus_1, _scenario_empty],
        ids=["chain_block_hit", "miss", "n_minus_1_partial_index", "empty_request"],
    )
    def test_telemetry_recording_never_changes_fetch_cache_behavior(
        self, monkeypatch, scenario
    ):
        with_telemetry = self._run(monkeypatch, scenario=scenario, disable_telemetry=False)
        without_telemetry = self._run(monkeypatch, scenario=scenario, disable_telemetry=True)
        assert with_telemetry == without_telemetry
