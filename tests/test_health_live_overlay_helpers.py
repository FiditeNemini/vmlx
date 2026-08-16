# SPDX-License-Identifier: Apache-2.0
"""Pins for the cached-health live-overlay helpers (campaign #181 rows 88/90).

The /health snapshot cache serves a pre-request payload while any request is
running or lingering; these helpers supply the LIVE records the cached branch
overlays. Attribute-read-only discipline — nothing here may call get_stats."""

from types import SimpleNamespace

from vmlx_engine.server import (
    _live_last_cache_execution,
    _live_ssm_prefix_lookup,
)


def test_live_lce_prefers_generator_record_then_admission():
    gen_stats = SimpleNamespace(
        last_cache_execution={"request_id": "gen", "cache_outcome": "hit"}
    )
    scheduler = SimpleNamespace(
        batch_generator=SimpleNamespace(_stats=gen_stats),
        _admission_cache_execution={"request_id": "adm"},
    )
    assert _live_last_cache_execution(scheduler)["request_id"] == "gen"
    gen_stats.last_cache_execution = None
    assert _live_last_cache_execution(scheduler)["request_id"] == "adm"


def test_live_lce_falls_back_to_text_scheduler_record_and_none():
    scheduler = SimpleNamespace(
        batch_generator=None,
        _last_cache_execution={"request_id": "txt"},
    )
    assert _live_last_cache_execution(scheduler)["request_id"] == "txt"
    assert _live_last_cache_execution(None) is None
    assert (
        _live_last_cache_execution(SimpleNamespace(batch_generator=None))
        is None
    )


def test_live_ssm_lookup_resolves_scheduler_then_generator_cache():
    lookup = {"request_id": "r1", "matched": True}
    scheduler = SimpleNamespace(
        _ssm_state_cache=SimpleNamespace(last_prefix_lookup=lookup)
    )
    assert _live_ssm_prefix_lookup(scheduler) == lookup
    scheduler2 = SimpleNamespace(
        _ssm_state_cache=None,
        batch_generator=SimpleNamespace(
            _ssm_state_cache=SimpleNamespace(last_prefix_lookup=lookup)
        ),
    )
    assert _live_ssm_prefix_lookup(scheduler2) == lookup
    assert _live_ssm_prefix_lookup(None) is None
