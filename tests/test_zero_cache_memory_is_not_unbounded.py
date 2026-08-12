# SPDX-License-Identifier: Apache-2.0
"""`--cache-memory-mb 0` must not buy an UNBOUNDED resident cache.

PagedCacheManager documents `max_resident_bytes == 0` as the legacy "unbounded"
sentinel. MemoryAwarePrefixCache reads the same 0 as "store nothing" (entry cap
0 * 0.95). So one user-facing flag had opposite meanings per tier, and the user
asking for the SMALLEST possible cache got the LARGEST one: no byte ceiling,
and every `enforces_byte_budget`-gated call site skipping.

The request now routes to the frugal policy, which is what "no resident RAM
payloads" already means in this file.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _manager(**kwargs):
    from vmlx_engine.paged_cache import PagedCacheManager

    defaults = dict(block_size=16, max_blocks=8, disk_store=None)
    defaults.update(kwargs)
    return PagedCacheManager(**defaults)


def test_zero_bytes_alone_is_still_the_unbounded_sentinel():
    """Documents the sentinel this fix routes AROUND rather than redefining."""
    pool = _manager(max_resident_bytes=0)
    assert pool.max_resident_bytes == 0
    assert pool.paged_frugal is False
    # This is the shape that made an explicit 0 dangerous.
    assert pool.enforces_byte_budget is False


def test_frugal_suppresses_persistent_payloads():
    pool = _manager(max_resident_bytes=0, frugal=True)
    assert pool.paged_frugal is True
    assert pool.ram_mirror_policy == "frugal_config"


def test_frugal_is_reported_on_the_stats_surface():
    """The user must be able to SEE which RAM policy is in force.

    get_memory_usage() is the dict that feeds /health and the cache pill;
    get_stats() returns a CacheStats object and carries counters, not policy.
    """
    usage = _manager(max_resident_bytes=0, frugal=True).get_memory_usage()
    assert usage["paged_frugal"] is True
    assert usage["ram_mirror_policy"] == "frugal_config"
    # Not disk-only: the SSD tier is a separate choice.
    assert usage["disk_only"] is False
    assert usage["paged_ram_enabled"] is True


def test_env_override_still_works_and_does_not_need_the_parameter(monkeypatch):
    monkeypatch.setenv("VMLX_PAGED_FRUGAL", "1")
    pool = _manager(max_resident_bytes=1 << 20)
    assert pool.paged_frugal is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_env_off_values_do_not_engage_frugal(monkeypatch, value):
    monkeypatch.setenv("VMLX_PAGED_FRUGAL", value)
    pool = _manager(max_resident_bytes=1 << 20)
    assert pool.paged_frugal is False


def test_scheduler_routes_an_explicit_zero_to_frugal():
    source = (ROOT / "vmlx_engine" / "scheduler.py").read_text()
    start = source.index("_explicit_zero_cache = (")
    block = source[start : start + 900]
    # Only an EXPLICIT zero counts — None means "auto-detect from percent".
    assert "self.config.cache_memory_mb is not None" in block
    assert "int(self.config.cache_memory_mb) == 0" in block
    # And disk-only already has its own zero-budget meaning; don't double up.
    assert "not _block_disk_only" in block
    assert "frugal=_explicit_zero_cache," in source


def test_disk_only_still_reports_disk_only_not_frugal():
    """disk_only implies frugal but must keep its own, more specific label."""
    from vmlx_engine.paged_cache import PagedCacheManager

    src = re.search(
        r"self\.ram_mirror_policy = \(\s*(.*?)\s*\)",
        Path(PagedCacheManager.__init__.__code__.co_filename).read_text(),
        re.S,
    )
    assert src is not None
    assert '"disk_only"' in src.group(1)
    assert src.group(1).index('"disk_only"') < src.group(1).index('"frugal_env"')
