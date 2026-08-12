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


def test_the_policy_rule_exists_once():
    """Only an EXPLICIT zero counts, and disk-only keeps its own meaning."""
    from vmlx_engine.memory_cache import resolve_paged_resident_policy
    import inspect

    src = inspect.getsource(resolve_paged_resident_policy)
    assert "config.cache_memory_mb is not None" in src
    assert "int(config.cache_memory_mb) == 0" in src
    assert "if disk_only:" in src


def test_the_resolver_returns_frugal_only_for_an_explicit_zero():
    from types import SimpleNamespace
    from vmlx_engine.memory_cache import resolve_paged_resident_policy

    def cfg(mb):
        return SimpleNamespace(cache_memory_mb=mb, cache_memory_percent=0.15)

    _, frugal = resolve_paged_resident_policy(cfg(0), disk_only=False)
    assert frugal is True, "explicit 0 must engage frugal"

    _, frugal = resolve_paged_resident_policy(cfg(None), disk_only=False)
    assert frugal is False, "None means auto-detect, not frugal"

    _, frugal = resolve_paged_resident_policy(cfg(4096), disk_only=False)
    assert frugal is False

    budget, frugal = resolve_paged_resident_policy(cfg(0), disk_only=True)
    assert (budget, frugal) == (0, False), "disk-only keeps its own zero meaning"


def test_BOTH_schedulers_use_the_shared_resolver():
    """The first version of this fix lived only in the text scheduler.

    MEASURED consequence: Gemma 4 (a VLM, hence the MLLM scheduler) still
    reported paged_frugal=False / ram_mirror_policy=resident on /health with
    --cache-memory-mb 0. Every unit test passed; the fix was inert on half the
    engine. Both paths must call the one resolver AND pass frugal through.
    """
    for name in ("scheduler.py", "mllm_scheduler.py"):
        source = (ROOT / "vmlx_engine" / name).read_text()
        assert "resolve_paged_resident_policy(" in source, name
        assert re.search(r"frugal=_\w*explicit_zero_cache,", source), name
        # and neither may rebuild the budget inline again
        assert "MemoryCacheConfig as _MemCacheCfg" not in source, name


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


def test_frugal_makes_byte_budget_enforcement_REACHABLE():
    """Frugal with a zero ceiling must still enforce — this was the hole.

    The native path-dependent families (DSV4, ZAYA CCA, rotating/mixed-SWA)
    override frugal with `keep_in_ram` so their composite state survives until
    its async L2 write is readable (prefix_cache.py). That pin is meant to be
    temporary and is released once L2 confirms — but if enforcement is
    unreachable, nothing ever runs the pass that releases it. The user asked
    for zero resident bytes and got an unbounded, unaccounted mirror on exactly
    the families with the largest per-block state.
    """
    pool = _manager(max_resident_bytes=0, frugal=True)
    assert pool.paged_frugal is True
    assert pool.enforces_byte_budget is True, (
        "explicit zero-cache must still run the byte-budget/pressure pass"
    )


def test_plain_zero_without_frugal_is_still_the_inert_sentinel():
    """Guard the boundary: only frugal/disk-only lift the zero ceiling."""
    pool = _manager(max_resident_bytes=0, frugal=False)
    assert pool.enforces_byte_budget is False


def test_enforcement_reachability_covers_all_three_reasons():
    from vmlx_engine.paged_cache import PagedCacheManager
    import inspect

    src = inspect.getsource(PagedCacheManager).split("def enforces_byte_budget")[1]
    src = src.split("def ")[0]
    assert "self.max_resident_bytes > 0" in src
    assert "self.disk_only" in src
    assert "self.paged_frugal" in src
