# SPDX-License-Identifier: Apache-2.0
"""The reconstruct memo must decline when a second copy will not fit.

The memo turns a repeated reconstruction into a lookup (3071ms -> 0.015ms at
413 blocks on the text path), but it pays with a full ``deepcopy``. That is
cheap for a compact DSV4 delta record and ruinous for a mixed-SWA L1, which is
~4.6GB at 86k tokens — retaining a second copy would roughly DOUBLE resident KV
at exactly the depth where memory is already tightest, and an OOM costs far more
than the reconstruction it was avoiding.

So the memo is opportunistic, and every uncertain case must decline: the fast
path's failure mode is an OOM, so the safe default is the slower one.
"""

import pytest

from vmlx_engine.prefix_cache import BlockAwarePrefixCache
from vmlx_engine.paged_cache import PagedCacheManager

GIB = 1024**3


@pytest.fixture
def cache():
    return BlockAwarePrefixCache(
        model=None, paged_cache_manager=PagedCacheManager(block_size=64, max_blocks=64)
    )


def _patch(monkeypatch, *, active, max_ws, copy_bytes):
    monkeypatch.setattr(
        "vmlx_engine.utils.memory_limits.get_effective_metal_working_set_bytes",
        lambda _mx=None: (active, max_ws),
    )
    monkeypatch.setattr(
        "vmlx_engine.memory_cache.estimate_kv_cache_memory",
        lambda _caches: copy_bytes,
    )


def test_declines_when_the_copy_would_break_the_budget(cache, monkeypatch):
    # 4.6GB copy on top of 70GB active, against a 100GB limit -> over the 70% budget
    _patch(monkeypatch, active=70 * GIB, max_ws=100 * GIB, copy_bytes=int(4.6 * GIB))
    assert cache._reconstruct_memo_fits_in_headroom(object()) is False


def test_allows_when_there_is_comfortable_room(cache, monkeypatch):
    _patch(monkeypatch, active=10 * GIB, max_ws=100 * GIB, copy_bytes=int(4.6 * GIB))
    assert cache._reconstruct_memo_fits_in_headroom(object()) is True


@pytest.mark.parametrize(
    "active,max_ws,copy_bytes",
    [
        (0, 100 * GIB, GIB),      # active unreadable
        (10 * GIB, 0, GIB),       # limit unknown
        (10 * GIB, 100 * GIB, 0), # size unknown
    ],
)
def test_unknown_readings_decline(cache, monkeypatch, active, max_ws, copy_bytes):
    """Declining costs a reconstruction; guessing wrong costs an OOM."""
    _patch(monkeypatch, active=active, max_ws=max_ws, copy_bytes=copy_bytes)
    assert cache._reconstruct_memo_fits_in_headroom(object()) is False


def test_budget_percentage_is_configurable(cache, monkeypatch):
    _patch(monkeypatch, active=60 * GIB, max_ws=100 * GIB, copy_bytes=15 * GIB)
    # 75GB projected: over the default 70% budget...
    assert cache._reconstruct_memo_fits_in_headroom(object()) is False
    # ...and under an explicitly raised one.
    monkeypatch.setenv("VMLX_RECONSTRUCT_MEMO_MAX_WS_PCT", "90")
    assert cache._reconstruct_memo_fits_in_headroom(object()) is True


def test_estimator_failure_declines_rather_than_raising(cache, monkeypatch):
    monkeypatch.setattr(
        "vmlx_engine.utils.memory_limits.get_effective_metal_working_set_bytes",
        lambda _mx=None: (10 * GIB, 100 * GIB),
    )

    def _boom(_caches):
        raise RuntimeError("cannot size this cache")

    monkeypatch.setattr("vmlx_engine.memory_cache.estimate_kv_cache_memory", _boom)
    assert cache._reconstruct_memo_fits_in_headroom(object()) is False
