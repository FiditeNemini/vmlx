# SPDX-License-Identifier: Apache-2.0
"""The reconstruct memo must budget against HEADROOM, not total active memory.

The memo turns the answer pass's repeated reconstruction into a lookup. It was
declining on every single turn, and the reason only became visible once the
decline was logged at INFO. Measured live, DSV4-Flash on the box:

    Reconstruct memo DECLINED: copy 0.54GB on top of active 95.44GB would pass
    the 70% working-set budget (75.26GB)

`active` includes the resident MODEL, so it was already ~20GB past the ceiling
before the copy was considered. `active + copy > ceiling` was therefore true for
any copy size at all — the memo could never be retained once a large model was
loaded, which is precisely when it is worth having. Meanwhile the answer pass
re-read the whole prefix from L2 every turn: 7.5s at 83k tokens, 21.7s at 166k,
reproducible to ~0.3s across runs.

The question that matters is whether the copy fits in what is LEFT.

Created by Jinho Jang (eric@jangq.ai).
"""

from __future__ import annotations

import pytest

import vmlx_engine.prefix_cache as pc

_GB = 1024**3


class _Manager:
    """Just enough object to call the predicate as a bound method."""

    _reconstruct_memo_fits_in_headroom = pc.BlockAwarePrefixCache._reconstruct_memo_fits_in_headroom


class _ArmManager:
    arm_reconstruct_memo = pc.BlockAwarePrefixCache.arm_reconstruct_memo

    def __init__(self, *, disk_only):
        self.paged_cache = type("Paged", (), {"disk_only": disk_only})()
        self._reconstruct_memo_arm = False
        self._reconstruct_memo = None


@pytest.fixture
def manager():
    return _Manager()


def _patch(monkeypatch, *, active, max_ws, copy_bytes):
    import vmlx_engine.utils.memory_limits as ml
    import vmlx_engine.memory_cache as mc

    monkeypatch.setattr(
        ml,
        "get_effective_metal_working_set_bytes",
        lambda mx_module=None: (active, max_ws),
    )
    monkeypatch.setattr(mc, "estimate_kv_cache_memory", lambda _caches: copy_bytes)


def test_ssd_only_mode_refuses_and_drops_reconstruct_memo(monkeypatch):
    """A full deep-copy memo is retained cache RAM, even if paged RAM is off."""
    monkeypatch.delenv("VMLX_DSV4_RECONSTRUCT_MEMO", raising=False)
    manager = _ArmManager(disk_only=True)
    manager._reconstruct_memo = ((1, 2), ["hidden-kv-copy"])

    manager.arm_reconstruct_memo(True)

    assert manager._reconstruct_memo_arm is False
    assert manager._reconstruct_memo is None


def test_ram_backed_mode_can_still_arm_reconstruct_memo(monkeypatch):
    monkeypatch.delenv("VMLX_DSV4_RECONSTRUCT_MEMO", raising=False)
    manager = _ArmManager(disk_only=False)

    manager.arm_reconstruct_memo(True)

    assert manager._reconstruct_memo_arm is True


def test_a_small_copy_is_kept_even_with_a_huge_model_resident(manager, monkeypatch):
    """The exact live reading that was being refused."""
    _patch(
        monkeypatch,
        active=int(95.44 * _GB),
        max_ws=int(107.5 * _GB),
        copy_bytes=int(0.54 * _GB),
    )
    # ~12GB of headroom for a 0.54GB copy. Refusing this is what left a 21.7s
    # stall on the table.
    assert manager._reconstruct_memo_fits_in_headroom(object()) is True


def test_the_larger_live_copy_also_fits(manager, monkeypatch):
    _patch(
        monkeypatch,
        active=int(95.90 * _GB),
        max_ws=int(107.5 * _GB),
        copy_bytes=int(1.07 * _GB),
    )
    assert manager._reconstruct_memo_fits_in_headroom(object()) is True


def test_a_copy_that_would_eat_the_remaining_headroom_is_refused(manager, monkeypatch):
    """The guard still has to guard: an OOM costs more than the replay."""
    _patch(
        monkeypatch,
        active=int(100 * _GB),
        max_ws=int(107.5 * _GB),
        # 7.5GB of headroom; a 6GB copy is over the 50% share.
        copy_bytes=int(6 * _GB),
    )
    assert manager._reconstruct_memo_fits_in_headroom(object()) is False


def test_no_headroom_at_all_is_refused(manager, monkeypatch):
    _patch(
        monkeypatch,
        active=int(110 * _GB),
        max_ws=int(107.5 * _GB),
        copy_bytes=int(0.1 * _GB),
    )
    assert manager._reconstruct_memo_fits_in_headroom(object()) is False


def test_the_share_of_headroom_is_configurable(manager, monkeypatch):
    _patch(
        monkeypatch,
        active=int(100 * _GB),
        max_ws=int(110 * _GB),
        copy_bytes=int(8 * _GB),
    )
    # 10GB headroom, 8GB copy: refused at the 50% default...
    assert manager._reconstruct_memo_fits_in_headroom(object()) is False
    # ...allowed when the operator raises the share.
    monkeypatch.setenv("VMLX_RECONSTRUCT_MEMO_MAX_WS_PCT", "90")
    assert manager._reconstruct_memo_fits_in_headroom(object()) is True


def test_unreadable_numbers_still_decline(manager, monkeypatch):
    """The fast path's failure mode is an OOM, so unknown means no."""
    _patch(monkeypatch, active=0, max_ws=0, copy_bytes=int(1 * _GB))
    assert manager._reconstruct_memo_fits_in_headroom(object()) is False

    _patch(monkeypatch, active=int(10 * _GB), max_ws=int(100 * _GB), copy_bytes=0)
    assert manager._reconstruct_memo_fits_in_headroom(object()) is False
