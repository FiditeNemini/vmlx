"""The SSD block-cache budget must never reach a scheduler as None.

`--block-disk-cache-max-gb` defaults to None so that a percent of the volume can
size the cache instead. That default is only safe if it is collapsed to a number
before anything reads it, and the failure when it is not is severe and
non-obvious:

    BlockDiskStore(max_size_gb=None) -> "'>' not supported between instances of
    'NoneType' and 'float'"

which the generic path swallows ("Continuing without disk cache") and the
disk-ONLY path — the shipping default since v1.6.35 — turns into a refusal to
start at all. Observed live on DeepSeek-V4-Flash: the startup summary printed
`max=NoneGB` and the engine exited during lifespan startup.

Resolving inside one helper was not enough; these tests pin the normalization at
the single parse point and its idempotency, because the original bug was a fix
that ran on one path while other paths kept reading the raw attribute.
"""
import argparse

import pytest

from vmlx_engine.cli import (
    DEFAULT_BLOCK_DISK_CACHE_PERCENT,
    normalize_block_disk_cache_budget,
    resolve_block_disk_cache_max_gb,
)


def _args(**kw):
    ns = argparse.Namespace(
        block_disk_cache_max_gb=None,
        block_disk_cache_max_percent=DEFAULT_BLOCK_DISK_CACHE_PERCENT,
        block_disk_cache_dir=None,
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_default_budget_is_a_number_not_none():
    args = _args()
    normalize_block_disk_cache_budget(args)
    assert args.block_disk_cache_max_gb is not None
    assert isinstance(args.block_disk_cache_max_gb, float)
    # The comparison BlockDiskStore performs, which is what raised live.
    assert args.block_disk_cache_max_gb > 0.0


def test_default_budget_is_the_configured_percent_of_the_volume(tmp_path):
    import shutil

    args = _args(block_disk_cache_dir=str(tmp_path))
    normalize_block_disk_cache_budget(args)
    total_gb = shutil.disk_usage(str(tmp_path)).total / (1024 ** 3)
    expected = total_gb * DEFAULT_BLOCK_DISK_CACHE_PERCENT / 100.0
    assert args.block_disk_cache_max_gb == pytest.approx(expected, rel=0.02)
    assert args.block_disk_cache_sized_by_percent is True


def test_explicit_gb_wins_over_percent():
    args = _args(block_disk_cache_max_gb=42.0, block_disk_cache_max_percent=90.0)
    normalize_block_disk_cache_budget(args)
    assert args.block_disk_cache_max_gb == 42.0
    assert args.block_disk_cache_sized_by_percent is False


def test_explicit_zero_gb_is_unlimited_not_a_percent():
    """0 GB means unlimited and must not be re-read as "unset, use the percent"."""
    args = _args(block_disk_cache_max_gb=0.0, block_disk_cache_max_percent=10.0)
    normalize_block_disk_cache_budget(args)
    assert args.block_disk_cache_max_gb == 0.0


def test_zero_percent_is_unlimited_matching_the_slider():
    """The Session Settings slider's Unlimited position sends 0.

    Returning the historical flat 10GB here would cap a user who explicitly
    asked for no cap — the shape of limit this project forbids inventing.
    """
    args = _args(block_disk_cache_max_percent=0)
    assert resolve_block_disk_cache_max_gb(args) == 0.0


def test_normalization_is_idempotent():
    """Both scheduler config builders call it; the second call must not treat
    the number the first one produced as an explicit user choice."""
    args = _args(block_disk_cache_dir="/")
    first = normalize_block_disk_cache_budget(args)
    assert args.block_disk_cache_sized_by_percent is True
    second = normalize_block_disk_cache_budget(args)
    assert first == second
    assert args.block_disk_cache_sized_by_percent is True


def test_unmeasurable_volume_falls_back_to_a_finite_budget(monkeypatch):
    """A failed measurement must not hand the cache the whole disk."""
    import vmlx_engine.cli as cli

    def boom(_path):
        raise OSError("no such volume")

    monkeypatch.setattr(cli.shutil, "disk_usage", boom)
    args = _args(block_disk_cache_dir="/nonexistent-volume-for-test")
    resolved = resolve_block_disk_cache_max_gb(args)
    assert resolved > 0.0
    assert resolved != float("inf")
