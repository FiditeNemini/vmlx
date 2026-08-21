"""Low-RAM Macs get told to prefer SSD-only caching — and ONLY told.

The in-RAM paged KV mirror shares unified memory with model weights, so on a
16-32GB Mac it often costs more than it saves. That is worth saying at session
startup.

It is NOT worth enforcing. An invented RAM guard once refused to load big
models across six releases on the exact hardware the product exists for, so
this advisory must never disable the paged cache, never mutate an argument,
and never refuse to start. Failed detection must stay silent rather than
becoming a warning of its own.
"""

import types

import pytest

from vmlx_engine.cli import _low_ram_cache_advice_lines


def _args(**kw):
    base = dict(use_paged_cache=True, enable_block_disk_cache=True)
    base.update(kw)
    return types.SimpleNamespace(**base)


@pytest.fixture
def ram(monkeypatch):
    def _set(total_gb):
        import psutil

        monkeypatch.setattr(
            psutil,
            "virtual_memory",
            lambda: types.SimpleNamespace(total=int(total_gb * 1024**3)),
        )

    return _set


@pytest.mark.parametrize("gb", [16, 18, 24, 32, 36])
def test_small_macs_are_advised(ram, gb):
    ram(gb)
    lines = _low_ram_cache_advice_lines(_args())
    assert lines, f"{gb}GB should get the SSD-only suggestion"
    joined = " ".join(lines)
    assert "--no-paged-cache" in joined
    assert "suggestion only" in joined


@pytest.mark.parametrize("gb", [48, 64, 96, 128])
def test_large_macs_are_left_alone(ram, gb):
    ram(gb)
    assert _low_ram_cache_advice_lines(_args()) == []


def test_nothing_is_said_when_already_ssd_only(ram):
    ram(16)
    assert _low_ram_cache_advice_lines(_args(use_paged_cache=False)) == []


def test_l2_off_is_called_out_so_the_advice_is_actionable(ram):
    ram(16)
    lines = _low_ram_cache_advice_lines(
        _args(enable_block_disk_cache=False)
    )
    assert any("--enable-block-disk-cache" in ln for ln in lines)


def test_failed_detection_says_nothing(monkeypatch):
    """A guess that cannot be made must not become a warning."""
    import psutil

    def boom():
        raise RuntimeError("no psutil for you")

    monkeypatch.setattr(psutil, "virtual_memory", boom)
    assert _low_ram_cache_advice_lines(_args()) == []


def test_the_advice_never_mutates_the_configuration(ram):
    """The whole point: it advises, it does not act."""
    ram(16)
    args = _args()
    before = dict(vars(args))
    _low_ram_cache_advice_lines(args)
    assert dict(vars(args)) == before, "the advisory changed the user's config"


def test_threshold_is_tunable_without_touching_behaviour(ram, monkeypatch):
    ram(64)
    assert _low_ram_cache_advice_lines(_args()) == []
    monkeypatch.setenv("VMLX_LOW_RAM_ADVISORY_GB", "96")
    assert _low_ram_cache_advice_lines(_args())


def test_a_zero_threshold_silences_it(ram, monkeypatch):
    ram(8)
    monkeypatch.setenv("VMLX_LOW_RAM_ADVISORY_GB", "0")
    assert _low_ram_cache_advice_lines(_args()) == []
