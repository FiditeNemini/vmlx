"""The GLM DFlash2 RAM prefix store keeps 4 snapshots by default (was 12).

Measured 2026-10-11 (box, GLM-5.3-Flash JANGHT2.4, 8-turn chat, request 8 with 11,147 cached, 3 alternating rounds):
ENTRIES 12 -> 3.68 s, 4 -> 3.46 s, 2 -> 3.39 s; after a RAM clear a 7-turn engine equals a fresh one. Older branch
points still restore from the SSD tier.
"""
import pytest

pytest.importorskip("mlx.core")


def test_default_ram_entries_is_four(monkeypatch):
    from vmlx_engine import glm_dflash2 as G

    monkeypatch.delenv("VMLX_GLM_DFLASH2_PREFIX_ENTRIES", raising=False)
    monkeypatch.setattr(G, "_PREFIX_STORES", {})
    monkeypatch.setitem(G._PREFIX_DISK, "root", None)
    target = object()
    assert G.prefix_store_for(target).max_entries == 4
    monkeypatch.setattr(G, "_PREFIX_STORES", {})
    monkeypatch.setenv("VMLX_GLM_DFLASH2_PREFIX_ENTRIES", "12")
    assert G.prefix_store_for(target).max_entries == 12      # the override still works
