"""The scheduler-free GLM SSD lane must not report mandatory full prefill."""
from types import SimpleNamespace

import pytest


@pytest.fixture
def active_native_ssd(monkeypatch):
    from vmlx_engine import server, speculative, glm_dflash2
    monkeypatch.setattr(server, "_engine", object())
    monkeypatch.setattr(speculative, "is_dflash2_enabled", lambda: True)
    monkeypatch.setattr(glm_dflash2, "_PREFIX_DISK", {
        "root": "/unused/status-only", "model_key": "target-and-drafter",
        "max_bytes": 1024,
    })
    def forbidden():
        raise AssertionError("status must not scan SSD or touch tensors")
    monkeypatch.setattr(glm_dflash2, "session_cache_stats", forbidden)
    return server, speculative, glm_dflash2


def test_active_native_ssd_has_reusable_typed_state_and_no_paged_ram(active_native_ssd):
    server, _, runtime = active_native_ssd
    status = server._native_cache_status(family="glm5_next")
    assert status["schema"] == runtime.PrefixDiskTier.FORMAT
    assert status["schema_implemented"] is True
    assert status["prefix"] is True
    assert status["native_checkpoint_ssd"] is True
    assert status["paged"] is False
    assert status["block_disk_l2"] is False  # native catalog, not generic KV blocks
    assert status["cache_store_policy"]["every_request_recomputes_full_prefix"] is False
    assert status["cache_store_policy"]["recompute_when_no_compatible_checkpoint"] is True


def test_health_topology_agrees_with_native_contract(active_native_ssd):
    server, _, _ = active_native_ssd
    status = server._native_cache_status(family="glm5_next_text")
    topology = server._cache_topology_configuration(None, {"native_cache": status})
    assert topology["native_cache"]["schema"] == status["schema"]
    assert topology["native_cache"]["prefix"] is True
    assert topology["native_cache"]["cache_store_policy"]["every_request_recomputes_full_prefix"] is False
    assert topology["instantiated"]["paged_ram_enabled"] is False


@pytest.mark.parametrize("missing", ["root", "model_key"])
def test_unconfigured_ssd_is_not_advertised(active_native_ssd, monkeypatch, missing):
    server, _, runtime = active_native_ssd
    monkeypatch.setitem(runtime._PREFIX_DISK, missing, None)
    assert not server._native_cache_status(family="glm5_next").get("native_checkpoint_ssd")


def test_dflash_off_keeps_scheduler_contract(active_native_ssd, monkeypatch):
    server, speculative, _ = active_native_ssd
    monkeypatch.setattr(speculative, "is_dflash2_enabled", lambda: False)
    assert server._native_cache_status(family="glm5_next")["schema_implemented"] is False


def test_batched_glm_does_not_use_simple_engine_backend(active_native_ssd):
    server, _, _ = active_native_ssd
    scheduler = SimpleNamespace(_uses_glm5_next_cache=False)
    assert server._native_cache_status(scheduler, family="glm5_next")["schema_implemented"] is False


def test_other_family_does_not_adopt_glm_backend(active_native_ssd):
    server, _, _ = active_native_ssd
    assert server._native_cache_status(family="naive_n05_flash").get("cache_backend") != "glm-dflash2"
