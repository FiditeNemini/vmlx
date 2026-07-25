# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for partial scheduler cache construction cleanup."""

from types import SimpleNamespace

import pytest


class KVCache:
    pass


class MambaCache:
    pass


class _HybridLanguageModel:
    config = SimpleNamespace(model_type="test_hybrid")

    def make_cache(self):
        return [KVCache(), MambaCache()]


def _assert_cache_resources_released(block_store, ssm_store, root):
    assert not block_store._writer_thread.is_alive()
    assert not ssm_store._writer_thread.is_alive()
    lease_dir = root / ".vmlx-global-cache-budget-leases"
    assert list(lease_dir.glob("*.json")) == []


@pytest.mark.parametrize("failure_stage", ["paged_manager", "block_aware"])
def test_text_scheduler_cache_construction_failure_releases_async_l2(
    tmp_path,
    monkeypatch,
    failure_stage,
):
    import vmlx_engine.scheduler as scheduler_module
    from vmlx_engine.block_disk_store import BlockDiskStore
    from vmlx_engine.scheduler import Scheduler, SchedulerConfig
    from vmlx_engine.utils.ssm_companion_disk_store import SSMCompanionDiskStore

    root = tmp_path / "text-block-root"
    created_blocks = []
    created_ssm = []
    created_managers = []

    def capture_block(*args, **kwargs):
        store = BlockDiskStore(*args, **kwargs)
        created_blocks.append(store)
        return store

    def capture_ssm(*args, **kwargs):
        store = SSMCompanionDiskStore(*args, **kwargs)
        created_ssm.append(store)
        return store

    real_paged_manager = scheduler_module.PagedCacheManager

    def paged_manager(*args, **kwargs):
        if failure_stage == "paged_manager":
            raise RuntimeError("injected paged manager failure")
        manager = real_paged_manager(*args, **kwargs)
        created_managers.append(manager)
        return manager

    def block_aware(*_args, **_kwargs):
        raise RuntimeError("injected block-aware cache failure")

    monkeypatch.setattr(scheduler_module, "BlockDiskStore", capture_block)
    monkeypatch.setattr(scheduler_module, "SSMCompanionDiskStore", capture_ssm)
    monkeypatch.setattr(scheduler_module, "PagedCacheManager", paged_manager)
    if failure_stage == "block_aware":
        monkeypatch.setattr(scheduler_module, "BlockAwarePrefixCache", block_aware)

    model = _HybridLanguageModel()
    tokenizer = SimpleNamespace(eos_token_id=0, eos_token_ids={0})
    try:
        with pytest.raises(RuntimeError, match="injected"):
            Scheduler(
                model=model,
                tokenizer=tokenizer,
                config=SchedulerConfig(
                    model_path="example/text-hybrid",
                    enable_prefix_cache=True,
                    use_paged_cache=True,
                    enable_block_disk_cache=True,
                    block_disk_cache_dir=str(root),
                    block_disk_cache_max_gb=0.001,
                    max_cache_blocks=8,
                ),
            )

        assert len(created_blocks) == 1
        assert len(created_ssm) == 1
        _assert_cache_resources_released(created_blocks[0], created_ssm[0], root)
        if created_managers:
            assert created_managers[0]._disk_store is None
    finally:
        for store in created_ssm:
            store.shutdown(timeout=5.0)
        for store in created_blocks:
            store.shutdown()

@pytest.mark.parametrize("failure_stage", ["paged_manager", "block_aware"])
def test_mllm_scheduler_cache_construction_failure_releases_async_l2_and_truth(
    tmp_path,
    monkeypatch,
    failure_stage,
):
    import vmlx_engine.block_disk_store as block_store_module
    import vmlx_engine.mllm_scheduler as scheduler_module
    import vmlx_engine.paged_cache as paged_cache_module
    import vmlx_engine.prefix_cache as prefix_cache_module
    from vmlx_engine.block_disk_store import BlockDiskStore
    from vmlx_engine.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig
    from vmlx_engine.utils.ssm_companion_disk_store import SSMCompanionDiskStore

    root = tmp_path / "mllm-block-root"
    created_blocks = []
    created_ssm = []
    created_managers = []

    def capture_block(*args, **kwargs):
        store = BlockDiskStore(*args, **kwargs)
        created_blocks.append(store)
        return store

    def capture_ssm(*args, **kwargs):
        store = SSMCompanionDiskStore(*args, **kwargs)
        created_ssm.append(store)
        return store

    real_paged_manager = paged_cache_module.PagedCacheManager

    def paged_manager(*args, **kwargs):
        if failure_stage == "paged_manager":
            raise RuntimeError("injected paged manager failure")
        manager = real_paged_manager(*args, **kwargs)
        created_managers.append(manager)
        return manager

    def block_aware(*_args, **_kwargs):
        raise RuntimeError("injected block-aware cache failure")

    monkeypatch.setattr(block_store_module, "BlockDiskStore", capture_block)
    monkeypatch.setattr(scheduler_module, "SSMCompanionDiskStore", capture_ssm)
    monkeypatch.setattr(paged_cache_module, "PagedCacheManager", paged_manager)
    if failure_stage == "block_aware":
        monkeypatch.setattr(prefix_cache_module, "BlockAwarePrefixCache", block_aware)

    model = SimpleNamespace(
        language_model=_HybridLanguageModel(),
        config=SimpleNamespace(model_type="test_hybrid_vlm"),
    )
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(eos_token_id=0, eos_token_ids={0})
    )
    scheduler = None
    try:
        scheduler = MLLMScheduler(
            model=model,
            processor=processor,
            config=MLLMSchedulerConfig(
                model_path="example/hybrid-vlm",
                enable_prefix_cache=True,
                use_paged_cache=True,
                enable_block_disk_cache=True,
                block_disk_cache_dir=str(root),
                block_disk_cache_max_gb=0.001,
                max_cache_blocks=8,
                step_executor=SimpleNamespace(_thread_name_prefix="test-worker"),
            ),
        )

        assert scheduler.paged_cache_manager is None
        assert scheduler.block_aware_cache is None
        assert scheduler._ssm_companion_disk_store is None
        assert scheduler._block_disk_l2_enabled is False
        assert len(created_blocks) == 1
        assert len(created_ssm) == 1
        _assert_cache_resources_released(created_blocks[0], created_ssm[0], root)
        if created_managers:
            assert created_managers[0]._disk_store is None
    finally:
        for store in created_ssm:
            store.shutdown(timeout=5.0)
        for store in created_blocks:
            store.shutdown()
