"""Allocator ownership regression: no model, tensor allocation, or GPU work."""
from types import SimpleNamespace

import pytest

from vmlx_engine.mlx_memory import configured_serving_cache_limit_bytes


@pytest.mark.parametrize('setting,expected', [
    (None, 4*1024**3), ('bad', 4*1024**3), ('0', None),
    ('128', 128*1024**2), ('8192', 8*1024**3),
])
def test_shared_policy_preserves_operator_override(monkeypatch, setting, expected):
    if setting is None:
        monkeypatch.delenv('VMLX_MLX_CACHE_LIMIT_MB', raising=False)
    else:
        monkeypatch.setenv('VMLX_MLX_CACHE_LIMIT_MB', setting)
    assert configured_serving_cache_limit_bytes(96*1024**3) == expected


def test_measured_headroom_can_only_tighten_default(monkeypatch):
    monkeypatch.delenv('VMLX_MLX_CACHE_LIMIT_MB', raising=False)
    assert configured_serving_cache_limit_bytes(96*1024**3, max_default_bytes=1024**3) == 1024**3


@pytest.fixture
def allocator(monkeypatch):
    import vmlx_engine.mllm_batch_generator as module
    state = {'limit': 4*1024**3, 'calls': []}

    def set_limit(value):
        old = state['limit']
        state['limit'] = value
        state['calls'].append(value)
        return old

    monkeypatch.setattr(module.mx.metal, 'is_available', lambda: True)
    monkeypatch.setattr(module.mx, 'set_cache_limit', set_limit)
    monkeypatch.setattr(module.mx, 'set_wired_limit', lambda value: 115*1024**3)
    monkeypatch.setattr(module.mx, 'synchronize', lambda *args: None)
    monkeypatch.setattr(module.MLLMBatchGenerator, '_stream', object())
    monkeypatch.setattr(module, 'get_effective_metal_working_set_bytes', lambda mx: (96*1024**3, 115*1024**3))
    monkeypatch.delenv('VMLX_MLX_CACHE_LIMIT_MB', raising=False)
    monkeypatch.delenv('VMLX_VLM_IMAGE_CACHE_LIMIT', raising=False)
    monkeypatch.delenv('VMLX_VLM_IMAGE_CACHE_LIMIT_GB', raising=False)
    monkeypatch.delenv('VMLX_VLM_IMAGE_CACHE_LIMIT_FREE_FRACTION', raising=False)
    monkeypatch.delenv('VMLX_VLM_IMAGE_CACHE_LIMIT_FLOOR_GB', raising=False)
    return module, state


def make_generator(module):
    model = SimpleNamespace(config=SimpleNamespace(model_type='allocator_test'))
    tokenizer = SimpleNamespace(eos_token_id=9, encode=lambda *args, **kwargs: [1])
    return module.MLLMBatchGenerator(model, SimpleNamespace(tokenizer=tokenizer),
                                    enable_prefix_cache=False, ssm_state_cache_size=0)


def test_lazy_generator_does_not_widen_post_load_bound(allocator):
    module, state = allocator
    gen = make_generator(module)
    try:
        # Old lazy-init policy set9.5GiB here despite the4GiB serving ceiling.
        assert gen._steady_cache_limit == state['limit'] == 4*1024**3
        assert gen._tight_memory_prefill_drain
        assert module._apply_vlm_image_request_cache_limit(gen._steady_cache_limit)
        assert state['limit'] == 1024**3
        gen._vlm_cache_limit_tightened = True
        assert gen._next() == []
        assert state['limit'] == 4*1024**3
    finally:
        gen.close()
    assert state['limit'] == 4*1024**3


def test_explicit_small_limit_survives_media_restore_and_close(allocator, monkeypatch):
    module, state = allocator
    monkeypatch.setenv('VMLX_MLX_CACHE_LIMIT_MB', '128')
    gen = make_generator(module)
    try:
        assert state['limit'] == 128*1024**2
        assert module._apply_vlm_image_request_cache_limit(gen._steady_cache_limit)
        assert state['limit'] == 128*1024**2
        gen._vlm_cache_limit_tightened = True
        gen._next()
        assert state['limit'] == 128*1024**2
    finally:
        gen.close()
    assert state['limit'] == 4*1024**3


@pytest.mark.parametrize("setting", ["0", "00", " 0 "])
def test_zero_override_does_not_take_allocator_ownership(allocator, monkeypatch, setting):
    module, state = allocator
    monkeypatch.setenv('VMLX_MLX_CACHE_LIMIT_MB', setting)
    gen = make_generator(module)
    try:
        assert gen._steady_cache_limit is None
        assert state['calls'] == []
        assert module._apply_vlm_image_request_cache_limit(gen._steady_cache_limit) is False
    finally:
        gen.close()
    assert state['calls'] == []


def test_default_preserves_stricter_prior_limit_after_residency_changes(allocator, monkeypatch):
    module, state = allocator
    state['limit'] = 512*1024**2
    monkeypatch.setattr(module, 'get_effective_metal_working_set_bytes',
                        lambda mx: (20*1024**3, 115*1024**3))
    gen = make_generator(module)
    try:
        assert state['calls'] == [1024**3, 512*1024**2]
        assert gen._steady_cache_limit == state['limit'] == 512*1024**2
    finally:
        gen.close()
    assert state['limit'] == 512*1024**2


def test_positive_override_can_replace_stricter_prior_limit(allocator, monkeypatch):
    module, state = allocator
    state['limit'] = 128*1024**2
    monkeypatch.setenv('VMLX_MLX_CACHE_LIMIT_MB', '1024')
    gen = make_generator(module)
    try:
        assert state['calls'] == [1024**3]
        assert state['limit'] == gen._steady_cache_limit == 1024**3
    finally:
        gen.close()
    assert state['limit'] == 128*1024**2
