# SPDX-License-Identifier: Apache-2.0
"""Naive output admission follows actual growing and rotating cache state."""
from types import SimpleNamespace

import pytest

from vmlx_engine.utils.memory_limits import (
    estimate_naive_decode_cache_memory_from_config,
    estimate_kv_bytes_per_token_from_config,
)


@pytest.fixture
def config():
    return dict(model_type="naive_n05_flash", num_hidden_layers=48,
                hybrid_layer_pattern=[0 if i in (0,5,11,17,23,29,35,41,47) else 1
                                      for i in range(48)],
                num_key_value_heads=4, head_dim=192, v_head_dim=128,
                swa_num_key_value_heads=8, swa_head_dim=192, swa_v_head_dim=128,
                sliding_window=128, index_head_dim=128, dtype="bfloat16")


@pytest.mark.parametrize("wrapper", ["dict", "attrs", "text"])
def test_native_decode_geometry_preserves_prompt_projection(config, wrapper):
    cfg = (config if wrapper == "dict" else SimpleNamespace(**config)
           if wrapper == "attrs" else {"text_config": config})
    estimate = estimate_naive_decode_cache_memory_from_config(cfg)
    assert (estimate.dsa_layers, estimate.swa_layers) == (9, 39)
    assert estimate.growth_bytes_per_token == 27_648
    assert estimate.output_reserve_bytes == 32_636_928
    # This is deliberately NOT a prefill workspace or prompt-admission change.
    assert estimate_kv_bytes_per_token_from_config(cfg) == 147_456


@pytest.mark.parametrize("field,value", [("hybrid_layer_pattern", []),
    ("hybrid_layer_pattern", [2]*48), ("v_head_dim", None),
    ("sliding_window", 0), ("index_head_dim", 0), ("model_type", "other")])
def test_unknown_native_geometry_keeps_generic_guard(config, field, value):
    config[field] = value
    assert estimate_naive_decode_cache_memory_from_config(config) is None


def test_weight_bits_do_not_shrink_indexer_or_attention_state(config):
    estimate = estimate_naive_decode_cache_memory_from_config(config)
    config["quantization"] = {"bits": 2}
    assert estimate_naive_decode_cache_memory_from_config(config) == estimate
    config["dtype"] = "float32"
    assert estimate_naive_decode_cache_memory_from_config(config).growth_bytes_per_token == 50_688


def test_real_native_cache_allocation_is_bounded_after_chunk_and_decode(config):
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.cache import KVCache, RotatingKVCache
    assert KVCache.step == 256
    kv, index, swa = KVCache(), KVCache(), RotatingKVCache(128)
    estimate = estimate_naive_decode_cache_memory_from_config(config)
    def append(n):
        kv.update_and_fetch(mx.zeros((1,4,n,192), mx.bfloat16),
                            mx.zeros((1,4,n,128), mx.bfloat16))
        index.update_and_fetch(mx.zeros((1,1,n,128), mx.float32),
                               mx.zeros((1,1,n,0), mx.float32))
        swa.update_and_fetch(mx.zeros((1,8,n,192), mx.bfloat16),
                             mx.zeros((1,8,n,128), mx.bfloat16))
        mx.eval(kv.state, index.state, swa.state)
    for chunk in (127, 129, 511, 256):
        append(chunk)
        append(1)  # Decode trims the prefill concat storage to the SWA ring.
        assert swa.keys.shape[2] <= 128
        actual = 9*(kv.nbytes + index.nbytes) + 39*swa.nbytes
        envelope = kv.offset*estimate.growth_bytes_per_token + estimate.output_reserve_bytes
        assert actual <= envelope
        assert index.values.shape[-1] == 0 and index.keys.dtype == mx.float32


def test_server_uses_native_decode_geometry_without_disabling_guard(config, monkeypatch):
    from fastapi import HTTPException
    from vmlx_engine import server
    monkeypatch.setattr(server, "_loaded_model_config_for_memory_projection", lambda: config)
    monkeypatch.setattr(server, "_metal_projection_stats", lambda: (96*2**30,108*2**30))
    monkeypatch.setattr(server, "_projection_env", lambda name, default: default)
    cap = server._metal_projected_output_token_cap("native-test")
    assert 50_000 < cap < 60_000
    assert server._apply_projected_output_guard(12_000, explicit=True) == 12_000
    with pytest.raises(HTTPException) as e:
        server._apply_projected_output_guard(cap + 1, explicit=True)
    assert e.value.status_code == 413
    config["model_type"] = "unknown"
    assert server._metal_projected_output_token_cap("generic-test") < 12_000
