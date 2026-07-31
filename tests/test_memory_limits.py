# SPDX-License-Identifier: Apache-2.0

"""Focused tests for shared Metal working-set guard helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from vmlx_engine.utils.memory_limits import (
    _parse_float_env,
    _parse_working_set_bytes,
    estimate_cache_token_capacity_from_config,
    estimate_dsv4_cache_memory_from_config,
    estimate_kv_bytes_per_token_from_config,
    get_effective_metal_working_set_bytes,
    get_metal_ws_guard_threshold,
    is_metal_ws_guard_enabled,
    projected_output_token_cap,
    resolve_working_set_override,
)


def _dsv4_config():
    return {
        "model_type": "deepseek_v4",
        "architectures": ["DeepseekV4ForCausalLM"],
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "hidden_size": 4096,
        "sliding_window": 128,
        "index_head_dim": 128,
        "torch_dtype": "bfloat16",
        "max_position_embeddings": 1_048_576,
        # The source bundle carries one trailing sentinel; only the first 43
        # entries are instantiated by the runtime.
        "compress_ratios": [0, 0]
        + [value for _ in range(20) for value in (4, 128)]
        + [4, 0],
    }


def test_resolve_working_set_override_clamps_to_base():
    base = 64 * (1024**3)
    with patch.dict(
        "os.environ",
        {"VMLX_METAL_WS_MAX_GB": "96"},
        clear=True,
    ):
        assert resolve_working_set_override(base) == base


def test_resolve_working_set_override_bytes_override():
    base = 128 * (1024**3)
    with patch.dict(
        "os.environ",
        {"VMLX_METAL_WS_MAX_BYTES": str(32 * 1024**3)},
        clear=True,
    ):
        assert resolve_working_set_override(base) == 32 * (1024**3)


def test_parse_working_set_bytes_rejects_invalid():
    assert _parse_working_set_bytes("abc") is None
    assert _parse_working_set_bytes("") is None
    assert _parse_working_set_bytes("12x") is None


def test_get_effective_metal_working_set_bytes_apply_override():
    mx = SimpleNamespace(
        get_active_memory=lambda: 2 * 1024**3,
        device_info=lambda: {"max_recommended_working_set_size": 64 * 1024**3},
    )
    with patch.dict(
        "os.environ",
        {"VMLX_METAL_WS_MAX_GB": "48"},
        clear=True,
    ):
        active, max_ws = get_effective_metal_working_set_bytes(mx)
        assert active == 2 * 1024**3
        assert max_ws == 48 * 1024**3


def test_guard_threshold_parse_default_and_override():
    assert get_metal_ws_guard_threshold() == 98.0
    assert get_metal_ws_guard_threshold(85.0) == 85.0
    with patch.dict("os.environ", {"VMLX_METAL_WS_REJECT_PCT": "30"}, clear=True):
        assert get_metal_ws_guard_threshold(85.0) == 30.0
    with patch.dict("os.environ", {"VMLX_METAL_WS_REJECT_PCT": "oops"}, clear=True):
        assert get_metal_ws_guard_threshold(85.0) == 85.0


def test_guard_is_enabled_default_and_disable():
    with patch.dict("os.environ", {}, clear=True):
        assert is_metal_ws_guard_enabled() is True
    with patch.dict("os.environ", {"VMLX_METAL_WS_GUARD": "0"}, clear=True):
        assert is_metal_ws_guard_enabled() is False
    with patch.dict("os.environ", {"VMLX_METAL_WS_GUARD": "1"}, clear=True):
        assert is_metal_ws_guard_enabled() is True


def test_parse_float_env_negative_reverts_to_default():
    assert _parse_float_env("MISSING", 42.0) == 42.0


def test_estimate_kv_bytes_per_token_from_text_config_dict():
    cfg = {
        "text_config": {
            "num_hidden_layers": 2,
            "num_key_value_heads": 4,
            "head_dim": 8,
            "torch_dtype": "bfloat16",
        }
    }

    assert estimate_kv_bytes_per_token_from_config(cfg) == 2 * 2 * 4 * 8 * 2


def test_estimate_kv_bytes_per_token_counts_looped_cache_slots():
    cfg = {
        "num_hidden_layers": 22,
        "num_loops": 2,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "torch_dtype": "bfloat16",
    }

    assert estimate_kv_bytes_per_token_from_config(cfg) == 44 * 2 * 8 * 128 * 2


def test_estimate_kv_bytes_prefers_runtime_total_loops():
    cfg = SimpleNamespace(
        num_hidden_layers=22,
        num_loops=1,
        total_loops=2,
        num_key_value_heads=8,
        head_dim=128,
        torch_dtype="bfloat16",
    )

    assert estimate_kv_bytes_per_token_from_config(cfg) == 44 * 2 * 8 * 128 * 2


def test_dsv4_estimator_uses_native_swa_csa_hca_topology():
    estimate = estimate_dsv4_cache_memory_from_config(
        _dsv4_config(),
        32_768,
        pool_quant_enabled=False,
    )

    assert estimate is not None
    assert estimate.ratio_zero_layers == 2
    assert estimate.ratio_four_layers == 21
    assert estimate.ratio_high_layers == 20
    assert estimate.local_swa_bytes == 43 * 128 * 2 * 1 * 512 * 2
    assert estimate.csa_pool_bytes == 21 * (32_768 // 4) * 512 * 2
    assert estimate.csa_indexer_bytes == 21 * (32_768 // 4) * 128 * 2
    assert estimate.hca_pool_bytes == 20 * (32_768 // 128) * 512 * 2
    # Conservative prompt admission retains the peak incomplete-window state:
    # 7 overlap rows for ratio-4 and 127 remainder rows for ratio-128.
    assert estimate.tail_bytes == 5_954_560
    assert estimate.total_bytes == 242_670_592


def test_dsv4_pool_q8_admission_counts_encoded_storage_without_full_bf16_view():
    config = _dsv4_config()
    bf16 = estimate_dsv4_cache_memory_from_config(
        config,
        32_768,
        pool_quant_enabled=False,
    )
    q8 = estimate_dsv4_cache_memory_from_config(
        config,
        32_768,
        pool_quant_enabled=True,
    )

    assert bf16 is not None and q8 is not None
    assert q8.total_bytes < bf16.total_bytes
    assert q8.local_swa_bytes == bf16.local_swa_bytes
    assert q8.tail_bytes == bf16.tail_bytes
    # The 512-wide ratio-4 compressor pool retains q8 codes/affine metadata
    # only. Attention uses bounded indexer tiles and selected-row decode, so a
    # full historical BF16 view is not a retained cache owner.
    assert q8.csa_pool_bytes < bf16.csa_pool_bytes
    # The 128-wide indexer and ratio-128 HCA branches are still at or below
    # their per-branch BF16 hot-tier threshold at 32K.
    assert q8.csa_indexer_bytes == bf16.csa_indexer_bytes
    assert q8.hca_pool_bytes == bf16.hca_pool_bytes


def test_dsv4_admission_envelope_is_monotonic_at_pool_and_window_boundaries():
    config = _dsv4_config()
    boundaries = (3, 4, 7, 8, 127, 128, 16_383, 16_384, 16_387, 16_388)
    estimates = [
        estimate_dsv4_cache_memory_from_config(
            config,
            tokens,
            pool_quant_enabled=True,
        ).total_bytes
        for tokens in boundaries
    ]

    assert estimates == sorted(estimates)


def test_dsv4_capacity_uses_native_non_linear_geometry_and_tiled_q8_view():
    config = _dsv4_config()
    budget = 4 * 1024**3

    q8_capacity = estimate_cache_token_capacity_from_config(
        config,
        budget,
        max_tokens=1_048_576,
        dsv4_pool_quant_enabled=True,
    )
    bf16_capacity = estimate_cache_token_capacity_from_config(
        config,
        budget,
        max_tokens=1_048_576,
        dsv4_pool_quant_enabled=False,
    )

    assert 32_768 < bf16_capacity < q8_capacity <= 1_048_576


def test_dsv4_pool_q8_one_million_active_cache_stays_below_fifteen_gib():
    estimate = estimate_dsv4_cache_memory_from_config(
        _dsv4_config(),
        1_048_576,
        pool_quant_enabled=True,
    )

    assert estimate is not None
    assert estimate.total_bytes < 15 * 1024**3


def test_dsv4_output_projection_is_architecture_aware_not_generic_full_kv():
    config = _dsv4_config()

    # 88,064 local-ring bytes plus average BF16 CSA/indexer/HCA growth.
    assert estimate_kv_bytes_per_token_from_config(config) == 94_944


def test_projected_output_token_cap_accounts_for_transient_multiplier():
    gib = 1024**3
    cap = projected_output_token_cap(
        active_bytes=int(105.41 * gib),
        max_working_set_bytes=int(107.52 * gib),
        bytes_per_token=256 * 1024,
        budget_fraction=0.50,
        transient_multiplier=4.0,
    )

    assert 1000 <= cap <= 1100


def test_scheduler_waiting_uses_shared_memory_helper():
    import inspect

    from vmlx_engine.mllm_scheduler import MLLMScheduler
    from vmlx_engine.scheduler import Scheduler

    assert "get_effective_metal_working_set_bytes" in inspect.getsource(Scheduler._schedule_waiting)
    assert "get_metal_ws_guard_threshold" in inspect.getsource(Scheduler._schedule_waiting)
    assert "get_metal_ws_guard_threshold(85.0)" not in inspect.getsource(
        Scheduler._schedule_waiting
    )
    assert (
        "get_effective_metal_working_set_bytes"
        in inspect.getsource(MLLMScheduler._schedule_waiting)
    )
    assert "get_metal_ws_guard_threshold" in inspect.getsource(MLLMScheduler._schedule_waiting)
    assert "get_metal_ws_guard_threshold(85.0)" not in inspect.getsource(
        MLLMScheduler._schedule_waiting
    )
