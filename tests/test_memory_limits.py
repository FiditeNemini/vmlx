# SPDX-License-Identifier: Apache-2.0

"""Focused tests for shared Metal working-set guard helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vmlx_engine.utils.memory_limits import (
    _parse_float_env,
    _parse_working_set_bytes,
    estimate_cache_bytes_for_tokens_from_config,
    estimate_cache_token_capacity_from_config,
    estimate_dsv4_delta_transport_bytes_from_config,
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


def test_dsv4_delta_transport_estimate_counts_pool_anchors_and_records():
    config = _dsv4_config()
    one_anchor = estimate_dsv4_delta_transport_bytes_from_config(
        config,
        0,
        2048,
        pool_quant_enabled=True,
    )
    two_anchors = estimate_dsv4_delta_transport_bytes_from_config(
        config,
        0,
        4096,
        pool_quant_enabled=True,
    )
    tail_only = estimate_dsv4_delta_transport_bytes_from_config(
        config,
        2048,
        2304,
        pool_quant_enabled=True,
    )

    assert one_anchor is not None and one_anchor > 0
    assert two_anchors is not None and two_anchors > one_anchor
    assert tail_only is not None and 0 < tail_only < two_anchors
    assert (
        estimate_dsv4_delta_transport_bytes_from_config(
            {"model_type": "llama"},
            0,
            4096,
        )
        is None
    )


def test_dsv4_delta_transport_estimate_counts_partial_terminal_and_aligned_predecessor():
    config = _dsv4_config()
    estimated = estimate_dsv4_delta_transport_bytes_from_config(
        config,
        0,
        331,
        pool_quant_enabled=True,
    )
    initial = estimate_dsv4_cache_memory_from_config(
        config,
        0,
        pool_quant_enabled=True,
    )
    final = estimate_dsv4_cache_memory_from_config(
        config,
        331,
        pool_quant_enabled=True,
    )
    assert estimated is not None and initial is not None and final is not None
    pool_delta = (
        final.csa_pool_bytes
        + final.csa_indexer_bytes
        + final.hca_pool_bytes
        - initial.csa_pool_bytes
        - initial.csa_indexer_bytes
        - initial.hca_pool_bytes
    )
    layer_count = (
        final.ratio_zero_layers
        + final.ratio_four_layers
        + final.ratio_high_layers
    )
    expected = (
        pool_delta
        + 2 * (final.local_swa_bytes + final.tail_bytes)
        + 2 * layer_count * 4096
    )
    assert estimated == expected


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


def test_dsv4_pool_q8_admission_counts_encoded_storage_without_full_bf16_view(
    monkeypatch,
):
    # Pin the BF16 retention threshold below the 32K compressor pool so the
    # promotion math is exercised; the runtime default (64 MiB) keeps 32K
    # pools BF16 entirely (see the default-threshold test below).
    monkeypatch.setenv("DSV4_POOL_BF16_MAX_BYTES", str(2 * 1024 * 1024))
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


def test_dsv4_pool_default_bf16_retention_matches_runtime_at_32k():
    # Runtime default keeps pools attention-ready BF16 up to 64 MiB per
    # retained state (jang_tools pool_quant_cache), so at 32K tokens the
    # pool-quant estimate must equal the BF16 estimate — the admission
    # estimator must not claim q8 savings the runtime no longer takes.
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
    assert q8.total_bytes == bf16.total_bytes


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


def test_dsv4_admission_with_block_records_shrinks_advertised_capacity():
    config = _dsv4_config()
    budget = int(6.6 * 1024**3)

    live_only = estimate_cache_token_capacity_from_config(
        config,
        budget,
        max_tokens=1_048_576,
        dsv4_pool_quant_enabled=True,
    )
    with_records = estimate_cache_token_capacity_from_config(
        config,
        budget,
        max_tokens=1_048_576,
        dsv4_pool_quant_enabled=True,
        include_dsv4_block_records=True,
    )

    # Live q8 pools alone claim the full declared 1M context fits, which is
    # exactly the over-admission that crashed the 430k-token gate: the paged
    # L1 block records captured during prefill were never budgeted.
    assert live_only == 1_048_576
    assert 300_000 < with_records < 400_000


def test_dsv4_block_record_bytes_match_measured_430k_growth():
    config = _dsv4_config()
    tokens = 430_080
    gib = 1024**3

    live = estimate_cache_bytes_for_tokens_from_config(
        config, tokens, dsv4_pool_quant_enabled=True
    )
    full = estimate_cache_bytes_for_tokens_from_config(
        config,
        tokens,
        dsv4_pool_quant_enabled=True,
        include_dsv4_block_records=True,
    )

    # Measured on the 128GB box at the crash point: live q8 pools 1.9GB plus
    # block records 5.6GB = 7.5GB of Metal active-memory growth.
    assert live < 2.1 * gib
    assert 7.0 * gib < full < 8.0 * gib


def test_dsv4_prefill_valve_threshold_is_pure_and_deterministic():
    from vmlx_engine.utils.dsv4_batch_generator import (
        DSV4PrefillMemoryError,
        dsv4_prefill_valve_check,
    )

    gib = 1024**3

    # Comfortable headroom: active + 1.25x observed transient fits.
    dsv4_prefill_valve_check(
        active_bytes=100 * gib,
        max_ws_bytes=107 * gib,
        observed_transient_bytes=3 * gib,
        min_margin_bytes=2 * gib,
        chunk_start=0,
        chunk_end=2048,
    )

    # Projected peak exceeds the working-set limit: abort before GPU submit.
    with pytest.raises(DSV4PrefillMemoryError):
        dsv4_prefill_valve_check(
            active_bytes=104 * gib,
            max_ws_bytes=107 * gib,
            observed_transient_bytes=3 * gib,
            min_margin_bytes=2 * gib,
            chunk_start=428_032,
            chunk_end=430_080,
        )

    # Unknown telemetry (zero readings) must never abort.
    dsv4_prefill_valve_check(
        active_bytes=0,
        max_ws_bytes=107 * gib,
        observed_transient_bytes=0,
        min_margin_bytes=2 * gib,
        chunk_start=0,
        chunk_end=2048,
    )
    dsv4_prefill_valve_check(
        active_bytes=104 * gib,
        max_ws_bytes=0,
        observed_transient_bytes=0,
        min_margin_bytes=2 * gib,
        chunk_start=0,
        chunk_end=2048,
    )


def test_dsv4_prefill_valve_error_is_not_treated_as_cache_corruption():
    import inspect

    from vmlx_engine.scheduler import CACHE_CORRUPTION_PATTERNS, Scheduler
    from vmlx_engine.utils.dsv4_batch_generator import (
        DSV4PrefillMemoryError,
        dsv4_prefill_valve_check,
    )

    gib = 1024**3
    with pytest.raises(DSV4PrefillMemoryError) as excinfo:
        dsv4_prefill_valve_check(
            active_bytes=104 * gib,
            max_ws_bytes=107 * gib,
            observed_transient_bytes=3 * gib,
            min_margin_bytes=2 * gib,
            chunk_start=428_032,
            chunk_end=430_080,
        )

    message = str(excinfo.value)
    # If the message matched a recover-and-reschedule pattern the scheduler
    # would clear caches and re-run a prefill that is doomed to fail forever.
    assert not any(pattern in message for pattern in CACHE_CORRUPTION_PATTERNS)
    # step() also carries an explicit type-based exclusion as a second guard.
    assert "DSV4PrefillMemoryError" in inspect.getsource(Scheduler.step)


def test_dsv4_prefill_adaptive_step_keeps_width_with_headroom():
    from vmlx_engine.utils.dsv4_batch_generator import dsv4_prefill_adaptive_step

    gib = 1024**3
    step, transient = dsv4_prefill_adaptive_step(
        2048,
        active_bytes=90 * gib,
        max_ws_bytes=107 * gib,
        observed_transient_bytes=3 * gib,
        min_margin_bytes=2 * gib,
    )
    assert step == 2048
    assert transient == 3 * gib

    # Unknown telemetry must never shrink.
    step, transient = dsv4_prefill_adaptive_step(
        2048,
        active_bytes=0,
        max_ws_bytes=107 * gib,
        observed_transient_bytes=0,
        min_margin_bytes=2 * gib,
    )
    assert step == 2048


def test_dsv4_prefill_adaptive_step_halves_until_projection_fits():
    """RUN A wall replay: 2048-wide chunk at ~524k ctx must shrink, not abort.

    Measured at the abort: active=98.79GB, transient(2048)~7GB, limit
    107.52GB. A single halving to 1024 already fits the conservative
    projection, so the request continues instead of dying at 524k.
    """
    from vmlx_engine.utils.dsv4_batch_generator import (
        dsv4_prefill_adaptive_step,
        dsv4_prefill_valve_check,
        dsv4_scale_transient_for_width,
    )

    gib = 1024**3
    active = int(98.79 * gib)
    max_ws = int(107.52 * gib)
    observed = 7 * gib

    step, transient = dsv4_prefill_adaptive_step(
        2048,
        active_bytes=active,
        max_ws_bytes=max_ws,
        observed_transient_bytes=observed,
        min_margin_bytes=2 * gib,
    )
    assert step == 1024
    assert transient == dsv4_scale_transient_for_width(observed, 2048, 1024)
    assert transient < observed

    # The valve accepts the shrunk projection — no abort.
    dsv4_prefill_valve_check(
        active,
        max_ws,
        transient,
        2 * gib,
        chunk_start=524_288,
        chunk_end=524_288 + step,
    )


def test_dsv4_prefill_adaptive_step_floors_at_native_block_then_valve_aborts():
    from vmlx_engine.utils.dsv4_batch_generator import (
        DSV4_PREFILL_MIN_STEP_TOKENS,
        DSV4PrefillMemoryError,
        dsv4_prefill_adaptive_step,
        dsv4_prefill_valve_check,
        dsv4_scale_transient_for_width,
    )

    gib = 1024**3
    active = 105 * gib
    max_ws = int(107.52 * gib)
    observed = 7 * gib

    step, transient = dsv4_prefill_adaptive_step(
        2048,
        active_bytes=active,
        max_ws_bytes=max_ws,
        observed_transient_bytes=observed,
        min_margin_bytes=2 * gib,
    )
    assert step == DSV4_PREFILL_MIN_STEP_TOKENS == 256
    expected = observed
    for old, new in ((2048, 1024), (1024, 512), (512, 256)):
        expected = dsv4_scale_transient_for_width(expected, old, new)
    assert transient == expected

    # Even the floor-width projection exceeds the limit: the valve aborts.
    with pytest.raises(DSV4PrefillMemoryError):
        dsv4_prefill_valve_check(
            active,
            max_ws,
            transient,
            2 * gib,
            chunk_start=700_000,
            chunk_end=700_000 + step,
        )


def test_dsv4_scale_transient_for_width_is_conservative_vs_measured():
    """Projection must stay above the measured narrow-chunk transient.

    Box measurements at ~530k ctx: transient(2048)=7.0GB, transient(512)=
    4.2GB. The invariant-fraction model predicts 0.775x for that 4x shrink
    where 0.60x was measured — conservative by construction.
    """
    from vmlx_engine.utils.dsv4_batch_generator import (
        dsv4_scale_transient_for_width,
    )

    gib = 1024**3
    projected = dsv4_scale_transient_for_width(7 * gib, 2048, 512)
    assert projected >= int(4.2 * gib)
    assert projected < 7 * gib

    # Never scales up, never goes negative.
    assert dsv4_scale_transient_for_width(7 * gib, 512, 2048) == 7 * gib
    assert dsv4_scale_transient_for_width(0, 2048, 512) == 0


def test_dsv4_prefill_loop_wires_adaptive_shrink():
    """The generator prefill loop must consult the adaptive step picker."""
    with open("vmlx_engine/utils/dsv4_batch_generator.py") as f:
        src = f.read()
    loop = src.split("off = 0\n        cur_step = step", 1)[1]
    assert "dsv4_prefill_adaptive_step(" in loop[:1200]
    assert "prefill adaptive shrink" in loop[:2000]
