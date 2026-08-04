"""Metal live-buffer ceiling projection (R21-13).

Metal caps the NUMBER of live MLX buffers, not just their bytes. On an M5 Max
`mx.device_info()["resource_limit"]` is 499000, and exceeding it raises
``[metal::malloc] Resource limit (499000) exceeded`` mid-generation, which used
to surface as a 500 after ~12 minutes of work.

Measured live, sampling the live-buffer count by allocator saturation while
decoding:

    Qwen3.6-27B (64 layers)   0.000 buffers/token   flat at 1978 over 600 steps
    DeepSeek-V4-Flash (43)    ~42   buffers/token   one per layer per token

A conventional mlx_lm KVCache grows in 256-token steps and reuses buffers;
DSV4 keeps cumulative per-layer compressor/indexer pool state. `mx.clear_cache()`
frees none of it, so the ceiling is hard and has to be projected rather than
recovered from.

These tests pin both halves: the projection maths, and the requirement that a
family whose retention has not been measured is never capped on a limit it
provably cannot reach.
"""
from vmlx_engine.utils.memory_limits import (
    projected_output_token_cap_by_buffers,
    retained_buffers_per_token,
)


class TestRetainedBuffersPerToken:
    def test_deepseek_v4_retains_one_buffer_per_layer_per_token(self):
        config = {"model_type": "deepseek_v4", "num_hidden_layers": 43}
        assert retained_buffers_per_token(config) == 43.0

    def test_deepseek_v4_reads_layers_from_text_config(self):
        config = {"text_config": {"model_type": "deepseek_v4",
                                  "num_hidden_layers": 61}}
        assert retained_buffers_per_token(config) == 61.0

    def test_conventional_kv_cache_families_report_no_retention(self):
        # Measured: Qwen3.6-27B held its live-buffer count flat across 600
        # decode steps. Reporting a rate here would cap a family on a ceiling
        # it never approaches.
        for model_type in ("qwen3", "llama", "gemma4", "minimax_m3", "laguna"):
            config = {"model_type": model_type, "num_hidden_layers": 64}
            assert retained_buffers_per_token(config) == 0.0, model_type

    def test_missing_or_malformed_config_is_not_capped(self):
        assert retained_buffers_per_token(None) == 0.0
        assert retained_buffers_per_token({}) == 0.0
        assert retained_buffers_per_token({"model_type": "deepseek_v4"}) == 0.0
        assert retained_buffers_per_token(
            {"model_type": "deepseek_v4", "num_hidden_layers": 0}) == 0.0


class TestProjectedOutputTokenCapByBuffers:
    def test_dsv4_cap_lands_below_the_measured_ceiling(self):
        # 12000 generated tokens succeeded; 14336 died. The cap must sit under
        # the failure point without being needlessly punitive.
        cap = projected_output_token_cap_by_buffers(
            resource_limit=499000,
            live_baseline_buffers=16384,
            buffers_per_token=43.0,
        )
        assert cap is not None
        assert 8000 <= cap < 12000, cap

    def test_zero_retention_returns_none_so_the_guard_stays_inert(self):
        assert projected_output_token_cap_by_buffers(
            resource_limit=499000,
            live_baseline_buffers=16384,
            buffers_per_token=0.0,
        ) is None

    def test_unknown_resource_limit_returns_none(self):
        assert projected_output_token_cap_by_buffers(
            resource_limit=0,
            live_baseline_buffers=0,
            buffers_per_token=43.0,
        ) is None

    def test_cap_scales_inversely_with_layer_count(self):
        few = projected_output_token_cap_by_buffers(
            resource_limit=499000, live_baseline_buffers=0,
            buffers_per_token=16.0)
        many = projected_output_token_cap_by_buffers(
            resource_limit=499000, live_baseline_buffers=0,
            buffers_per_token=64.0)
        assert few > many

    def test_baseline_is_deducted_from_the_budget(self):
        no_baseline = projected_output_token_cap_by_buffers(
            resource_limit=499000, live_baseline_buffers=0,
            buffers_per_token=43.0)
        with_baseline = projected_output_token_cap_by_buffers(
            resource_limit=499000, live_baseline_buffers=100000,
            buffers_per_token=43.0)
        assert with_baseline < no_baseline

    def test_baseline_larger_than_budget_yields_zero_not_negative(self):
        assert projected_output_token_cap_by_buffers(
            resource_limit=499000,
            live_baseline_buffers=10_000_000,
            buffers_per_token=43.0,
        ) == 0

    def test_malformed_inputs_return_none(self):
        assert projected_output_token_cap_by_buffers(
            resource_limit="nope", live_baseline_buffers=0,
            buffers_per_token=43.0) is None
        assert projected_output_token_cap_by_buffers(
            resource_limit=499000, live_baseline_buffers=0,
            buffers_per_token="nope") is None

    def test_safety_fraction_reduces_the_cap(self):
        strict = projected_output_token_cap_by_buffers(
            resource_limit=499000, live_baseline_buffers=0,
            buffers_per_token=43.0, safety_fraction=0.5)
        loose = projected_output_token_cap_by_buffers(
            resource_limit=499000, live_baseline_buffers=0,
            buffers_per_token=43.0, safety_fraction=1.0)
        assert strict < loose

    def test_out_of_range_safety_fraction_falls_back_to_default(self):
        default = projected_output_token_cap_by_buffers(
            resource_limit=499000, live_baseline_buffers=0,
            buffers_per_token=43.0)
        for bad in (0.0, -1.0, 1.5):
            assert projected_output_token_cap_by_buffers(
                resource_limit=499000, live_baseline_buffers=0,
                buffers_per_token=43.0, safety_fraction=bad) == default
