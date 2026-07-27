import json
from types import SimpleNamespace


def _qwen_hybrid_model_config():
    return {
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
        },
    }


def _lfm_hybrid_model_config():
    return {
        "model_type": "lfm2_moe",
        "num_hidden_layers": 6,
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "layer_types": [
            "conv",
            "full_attention",
            "conv",
            "conv",
            "full_attention",
            "conv",
        ],
    }


def test_qwen_hybrid_auto_keeps_live_transition_off(monkeypatch):
    from vmlx_engine.utils.turboquant_config import (
        QWEN_HYBRID_LIVE_TQ_COMPRESS_AFTER,
        resolve_compress_after,
    )

    monkeypatch.delenv("VMLINUX_TQ_COMPRESS_AFTER", raising=False)

    assert resolve_compress_after({}, _qwen_hybrid_model_config()) == 0
    assert QWEN_HYBRID_LIVE_TQ_COMPRESS_AFTER == 0


def test_explicit_qwen_hybrid_live_tq_zero_remains_disabled(monkeypatch):
    from vmlx_engine.utils.turboquant_config import resolve_compress_after

    monkeypatch.delenv("VMLINUX_TQ_COMPRESS_AFTER", raising=False)

    assert resolve_compress_after(
        {"compress_after": 0}, _qwen_hybrid_model_config()
    ) == 0


def test_qwen_full_attention_only_does_not_take_hybrid_default(monkeypatch):
    from vmlx_engine.utils.turboquant_config import resolve_compress_after

    monkeypatch.delenv("VMLINUX_TQ_COMPRESS_AFTER", raising=False)
    config = _qwen_hybrid_model_config()
    config["text_config"]["layer_types"] = ["full_attention"] * 4

    assert resolve_compress_after({}, config) == 0


def test_qwen_cache_classifier_distinguishes_subtypes():
    from vmlx_engine.utils.hybrid_tq_cache import (
        classify_qwen_cache_architecture,
    )

    hybrid = ["ssm", "ssm", "ssm", "attention"]
    assert classify_qwen_cache_architecture(
        _qwen_hybrid_model_config(), hybrid
    ) == "qwen3_5_hybrid_gated_delta"
    assert classify_qwen_cache_architecture(
        {"model_type": "qwen3_next"}, hybrid
    ) == "qwen3_next_hybrid_gated_delta"
    assert classify_qwen_cache_architecture(
        {"model_type": "qwen3"}, ["attention"] * 4
    ) == "qwen_full_kv"
    assert classify_qwen_cache_architecture(
        {"model_type": "qwen3_moe"}, ["attention"] * 4
    ) == "qwen_full_kv"
    assert classify_qwen_cache_architecture(
        {"model_type": "qwen_mamba"}, ["ssm"] * 4
    ) == "qwen_cumulative_only"
    assert classify_qwen_cache_architecture(
        {"model_type": "qwen3_vl"}, hybrid
    ) == "unsupported"


def test_lfm_hybrid_is_selective_tq_supported_only_with_attention_and_companions():
    from vmlx_engine.utils.hybrid_tq_cache import is_selective_hybrid_tq_supported

    config = _lfm_hybrid_model_config()
    assert is_selective_hybrid_tq_supported(
        config, ["ssm", "attention", "ssm"]
    ) is True
    assert is_selective_hybrid_tq_supported(config, ["ssm"] * 3) is False
    assert is_selective_hybrid_tq_supported(config, ["attention"] * 3) is False


def test_uncalibrated_qwen_hybrid_auto_uses_tq4_on_real_kv_positions():
    from vmlx_engine.utils.turboquant_config import (
        apply_uncalibrated_auto_tq_policy,
    )

    layer_types = ["ssm", "ssm", "ssm", "attention"] * 16
    resolved = apply_uncalibrated_auto_tq_policy(
        {"enabled": True, "default_key_bits": 3, "seed": 42},
        _qwen_hybrid_model_config(),
        layer_types,
    )

    assert resolved["default_key_bits"] == 4
    assert resolved["default_value_bits"] == 4
    assert resolved["critical_key_bits"] == 4
    assert resolved["critical_value_bits"] == 4
    assert resolved["critical_layers"] == list(range(3, 64, 4))
    assert resolved["sink_tokens"] == 0
    assert resolved["compress_after"] == 0
    assert resolved["auto_policy"] == "qwen_hybrid_attention_kv_storage_tq4"


def test_bonsai_qwen_hybrid_auto_retains_tq8_exception():
    from vmlx_engine.utils.turboquant_config import (
        apply_uncalibrated_auto_tq_policy,
    )

    config = _qwen_hybrid_model_config()
    config["_name_or_path"] = "Bonsai-27b-1bit-JANG"
    layer_types = ["ssm", "ssm", "ssm", "attention"] * 16

    resolved = apply_uncalibrated_auto_tq_policy(
        {"enabled": True, "default_key_bits": 3, "seed": 42},
        config,
        layer_types,
    )

    assert resolved["default_key_bits"] == 8
    assert resolved["default_value_bits"] == 8
    assert resolved["critical_layers"] == list(range(3, 64, 4))
    assert resolved["auto_policy"] == "bonsai_hybrid_attention_kv_storage_tq8"


def test_uncalibrated_qwen_full_kv_auto_uses_tq4_with_q8_critical_layers():
    from vmlx_engine.utils.turboquant_config import (
        apply_uncalibrated_auto_tq_policy,
    )

    layer_types = ["attention"] * 28
    resolved = apply_uncalibrated_auto_tq_policy(
        {
            "enabled": True,
            "default_key_bits": 3,
            "default_value_bits": 3,
            "critical_key_bits": 4,
            "critical_value_bits": 4,
            "seed": 42,
        },
        {"model_type": "qwen3"},
        layer_types,
    )

    assert resolved["default_key_bits"] == 4
    assert resolved["default_value_bits"] == 4
    assert resolved["critical_key_bits"] == 8
    assert resolved["critical_value_bits"] == 8
    assert resolved["critical_layers"] == [
        0, 1, 2, 3, 4, 5, -6, -5, -4, -3, -2, -1
    ]
    assert resolved["sink_tokens"] == 0
    assert resolved["compress_after"] == 0
    assert resolved["auto_policy"] == "qwen_full_kv_storage_tq4_critical_tq8"


def test_qwen_full_kv_mixed_policy_builds_q4_bulk_and_q8_critical_slots():
    from vmlx_engine.utils.turboquant_config import (
        TurboQuantConfig,
        apply_uncalibrated_auto_tq_policy,
        make_turboquant_cache,
    )

    layer_types = ["attention"] * 28
    resolved = apply_uncalibrated_auto_tq_policy(
        {"enabled": True, "seed": 42},
        {"model_type": "qwen3"},
        layer_types,
    )
    config = TurboQuantConfig.from_jang_config(
        {"turboquant": resolved}, n_layers=28
    )
    assert config is not None
    cache = make_turboquant_cache(
        config,
        n_layers=28,
        key_dims=[128] * 28,
        value_dims=[128] * 28,
        layer_types=layer_types,
    )

    assert [layer.key_bits for layer in cache[:7]] == [8, 8, 8, 8, 8, 8, 4]
    assert [layer.value_bits for layer in cache[-7:]] == [4, 8, 8, 8, 8, 8, 8]


def test_uncalibrated_mixed_swa_auto_uses_q4_on_full_attention_only():
    from vmlx_engine.utils.turboquant_config import apply_mixed_swa_auto_tq_policy

    resolved = apply_mixed_swa_auto_tq_policy(
        {"enabled": True, "default_key_bits": 8, "default_value_bits": 8},
        ["sliding", "sliding", "attention", "sliding", "attention"],
    )

    assert resolved["default_key_bits"] == 4
    assert resolved["default_value_bits"] == 4
    assert resolved["critical_key_bits"] == 4
    assert resolved["critical_value_bits"] == 4
    assert resolved["critical_layers"] == [2, 4]
    assert resolved["compress_after"] == 0
    assert resolved["auto_policy"] == "mixed_swa_full_attention_kv_storage_tq4"


def test_uncalibrated_qwen_cumulative_only_does_not_get_fake_tq_slots():
    from vmlx_engine.utils.turboquant_config import (
        apply_uncalibrated_auto_tq_policy,
    )

    original = {"enabled": True, "default_key_bits": 3, "seed": 42}
    resolved = apply_uncalibrated_auto_tq_policy(
        original,
        {"model_type": "qwen_mamba"},
        ["ssm"] * 8,
    )

    assert resolved == original


def test_uncalibrated_laguna_full_kv_auto_uses_tq4_on_all_attention_slots():
    from vmlx_engine.utils.turboquant_config import (
        apply_uncalibrated_auto_tq_policy,
    )

    layer_types = ["attention"] * 70
    resolved = apply_uncalibrated_auto_tq_policy(
        {
            "enabled": True,
            "default_key_bits": 3,
            "default_value_bits": 3,
            "critical_key_bits": 4,
            "critical_value_bits": 4,
            "critical_layers": [0, 1, 2, -3, -2, -1],
            "sink_tokens": 4,
            "seed": 42,
        },
        {"model_type": "laguna", "num_hidden_layers": 70},
        layer_types,
    )

    assert resolved["default_key_bits"] == 4
    assert resolved["default_value_bits"] == 4
    assert resolved["critical_key_bits"] == 4
    assert resolved["critical_value_bits"] == 4
    assert resolved["critical_layers"] == list(range(70))
    assert resolved["sink_tokens"] == 0
    assert resolved["compress_after"] == 0
    assert resolved["auto_policy"] == "uncalibrated_full_kv_storage_tq4"


def test_uncalibrated_hy3_full_kv_auto_uses_tq4_on_all_attention_slots():
    from vmlx_engine.utils.turboquant_config import (
        apply_uncalibrated_auto_tq_policy,
    )

    layer_types = ["attention"] * 80
    resolved = apply_uncalibrated_auto_tq_policy(
        {
            "enabled": True,
            "default_key_bits": 3,
            "default_value_bits": 3,
            "critical_key_bits": 4,
            "critical_value_bits": 4,
            "seed": 42,
        },
        {"model_type": "hy_v3", "num_hidden_layers": 80},
        layer_types,
    )

    assert resolved["default_key_bits"] == 4
    assert resolved["default_value_bits"] == 4
    assert resolved["critical_key_bits"] == 4
    assert resolved["critical_value_bits"] == 4
    assert resolved["critical_layers"] == list(range(80))
    assert resolved["sink_tokens"] == 0
    assert resolved["compress_after"] == 0
    assert resolved["auto_policy"] == "hy3_full_kv_storage_tq4"


def test_uncalibrated_minimax_m2_full_kv_auto_uses_tq8_safety_policy():
    from vmlx_engine.utils.turboquant_config import (
        apply_uncalibrated_auto_tq_policy,
    )

    layer_types = ["attention"] * 62
    resolved = apply_uncalibrated_auto_tq_policy(
        {
            "enabled": True,
            "default_key_bits": 3,
            "default_value_bits": 3,
            "critical_key_bits": 4,
            "critical_value_bits": 4,
            "seed": 42,
        },
        {"model_type": "minimax_m2", "num_hidden_layers": 62},
        layer_types,
    )

    assert resolved["default_key_bits"] == 8
    assert resolved["default_value_bits"] == 8
    assert resolved["critical_key_bits"] == 8
    assert resolved["critical_value_bits"] == 8
    assert resolved["critical_layers"] == list(range(62))
    assert resolved["sink_tokens"] == 0
    assert resolved["compress_after"] == 0
    assert resolved["auto_policy"] == "minimax_m2_full_kv_storage_tq8"


def test_tq_codec_signature_separates_bits_and_prefix_cache_namespaces():
    from vmlx_engine.prefix_cache import compute_model_cache_key
    from vmlx_engine.utils.turboquant_config import (
        TurboQuantConfig,
        turboquant_storage_signature,
    )

    class Model:
        pass

    def model_with_bits(bits):
        config = TurboQuantConfig(
            n_layers=4,
            default_key_bits=bits,
            default_value_bits=bits,
            critical_key_bits=bits,
            critical_value_bits=bits,
            critical_layers=[0, 1, 2, 3],
            sink_tokens=0,
            seed=42,
            compress_after=0,
        )

        def make_cache():
            return []

        make_cache._vmlx_tq_storage_signature = turboquant_storage_signature(
            config, "uncalibrated_full_kv_storage_tq4"
        )
        model = Model()
        model.make_cache = make_cache
        return model

    key3 = compute_model_cache_key(model_with_bits(3), tq_enabled=True)
    key8 = compute_model_cache_key(model_with_bits(8), tq_enabled=True)

    assert key3 != key8


class NativeGatedDeltaState:
    """Sentinel for Qwen hybrid non-KV companion state."""


class FakeHybridModel:
    def __init__(self, n_layers=4):
        self.layers = [object()] * n_layers

    def make_cache(self):
        from mlx_lm.models.cache import KVCache

        return [
            KVCache(),
            NativeGatedDeltaState(),
            KVCache(),
            NativeGatedDeltaState(),
        ]


class FakeLFMModel:
    def __init__(self):
        self.layers = [object()] * 6

    def make_cache(self):
        from mlx_lm.models.cache import ArraysCache, KVCache

        return [
            ArraysCache(size=1),
            KVCache(),
            ArraysCache(size=1),
            ArraysCache(size=1),
            KVCache(),
            ArraysCache(size=1),
        ]


class FakeBonsaiModel:
    def __init__(self):
        self.layers = [object()] * 64

    def make_cache(self):
        from mlx_lm.models.cache import KVCache

        return [
            KVCache() if index % 4 == 3 else NativeGatedDeltaState()
            for index in range(64)
        ]


class FakeMixedSwaModel:
    def __init__(self):
        self.layers = [object(), object()]

    def make_cache(self):
        from mlx_lm.models.cache import KVCache, RotatingKVCache

        return [RotatingKVCache(max_size=1024), KVCache()]


class FakeFullKVModel:
    def __init__(self, n_layers=4):
        self.layers = [object()] * n_layers

    def make_cache(self):
        from mlx_lm.models.cache import KVCache

        return [KVCache() for _ in self.layers]


def _write_qwen36_hybrid_config(path):
    config = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 4,
            "head_dim": 128,
            "layer_types": [
                "full_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
            ],
        },
    }
    (path / "config.json").write_text(json.dumps(config))
    return config


def _write_lfm_hybrid_config(path):
    config = _lfm_hybrid_model_config()
    (path / "config.json").write_text(json.dumps(config))
    return config


def test_standard_qwen_hybrid_tq_patches_attention_cache_only(tmp_path, monkeypatch):
    from vmlx_engine.utils.tokenizer import _apply_turboquant_to_model

    _write_qwen36_hybrid_config(tmp_path)
    model = FakeHybridModel()
    monkeypatch.delenv("VMLX_DISABLE_TQ_KV", raising=False)
    monkeypatch.delenv("VMLX_ALLOW_HYBRID_KV_QUANT", raising=False)

    _apply_turboquant_to_model(model, str(tmp_path))

    cache = model.make_cache()
    assert [type(slot).__name__ for slot in cache] == [
        "TurboQuantKVCache",
        "NativeGatedDeltaState",
        "TurboQuantKVCache",
        "NativeGatedDeltaState",
    ]
    assert getattr(model.make_cache, "_vmlx_hybrid_tq_policy") == "attention_kv_only"
    assert getattr(model.make_cache, "_vmlx_hybrid_tq_attention_layers") == (0, 2)
    assert getattr(model.make_cache, "_vmlx_hybrid_tq_companion_layers") == (1, 3)


def test_standard_lfm_hybrid_auto_uses_q4_tq_on_real_kv_slots_only(
    tmp_path, monkeypatch
):
    from vmlx_engine.utils.tokenizer import _apply_turboquant_to_model

    _write_lfm_hybrid_config(tmp_path)
    model = FakeLFMModel()
    monkeypatch.delenv("VMLX_DISABLE_TQ_KV", raising=False)
    monkeypatch.setenv("VMLX_FORCE_TQ_AUTO", "1")

    _apply_turboquant_to_model(model, str(tmp_path))

    cache = model.make_cache()
    assert [type(slot).__name__ for slot in cache] == [
        "ArraysCache",
        "TurboQuantKVCache",
        "ArraysCache",
        "ArraysCache",
        "TurboQuantKVCache",
        "ArraysCache",
    ]
    assert getattr(model.make_cache, "_vmlx_hybrid_tq_attention_layers") == (1, 4)
    assert getattr(model.make_cache, "_vmlx_hybrid_tq_companion_layers") == (
        0,
        2,
        3,
        5,
    )
    assert getattr(model.make_cache, "_vmlx_tq_auto_policy") == (
        "uncalibrated_selective_attention_kv_storage_tq4"
    )
    for index in (1, 4):
        assert cache[index].key_bits == 4
        assert cache[index].value_bits == 4
        assert cache[index].key_dim == 64
        assert cache[index].value_dim == 64
        assert cache[index].compress_after == 0


def test_jang_qwen_hybrid_tq_patches_attention_cache_only(monkeypatch):
    from vmlx_engine.utils.jang_loader import _patch_turboquant_make_cache

    model = FakeHybridModel()
    model_config = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 4,
            "head_dim": 128,
            "layer_types": [
                "full_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
            ],
        },
    }
    monkeypatch.delenv("VMLX_DISABLE_TQ_KV", raising=False)
    monkeypatch.delenv("VMLX_ALLOW_HYBRID_KV_QUANT", raising=False)

    _patch_turboquant_make_cache(
        model,
        {
            "turboquant": {
                "enabled": True,
                "default_key_bits": 3,
                "default_value_bits": 3,
                "critical_key_bits": 4,
                "critical_value_bits": 4,
                "critical_layers": [0, -1],
                "seed": 42,
            }
        },
        model_config,
    )

    cache = model.make_cache()
    assert [type(slot).__name__ for slot in cache] == [
        "TurboQuantKVCache",
        "NativeGatedDeltaState",
        "TurboQuantKVCache",
        "NativeGatedDeltaState",
    ]
    assert getattr(model.make_cache, "_vmlx_hybrid_tq_policy") == "attention_kv_only"
    assert getattr(model.make_cache, "_vmlx_hybrid_tq_attention_layers") == (0, 2)
    assert getattr(model.make_cache, "_vmlx_hybrid_tq_companion_layers") == (1, 3)


def test_jang_mixed_swa_auto_tq_patches_only_full_attention_slot(monkeypatch):
    from vmlx_engine.utils.jang_loader import _patch_turboquant_make_cache

    model = FakeMixedSwaModel()
    model_config = {
        "model_type": "gemma4_unified",
        "text_config": {
            "model_type": "gemma4_unified_text",
            "num_hidden_layers": 2,
            "head_dim": 256,
            "global_head_dim": 512,
            "layer_types": ["sliding_attention", "full_attention"],
        },
    }
    monkeypatch.setenv("VMLX_FORCE_TQ_AUTO", "1")
    monkeypatch.delenv("VMLX_SWA_TQ", raising=False)
    monkeypatch.delenv("VMLX_DISABLE_TQ_KV", raising=False)

    _patch_turboquant_make_cache(model, {}, model_config)

    cache = model.make_cache()
    assert [type(slot).__name__ for slot in cache] == [
        "RotatingKVCache",
        "TurboQuantKVCache",
    ]
    assert getattr(model.make_cache, "_vmlx_hybrid_tq_attention_layers") == (1,)
    assert getattr(model.make_cache, "_vmlx_hybrid_tq_companion_layers") == (0,)
    assert getattr(model.make_cache, "_vmlx_tq_auto_policy") == (
        "mixed_swa_full_attention_kv_storage_tq4"
    )
    assert cache[1].key_bits == 4
    assert cache[1].value_bits == 4


def test_jang_bonsai_auto_uses_tq8_only_on_attention_slots(monkeypatch):
    from vmlx_engine.utils.jang_loader import _patch_turboquant_make_cache

    model = FakeBonsaiModel()
    model_config = {
        "model_type": "qwen3_5",
        "_name_or_path": "Bonsai-27b-1bit-JANG",
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 64,
            "head_dim": 256,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        },
    }
    model_config["text_config"]["layer_types"] *= 16
    monkeypatch.setenv("VMLX_FORCE_TQ_AUTO", "1")
    monkeypatch.delenv("VMLX_DISABLE_TQ_KV", raising=False)
    monkeypatch.delenv("VMLINUX_TQ_COMPRESS_AFTER", raising=False)

    _patch_turboquant_make_cache(model, {}, model_config)

    cache = model.make_cache()
    attention_positions = tuple(range(3, 64, 4))
    assert getattr(model.make_cache, "_vmlx_hybrid_tq_attention_layers") == attention_positions
    assert getattr(model.make_cache, "_vmlx_hybrid_tq_companion_layers") == tuple(
        index for index in range(64) if index not in attention_positions
    )
    assert getattr(model.make_cache, "_vmlx_tq_auto_policy") == (
        "bonsai_hybrid_attention_kv_storage_tq8"
    )
    for index, layer in enumerate(cache):
        if index in attention_positions:
            assert type(layer).__name__ == "TurboQuantKVCache"
            assert layer.key_bits == 8
            assert layer.value_bits == 8
            assert layer.compress_after == 0
            assert layer.sink_tokens == 0
        else:
            assert isinstance(layer, NativeGatedDeltaState)


def test_jang_hy3_auto_uses_tq4_on_all_attention_slots(monkeypatch):
    from vmlx_engine.utils.jang_loader import _patch_turboquant_make_cache

    model = FakeFullKVModel(n_layers=4)
    model_config = {
        "model_type": "hy_v3",
        "num_hidden_layers": 4,
        "head_dim": 128,
    }
    monkeypatch.setenv("VMLX_FORCE_TQ_AUTO", "1")
    monkeypatch.delenv("VMLX_DISABLE_TQ_KV", raising=False)
    monkeypatch.delenv("VMLINUX_TQ_COMPRESS_AFTER", raising=False)

    _patch_turboquant_make_cache(model, {}, model_config)

    cache = model.make_cache()
    assert [type(slot).__name__ for slot in cache] == ["TurboQuantKVCache"] * 4
    assert getattr(model.make_cache, "_vmlx_tq_auto_policy") == (
        "hy3_full_kv_storage_tq4"
    )
    assert all(layer.key_bits == 4 for layer in cache)
    assert all(layer.value_bits == 4 for layer in cache)


def test_turboquant_batch_cache_deepcopy_preserves_dtype_and_array_ownership():
    import copy

    import mlx.core as mx

    from vmlx_engine.utils.turboquant_config import (
        TurboQuantConfig,
        make_turboquant_cache,
    )

    config = TurboQuantConfig(
        n_layers=1,
        default_key_bits=4,
        default_value_bits=4,
        critical_key_bits=4,
        critical_value_bits=4,
        critical_layers=[0],
        sink_tokens=0,
        seed=42,
        compress_after=0,
    )
    cache = make_turboquant_cache(
        config,
        n_layers=1,
        key_dims=[8],
        value_dims=[8],
        layer_types=["attention"],
    )[0]
    cache.prepare(left_padding=[0])
    keys = mx.arange(8, dtype=mx.float16).reshape(1, 1, 1, 8)
    values = (keys + 10).astype(mx.bfloat16)
    cache.update_and_fetch(keys, values)

    clone = copy.deepcopy([cache])[0]

    assert clone is not cache
    assert clone._vmlx_tq_key_dtype == mx.float16
    assert clone._vmlx_tq_value_dtype == mx.bfloat16
    assert clone.keys is not cache.keys
    assert clone.values is not cache.values
    clone.keys[..., 0, 0] = 99
    mx.eval(cache.keys, clone.keys)
    assert float(cache.keys[..., 0, 0].item()) == 0.0
    assert float(clone.keys[..., 0, 0].item()) == 99.0

    clone.filter([0])
    assert clone._idx is None
    assert cache._idx == 1


def test_native_cache_status_reports_hybrid_live_attention_tq():
    from vmlx_engine.server import _native_cache_status

    scheduler = SimpleNamespace(
        config=SimpleNamespace(kv_cache_quantization="q4"),
        _model_type_for_runtime="qwen3_5_moe",
        _is_hybrid=True,
        _uses_dsv4_cache=False,
        _uses_zaya_cache=False,
        _tq_active=True,
        _hybrid_live_tq_policy="attention_kv_only",
        _hybrid_live_tq_attention_layers=[0, 2],
        _hybrid_live_tq_companion_layers=[1, 3],
        # compress_after=0 => objects-only, no live decode-time encoding.
        _hybrid_live_tq_compress_after=0,
        _hybrid_tq_auto_policy="qwen_hybrid_attention_kv_storage_tq4",
        _hybrid_tq_default_key_bits=4,
        _hybrid_tq_default_value_bits=4,
        _hybrid_kv_positions=[0, 2],
        _kv_cache_bits=4,
        _kv_cache_group_size=64,
        _ssm_state_cache=SimpleNamespace(_store={"prompt": object()}),
        block_aware_cache=object(),
        paged_cache_manager=SimpleNamespace(_disk_store=object()),
    )

    status = _native_cache_status(scheduler)

    assert status["generic_turboquant_kv"] == {
        "enabled": True,
        "reason": "hybrid_attention_kv_only",
    }
    # Objects-only run (compress_after=0): must NOT over-claim "live_decode".
    assert status["live_attention_tq_kv"] == {
        "enabled": True,
        "mode": "attention_kv_objects",
        "compress_after": 0,
        "applies_to": "attention_kv_layers_only",
        "ssm_policy": "native_full_precision_companion_state",
        "attention_layers": [0, 2],
        "companion_layers": [1, 3],
    }
    assert status["attention_kv_storage_quantization"]["mode"] == "storage_boundary"
    assert status["attention_kv_storage_quantization"] == {
        "enabled": True,
        "mode": "storage_boundary",
        "codec": "turboquant_native",
        "bits": 4,
        "value_bits": 4,
        "group_size": None,
        "auto_policy": "qwen_hybrid_attention_kv_storage_tq4",
        "applies_to": "attention_kv_layers_only",
        "ssm_policy": "native_companion_state",
        "rederive": "async_clean_prefill_on_miss_or_warm_pass",
    }

    # When compress_after > 0, live encode is genuinely enabled -> "live_decode".
    scheduler._hybrid_live_tq_compress_after = 16
    status_live = _native_cache_status(scheduler)
    assert status_live["live_attention_tq_kv"]["mode"] == "live_decode"
    assert status_live["live_attention_tq_kv"]["compress_after"] == 16


def test_mllm_hybrid_prompt_truncation_preserves_tq_storage_metadata():
    import mlx.core as mx
    from jang_tools.turboquant.cache import TurboQuantKVCache
    from mlx_lm.models.cache import KVCache
    from vmlx_engine.mllm_scheduler import MLLMScheduler

    layer = TurboQuantKVCache(
        key_dim=16,
        value_dim=16,
        key_bits=8,
        value_bits=8,
        seed=49,
        compress_after=0,
        sink_tokens=0,
    )
    raw = KVCache()
    raw.keys = mx.arange(1 * 2 * 12 * 16).reshape(1, 2, 12, 16).astype(mx.float16)
    raw.values = (raw.keys + 1).astype(mx.float16)
    raw.offset = 12

    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler._tq_active = True
    scheduler.batch_generator = SimpleNamespace(
        language_model=SimpleNamespace(make_cache=lambda: [layer])
    )
    scheduler._detect_n_kv_heads = lambda: 2
    scheduler._detect_allowed_n_kv_heads = lambda: set()

    prepared = scheduler._prepare_tq_cache_for_storage([raw])
    assert prepared is not None
    assert type(prepared[0]).__name__ == "TurboQuantKVCache"
    truncated = scheduler._truncate_hybrid_cache(prepared, prompt_len=9)
    assert truncated is not None
    assert len(truncated) == 1
    stored = truncated[0]
    assert type(stored).__name__ == "TurboQuantKVCache"
    assert stored.offset == 8
    assert stored.key_bits == 8
    assert stored.value_bits == 8
    assert stored._seed == 49
    assert stored.sink_tokens == 0

    extracted = scheduler._extract_cache_states(truncated)
    assert extracted[0]["class_name"] == "TurboQuantKVCache"
    assert extracted[0]["tq_config"] == {
        "key_bits": 8,
        "value_bits": 8,
        "seed": 49,
    }
