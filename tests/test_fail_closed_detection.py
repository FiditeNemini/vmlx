"""H5: cache/parser architecture detection must not fail open."""

import json

import pytest

from vmlx_engine.mllm_scheduler import MLLMScheduler
from vmlx_engine.model_config_registry import ModelConfig, ModelConfigRegistry
from vmlx_engine.scheduler import Scheduler
from vmlx_engine.utils.jang_loader import _patch_turboquant_make_cache
from vmlx_engine.utils.model_inspector import is_mla_model


class _BrokenCacheModel:
    layers = [object()]

    def make_cache(self):
        raise RuntimeError("cache construction failed")


def test_llm_hybrid_detector_refuses_plain_kv_fallback():
    with pytest.raises(RuntimeError, match="refusing to classify"):
        Scheduler._is_hybrid_model(_BrokenCacheModel())


def test_mllm_hybrid_detector_refuses_plain_kv_fallback():
    with pytest.raises(RuntimeError, match="refusing to classify"):
        MLLMScheduler._is_hybrid_model(_BrokenCacheModel())


def test_mla_detector_treats_unreadable_or_unknown_metadata_conservatively(tmp_path):
    (tmp_path / "config.json").write_text("{not-json")
    assert is_mla_model(tmp_path) is True
    assert is_mla_model(None) is True
    assert is_mla_model({"model_type": "llama"}) is False


def test_turboquant_keeps_native_make_cache_when_native_layout_probe_fails():
    model = _BrokenCacheModel()
    original = model.make_cache
    _patch_turboquant_make_cache(
        model,
        {"turboquant": {"enabled": True}},
        {"model_type": "llama", "num_hidden_layers": 1},
    )
    assert model.make_cache.__func__ is original.__func__


def test_invalid_jang_stamp_fails_closed_instead_of_falling_through(tmp_path):
    (tmp_path / "jang_config.json").write_text("{not-json")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    with pytest.raises(RuntimeError, match="invalid authoritative JANG stamp"):
        ModelConfigRegistry().lookup(str(tmp_path))


def test_incomplete_jang_stamp_falls_back_to_structural_config_family(
    tmp_path, monkeypatch, caplog
):
    """Pre-family JANG sidecars must not make an otherwise known model unloadable."""
    import vmlx_engine.model_config_registry as registry_module

    model_config = {"model_type": "lfm2_moe"}
    (tmp_path / "config.json").write_text(json.dumps(model_config))
    (tmp_path / "jang_config.json").write_text(
        json.dumps(
            {
                "capabilities": {
                    "cache_type": "kv",
                    "reasoning_parser": "deepseek_r1",
                    "tool_parser": "deepseek",
                }
            }
        )
    )
    monkeypatch.setattr(registry_module, "load_config", lambda _path: model_config)
    registry = ModelConfigRegistry()
    registry.register(
        ModelConfig(
            family_name="lfm2",
            model_types=["lfm2", "lfm2_moe"],
            cache_type="hybrid",
            cache_subtype="lfm2_moe_hybrid_ssm",
            reasoning_parser="qwen3",
            tool_parser="lfm2",
            priority=10,
        )
    )

    result = registry.lookup(str(tmp_path))

    assert result.family_name == "lfm2"
    assert result.cache_type == "hybrid"
    assert result.cache_subtype == "lfm2_moe_hybrid_ssm"
    assert result.reasoning_parser == "qwen3"
    assert result.tool_parser == "lfm2"
    assert "falling back to config.json architecture detection" in caplog.text


def test_partial_capabilities_accept_older_top_level_model_family(tmp_path):
    """DSV4-era top-level model_family remains an authoritative identity."""
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "unregistered_wrapper"})
    )
    (tmp_path / "jang_config.json").write_text(
        json.dumps(
            {
                "model_family": "legacy_family",
                "capabilities": {
                    "cache_type": "hybrid",
                    "reasoning_parser": "qwen3",
                    "tool_parser": "qwen",
                },
            }
        )
    )
    registry = ModelConfigRegistry()
    registry.register(
        ModelConfig(
            family_name="legacy_family",
            model_types=["legacy_model_type"],
            cache_type="kv",
            priority=10,
        )
    )

    result = registry.lookup(str(tmp_path))

    assert result.family_name == "legacy_family"
    assert result.cache_type == "hybrid"
    assert result.reasoning_parser == "qwen3"
    assert result.tool_parser == "qwen"


def test_incomplete_jang_stamp_unknown_architecture_stays_conservative(
    tmp_path, monkeypatch
):
    """A partial stamp cannot promote unsafe cache or parser guesses."""
    import vmlx_engine.model_config_registry as registry_module

    model_config = {"model_type": "totally_unknown_type"}
    (tmp_path / "config.json").write_text(json.dumps(model_config))
    (tmp_path / "jang_config.json").write_text(
        json.dumps(
            {
                "capabilities": {
                    "cache_type": "hybrid",
                    "reasoning_parser": "qwen3",
                    "tool_parser": "qwen",
                }
            }
        )
    )
    monkeypatch.setattr(registry_module, "load_config", lambda _path: model_config)

    result = ModelConfigRegistry().lookup(str(tmp_path))

    assert result.family_name == "unknown"
    assert result.cache_type == "native"
    assert result.reasoning_parser is None
    assert result.tool_parser is None
    assert result.architecture_hints["force_native_cache"] is True


def test_incomplete_chat_stamp_falls_back_to_structural_config_family(
    tmp_path, monkeypatch
):
    """The older chat schema is also optional when model_family is absent."""
    import vmlx_engine.model_config_registry as registry_module

    model_config = {"model_type": "lfm2_moe"}
    (tmp_path / "config.json").write_text(json.dumps(model_config))
    (tmp_path / "jang_config.json").write_text(
        json.dumps({"chat": {"reasoning": {"supported": True}}})
    )
    monkeypatch.setattr(registry_module, "load_config", lambda _path: model_config)
    registry = ModelConfigRegistry()
    registry.register(
        ModelConfig(
            family_name="lfm2",
            model_types=["lfm2", "lfm2_moe"],
            cache_type="hybrid",
            tool_parser="lfm2",
            priority=10,
        )
    )

    result = registry.lookup(str(tmp_path))

    assert result.family_name == "lfm2"
    assert result.cache_type == "hybrid"
    assert result.tool_parser == "lfm2"


def test_non_string_jang_family_is_invalid_instead_of_guessed(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "lfm2_moe"}))
    (tmp_path / "jang_config.json").write_text(
        json.dumps({"capabilities": {"family": ["lfm2_moe"]}})
    )

    with pytest.raises(RuntimeError, match="family must be a string when present"):
        ModelConfigRegistry().lookup(str(tmp_path))


def test_unknown_family_uses_native_cache_and_no_automatic_parser(monkeypatch):
    import vmlx_engine.model_config_registry as registry_module

    monkeypatch.setattr(
        registry_module,
        "load_config",
        lambda _path: {"model_type": "totally_unknown_type"},
    )
    config = ModelConfigRegistry().lookup("totally-unknown-model")
    assert config.family_name == "unknown"
    assert config.cache_type == "native"
    assert config.tool_parser is None
    assert config.reasoning_parser is None
    assert config.architecture_hints["force_native_cache"] is True
