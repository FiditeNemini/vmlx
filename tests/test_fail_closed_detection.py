"""H5: cache/parser architecture detection must not fail open."""

import json

import pytest

from vmlx_engine.mllm_scheduler import MLLMScheduler
from vmlx_engine.model_config_registry import ModelConfigRegistry
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

