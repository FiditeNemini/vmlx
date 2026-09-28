"""Small native-family contracts; no full model bundle is loaded."""
import importlib
import json

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")


def test_native_cache_slots_and_incremental_forward():
    from vmlx_engine.models.naive_n05_flash.register import register_naive_n05_flash_runtime

    register_naive_n05_flash_runtime()
    module = importlib.import_module("mlx_lm.models.naive_n05_flash")
    args = module.ModelArgs(
        vocab_size=32, hidden_size=64, num_hidden_layers=2,
        intermediate_size=96, num_attention_heads=2, num_key_value_heads=1,
        head_dim=64, v_head_dim=32, swa_num_attention_heads=2,
        swa_num_key_value_heads=2, swa_head_dim=64, swa_v_head_dim=32,
        index_n_heads=2, index_head_dim=64, partial_rotary_factor=0.5,
        index_top_k=4, sliding_window=4,
        hybrid_layer_pattern=[0, 1], moe_layer_freq=[0, 0],
    )
    model = module.Model(args)
    assert model.cache_list_head_counts == ((1, 1), None)
    caches = model.make_cache()
    assert type(caches[0]).__name__ == "CacheList"
    assert type(caches[1]).__name__ == "RotatingKVCache"
    for tokens in [mx.array([[1, 2, 3]]), mx.array([[4, 5, 6]]), mx.array([[7]])]:
        logits = model(tokens, cache=caches)
        mx.eval(logits)
        assert logits.shape == (1, tokens.shape[1], 32)
        assert bool(mx.all(mx.isfinite(logits)))
    assert caches[0][0].offset == caches[0][1].offset == caches[1].offset == 7
    assert caches[0][1].state[0].dtype == mx.float32
    assert caches[0][1].state[1].shape[-1] == 0


def test_packed_text_quantization_preserves_module_and_rejects_mismatch():
    from vmlx_engine.jangh.switch import TQSwitchLinear

    module = TQSwitchLinear(64, 32, 2, 2, "hadamard32")
    before = module.tq2_packed
    assert module.to_quantized(mode="jangtq2", bits=2, rotation="hadamard32") is module
    assert module.tq2_packed is before
    for bad in [dict(mode="affine"), dict(bits=3), dict(rotation="none")]:
        with pytest.raises(ValueError, match="entry differs"):
            module.to_quantized(**bad)


def test_native_settings_and_text_codec_detection(tmp_path):
    from vmlx_engine.model_config_registry import get_model_config_registry
    from vmlx_engine.utils.jang_loader import is_jang_model

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "naive_n05_flash"}))
    (tmp_path / "jang_config.json").write_text(json.dumps({"format": "jangtq2"}))
    assert is_jang_model(tmp_path)
    config = get_model_config_registry().lookup(str(tmp_path))
    assert config.family_name == "naive_n05_flash"
    assert config.supported_reasoning_efforts == ["low", "high", "max"]
    assert config.reasoning_parser == "think_xml"
    assert config.tool_parser == "xml_function"
    assert config.think_in_template is False
    assert config.cache_subtype == "naive_n05_swa_dsa"
    assert config.is_mllm is False


@pytest.mark.parametrize("damage", [None, "missing_regular", "bad_packed_shape"])
def test_text_jangh_loader_roundtrip(tmp_path, monkeypatch, damage):
    from dataclasses import asdict
    from mlx.utils import tree_flatten
    import mlx_lm.utils
    from tests.test_jangtq2_contract import bundle
    from vmlx_engine.models.naive_n05_flash.register import register_naive_n05_flash_runtime
    from vmlx_engine.utils.jang_loader import _load_jang_v2

    register_naive_n05_flash_runtime()
    module = importlib.import_module("mlx_lm.models.naive_n05_flash")
    contract = bundle()
    contract["quantization"] = {
        k.replace("layers.3.", "layers.1."): v
        for k, v in contract["quantization"].items()
    }
    args = module.ModelArgs(
        vocab_size=32, hidden_size=64, num_hidden_layers=2,
        intermediate_size=64, moe_intermediate_size=64,
        n_routed_experts=2, num_experts_per_tok=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=64, v_head_dim=32,
        swa_num_attention_heads=2, swa_num_key_value_heads=2,
        swa_head_dim=64, swa_v_head_dim=32, index_n_heads=2, index_head_dim=64,
        partial_rotary_factor=0.5, hybrid_layer_pattern=[0, 1], moe_layer_freq=[0, 1],
        **contract,
    )
    original = module.Model(args)
    weights = dict(tree_flatten(original.parameters()))
    if damage == "missing_regular":
        weights.pop("model.embed_tokens.weight")
    elif damage == "bad_packed_shape":
        key = "model.layers.1.mlp.switch_mlp.gate_proj.tq2_packed"
        weights[key] = mx.zeros((2, 64, 5), dtype=mx.uint32)
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(json.dumps(asdict(args)))
    stamp = {"format": "jangtq2", "format_version": 2}
    (tmp_path / "jang_config.json").write_text(json.dumps(stamp))
    tokenizer = object()
    monkeypatch.setattr(mlx_lm.utils, "load_tokenizer", lambda *a, **k: tokenizer)
    if damage:
        reason = "Naive weight inventory mismatch" if damage == "missing_regular" else "invalid shape or dtype"
        with pytest.raises(ValueError, match=reason):
            _load_jang_v2(tmp_path, stamp, skip_eval=True)
        return
    from vmlx_engine.utils import jang_loader

    loading_calls = []
    original_eval = jang_loader._chunked_eval_params

    def evaluate_parameters(*args, **kwargs):
        loading_calls.append(kwargs)
        return original_eval(*args, **kwargs)

    monkeypatch.setattr(jang_loader, "_chunked_eval_params", evaluate_parameters)
    monkeypatch.setattr(jang_loader, "_set_wired_limit_for_model", lambda *a: None)
    loaded, actual_tokenizer = _load_jang_v2(tmp_path, stamp, skip_eval=False)
    assert loading_calls == [{"chunk_size": 25, "pipeline_gpu": True}]
    assert actual_tokenizer is tokenizer
    assert loaded.jangh_modules == 1
    actual = dict(tree_flatten(loaded.parameters()))
    assert set(actual) == set(weights)
    for key in weights:
        assert bool(mx.array_equal(actual[key], weights[key])), key
