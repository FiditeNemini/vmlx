import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest


def _qlinear(input_dims: int, output_dims: int) -> nn.QuantizedLinear:
    dense = nn.Linear(input_dims, output_dims, bias=False)
    dense.weight = dense.weight.astype(mx.bfloat16)
    return dense.to_quantized(group_size=32, bits=4)


def _text_config():
    _register_qwen35()
    from mlx_vlm.models.qwen3_5.config import TextConfig

    return TextConfig(
        model_type="qwen3_5_text",
        hidden_size=64,
        intermediate_size=128,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        num_hidden_layers=4,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=199,
        num_key_value_heads=2,
        max_position_embeddings=8192,
        head_dim=16,
        full_attention_interval=4,
    )


def _register_qwen35():
    from vmlx_engine.models.qwen3_5_family import register_qwen3_5_family_runtime

    assert register_qwen3_5_family_runtime()


def _language_module():
    _register_qwen35()
    from mlx_vlm.models.qwen3_5 import language

    return language


def _assert_exact(reference, candidate):
    mx.eval(*reference, *candidate)
    for expected, actual in zip(reference, candidate):
        np.testing.assert_array_equal(
            np.asarray(actual.astype(mx.float32)),
            np.asarray(expected.astype(mx.float32)),
        )


def test_qwen35_gdn_input_projection_group_is_exact():
    gdn_type = _language_module().Qwen3_5GatedDeltaNet

    module = gdn_type(_text_config())
    module.in_proj_qkv = _qlinear(64, 128)
    module.in_proj_z = _qlinear(64, 64)
    module.in_proj_b = _qlinear(64, 4)
    module.in_proj_a = _qlinear(64, 4)
    decode = (mx.arange(64, dtype=mx.bfloat16) / 127.0).reshape(1, 1, 64)
    prefill = (mx.arange(128, dtype=mx.bfloat16) / 127.0).reshape(1, 2, 64)
    decode_reference = tuple(
        projection(decode)
        for projection in (
            module.in_proj_qkv,
            module.in_proj_z,
            module.in_proj_b,
            module.in_proj_a,
        )
    )
    prefill_reference = tuple(
        projection(prefill)
        for projection in (
            module.in_proj_qkv,
            module.in_proj_z,
            module.in_proj_b,
            module.in_proj_a,
        )
    )

    assert module.prepare_runtime() is True
    decode_candidate = module._project_inputs(decode)
    prefill_candidate = module._project_inputs(prefill)

    assert module.in_proj_qkv is None
    assert module.in_proj_z is module.in_proj_b is module.in_proj_a is None
    _assert_exact(decode_reference, decode_candidate)
    _assert_exact(prefill_reference, prefill_candidate)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_qwen35_gdn_conv_fusion_matches_stock(dtype):
    from vmlx_engine.metal.gdn_conv_decode import (
        qwen35_gdn_conv_decode,
        qwen35_gdn_conv_status,
    )

    channels = 96
    kernel_size = 4
    conv = nn.Conv1d(
        channels,
        channels,
        kernel_size=kernel_size,
        groups=channels,
        bias=False,
    )
    conv.weight = conv.weight.astype(dtype)
    state = (
        mx.arange((kernel_size - 1) * channels, dtype=mx.float32)
        .reshape(1, kernel_size - 1, channels)
        .astype(dtype)
        / 127.0
    )
    token = (
        mx.arange(channels, dtype=mx.float32).reshape(1, 1, channels)
        .astype(dtype)
        / 191.0
    )
    full = mx.concatenate([state, token], axis=1)
    reference_conv = nn.silu(conv(full))
    reference_state = mx.contiguous(full[:, -(kernel_size - 1) :, :])
    candidate = qwen35_gdn_conv_decode(
        token,
        state,
        conv.weight,
        enabled=True,
    )
    assert candidate is not None
    assert qwen35_gdn_conv_status()["observed_calls"] == 1
    candidate_conv, candidate_state = candidate
    mx.eval(reference_conv, reference_state, candidate_conv, candidate_state)
    np.testing.assert_array_equal(
        np.asarray(candidate_state.astype(mx.float32)),
        np.asarray(reference_state.astype(mx.float32)),
    )
    np.testing.assert_array_equal(
        np.asarray(candidate_conv.astype(mx.float32)),
        np.asarray(reference_conv.astype(mx.float32)),
    )


def test_qwen35_gdn_conv_fusion_refuses_prefill():
    from vmlx_engine.metal.gdn_conv_decode import qwen35_gdn_conv_decode

    assert qwen35_gdn_conv_decode(
        mx.zeros((1, 2, 64), dtype=mx.float16),
        mx.zeros((1, 3, 64), dtype=mx.float16),
        mx.zeros((64, 4, 1), dtype=mx.float16),
        enabled=True,
    ) is None


def test_qwen35_projection_preparation_walks_backbone_and_draft_siblings():
    language = _language_module()
    gdn_type = language.Qwen3_5GatedDeltaNet
    prepare_groups = language.prepare_quantized_projection_groups

    holder = nn.Module()
    holder.base_gdn = gdn_type(_text_config())
    holder.mtp = nn.Module()
    holder.mtp.gdn = gdn_type(_text_config())

    for module, names in (
        (
            holder.base_gdn,
            ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),
        ),
        (
            holder.mtp.gdn,
            ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),
        ),
    ):
        for name in names:
            projection = getattr(module, name)
            input_dims = int(projection.weight.shape[-1])
            output_dims = int(projection.weight.shape[0])
            setattr(module, name, _qlinear(input_dims, output_dims))

    assert prepare_groups(holder) == {
        "gdn_inputs": 2,
    }
    assert prepare_groups(holder) == {
        "gdn_inputs": 0,
    }


def test_qwen35_vlm_mtp_wrappers_consume_prepared_projection_groups():
    import inspect

    from vmlx_engine.patches.mlx_vlm_mtp import qwen35_vl

    gdn_source = inspect.getsource(qwen35_vl._patch_gated_delta_net)

    assert 'getattr(self, "_project_inputs", None)' in gdn_source
    assert "qwen35_gdn_conv_decode" in gdn_source
    assert "lengths is None" in gdn_source


def test_qwen35_text_mtp_wrapper_consumes_exact_decode_conv_candidate():
    import inspect

    from vmlx_engine.patches.mlx_lm_mtp import qwen35_model

    gdn_source = inspect.getsource(qwen35_model._patch_gated_delta_net)

    assert "qwen35_gdn_conv_decode" in gdn_source
    assert "lengths is None" in gdn_source


def test_generic_loader_prepares_acceleration_after_weight_hydration(
    monkeypatch, tmp_path
):
    import json

    import mlx_lm

    from vmlx_engine import mlx_memory
    from vmlx_engine.utils import jang_loader, nanbeige_runtime, tokenizer

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "test_model"}))
    events = []

    class PreparedModel:
        def prepare_acceleration(self):
            events.append("prepare")
            return {"projection_groups": 3}

    model = PreparedModel()

    def fake_load(*_args, **_kwargs):
        events.append("load")
        return model, object()

    monkeypatch.setattr(mlx_lm, "load", fake_load)
    monkeypatch.setattr(
        tokenizer, "_register_mimo_v2_runtime_for_mlx_lm", lambda: False
    )
    monkeypatch.setattr(tokenizer, "_needs_tokenizer_fallback", lambda _path: False)
    monkeypatch.setattr(
        tokenizer, "_inject_chat_template_if_missing", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(jang_loader, "is_jang_model", lambda _path: False)
    monkeypatch.setattr(
        nanbeige_runtime, "ensure_nanbeige_runtime_registered", lambda _path: None
    )
    monkeypatch.setattr(
        nanbeige_runtime,
        "validate_nanbeige_loop_cache_contract",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        mlx_memory,
        "maybe_harmonize_quant_metadata_dtypes",
        lambda *_args, **_kwargs: events.append("harmonize"),
    )

    loaded, _ = tokenizer.load_model_with_fallback(
        str(tmp_path), skip_turboquant=True
    )

    assert loaded is model
    assert events == ["load", "harmonize", "prepare"]
