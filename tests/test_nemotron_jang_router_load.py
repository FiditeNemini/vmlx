"""Exercise JANG skeleton creation and packed router binding with real MLX modules."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.mark.parametrize("latent_size", [None, 64])
def test_packed_nemotron_router_survives_jang_skeleton(tmp_path, latent_size):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.models.nemotron_h import Model, ModelArgs
    from mlx_lm.utils import load_model
    from vmlx_engine.utils.jang_loader import _load_jang_v2

    config = dict(
        model_type="nemotron_h", vocab_size=128, hidden_size=128,
        intermediate_size=128, num_hidden_layers=1, max_position_embeddings=128,
        num_attention_heads=2, num_key_value_heads=1, attention_bias=False,
        mamba_num_heads=2, mamba_head_dim=64, mamba_proj_bias=False,
        ssm_state_size=16, conv_kernel=4, n_groups=1, mlp_bias=False,
        layer_norm_epsilon=1e-5, use_bias=False, use_conv_bias=True,
        hybrid_override_pattern="E", head_dim=64, moe_intermediate_size=128,
        moe_shared_expert_intermediate_size=128, moe_latent_size=latent_size,
        n_group=1, n_routed_experts=4, n_shared_experts=1, topk_group=1,
        num_experts_per_tok=2, norm_topk_prob=True, routed_scaling_factor=1.0,
    )
    mx.random.seed(17)
    model = Model(ModelArgs.from_dict(config))
    router = model.backbone.layers[0].mixer.gate
    router.weight = mx.random.normal(router.weight.shape).astype(mx.bfloat16)
    nn.quantize(model, bits=4, group_size=64)
    weights = dict(tree_flatten(model.parameters()))
    base = "backbone.layers.0.mixer.gate"
    packed, scales, biases = mx.quantize(router.weight, bits=8, group_size=64)
    expected = mx.dequantize(packed, scales, biases, bits=8, group_size=64).astype(mx.bfloat16)
    weights.update({base+".weight":packed, base+".scales":scales, base+".biases":biases})
    config["quantization"] = {"bits":4,"group_size":64,base:{"bits":8,"group_size":64}}
    mx.save_safetensors(str(tmp_path/"model-00001-of-00001.safetensors"),weights)
    (tmp_path/"config.json").write_text(json.dumps(config))
    (tmp_path/"model.safetensors.index.json").write_text(json.dumps({"weight_map":{k:"model-00001-of-00001.safetensors" for k in weights}}))
    jang = {"format":"jang","format_version":"2.0","quantization":{"bit_widths_used":[4,8],"block_size":64},"architecture":{"has_moe":True}}
    (tmp_path/"jang_config.json").write_text(json.dumps(jang))
    original_config = (tmp_path/"config.json").read_bytes()
    with pytest.raises(ValueError, match="Unable to quantize.*MoEGate"):
        load_model(tmp_path, lazy=True, strict=False)
    with patch("mlx_lm.utils.load_tokenizer", return_value=SimpleNamespace()):
        restored, _ = _load_jang_v2(tmp_path, jang, skip_eval=True)
    actual = restored.backbone.layers[0].mixer.gate.weight
    assert actual.dtype == mx.bfloat16
    assert mx.array_equal(actual, expected).item()
    assert isinstance(restored.backbone.embeddings, nn.QuantizedEmbedding)
    assert mx.isfinite(restored(mx.array([[1,2,3]]))).all().item()
    assert (tmp_path/"config.json").read_bytes() == original_config


def test_doctor_recognizes_nemotron_embeddings_without_masking_missing(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file
    from vmlx_engine.commands.doctor import _check_weights
    from vmlx_engine.utils.model_inspector import ModelInfo

    filename = "model-00001-of-00001.safetensors"
    weights = {"backbone.embeddings.weight":np.zeros((8,8),dtype=np.float16),"backbone.layers.0.mixer.weight":np.zeros((8,8),dtype=np.float16)}
    info=ModelInfo(model_path=str(tmp_path),model_type="nemotron_h",architecture="NemotronHForCausalLM",weight_files=[filename],config={"tie_word_embeddings":True})
    save_file(weights,str(tmp_path/filename))
    issues,warnings=_check_weights(str(tmp_path),info)
    assert not issues
    assert not any("embedding layer" in w for w in warnings)
    del weights["backbone.embeddings.weight"]
    save_file(weights,str(tmp_path/filename))
    _,warnings=_check_weights(str(tmp_path),info)
    assert any("embedding layer" in w for w in warnings)


@pytest.mark.parametrize("latent", [None, 64])
def test_active_parameter_estimate_only_discounts_routed_matrices(latent):
    from vmlx_engine.utils.model_inspector import _estimate_param_count
    config = dict(model_type="nemotron_h",hidden_size=128,num_hidden_layers=3,
                  vocab_size=128,num_attention_heads=2,num_key_value_heads=1,
                  intermediate_size=128,moe_intermediate_size=256,moe_latent_size=latent,
                  n_routed_experts=8,num_experts_per_tok=2,n_shared_experts=1,
                  moe_shared_expert_intermediate_size=128,hybrid_override_pattern="E*E")
    total = _estimate_param_count(config)
    active = _estimate_param_count(config,active_experts=True)
    # Two MoE layers, six inactive experts, two ReLU-squared expert matrices.
    inactive = 2 * (8-2) * 2 * (latent or 128) * 256 / 1e9
    assert total-active == pytest.approx(inactive)
    config["num_experts_per_tok"] = 8
    assert _estimate_param_count(config,active_experts=True) == total
