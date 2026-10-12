"""--text-only GLM-5.3 JANGHT load: the text Model installs JANGT/JANGH routed stacks like the VLM wrapper.

Before: the text runtime built plain SwitchGLU experts, so nn.quantize handed {"mode": "jangt", "kbits": ...} bundle
entries to SwitchLinear.to_quantized ("unexpected keyword argument 'kbits'") and the engine refused to start.
"""
import pytest

mx = pytest.importorskip("mlx.core")

from vmlx_engine.jangt.switch import JTMixedSwitchGLU  # noqa: E402
from vmlx_engine.models.glm5_next.glm5_next import Model, ModelArgs  # noqa: E402

LAYER_TYPES = ["linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention"]
TEXT = dict(hidden_size=128, num_hidden_layers=4, vocab_size=256, num_attention_heads=2, q_lora_rank=64,
            kv_lora_rank=64, qk_nope_head_dim=32, v_head_dim=32, index_n_heads=2, index_head_dim=32,
            n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=64, intermediate_size=128,
            first_k_dense_replace=1, layer_types=LAYER_TYPES, qk_rope_head_dim=0)
JANGT = {"version": 1, "code": "v2_halfbits", "state_bits": 12, "outlier_columns": {}}


def _quant():
    q = {"mode": "affine", "bits": 8, "group_size": 64}
    for L in (1, 2, 3):                       # routed layers (layer 0 is dense)
        p = f"model.layers.{L}.mlp.switch_mlp"
        q[f"{p}.gate_proj"] = q[f"{p}.up_proj"] = {"mode": "jangt", "kbits": 2}
        q[f"{p}.down_proj"] = {"mode": "jangtq2", "bits": 3} if L == 2 else {"mode": "jangt", "kbits": 2.5}
    return q


def test_from_dict_keeps_the_jangt_contract_from_the_full_config():
    args = ModelArgs.from_dict({"model_type": "glm5_next", "text_config": dict(TEXT), "jangt": JANGT,
                                "quantization": _quant()})
    assert args.jangt == JANGT and args.quantization["model.layers.2.mlp.switch_mlp.down_proj"]["mode"] == "jangtq2"
    plain = ModelArgs.from_dict({"model_type": "glm5_next", "text_config": dict(TEXT), "quantization": _quant()})
    assert plain.jangt is None and plain.quantization is None


def test_text_model_installs_jangt_routed_stacks():
    args = ModelArgs.from_dict({"model_type": "glm5_next", "text_config": dict(TEXT), "jangt": JANGT,
                                "quantization": _quant()})
    m = Model(args)
    assert m.jangt_modules == 3
    for L in (1, 2, 3):
        assert isinstance(m.model.layers[L].mlp.switch_mlp, JTMixedSwitchGLU)
    assert m.model.layers[2].mlp.switch_mlp.dn_fmt == "jangtq2"
    # The bundle entries now reach modules that accept them (the old path raised on 'kbits').
    sw = m.model.layers[1].mlp.switch_mlp
    assert sw.gate_proj.to_quantized(mode="jangt", kbits=2) is sw.gate_proj


def test_non_jangt_text_model_is_unchanged():
    m = Model(ModelArgs.from_dict({"model_type": "glm5_next", "text_config": dict(TEXT)}))
    assert not hasattr(m, "jangt_modules")
    assert not isinstance(m.model.layers[1].mlp.switch_mlp, JTMixedSwitchGLU)
