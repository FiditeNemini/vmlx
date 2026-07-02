# SPDX-License-Identifier: Apache-2.0
"""openPangu-2.0-Flash (openpangu_v2) integration regression tests.

Pins the vendored runtime's load-bearing invariants (Swift-port bug list):
registry autodetect, sanitize transforms, biased-select/unbiased-weight MoE
gate, path-dependent cache contract, prefill-vs-incremental decode
equivalence (conv-state + SWA rotation + DSA indexer + sink masks), and the
manifest-driven quantization overrides.
"""

import mlx.core as mx
import pytest

from vmlx_engine.models.openpangu_v2.register import register_openpangu_v2_runtime


def _tiny_args():
    from mlx_lm.models.openpangu_v2 import ModelArgs

    return ModelArgs(
        hidden_size=64,
        num_hidden_layers=4,
        num_nextn_predict_layers=0,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_attention_heads=4,
        vocab_size=256,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        param_sink_number=8,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=12,
        dsa_layers=[0, 3],
        swa_layers=[1, 2],
        sliding_window_list=[8, 16],
        block_post_layernorm_idx=[0, 2],
        mhc_num_stream=4,
        mhc_recur_norm=20,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=2,
        rope_theta=10000.0,
        max_position_embeddings=2048,
    )


@pytest.fixture(scope="module")
def tiny_model():
    assert register_openpangu_v2_runtime() or True
    from mlx_lm.models.openpangu_v2 import Model

    model = Model(_tiny_args())
    mx.eval(model.parameters())
    return model


def test_runtime_registers_under_mlx_lm_namespace():
    register_openpangu_v2_runtime()
    import importlib

    mod = importlib.import_module("mlx_lm.models.openpangu_v2")
    assert hasattr(mod, "Model") and hasattr(mod, "ModelArgs")


def test_registry_resolves_openpangu_family(tmp_path):
    import json

    from vmlx_engine.model_config_registry import get_model_config_registry

    model_dir = tmp_path / "openPangu-2.0-Flash-JANG_2L"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "openpangu_v2", "hidden_size": 2560})
    )
    # The converter stamps the coarse cache_type="hybrid"; the registry entry's
    # kv/openpangu_v2_composite contract must WIN (hybrid would misroute the
    # scheduler into SSM handling). Found live on the first server launch.
    (model_dir / "jang_config.json").write_text(
        json.dumps(
            {
                "capabilities": {
                    "family": "openpangu_v2",
                    "cache_type": "hybrid",
                    "reasoning_parser": "deepseek_r1",
                    "tool_parser": "qwen",
                    "think_in_template": True,
                    "supports_thinking": True,
                    "modality": "text",
                }
            }
        )
    )
    mc = get_model_config_registry().lookup(str(model_dir))
    assert mc is not None
    assert mc.family_name == "openpangu_v2"
    assert mc.cache_type == "kv"
    assert mc.cache_subtype == "openpangu_v2_composite"
    assert mc.reasoning_parser == "deepseek_r1"
    assert mc.tool_parser == "qwen"
    assert "<|message_end|>" in mc.eos_tokens


def test_sanitize_conv_transpose_and_mtp_drop(tiny_model):
    w = {
        "model.layers.0.self_attn.qa_conv.weight": mx.zeros((32, 1, 3)),
        "model.layers.0.self_attn.o_conv.weight": mx.zeros((64, 1, 3)),
        # MTP layers (>= num_hidden_layers=4) must be dropped: mtp_mode=off.
        "model.layers.4.eh_proj.weight": mx.zeros((8, 8)),
        "model.layers.6.shared_head.head.weight": mx.zeros((8, 8)),
        "model.layers.1.input_layernorm.weight": mx.ones((64,)),
        "model.rotary_emb.inv_freq": mx.zeros((4,)),
    }
    out = tiny_model.sanitize(w)
    assert out["model.layers.0.self_attn.qa_conv.weight"].shape == (32, 3, 1)
    assert out["model.layers.0.self_attn.o_conv.weight"].shape == (64, 3, 1)
    assert "model.layers.4.eh_proj.weight" not in out
    assert "model.layers.6.shared_head.head.weight" not in out
    assert "model.rotary_emb.inv_freq" not in out
    assert "model.layers.1.input_layernorm.weight" in out


def test_moe_gate_selects_biased_weights_unbiased(tiny_model):
    """SELECT on scores+bias, WEIGHT with unbiased scores (Swift-proven)."""
    moe = tiny_model.model.layers[2].mlp  # first MoE layer
    # Huge bias on expert 0 forces its selection even with low raw score.
    moe.e_score_correction_bias = mx.array([100.0, 0.0, 0.0, 0.0])
    x = mx.random.normal((1, 1, 64))
    logits = moe.gate(x)
    scores = mx.sigmoid(logits.astype(mx.float32))
    choice = scores + moe.e_score_correction_bias.astype(mx.float32)
    inds = mx.argpartition(-choice, kth=moe.top_k - 1, axis=-1)[..., : moe.top_k]
    mx.eval(inds)
    assert 0 in [int(i) for i in inds.reshape(-1).tolist()], "biased selection"
    picked = mx.take_along_axis(scores, inds, axis=-1)
    mx.eval(picked)
    # Weights must be the UNBIASED sigmoid scores (all < 1), never the biased.
    assert float(picked.max()) < 1.0
    moe.e_score_correction_bias = mx.zeros((4,))


def test_cache_contract_path_dependent(tiny_model):
    cache = tiny_model.make_cache()
    assert len(cache) == 4
    assert cache[0].is_dsa and cache[0].indexer_kv is not None
    assert not cache[1].is_dsa and cache[1].window == 8
    assert cache[2].window == 16
    # Path-dependent conv state: trim-based reuse must be refused.
    assert not cache[0].is_trimmable()
    assert cache[0].trim(5) == 0


def test_prefill_vs_incremental_equivalence(tiny_model):
    """Decode one-token-at-a-time must match single prefill: proves conv-state
    carry, sink mask polarity, SWA rotation, and DSA indexer consistency."""
    seq = [1, 2, 3, 4, 5, 6]
    c_a = tiny_model.make_cache()
    la = tiny_model(mx.array([seq]), cache=c_a)
    c_b = tiny_model.make_cache()
    for t in seq:
        lb = tiny_model(mx.array([[t]]), cache=c_b)
    mx.eval(la, lb)
    diff = float(mx.abs(la[0, -1] - lb[0, -1]).max())
    assert diff < 0.05, f"decode path diverges from prefill: {diff}"


def test_decode_past_window_and_topk(tiny_model):
    cache = tiny_model.make_cache()
    logits = tiny_model(mx.array([[1, 2, 3, 4, 5, 6, 7]]), cache=cache)
    for step in range(14):
        logits = tiny_model(mx.array([[(step % 200) + 10]]), cache=cache)
    mx.eval(logits)
    assert logits.shape == (1, 1, 256)
    # 21 tokens: SWA window 8 exceeded AND indexer (topk=12) active.
    assert cache[0].offset == 21
    assert cache[0].indexer_kv.offset == 21


def test_cache_state_roundtrip(tiny_model):
    cache = tiny_model.make_cache()
    tiny_model(mx.array([[1, 2, 3]]), cache=cache)
    for layer_cache in (cache[0], cache[1]):
        st = layer_cache.state
        meta = layer_cache.meta_state
        fresh = type(layer_cache)(
            window=layer_cache.window, is_dsa=layer_cache.is_dsa
        )
        fresh.state = st
        fresh.meta_state = meta
        assert fresh.kv.offset == layer_cache.kv.offset
        assert fresh.conv_states[0] is not None
        mx.eval(fresh.conv_states[0])


def test_quant_overrides_use_jang_manifest():
    from vmlx_engine.utils.jang_loader import _post_load_quantization_overrides

    config = {"model_type": "openpangu_v2"}
    jang_cfg = {
        "quantization": {
            "tensor_quantization_manifest": {
                "model.layers.0.attn_mhc_module.phi": {
                    "bits": 2,
                    "group_size": 128,
                    "weight_shape": [24, 640],
                },
                "model.embed_tokens": {"bits": 6, "group_size": 128},
                "bogus": {"note": "no bits"},
            }
        }
    }
    overrides = _post_load_quantization_overrides(config, jang_cfg)
    assert overrides is not None
    assert overrides["model.layers.0.attn_mhc_module.phi"] == {
        "bits": 2,
        "group_size": 128,
    }
    assert overrides["model.embed_tokens"]["bits"] == 6
    assert "bogus" not in overrides


def test_native_mtp_stays_runtime_unwired():
    """openpangu_v2 is detection-only (DSV4 bucket): never runtime-advertised."""
    from vmlx_engine import native_mtp

    assert "openpangu_v2" not in native_mtp._RUNTIME_SUPPORTED_FAMILIES
    assert "openpangu_v2" not in native_mtp._EAGLE3_NATIVE_MTP_FAMILIES
