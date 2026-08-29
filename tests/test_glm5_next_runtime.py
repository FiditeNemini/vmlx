# SPDX-License-Identifier: Apache-2.0
"""glm5_next (GLM-5.3-Flash) vendored text runtime — synthetic contract tests.

Real-bundle load/generation receipts live in the private campaign evidence;
these tests pin the source contract on a tiny random model: registration,
config parsing, cache layout, prefill/decode path agreement (the chunked
KDA prefill vs split-recurrent must be BIT-exact — same fp32 recurrence),
the dense-DSA bound refusal, sanitize drops, and registry detection.
"""

import json
import random

import mlx.core as mx
import pytest

from vmlx_engine.models.glm5_next.register import register_glm5_next_runtime


TINY_CFG = {
    "model_type": "glm5_next",
    "text_config": {
        "model_type": "glm5_next_text",
        "hidden_size": 64,
        "num_hidden_layers": 5,
        "vocab_size": 512,
        "num_attention_heads": 4,
        "q_lora_rank": 32,
        "kv_lora_rank": 16,
        "qk_nope_head_dim": 16,
        "v_head_dim": 16,
        "n_routed_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "intermediate_size": 96,
        "linear_attn_config": {
            "num_heads": 4,
            "head_dim": 16,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        },
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
        ],
        "qk_rope_head_dim": 0,
        "index_topk": 200,
    },
}


@pytest.fixture(scope="module")
def glm5():
    register_glm5_next_runtime()
    import mlx_lm.models.glm5_next as g

    return g


@pytest.fixture(scope="module")
def tiny_model(glm5):
    model = glm5.Model(glm5.ModelArgs.from_dict(TINY_CFG))
    mx.eval(model.parameters())
    return model


class TestRegistration:
    def test_registers_under_mlx_lm_namespace(self, glm5):
        import sys

        assert "mlx_lm.models.glm5_next" in sys.modules
        assert glm5.Model is not None and glm5.ModelArgs is not None

    def test_args_from_nested_text_config(self, glm5):
        args = glm5.ModelArgs.from_dict(TINY_CFG)
        assert args.model_type == "glm5_next"
        assert args.num_hidden_layers == 5
        assert args.linear_num_heads == 4
        assert args.index_topk == 200

    def test_rejects_rope_and_grouped_router(self, glm5):
        bad = json.loads(json.dumps(TINY_CFG))
        bad["text_config"]["qk_rope_head_dim"] = 64
        with pytest.raises(ValueError):
            glm5.ModelArgs.from_dict(bad)
        bad2 = json.loads(json.dumps(TINY_CFG))
        bad2["text_config"]["n_group"] = 8
        with pytest.raises(ValueError):
            glm5.ModelArgs.from_dict(bad2)


class TestCacheAndForward:
    def test_cache_layout(self, tiny_model):
        from mlx_lm.models.cache import ArraysCache, KVCache

        cache = tiny_model.make_cache()
        kinds = [type(c) for c in cache]
        assert kinds == [ArraysCache, ArraysCache, ArraysCache, KVCache, ArraysCache]
        assert len(cache[0].cache) == 4  # conv_q, conv_k, conv_v, S

    def test_prefill_and_decode_shapes(self, tiny_model):
        cache = tiny_model.make_cache()
        out = tiny_model(mx.array([[1, 2, 3, 4, 5, 6, 7]]), cache=cache)
        assert out.shape == (1, 7, 512)
        out2 = tiny_model(mx.array([[8]]), cache=cache)
        assert out2.shape == (1, 1, 512)
        # KDA state is fp32 and fixed-shape after decode
        assert cache[0].cache[3].dtype == mx.float32
        assert cache[0].cache[3].shape == (1, 4, 16, 16)

    def test_chunked_prefill_bit_exact_vs_split_recurrent(self, tiny_model):
        """T=100 goes through kda_chunked; 50+50 goes through kda_recurrent
        twice. Same fp32 recurrence — logits and state must be BIT-exact."""
        random.seed(7)
        toks = [random.randint(1, 500) for _ in range(100)]
        ids = mx.array([toks])
        c1 = tiny_model.make_cache()
        o1 = tiny_model(ids, cache=c1)
        c2 = tiny_model.make_cache()
        tiny_model(ids[:, :50], cache=c2)
        o2 = tiny_model(ids[:, 50:], cache=c2)
        mx.eval(o1, o2)
        assert float(mx.abs(o1[0, -1] - o2[0, -1]).max()) == 0.0
        assert float(mx.abs(c1[0].cache[3] - c2[0].cache[3]).max()) == 0.0
        # MLA KV offsets agree
        assert c1[3].offset == c2[3].offset == 100

    def test_dense_dsa_bound_refuses_loudly(self, tiny_model):
        with pytest.raises(ValueError, match="dense-attention path is exact"):
            tiny_model(mx.array([[1] * 201]))

    def test_sanitize_drops_visual_mtp_indexer_only(self, tiny_model):
        weights = {
            "visual.blocks.0.attn.proj.weight": mx.zeros((1,)),
            "model.layers.5.mlp.gate.weight": mx.zeros((1,)),  # MTP layer (== num_hidden_layers)
            "model.layers.3.self_attn.indexer.wk.weight": mx.zeros((1,)),
            "model.layers.0.self_attn.q_proj.weight": mx.zeros((1,)),
            "lm_head.weight": mx.zeros((1,)),
        }
        kept = tiny_model.sanitize(weights)
        assert set(kept) == {
            "model.layers.0.self_attn.q_proj.weight",
            "lm_head.weight",
        }


class TestRegistryDetection:
    def test_family_row(self, tmp_path):
        from vmlx_engine.model_config_registry import get_model_config_registry

        (tmp_path / "config.json").write_text(json.dumps(TINY_CFG))
        registry = get_model_config_registry()
        registry.clear_cache()
        conf = registry.lookup(str(tmp_path))
        assert conf.family_name == "glm5_next"
        assert conf.cache_type == "hybrid"
        assert conf.cache_subtype == "glm5_next_native_v1"
        assert conf.tool_parser == "glm47"
        assert conf.reasoning_parser == "deepseek_r1"
        assert conf.think_in_template is True
        assert conf.supported_reasoning_efforts == ["low", "high", "max"]

    def test_text_route_override(self, tmp_path):
        from vmlx_engine.api.utils import is_mllm_model

        cfg = json.loads(json.dumps(TINY_CFG))
        cfg["vision_config"] = {"model_type": "glm5_next"}
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        assert is_mllm_model(str(tmp_path)) is False
        assert is_mllm_model(str(tmp_path), force_mllm=True) is False
