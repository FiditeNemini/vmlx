# SPDX-License-Identifier: Apache-2.0
"""Native GLM admission must describe runtime state, not generic full K/V."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from vmlx_engine.utils.memory_limits import (
    estimate_cache_bytes_for_tokens_from_config,
    estimate_cache_token_capacity_from_config,
    estimate_glm5_cache_memory_from_config,
    estimate_kv_bytes_per_token_from_config,
)


@pytest.fixture
def config(monkeypatch):
    monkeypatch.delenv("VMLINUX_GLM5_MLA_ABSORB", raising=False)
    monkeypatch.delenv("VMLX_GLM5_MLA_ABSORB", raising=False)
    monkeypatch.delenv("VMLX_GLM5_DSA_BF16", raising=False)
    return {
        "model_type":"glm5_next_text", "dtype":"bfloat16",
        "num_hidden_layers":45, "num_attention_heads":64,
        "num_key_value_heads":64, "hidden_size":4096, "head_dim":0,
        "layer_types":(["linear_attention"]*3+["deepseek_sparse_attention"])*11
                       + ["linear_attention"],
        "linear_attn_config":{"num_heads":64,"head_dim":128,"short_conv_kernel_size":4},
        "kv_lora_rank":512, "qk_nope_head_dim":256, "qk_rope_head_dim":0,
        "v_head_dim":256, "index_head_dim":128, "index_kpool":4,
    }


def test_native_growth_counts_latent_packed_pool_not_all_attention(config):
    estimate = estimate_glm5_cache_memory_from_config(config, 4096)
    assert estimate.kda_layers == 34 and estimate.mla_layers == 11
    assert estimate.absorbed and estimate.dsa_scalar_bytes == 4
    assert estimate.growth_bytes_per_token == 23_936
    assert estimate.kda_bytes == 34 * (64*128*128*4 + 3*3*64*128*2)
    assert estimate.mla_bytes == 11*4096*512*2
    assert estimate.packed_bytes == 11*4096*256*4
    assert estimate.pool_bytes == 11*1024*128*4
    assert estimate.total_bytes == sum((estimate.kda_bytes,estimate.mla_bytes,
                                       estimate.packed_bytes,estimate.pool_bytes))
    assert estimate_kv_bytes_per_token_from_config(config) == estimate.growth_bytes_per_token
    assert estimate_cache_bytes_for_tokens_from_config(config,4096) == estimate.total_bytes
    assert estimate.output_reserve_bytes == estimate.kda_bytes + 2048*23_936


@pytest.mark.parametrize("wrapper", ["dict","attrs","text_attrs"])
def test_native_config_wrappers(config, wrapper):
    wrapped = ({"model_type":"glm5_next","text_config":config} if wrapper=="dict" else
               SimpleNamespace(**config) if wrapper=="attrs" else
               SimpleNamespace(model_type="glm5_next",text_config=SimpleNamespace(**config)))
    assert estimate_glm5_cache_memory_from_config(wrapped,100) == estimate_glm5_cache_memory_from_config(config,100)


@pytest.mark.parametrize("bits", [2,4,6,8])
def test_weight_quant_does_not_change_state_geometry(config,bits):
    changed = deepcopy(config)
    changed["quantization"]={"bits":bits,"group_size":32 if bits==6 else 64}
    assert estimate_glm5_cache_memory_from_config(config,2049) == estimate_glm5_cache_memory_from_config(changed,2049)


def test_compact_capacity_slack_and_fixed_state_are_budgeted(config):
    zero = estimate_glm5_cache_memory_from_config(config,0)
    edge = estimate_glm5_cache_memory_from_config(config,2048)
    after = estimate_glm5_cache_memory_from_config(config,2049)
    assert zero.total_bytes == 0 and zero.output_reserve_bytes > 0
    assert after.total_bytes > edge.total_bytes
    assert after.total_bytes-edge.total_bytes <= after.output_reserve_bytes
    assert estimate_cache_token_capacity_from_config(config,edge.total_bytes,max_tokens=10000)==2048
    # At2052 the completed pool count reaches513, growing its separate
    # 512-row capacity block. Latent/packed slack alone is not sufficient.
    assert estimate_cache_token_capacity_from_config(config,after.total_bytes,max_tokens=3000)==2051
    third=estimate_glm5_cache_memory_from_config(config,3000)
    assert estimate_cache_token_capacity_from_config(config,third.total_bytes,max_tokens=3000)==3000
    assert estimate_cache_token_capacity_from_config(config,1,max_tokens=10000)==0


@pytest.mark.parametrize("dtype,scalar", [("bfloat16",2),("float16",2),("float32",4),("float8",2),(None,4)])
def test_activation_dtype_not_quantized_weight_width(config,dtype,scalar):
    config["dtype"]=dtype
    estimate=estimate_glm5_cache_memory_from_config(config,100)
    assert estimate.mla_bytes == 11*2048*512*scalar
    assert estimate.dsa_scalar_bytes == 4


def test_legacy_env_wins_and_expanded_kv_is_counted(config,monkeypatch):
    monkeypatch.setenv("VMLX_GLM5_MLA_ABSORB","1")
    monkeypatch.setenv("VMLINUX_GLM5_MLA_ABSORB","0")
    estimate=estimate_glm5_cache_memory_from_config(config,3000)
    assert not estimate.absorbed
    assert estimate.growth_bytes_per_token == 733_568
    assert estimate.mla_bytes==11*3000*64*(256+256)*2
    assert estimate.output_reserve_bytes==estimate.kda_bytes


def test_dsa_precision_policy_is_counted_separately(config,monkeypatch):
    monkeypatch.setenv("VMLX_GLM5_DSA_BF16","true")
    estimate=estimate_glm5_cache_memory_from_config(config,4096)
    assert estimate.dsa_scalar_bytes==2
    assert estimate.growth_bytes_per_token==17_600
    assert estimate.mla_bytes==11*4096*512*2


@pytest.mark.parametrize("field,value", [
    ("model_type","qwen4_exp"),("layer_types",[]),
    ("layer_types",["unknown"]*45),("kv_lora_rank",0),
    ("index_head_dim",None),("index_kpool",0),("qk_rope_head_dim",64),
    ("linear_attn_config",{}),
])
def test_unknown_geometry_keeps_existing_fallback(config,field,value):
    config[field]=value
    assert estimate_glm5_cache_memory_from_config(config,4096) is None


@pytest.mark.parametrize("absorbed,bf16", [(False,False),(True,False),(True,True)])
def test_real_cache_capacity_is_bounded_by_estimate(config,monkeypatch,absorbed,bf16):
    mx=pytest.importorskip("mlx.core")
    from vmlx_engine.models.glm5_next.glm5_next import Glm5KDACache,Glm5MLACache
    monkeypatch.setenv("VMLX_GLM5_MLA_ABSORB",str(int(absorbed)))
    monkeypatch.setenv("VMLX_GLM5_DSA_BF16",str(int(bf16)))
    kd=Glm5KDACache()
    kd.cache=[mx.full((1,3,8192),i+1,dtype=mx.bfloat16) for i in range(3)]
    kd.cache.append(mx.zeros((1,64,128,128),dtype=mx.float32))
    mla=Glm5MLACache(4)
    state_dtype=mx.bfloat16 if bf16 else mx.float32
    previous=0; pools=0
    for count in (1,4,2048,2049,4096,4097):
        length=count-previous
        if absorbed:
            mla.update_latent(mx.full((1,1,length,512),1,dtype=mx.bfloat16))
        else:
            mla.update_kv(mx.full((1,64,length,256),1,dtype=mx.bfloat16),
                          mx.full((1,64,length,256),2,dtype=mx.bfloat16))
        mla.update_packed(mx.full((1,length,256),3,dtype=state_dtype))
        if count>2048:
            complete=count//4
            if complete>pools:
                mla.update_pool_keys(mx.full((1,complete-pools,128),4,dtype=state_dtype))
            pools=complete
        mx.eval(kd.state,mla.state)
        actual=34*kd.nbytes+11*mla.nbytes
        estimate=estimate_glm5_cache_memory_from_config(config,count)
        assert actual<=estimate.total_bytes,(count,actual,estimate)
        if count>=4096: assert actual==estimate.total_bytes
        previous=count


def test_output_guard_keeps_reserves_and_rejects_real_exhaustion(config,monkeypatch,caplog):
    from fastapi import HTTPException
    from vmlx_engine import server
    caplog.set_level("INFO",logger="vmlx_engine.server")
    monkeypatch.setattr(server,"_native_cache_projection_logged",set())
    monkeypatch.setattr(server,"_loaded_model_config_for_memory_projection",lambda:config)
    monkeypatch.setattr(server,"_metal_projection_stats",lambda:(int(97.56*1024**3),int(108.40*1024**3)))
    monkeypatch.setenv("VMLX_METAL_PROJECTED_OUTPUT_GUARD","1")
    monkeypatch.setenv("VMLX_METAL_PROJECTED_TOKEN_BUDGET_FRACTION","0.5")
    monkeypatch.setenv("VMLX_METAL_PROJECTED_TOKEN_TRANSIENT_MULTIPLIER","4")
    cap=server._metal_projected_output_token_cap()
    assert cap is not None and cap>4096
    assert server._apply_projected_output_guard(4096,explicit=True)==4096
    assert sum("Native GLM cache projection:" in r.message for r in caplog.records)==1
    monkeypatch.setattr(server,"_metal_projection_stats",lambda:(108*1024**3,108*1024**3))
    with pytest.raises(HTTPException) as rejected:
        server._apply_projected_output_guard(4096,explicit=True)
    assert rejected.value.status_code==413
