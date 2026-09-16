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


def _grown_native_cache():
    mx = pytest.importorskip("mlx.core")
    from vmlx_engine.models.glm5_next.glm5_next import Glm5MLACache

    cache = Glm5MLACache(4, absorbed=True)
    for count in (2048, 1):
        cache.update_latent(mx.ones((1, 1, count, 8), mx.bfloat16))
        cache.update_packed(mx.full((1, count, 8), 2, mx.float32))
    mx.eval(cache.state)
    return cache


def test_grown_native_cache_resident_estimate():
    from vmlx_engine.memory_cache import _CacheEntry, estimate_kv_cache_memory

    cache = _grown_native_cache()
    assert sum(x.nbytes for x in cache.state) == 98_352
    assert cache.nbytes == 196_608
    assert estimate_kv_cache_memory([cache]) == cache.nbytes
    assert _CacheEntry.create([1, 2, 3], [cache]).memory_bytes == cache.nbytes
    assert estimate_kv_cache_memory([cache], resident=False) == 98_352


@pytest.mark.parametrize("operation", ["clone", "extract", "merge"])
def test_grown_native_cache_shared_resident_bound(operation):
    from vmlx_engine.memory_cache import estimate_kv_cache_memory
    from vmlx_engine.models.glm5_next.glm5_next import (
        Glm5MLACache,
        clone_glm5_next_layer_cache,
    )
    from vmlx_engine.utils.ssm_companion_cache import SSMCompanionCache

    source = _grown_native_cache()
    if operation == "clone":
        shared = clone_glm5_next_layer_cache(source, copy_fn=lambda x: x)
    elif operation == "extract":
        shared = source.extract(0)
    else:
        shared = Glm5MLACache.merge([source])
    assert shared.nbytes == source.nbytes
    assert estimate_kv_cache_memory([shared]) == source.nbytes
    assert SSMCompanionCache._estimate_state_nbytes([shared]) == source.nbytes


def test_grown_native_cache_l1_budget_counts_shared_capacity():
    from vmlx_engine.utils.ssm_companion_cache import SSMCompanionCache

    source = _grown_native_cache()
    companion = SSMCompanionCache(max_entries=1, max_bytes=120_000, disk_store=False)
    companion.store([1, 2, 3], 3, [source])
    assert companion.size == 0
    assert companion.total_nbytes == 0


def test_grown_native_cache_detach_copies_and_trim():
    mx = pytest.importorskip("mlx.core")
    from vmlx_engine.models.glm5_next.glm5_next import clone_glm5_next_layer_cache
    from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator

    source = _grown_native_cache()
    copied = SingleBatchGenerator._clone_cache_object(source)
    assert copied.nbytes == 98_352
    source.trim(2048)
    assert source.nbytes == 196_608
    # Even a distinct array view is not an independently copied allocation.
    shared = clone_glm5_next_layer_cache(source, copy_fn=lambda x: x[...])
    assert shared.nbytes == 196_608
    shared.update_latent(mx.zeros((1, 1, 1, 8), mx.bfloat16))
    mx.eval(shared.state)
    assert shared.nbytes == 32_768 + 131_072
    assert source.nbytes == 196_608
    assert copied.offset == 2049
    assert source.offset == 1
    assert mx.array_equal(source.cache[0], mx.ones((1, 1, 1, 8), mx.bfloat16))
    shared.update_packed(mx.zeros((1, 1, 8), mx.float32))
    mx.eval(shared.state)
    assert shared.nbytes == 98_304
    source.trim(1)
    assert source.nbytes == 196_608
    assert source.extract(0).nbytes == 196_608
    empty = clone_glm5_next_layer_cache(source, copy_fn=lambda x: x)
    assert empty.nbytes == 0


@pytest.mark.parametrize("basic_index", [slice(0, 1), Ellipsis, (slice(0, 1),)])
def test_grown_native_cache_filter_extend_and_metadata(basic_index):
    mx = pytest.importorskip("mlx.core")
    from vmlx_engine.memory_cache import estimate_kv_cache_memory
    from vmlx_engine.models.glm5_next.glm5_next import Glm5MLACache
    from vmlx_engine.utils.ssm_companion_cache import SSMCompanionCache

    source = _grown_native_cache()
    source.filter(basic_index)
    mx.eval(source.state)
    assert source.nbytes == 196_608
    source.filter(mx.array([0]))
    mx.eval(source.state)
    assert source.nbytes == 98_352
    source.extend(_grown_native_cache())
    mx.eval(source.state)
    assert source.nbytes == 196_704
    row = source.extract(0)
    assert row.nbytes == source.nbytes  # Do not divide a shared allocation.
    merged = Glm5MLACache.merge([row, row])
    mx.eval(merged.state)
    assert merged.nbytes == 196_704
    row.left_padding = row.lengths = mx.array([0], mx.int32)
    assert estimate_kv_cache_memory([row]) == row.nbytes + 4
    assert SSMCompanionCache._estimate_state_nbytes([row]) == row.nbytes + 4


def test_grown_native_cache_snapshot_budget_stays_logical():
    from vmlx_engine.memory_cache import estimate_kv_cache_memory
    from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator

    source = _grown_native_cache()
    generator = object.__new__(SingleBatchGenerator)
    generator.prompt_snapshot_max_bytes = 120_000
    generator.prompt_snapshot_oversize_skips = 0
    snapshot = generator._clone_admissible_prompt_cache_snapshot([source])
    assert snapshot is not None
    assert generator.prompt_snapshot_oversize_skips == 0
    assert generator.prompt_snapshot_last_estimated_bytes == 98_352
    assert snapshot[0].nbytes == 98_352
    nested = SimpleNamespace(caches=[source])
    assert estimate_kv_cache_memory([nested]) == 196_608
    assert estimate_kv_cache_memory([nested], resident=False) == 98_352


@pytest.mark.parametrize("bits", [4, 8])
def test_native_capacity_does_not_override_quantized_kv_accounting(bits):
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.cache import QuantizedKVCache

    from vmlx_engine.memory_cache import estimate_kv_cache_memory

    cache = QuantizedKVCache(group_size=32, bits=bits)
    value = mx.ones((1, 1, 3, 64), mx.bfloat16)
    cache.update_and_fetch(value, value)
    mx.eval(cache.state)
    packed = sum(x.nbytes for side in (cache.keys, cache.values) for x in side)
    assert estimate_kv_cache_memory([cache]) == packed
    assert estimate_kv_cache_memory([cache], resident=False) == packed
    plain = SimpleNamespace(cache=[value], nbytes=123_456_789)
    assert estimate_kv_cache_memory([plain]) == value.nbytes


def test_grown_native_cache_ssd_budget_uses_logical_state(tmp_path):
    mx = pytest.importorskip("mlx.core")
    from vmlx_engine.models.glm5_next.glm5_next import Glm5KDACache
    from vmlx_engine.utils.glm5_native_prefix_cache import (
        Glm5NativePrefixCache,
        glm5_native_layout,
    )

    source = _grown_native_cache()
    kda = Glm5KDACache()
    kda.cache = [mx.ones((1, 3, 8), mx.bfloat16) for _ in range(3)]
    kda.cache.append(mx.ones((1, 2, 4, 4), mx.float32))
    original = [kda, source]
    tokens = list(range(2049))
    options = dict(root=tmp_path, max_size_bytes=120_000,
                   model_key="native-capacity-test", layout=glm5_native_layout(original))
    cache = Glm5NativePrefixCache(**options)
    try:
        result = cache.store(tokens, len(tokens), original)
        assert result["outcome"] == "stored"
        assert result["durable"] is True
        assert cache.lookup.total_nbytes == 0
    finally:
        cache.close()
    reopened = Glm5NativePrefixCache(**options)
    try:
        # Serving restores at most N-1, leaving the next prompt token to run.
        boundary, layers = reopened.fetch(tokens + [2050])
        assert boundary == 2049
        restored = layers[1]
        assert restored.nbytes == 98_352
        assert source.nbytes == 196_608
        for before, after in zip(source.state, restored.state):
            assert before.dtype == after.dtype
            assert before.shape == after.shape
            assert mx.array_equal(before, after)
    finally:
        reopened.close()
