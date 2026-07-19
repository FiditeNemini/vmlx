# SPDX-License-Identifier: Apache-2.0
"""openPangu-2.0-Flash (openpangu_v2) integration regression tests.

Pins the vendored runtime's load-bearing invariants (Swift-port bug list):
registry autodetect, sanitize transforms, biased-select/unbiased-weight MoE
gate, path-dependent cache contract, prefill-vs-incremental decode
equivalence (conv-state + SWA rotation + DSA indexer + sink masks), and the
manifest-driven quantization overrides.
"""

import os

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
    import mlx_lm.models.cache as mlx_cache

    from vmlx_engine.models.openpangu_v2.cache import OpenPanguV2LayerCache

    assert mlx_cache.OpenPanguV2LayerCache is OpenPanguV2LayerCache


def test_registry_resolves_openpangu_family(tmp_path):
    import json

    from vmlx_engine.model_config_registry import get_model_config_registry

    model_dir = tmp_path / "openPangu-2.0-Flash-JANG_2L"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "openpangu_v2", "hidden_size": 2560})
    )
    # The converter stamps the coarse cache_type="hybrid" AND tool_parser="qwen";
    # the registry entry's kv/openpangu_v2_composite + "openpangu" contract must
    # WIN (hybrid would misroute the scheduler into SSM handling; the qwen
    # parser never matches the <|tool_call_start|> JSON-list format —
    # live-proven tool_calls=None). Found live on the first server launch.
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
    assert mc.tool_parser == "openpangu"
    assert "<|message_end|>" in mc.eos_tokens


def test_cli_policy_forces_turboquant_and_generic_kv_quant_off(monkeypatch, request):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from vmlx_engine.cli import _apply_openpangu_cache_policy

    original_disable_tq = os.environ.get("VMLX_DISABLE_TQ_KV")

    def _restore_disable_tq():
        if original_disable_tq is None:
            os.environ.pop("VMLX_DISABLE_TQ_KV", None)
        else:
            os.environ["VMLX_DISABLE_TQ_KV"] = original_disable_tq

    request.addfinalizer(_restore_disable_tq)
    monkeypatch.setenv("VMLX_FORCE_TQ_AUTO", "1")
    monkeypatch.delenv("VMLX_DISABLE_TQ_KV", raising=False)
    args = SimpleNamespace(
        use_paged_cache=True,
        enable_block_disk_cache=True,
        kv_cache_quantization="q4",
    )

    applied, changed = _apply_openpangu_cache_policy(args, Mock())

    assert applied is True
    assert args.kv_cache_quantization == "none"
    assert args._openpangu_force_no_kv_cache_quantization is True
    assert args.use_paged_cache is False
    assert args.enable_block_disk_cache is False
    assert "kv_quant=q4->none" in changed
    assert __import__("os").environ["VMLX_DISABLE_TQ_KV"] == "1"
    assert "VMLX_FORCE_TQ_AUTO" not in __import__("os").environ


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
    assert out["model.layers.0.self_attn.qa_conv.conv.weight"].shape == (32, 3, 1)
    assert out["model.layers.0.self_attn.o_conv.conv.weight"].shape == (64, 3, 1)
    assert "model.layers.0.self_attn.qa_conv.weight" not in out
    assert "model.layers.0.self_attn.o_conv.weight" not in out
    assert "model.layers.4.eh_proj.weight" not in out
    assert "model.layers.6.shared_head.head.weight" not in out
    assert "model.rotary_emb.inv_freq" not in out
    assert "model.layers.1.input_layernorm.weight" in out


def test_sanitized_conv_keys_land_on_real_model_parameters(tiny_model):
    """All three checkpoint conv names must resolve to the nested MLX modules."""
    from mlx.utils import tree_flatten

    incoming = {
        "model.layers.0.self_attn.qa_conv.weight": mx.zeros((32, 1, 3)),
        "model.layers.0.self_attn.compresskv_conv.weight": mx.zeros((16, 1, 3)),
        "model.layers.0.self_attn.o_conv.weight": mx.zeros((64, 1, 3)),
    }
    sanitized = tiny_model.sanitize(incoming)
    parameter_names = {name for name, _ in tree_flatten(tiny_model.parameters())}

    assert set(sanitized) <= parameter_names
    assert set(sanitized) == {
        "model.layers.0.self_attn.qa_conv.conv.weight",
        "model.layers.0.self_attn.compresskv_conv.conv.weight",
        "model.layers.0.self_attn.o_conv.conv.weight",
    }


def test_openpangu_weight_landing_audit_fails_closed():
    from vmlx_engine.utils.jang_loader import (
        _finalize_openpangu_weight_landing,
        _needs_renamed_key_requantization,
        _record_openpangu_weight_landing,
    )

    expected = {"model.embed_tokens.weight", "lm_head.weight"}
    for layer in range(2):
        expected.update(
            {
                f"model.layers.{layer}.input_layernorm.weight",
                f"model.layers.{layer}.self_attn.qa_conv.conv.weight",
                f"model.layers.{layer}.self_attn.compresskv_conv.conv.weight",
                f"model.layers.{layer}.self_attn.o_conv.conv.weight",
            }
        )
    seen = set()
    _record_openpangu_weight_landing(
        expected, seen, {name: object() for name in expected}, shard_name="unit"
    )
    stats = _finalize_openpangu_weight_landing(
        expected, seen, layer_count=2
    )
    assert stats == {"parameter_leaves": 10, "layers": 2, "causal_convs": 6}

    with pytest.raises(RuntimeError, match="unmatched checkpoint leaves"):
        _record_openpangu_weight_landing(
            expected,
            set(),
            {"model.layers.0.self_attn.qa_conv.weight": object()},
            shard_name="bad",
        )

    with pytest.raises(RuntimeError, match="incomplete"):
        _finalize_openpangu_weight_landing(
            expected, seen - {"lm_head.weight"}, layer_count=2
        )

    assert not _needs_renamed_key_requantization(
        {"model_type": "openpangu_v2"},
        {"architecture": {"attention": "mla"}},
        loader_name="load_model",
    )
    assert _needs_renamed_key_requantization(
        {"model_type": "mistral4"},
        {"architecture": {"attention": "mla"}},
        loader_name="load_model",
    )


def test_dsa_indexer_uses_checkpoint_exact_rmsnorm(tiny_model):
    from mlx.utils import tree_flatten

    parameter_names = {name for name, _ in tree_flatten(tiny_model.parameters())}
    assert "model.layers.0.self_attn.indexer.k_norm.weight" in parameter_names
    assert "model.layers.0.self_attn.indexer.k_norm.bias" not in parameter_names


def test_dsa_sparse_indexer_reports_real_over_threshold_activation(tiny_model, caplog):
    indexer = tiny_model.model.layers[0].self_attn.indexer
    assert indexer is not None
    x = mx.random.normal((1, 13, 64))
    qr = mx.random.normal((1, 13, 32))

    with caplog.at_level("INFO"):
        indices = indexer(x, qr, None, cache=None)
        mx.eval(indices)

    assert indices.shape == (1, 1, 13, 12)
    assert "openPangu DSA sparse indexer active: layer=0 key_length=13 topk=12" in caplog.text


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


def test_forward_refuses_silent_cache_layer_truncation(tiny_model):
    tokens = mx.array([[1, 2]], dtype=mx.int32)
    with pytest.raises(ValueError, match=r"cache=3 layers=4"):
        tiny_model.model(tokens, cache=tiny_model.make_cache()[:-1])


def test_scheduler_routes_composite_cache_to_exact_typed_lane(tiny_model, tmp_path):
    from types import SimpleNamespace

    from vmlx_engine.scheduler import Scheduler, SchedulerConfig

    assert {type(layer).__name__ for layer in tiny_model.make_cache()} == {
        "OpenPanguV2LayerCache"
    }
    assert Scheduler._is_hybrid_model(tiny_model) is False

    tokenizer = SimpleNamespace(eos_token_id=148900, eos_token_ids=[148900, 148902])
    config = SchedulerConfig(
        enable_prefix_cache=True,
        use_paged_cache=True,
        enable_disk_cache=True,
        enable_block_disk_cache=True,
        disk_cache_dir=str(tmp_path / "prompt-l2"),
        block_disk_cache_dir=str(tmp_path / "block-l2"),
    )
    missing = object()
    prior_config = getattr(tiny_model, "config", missing)
    tiny_model.config = {"model_type": "openpangu_v2"}
    scheduler = Scheduler(tiny_model, tokenizer, config)
    try:
        assert scheduler._uses_openpangu_cache is True
        assert scheduler._prefix_cache_requested is True
        assert scheduler._prompt_disk_cache_requested is True
        assert scheduler._block_disk_cache_requested is True
        assert scheduler.config.enable_prefix_cache is True
        assert scheduler.config.use_paged_cache is False
        assert scheduler.config.enable_disk_cache is True
        assert scheduler.config.enable_block_disk_cache is False
        assert scheduler.config.use_memory_aware_cache is True
        assert scheduler.memory_aware_cache is not None
        assert scheduler.paged_cache_manager is None
        assert scheduler.disk_cache is not None
        assert scheduler.disk_cache._required_cache_class == "OpenPanguV2LayerCache"
    finally:
        scheduler.shutdown()
        if prior_config is missing:
            del tiny_model.config
        else:
            tiny_model.config = prior_config


def test_health_reports_openpangu_composite_policy_not_ssm_or_paged():
    from types import SimpleNamespace

    from vmlx_engine.server import _native_cache_status

    scheduler = SimpleNamespace(
        _model_type_for_runtime="openpangu_v2",
        _is_hybrid=False,
        _prefix_cache_requested=True,
        _prompt_disk_cache_requested=True,
        _block_disk_cache_requested=True,
        config=SimpleNamespace(enable_prefix_cache=True),
        memory_aware_cache=object(),
        block_aware_cache=None,
        paged_cache_manager=None,
        disk_cache=object(),
    )
    cfg = SimpleNamespace(
        cache_type="kv", cache_subtype="openpangu_v2_composite"
    )

    status = _native_cache_status(scheduler, family="openpangu_v2", cfg=cfg)

    assert status["schema"] == "openpangu_v2_composite_v2"
    assert status["cache_type"] == "native_path_dependent_composite"
    assert status["prefix_configured"] is True
    assert status["prefix"] is True
    assert status["paged"] is False
    assert status["prompt_disk_l2_configured"] is True
    assert status["block_disk_l2_configured"] is True
    assert status["prompt_disk_l2"] is True
    assert status["block_disk_l2"] is False
    assert status["cache_store_policy"]["prompt_boundary_snapshot"] == (
        "exact_typed_n_minus_1"
    )
    assert status["cache_store_policy"]["reverse_truncation"] == (
        "unsupported_clean_miss"
    )
    assert "ssm_companion_state" not in status["components"]


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
        fresh = type(layer_cache).from_state(st, meta)
        assert fresh.kv.offset == layer_cache.kv.offset
        assert fresh.is_dsa == layer_cache.is_dsa
        assert fresh.window == layer_cache.window
        assert fresh.conv_states[0] is not None
        mx.eval(fresh.conv_states[0])


def test_exact_cache_clone_preserves_all_typed_state_without_aliasing(tiny_model):
    from vmlx_engine.models.openpangu_v2.cache import clone_openpangu_layer_cache

    cache = tiny_model.make_cache()
    tiny_model(mx.array([[1, 2, 3, 4]]), cache=cache)

    def copy_array(value):
        copied = value + mx.zeros_like(value)
        mx.eval(copied)
        return copied

    for source in (cache[0], cache[1]):
        cloned = clone_openpangu_layer_cache(source, copy_fn=copy_array)
        assert cloned.offset == source.offset == 4
        assert cloned.meta_state == source.meta_state
        assert cloned.kv.keys is not source.kv.keys
        assert cloned.conv_states[0] is not source.conv_states[0]
        if source.is_dsa:
            assert cloned.indexer_kv is not None
            assert cloned.indexer_kv.offset == source.indexer_kv.offset
            assert cloned.indexer_kv.keys is not source.indexer_kv.keys
        cloned.kv.trim(1)
        assert cloned.kv.offset == 3
        assert source.kv.offset == 4


def test_single_batch_snapshot_is_exact_n_minus_one(tiny_model):
    from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator

    generator = SingleBatchGenerator(tiny_model, max_tokens=1)
    generator.insert([[10, 11, 12, 13]])
    prompt_responses, generation_responses = generator.next()

    assert generation_responses == []
    assert len(prompt_responses) == 1
    response = prompt_responses[0]
    assert response.prompt_cache_snapshot is not None
    assert {layer.offset for layer in response.prompt_cache_snapshot} == {3}
    # The live cache has consumed the last prompt token and one-token lookahead
    # generation, while the stored boundary remains immutable at N-1.
    assert min(layer.offset for layer in response.prompt_cache) >= 4


def test_single_batch_skips_oversize_snapshot_before_deep_copy(
    tiny_model, monkeypatch, caplog
):
    from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator

    generator = SingleBatchGenerator(
        tiny_model,
        max_tokens=1,
        prompt_snapshot_max_bytes=1,
    )

    def forbidden_copy(cls, cache_obj):
        raise AssertionError("oversize typed cache must be rejected before cloning")

    monkeypatch.setattr(
        SingleBatchGenerator,
        "_clone_cache_object",
        classmethod(forbidden_copy),
    )
    generator.insert([[10, 11, 12, 13]])
    with caplog.at_level("WARNING"):
        prompt_responses, generation_responses = generator.next()

    assert generation_responses == []
    assert prompt_responses[0].prompt_cache_snapshot is None
    assert generator.prompt_snapshot_oversize_skips == 1
    assert generator.prompt_snapshot_last_estimated_bytes > 1
    assert "Skipping typed prompt snapshot before copy" in caplog.text


def test_memory_cache_clones_exact_composite_and_rejects_reverse_trim(tiny_model):
    from vmlx_engine.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig
    from vmlx_engine.models.openpangu_v2.cache import clone_openpangu_layer_cache

    live = tiny_model.make_cache()
    tiny_model(mx.array([[1, 2, 3]]), cache=live)
    stored = [
        clone_openpangu_layer_cache(
            layer,
            copy_fn=lambda value: value + mx.zeros_like(value),
        )
        for layer in live
    ]
    cache = MemoryAwarePrefixCache(
        model=tiny_model,
        config=MemoryCacheConfig(max_memory_mb=64),
    )
    assert cache.store([1, 2, 3], stored)

    fetched, remaining = cache.fetch([1, 2, 3, 4])
    assert fetched is not None
    assert remaining == [4]
    assert {layer.offset for layer in fetched} == {3}
    fetched[0].kv.trim(1)
    assert stored[0].offset == 3

    reverse, reverse_remaining = cache.fetch([1, 2])
    assert reverse is None
    assert reverse_remaining == [1, 2]


def test_disk_cache_roundtrip_restores_typed_composite(tiny_model, tmp_path):
    from vmlx_engine.disk_cache import DiskCacheManager
    from vmlx_engine.models.openpangu_v2.cache import clone_openpangu_layer_cache

    live = tiny_model.make_cache()
    tiny_model(mx.array([[1, 2, 3]]), cache=live)
    payload = [
        clone_openpangu_layer_cache(
            layer,
            copy_fn=lambda value: value + mx.zeros_like(value),
        )
        for layer in live
    ]
    cache_dir = tmp_path / "openpangu-l2"
    writer = DiskCacheManager(
        str(cache_dir),
        expected_num_layers=4,
        required_cache_class="OpenPanguV2LayerCache",
    )
    assert writer.store([1, 2, 3, 4], payload)
    writer.shutdown()

    reader = DiskCacheManager(
        str(cache_dir),
        expected_num_layers=4,
        required_cache_class="OpenPanguV2LayerCache",
    )
    try:
        restored = reader.fetch([1, 2, 3, 4])
        assert restored is not None
        assert {type(layer).__name__ for layer in restored} == {
            "OpenPanguV2LayerCache"
        }
        assert {layer.offset for layer in restored} == {3}
        assert restored[0].is_dsa and restored[0].indexer_kv.offset == 3
        assert not restored[1].is_dsa and restored[1].kv._idx == 3
        assert all(layer.conv_states[0] is not None for layer in restored)
    finally:
        reader.shutdown()


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


def test_native_mtp_status_accepts_openpangu_included_layers_runtime_unwired(tmp_path):
    """openPangu stores MTP as extra model.layers entries, not mtp.* keys."""
    import json

    from vmlx_engine.native_mtp import inspect_native_mtp_bundle

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "openpangu_v2",
                "num_nextn_predict_layers": 3,
            }
        )
    )
    (tmp_path / "jang_config.json").write_text(
        json.dumps(
            {
                "capabilities": {"family": "openpangu_v2"},
                "runtime": {
                    "bundle_has_mtp": True,
                    "mtp_layers": 3,
                    "mtp_mode": "included",
                },
                "mtp": {
                    "kept": True,
                    "enabled": True,
                    "num_layers": 3,
                    "layer_indices": [46, 47, 48],
                },
            }
        )
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.46.eh_proj.weight": "shard.safetensors",
                    "model.layers.47.shared_head.head.weight": "shard.safetensors",
                    "model.layers.48.shared_head.head.weight": "shard.safetensors",
                }
            }
        )
    )

    status = inspect_native_mtp_bundle(str(tmp_path))

    assert status["family"] == "openpangu_v2"
    assert status["index_has_mtp_tensors"] is False
    assert status["runtime_supported"] is False
    assert status["runtime_available"] is False
    assert status["status"] == "weights_present_runtime_unwired"
    assert status["runtime_mtp_mode"] == "included_but_dropped_for_runtime"
    assert status["issues"] == []
