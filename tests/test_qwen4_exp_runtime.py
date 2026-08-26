import mlx.core as mx
import numpy as np
import pytest

from vmlx_engine.models.qwen4_exp.language import LanguageModel, Qwen4ExpTextArgs


def _tiny_args() -> Qwen4ExpTextArgs:
    return Qwen4ExpTextArgs(
        hidden_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=997,
        full_attention_interval=4,
        linear_num_value_heads=6,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        num_experts=8,
        num_experts_per_tok=3,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        hc_count=4,
        hc_lowrank=16,
        ple_layer_ids=[2],
        ple_embed_dim=64,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=1009,
        make_ngram_vocab_size_divisible_by=128,
        split_ngram_parts=12,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=32,
        indexer_budget=8,
        indexer_compress_ratio=4,
        rope_theta=10_000.0,
        partial_rotary_factor=0.25,
        mrope_section=[2, 1, 1],
        eos_token_id=7,
        mtp_num_hidden_layers=1,
    )


def _randomize(model, scale=0.5):
    from mlx.utils import tree_map

    mx.random.seed(11)

    def random_parameter(parameter):
        if parameter.dtype in (mx.int32, mx.int64, mx.uint32):
            return parameter
        return (
            mx.random.normal(parameter.shape).astype(parameter.dtype)
            * scale
            / max(1, parameter.shape[-1]) ** 0.5
        )

    model.update(tree_map(random_parameter, model.parameters()))


def _logits(model, ids, cache=None):
    return model(ids, cache=cache).logits


def test_qwen4_exp_native_state_chunk_parity_off_boundaries():
    args = _tiny_args()
    model = LanguageModel(args)
    _randomize(model)
    mx.eval(model.parameters())

    rng = np.random.default_rng(3)
    for sequence_length, chunks in ((29, (13, 9, 7)), (61, (17, 19, 25))):
        ids_np = rng.integers(0, args.vocab_size, size=(1, sequence_length))
        ids_np[0, 11] = args.eos_token_id
        ids = mx.array(ids_np)
        reference = _logits(model, ids)
        mx.eval(reference)
        assert np.asarray(reference).size > 0
        assert not np.isnan(np.asarray(reference)).any()

        cache = model.make_cache()
        cached = _logits(model, ids, cache=cache)
        mx.eval(cached)
        assert np.max(np.abs(np.asarray(cached - reference))) < 1e-4

        cache = model.make_cache()
        outputs = []
        offset = 0
        for chunk_length in chunks:
            outputs.append(
                _logits(model, ids[:, offset : offset + chunk_length], cache=cache)
            )
            offset += chunk_length
        chunked = mx.concatenate(outputs, axis=1)
        mx.eval(chunked)
        assert np.max(np.abs(np.asarray(chunked - reference))) < 1e-4

        cache = model.make_cache()
        stepped = mx.concatenate(
            [
                _logits(model, ids[:, index : index + 1], cache=cache)
                for index in range(sequence_length)
            ],
            axis=1,
        )
        mx.eval(stepped)
        assert np.max(np.abs(np.asarray(stepped - reference))) < 1e-4

    main_logits, hidden = model(
        ids[:, :5], cache=model.make_cache(), return_hidden=True
    )
    mtp_logits = model.mtp_forward(
        hidden[:, -1:, :], ids[:, 5:6], model.make_mtp_cache()
    )
    mx.eval(main_logits, hidden, mtp_logits)
    assert mtp_logits.shape == (1, 1, args.vocab_size)
    assert not np.isnan(np.asarray(mtp_logits)).any()


def test_qwen4_exp_mrope_and_sparse_index_state_survive_chunking():
    args = _tiny_args()
    model = LanguageModel(args)
    _randomize(model)
    mx.eval(model.parameters())

    ids = mx.array(np.arange(23, dtype=np.int32)[None, :] % args.vocab_size)
    text = np.arange(23, dtype=np.int32)
    positions = mx.array(
        np.stack(
            [
                text,
                np.where((text >= 5) & (text < 13), text // 2, text),
                np.where((text >= 5) & (text < 13), text % 4, text),
            ],
            axis=0,
        )[:, None, :]
    )

    reference = model(ids, position_ids=positions).logits
    cache = model.make_cache()
    pieces = []
    start = 0
    for width in (7, 9, 7):
        pieces.append(
            model(
                ids[:, start : start + width],
                cache=cache,
                position_ids=positions[:, :, start : start + width],
            ).logits
        )
        start += width
    chunked = mx.concatenate(pieces, axis=1)
    mx.eval(reference, chunked)
    assert np.max(np.abs(np.asarray(chunked - reference))) < 1e-4

    qsa_layers = [
        layer_cache
        for layer_cache, layer_type in zip(cache, args.layer_types)
        if layer_type == "full_attention"
    ]
    assert qsa_layers
    for layer_cache in qsa_layers:
        keys, values, indexer_keys = layer_cache.state
        assert layer_cache.offset == 23
        assert keys.shape[2] == values.shape[2] == indexer_keys.shape[2] == 23
        assert indexer_keys.shape[1] == 1


def test_qwen4_exp_qsa_and_ple_support_synchronous_batch_chunking():
    args = _tiny_args()
    model = LanguageModel(args)
    _randomize(model)
    mx.eval(model.parameters())

    ids = mx.array(
        np.stack(
            [
                np.arange(19, dtype=np.int32) % args.vocab_size,
                (np.arange(19, dtype=np.int32) * 7 + 3) % args.vocab_size,
            ]
        )
    )
    reference = model(ids).logits
    cache = model.make_cache()
    chunked = mx.concatenate(
        [
            model(ids[:, :7], cache=cache).logits,
            model(ids[:, 7:12], cache=cache).logits,
            model(ids[:, 12:], cache=cache).logits,
        ],
        axis=1,
    )
    mx.eval(reference, chunked)
    assert np.max(np.abs(np.asarray(chunked - reference))) < 1e-4

    qsa_cache = next(
        layer_cache
        for layer_cache, layer_type in zip(cache, args.layer_types)
        if layer_type == "full_attention"
    )
    assert qsa_cache.state[2].shape[:3] == (2, 1, 19)
    ple_cache = cache[args.ple_layer_ids[0] - 1]
    assert ple_cache.cache[2].shape == (2, args.ngram_size - 1)


def test_qwen4_exp_registry_and_runtime_registration_are_source_available():
    from vmlx_engine.model_config_registry import ModelConfigRegistry
    from vmlx_engine.model_configs import register_all
    from vmlx_engine.models.qwen4_exp.register import (
        qwen4_exp_runtime_available,
        register_qwen4_exp_runtime,
    )

    registry = ModelConfigRegistry()
    register_all(registry)
    config = next(item for item in registry._configs if item.family_name == "qwen4_exp")
    assert config.family_name == "qwen4_exp"
    assert config.cache_type == "hybrid"
    assert config.cache_subtype == "qsa_gdn_ple_v1"
    assert config.is_mllm is True
    assert config.tool_parser == "qwen"
    assert config.reasoning_parser == "qwen3"
    assert config.architecture_hints["ple_storage"] == "ssd_row_addressed"
    assert config.architecture_hints["cache_precision"] == "full"

    assert qwen4_exp_runtime_available() is True
    assert register_qwen4_exp_runtime() is True


def test_qwen4_exp_loader_never_requests_ple_table_tensors():
    from pathlib import Path

    from vmlx_engine.models.qwen4_exp.loader import _load_non_table_weight_files

    class FakeHandle:
        def __init__(self, tensors):
            self.tensors = tensors
            self.requested = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def metadata(self):
            return {"format": "mlx"}

        def keys(self):
            return list(self.tensors)

        def get_tensor(self, key):
            self.requested.append(key)
            return self.tensors[key]

    prefix = "model.language_model.layers.1.ple.ple_embedding."
    tensors = {
        prefix + "ngram_embedding.shard_0.weight": mx.zeros((3, 20), mx.uint32),
        prefix + "ngram_embedding.shard_0.scales": mx.zeros((3, 3)),
        prefix + "ngram_embedding.shard_0.biases": mx.zeros((3, 3)),
        prefix + "layer_multipliers": mx.array([1, 3, 5]),
        prefix + "ngram_heads_vocab_sizes": mx.array([11, 13]),
        prefix + "ngram_heads_offsets": mx.array([0, 11]),
        "model.language_model.layers.0.self_attn.q_proj.weight": mx.ones((2, 2)),
    }
    handle = FakeHandle(tensors)

    def fake_open(_path, framework):
        assert framework == "mlx"
        return handle

    weights, is_mlx, buffers = _load_non_table_weight_files(
        [Path("synthetic-00001.safetensors")],
        frozenset(),
        safe_open_fn=fake_open,
    )

    assert is_mlx is True
    assert set(weights) == {"model.language_model.layers.0.self_attn.q_proj.weight"}
    assert set(buffers) == {
        "layer_multipliers",
        "ngram_heads_vocab_sizes",
        "ngram_heads_offsets",
    }
    assert all("ngram_embedding.shard_0" not in key for key in handle.requested)


def test_qwen4_exp_ple_layout_resolution_is_complete_and_unambiguous():
    from vmlx_engine.models.qwen4_exp.loader import _resolve_ple_module_key_format

    raw_prefix = (
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_"
    )
    weight_map = {
        f"{raw_prefix}{shard}.{suffix}": f"part-{shard}.safetensors"
        for shard in range(3)
        for suffix in ("weight", "scales", "biases")
    }
    assert _resolve_ple_module_key_format(weight_map, 3) == raw_prefix + "{}"

    del weight_map[f"{raw_prefix}2.biases"]
    with pytest.raises(ValueError, match="no complete indexed PLE"):
        _resolve_ple_module_key_format(weight_map, 3)


def test_qwen4_exp_ple_affine_layout_uses_exact_160_dimensional_rows():
    from vmlx_engine.models.qwen4_exp.table_reader import _validate_affine_layout

    # The old scale-derived formula yielded 64 * 3 = 192. The runtime contract
    # is PLE 2560 / 16 bigram+trigram heads = exactly 160 dimensions per row.
    _validate_affine_layout(
        weight_shape=(7, 20),
        weight_dtype="U32",
        scales_shape=(7, 3),
        biases_shape=(7, 3),
        group_size=64,
        logical_bits=4,
        storage_bits=4,
        head_dim=160,
    )
    with pytest.raises(ValueError, match="packed width"):
        _validate_affine_layout(
            weight_shape=(7, 20),
            weight_dtype="U32",
            scales_shape=(7, 3),
            biases_shape=(7, 3),
            group_size=64,
            logical_bits=4,
            storage_bits=4,
            head_dim=192,
        )


def test_qwen4_exp_nested_config_maps_qsa_ple_and_mtp_contracts():
    cfg = {
        "model_type": "qwen4_exp_text",
        "num_hidden_layers": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "partial_rotary_factor": 0.25,
        "mrope_section": [2, 1, 1],
        "full_attention_interval": 4,
        "sparse_attention_config": {
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "budget": 32,
            "block_size": 4,
        },
        "ple_config": {
            "ple_layer_ids": [2],
            "ple_embed_dim": 64,
            "ngram_size": 3,
            "heads_per_ngram": 8,
            "split_ngram_parts": 4,
        },
        "num_nextn_predict_layers": 1,
    }
    args = Qwen4ExpTextArgs.from_config(cfg)
    assert args.indexer_n_heads == 2
    assert args.indexer_kv_heads == 1
    assert args.indexer_head_dim == 16
    assert args.indexer_budget == 32
    assert args.indexer_compress_ratio == 4
    assert args.ple_layer_ids == [2]
    assert args.mtp_num_hidden_layers == 1
    assert args.indexer_budget // args.indexer_compress_ratio == 8


def test_qwen4_exp_ple_companion_is_ssd_only_and_survives_restart(tmp_path):
    from vmlx_engine.utils.ssm_companion_cache import SSMCompanionCache
    from vmlx_engine.utils.ssm_companion_disk_store import SSMCompanionDiskStore

    args = _tiny_args()
    model = LanguageModel(args)
    _randomize(model)
    mx.eval(model.parameters())

    token_ids = list(range(29))
    cache = model.make_cache()
    logits = model(mx.array([token_ids]), cache=cache).logits
    mx.eval(logits)

    # Layer 2 (one-based) is the PLE-bearing GDN layer. Its cumulative state is
    # exactly: GDN conv, GDN recurrent delta, PLE prior token IDs, and PLE
    # dilated-convolution state. All four are required for answer-preserving
    # prefix restore; none may be reconstructed from QSA K/V blocks.
    ple_cache = cache[args.ple_layer_ids[0] - 1]
    assert len(ple_cache.cache) == 4
    assert all(value is not None for value in ple_cache.cache)
    expected = [np.asarray(value) for value in ple_cache.cache]

    disk_dir = tmp_path / "qwen4-ple-companion"
    first_disk = SSMCompanionDiskStore(
        directory=disk_dir,
        budget_bytes=64 * 1024 * 1024,
    )
    first = SSMCompanionCache(
        max_entries=0,
        max_bytes=0,
        model_key="qwen4-exp-synthetic",
        disk_store=first_disk,
    )
    assert first.ram_enabled is False
    first.store(token_ids, len(token_ids), [ple_cache], is_complete=True)
    assert first.size == 0
    assert first.total_nbytes == 0
    assert first_disk.wait_for_pending(timeout=10.0)
    assert first_disk.shutdown(timeout=10.0)

    # New objects emulate an engine restart: no L1 state or in-process index is
    # retained. The SSD entry must refault with its native dtype/value intact.
    second_disk = SSMCompanionDiskStore(
        directory=disk_dir,
        budget_bytes=64 * 1024 * 1024,
    )
    second = SSMCompanionCache(
        max_entries=0,
        max_bytes=0,
        model_key="qwen4-exp-synthetic",
        disk_store=second_disk,
    )
    restored = second.fetch(token_ids, len(token_ids))
    assert restored is not None
    states, is_complete = restored
    assert is_complete is True
    assert second.size == 0
    assert second.total_nbytes == 0
    assert len(states) == 1
    assert len(states[0].cache) == 4
    for got, want in zip(states[0].cache, expected):
        got_np = np.asarray(got)
        assert got_np.dtype == want.dtype
        np.testing.assert_array_equal(got_np, want)
    assert second_disk.shutdown(timeout=10.0)


def test_qwen4_exp_qsa_cache_requires_all_three_native_lanes():
    from vmlx_engine.cache_record_validator import validate_live_cache

    args = _tiny_args()
    model = LanguageModel(args)
    _randomize(model)
    cache = model.make_cache()
    logits = model(mx.array([list(range(13))]), cache=cache).logits
    mx.eval(logits)

    qsa_cache = next(
        layer_cache
        for layer_cache, layer_type in zip(cache, args.layer_types)
        if layer_type == "full_attention"
    )
    ok, reason, _ = validate_live_cache([qsa_cache], expected_num_layers=1)
    assert ok, reason

    # A K/V-only restore is unsafe: the QSA selector would score against a
    # different prefix than attention. Validation must fail closed.
    qsa_cache.idx_keys = None
    ok, reason, _ = validate_live_cache([qsa_cache], expected_num_layers=1)
    assert ok is False
    assert "idx_keys" in reason


def test_qwen4_exp_qsa_three_lane_prompt_cache_survives_restart(tmp_path):
    from vmlx_engine.disk_cache import DiskCacheManager

    args = _tiny_args()
    model = LanguageModel(args)
    _randomize(model)
    token_ids = list(range(13))
    live = model.make_cache()
    logits = model(mx.array([token_ids]), cache=live).logits
    mx.eval(logits)

    qsa_index = args.layer_types.index("full_attention")
    live_qsa = live[qsa_index]
    assert live_qsa.offset == 13
    assert live_qsa.state[0].shape[2] == 13
    assert live_qsa.state[1].shape[2] == 13
    assert live_qsa.state[2].shape[2] == 13

    first = DiskCacheManager(cache_dir=str(tmp_path), max_size_gb=1.0)
    try:
        assert first.store(token_ids, [live_qsa])
    finally:
        first.shutdown()

    second = DiskCacheManager(cache_dir=str(tmp_path), max_size_gb=1.0)
    try:
        restored = second.fetch(token_ids)
        assert restored is not None
        restored_qsa = restored[0]
        keys, values, indexer_keys = restored_qsa.state
        assert restored_qsa.offset == 13
        assert keys.shape[2] == values.shape[2] == indexer_keys.shape[2] == 13
        np.testing.assert_array_equal(
            np.asarray(indexer_keys), np.asarray(live_qsa.state[2])
        )
    finally:
        second.shutdown()


def test_qwen4_exp_native_mtp_runtime_is_detected_from_the_attached_head():
    from vmlx_engine.native_mtp import model_has_native_mtp_runtime

    model = LanguageModel(_tiny_args())
    assert model_has_native_mtp_runtime(model) is True


def test_qwen4_exp_vlm_config_builds_text_image_and_video_contracts():
    from mlx_vlm.utils import update_module_configs

    from vmlx_engine.models.qwen4_exp.register import register_qwen4_exp_runtime

    assert register_qwen4_exp_runtime() is True
    import mlx_vlm.models.qwen4_exp as model_class

    text = dict(_tiny_args().__dict__)
    text.pop("rotary_dim", None)
    config = {
        "model_type": "qwen4_exp",
        "text_config": text,
        "vision_config": {
            "model_type": "qwen4_exp",
            "depth": 1,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        "image_token_id": 901,
        "video_token_id": 902,
    }
    parsed = model_class.ModelConfig.from_dict(config)
    parsed = update_module_configs(parsed, model_class, config, ["text", "vision"])
    # Runtime registration loads the same source under mlx_vlm.models, so the
    # class has a distinct Python identity from the vmlx_engine import above.
    assert type(parsed.text_config).__name__ == "Qwen4ExpTextArgs"
    assert parsed.text_config.indexer_budget == _tiny_args().indexer_budget
    assert parsed.vision_config.model_type == "qwen4_exp"
    assert parsed.image_token_index == 901
    assert parsed.video_token_index == 902


def test_qwen4_exp_reasoning_and_tools_do_not_invent_bundle_defaults():
    from vmlx_engine.model_config_registry import ModelConfigRegistry
    from vmlx_engine.model_configs import register_all

    registry = ModelConfigRegistry()
    register_all(registry)
    config = next(item for item in registry._configs if item.family_name == "qwen4_exp")
    assert config.reasoning_parser == "qwen3"
    assert config.tool_parser == "qwen"
    assert config.supports_thinking is True
    assert config.supports_native_tools is True
    assert config.architecture_hints["default_enable_thinking"] is None
    assert config.architecture_hints["modalities"] == ["text", "image", "video"]
    assert config.architecture_hints["audio_input"] is False


def test_qwen4_exp_ple_manifest_aliases_are_deterministic_and_fail_closed():
    from vmlx_engine.models.qwen4_exp.table_reader import (
        _module_aliases,
        _unique_mapping_override,
    )

    official = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0"
    aliases = _module_aliases(official)
    runtime = official.replace("model.language_model", "language_model.model", 1)
    short = official.replace(".ple.ple_embedding.", ".ple.")
    assert aliases[0] == official
    assert len(aliases) == len(set(aliases))
    same = {official: {"bits": 4}, runtime: {"bits": 4}}
    assert _unique_mapping_override(same, aliases, label="test") == {"bits": 4}
    with pytest.raises(ValueError, match="conflicting test aliases"):
        _unique_mapping_override(
            {official: {"bits": 4}, short: {"bits": 2}},
            aliases,
            label="test",
        )
