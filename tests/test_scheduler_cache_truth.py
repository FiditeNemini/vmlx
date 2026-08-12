# SPDX-License-Identifier: Apache-2.0
"""Focused truthfulness guards for scheduler cache identity and TTFT stats."""

import inspect
import json
import os
from types import SimpleNamespace


def test_ttft_ewma_seeds_first_sample_before_smoothing():
    from vmlx_engine.scheduler import Scheduler

    scheduler = object.__new__(Scheduler)
    scheduler._ewma_ttft = 0.0
    scheduler._ttft_sample_count = 0
    scheduler._ttft_alpha = 0.1

    scheduler._record_ttft_sample(12.0)

    assert scheduler._ewma_ttft == 12.0
    assert scheduler._ttft_sample_count == 1

    scheduler._record_ttft_sample(2.0)

    assert scheduler._ewma_ttft == 11.0
    assert scheduler._ttft_sample_count == 2
    assert "self._record_ttft_sample(ttft)" in inspect.getsource(
        Scheduler._process_batch_responses
    )


def test_scheduler_block_l2_scope_uses_loaded_bundle_cache_identity():
    """The scope key must bind to the loaded BUNDLE, not just its path.

    The construction moved into ``prefix_cache.build_block_cache_namespace`` so
    the MLLM scheduler could stop keeping a second, weaker copy of it (that one
    omitted ``bundle=`` entirely, so an in-place VLM swap replayed stale KV).
    This asserts both halves: the scheduler passes the loaded model and its
    quantization identity, and the shared builder folds them into the key.
    """
    from vmlx_engine.scheduler import Scheduler
    from vmlx_engine.prefix_cache import build_block_cache_namespace

    source = inspect.getsource(Scheduler.__init__)
    start = source.index("block_scope_key = build_block_cache_namespace(")
    call = source[start : start + 900]

    assert "model=self.model" in call
    assert "model_path=self.config.model_path" in call
    assert "smelt_enabled=self.config.smelt_enabled" in call
    assert "kv_quant_bits=self._kv_cache_bits" in call

    builder = inspect.getsource(build_block_cache_namespace)
    assert "compute_model_cache_key(" in builder
    assert 'f":bundle={bundle_cache_key}"' in builder


def test_model_cache_identity_changes_with_config_and_weight_index(tmp_path):
    from vmlx_engine.prefix_cache import compute_model_cache_key

    model = SimpleNamespace(
        config=SimpleNamespace(
            model_type="deepseek_v4",
            num_hidden_layers=43,
            hidden_size=7168,
        )
    )
    config_path = tmp_path / "config.json"
    jang_path = tmp_path / "jang_config.json"
    index_path = tmp_path / "model.safetensors.index.json"
    config_path.write_text(
        json.dumps({"model_type": "deepseek_v4", "revision": "a"}),
        encoding="utf-8",
    )
    jang_path.write_text(json.dumps({"format": "jang"}), encoding="utf-8")
    index_path.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1},
                "weight_map": {"model.layers.0.weight": "model-00001.safetensors"},
            }
        ),
        encoding="utf-8",
    )

    first = compute_model_cache_key(model, model_path=str(tmp_path))

    config_path.write_text(
        json.dumps({"model_type": "deepseek_v4", "revision": "b"}),
        encoding="utf-8",
    )
    config_stat = config_path.stat()
    os.utime(
        config_path,
        ns=(config_stat.st_atime_ns, config_stat.st_mtime_ns + 1_000_000_000),
    )
    changed_config = compute_model_cache_key(model, model_path=str(tmp_path))

    index_path.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 2},
                "weight_map": {"model.layers.0.weight": "model-00002.safetensors"},
            }
        ),
        encoding="utf-8",
    )
    changed_weights = compute_model_cache_key(model, model_path=str(tmp_path))

    assert changed_config != first
    assert changed_weights != changed_config


def test_indexed_model_cache_identity_tracks_same_size_shard_replacement(tmp_path):
    from vmlx_engine.prefix_cache import compute_model_cache_key

    model = SimpleNamespace(
        config=SimpleNamespace(
            model_type="deepseek_v4",
            num_hidden_layers=43,
            hidden_size=7168,
        )
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "deepseek_v4"}),
        encoding="utf-8",
    )
    shard_path = tmp_path / "model-00001-of-00001.safetensors"
    shard_path.write_bytes(b"AAAA")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 4},
                "weight_map": {
                    "model.layers.0.weight": shard_path.name,
                    "model.layers.1.weight": shard_path.name,
                },
            }
        ),
        encoding="utf-8",
    )

    first = compute_model_cache_key(model, model_path=str(tmp_path))
    unchanged = compute_model_cache_key(model, model_path=str(tmp_path))
    assert unchanged == first

    prior_stat = shard_path.stat()
    shard_path.write_bytes(b"BBBB")
    assert shard_path.stat().st_size == prior_stat.st_size
    os.utime(
        shard_path,
        ns=(prior_stat.st_atime_ns, prior_stat.st_mtime_ns + 1_000_000_000),
    )

    replaced = compute_model_cache_key(model, model_path=str(tmp_path))
    assert replaced != first


def test_block_l2_directory_changes_when_bundle_is_replaced_in_place(tmp_path):
    from vmlx_engine.scheduler import Scheduler, SchedulerConfig

    class KVCache:
        pass

    class Model:
        args = SimpleNamespace(
            model_type="qwen3",
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            hidden_size=8,
            vocab_size=16,
        )
        config = args

        def make_cache(self):
            return [KVCache()]

    class Tokenizer:
        eos_token_id = 1
        name_or_path = "cache-truth-tokenizer"

        def encode(self, *_args, **_kwargs):
            return [1]

    model_path = tmp_path / "model"
    cache_root = tmp_path / "block-cache"
    model_path.mkdir()
    config_path = model_path / "config.json"
    config_path.write_text(
        json.dumps({"model_type": "qwen3", "revision": "a"}),
        encoding="utf-8",
    )
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1},
                "weight_map": {"model.layers.0.weight": "model.safetensors"},
            }
        ),
        encoding="utf-8",
    )

    config = SchedulerConfig(
        enable_prefix_cache=True,
        use_paged_cache=False,
        use_memory_aware_cache=False,
        enable_block_disk_cache=True,
        block_disk_cache_dir=str(cache_root),
        block_disk_cache_max_gb=0.01,
        max_cache_blocks=4,
        model_path=str(model_path),
    )

    first_scheduler = Scheduler(Model(), Tokenizer(), config)
    try:
        first_dir = first_scheduler.paged_cache_manager._disk_store.cache_dir
    finally:
        first_scheduler.shutdown()

    config_path.write_text(
        json.dumps({"model_type": "qwen3", "revision": "b"}),
        encoding="utf-8",
    )
    config_stat = config_path.stat()
    os.utime(
        config_path,
        ns=(config_stat.st_atime_ns, config_stat.st_mtime_ns + 1_000_000_000),
    )

    second_scheduler = Scheduler(Model(), Tokenizer(), config)
    try:
        second_dir = second_scheduler.paged_cache_manager._disk_store.cache_dir
    finally:
        second_scheduler.shutdown()

    assert second_dir != first_dir
