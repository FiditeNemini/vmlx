from __future__ import annotations

from dataclasses import dataclass

from vmlx_engine.cache.codebook_cache import CodebookWeightCache


@dataclass
class _SizedValue:
    nbytes: int


def test_codebook_tuple_bytes_are_bounded_in_ram_without_disk_spill(tmp_path):
    deprecated_disk_dir = tmp_path / "must-not-be-created"
    cache = CodebookWeightCache(
        config={"memory_limit_mb": 1},
        disk_cache_dir=str(deprecated_disk_dir),
        disk_max_gb=1,
        eviction_batch_size=1,
    )
    first = (_SizedValue(400_000), _SizedValue(400_000))
    second = (_SizedValue(450_000), _SizedValue(450_000))

    cache.put((1, "gate_proj"), first)
    assert cache.get((1, "gate_proj")) is first
    cache.put((2, "up_proj"), second)

    assert cache.get((1, "gate_proj")) is None
    assert cache.get((2, "up_proj")) is second
    assert cache.memory_bytes == 900_000
    assert cache.disk_bytes == 0
    assert not deprecated_disk_dir.exists()
    stats = cache.get_stats()
    assert stats["disk_persistence"] is False
    assert stats["disk_entries"] == 0
    assert stats["disk_bytes"] == 0


def test_oversized_codebook_tuple_is_used_but_not_retained():
    cache = CodebookWeightCache(config={"memory_limit_mb": 0.25})
    value = (_SizedValue(200_000), _SizedValue(200_000))

    cache.put((3, "down_proj"), value)

    assert cache.get((3, "down_proj")) is None
    assert cache.memory_bytes == 0


def test_nested_codebook_value_size_counts_every_tensor():
    value = {
        "experts": [
            (_SizedValue(11), _SizedValue(13)),
            (_SizedValue(17), _SizedValue(19)),
        ]
    }

    assert CodebookWeightCache._entry_nbytes(value) == 60


def test_zero_memory_limit_preserves_documented_unlimited_semantics():
    cache = CodebookWeightCache(config={"memory_limit_mb": 0})
    value = (_SizedValue(1024), _SizedValue(2048))

    cache.put((0, "gate_proj"), value)

    assert cache.get((0, "gate_proj")) is value
    assert cache.memory_bytes == 3072


def test_oversized_replacement_does_not_leave_stale_unaccounted_value():
    cache = CodebookWeightCache(config={"memory_limit_mb": 0.25})
    original = (_SizedValue(50_000), _SizedValue(50_000))
    replacement = (_SizedValue(200_000), _SizedValue(200_000))

    cache.put((3, "down_proj"), original)
    cache.put((3, "down_proj"), replacement)

    assert cache.get((3, "down_proj")) is None
    assert cache.memory_bytes == 0


def test_model_eviction_reloads_original_codebook_shard(tmp_path, monkeypatch):
    from vmlx_engine.models import codebook as codebook_module

    first_path = tmp_path / "codebook-layer-001-gate_proj.safetensors"
    second_path = tmp_path / "codebook-layer-002-up_proj.safetensors"
    first_path.touch()
    second_path.touch()
    first = (_SizedValue(400_000), _SizedValue(400_000))
    second = (_SizedValue(450_000), _SizedValue(450_000))
    values = {str(first_path): first, str(second_path): second}
    loads = []

    def fake_load(path):
        loads.append(path)
        codebook, indices = values[path]
        return {"codebook": codebook, "indices": indices}

    class _Config:
        def get(self, key, default=None):
            return {
                "memory.codebook_cache": {
                    "memory_limit_mb": 1,
                    "eviction_batch_size": 1,
                },
                "codebook.kernel": "mlx",
                "codebook.kernel_config": {},
            }.get(key, default)

    monkeypatch.setattr(codebook_module.mx, "load", fake_load)
    model = codebook_module.CodebookVQLanguageModel(
        model_path=tmp_path,
        base_model=object(),
        tokenizer=object(),
        jang_config={"quantization": {}},
        config_manager=_Config(),
    )

    assert model.load_codebook_layer(1, "gate_proj") == first
    assert model.load_codebook_layer(2, "up_proj") == second
    assert model.load_codebook_layer(1, "gate_proj") == first
    assert loads == [str(first_path), str(second_path), str(first_path)]
