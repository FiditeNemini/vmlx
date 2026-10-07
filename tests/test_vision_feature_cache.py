"""Per-item SSD vision-feature cache (audit 2026-10-07)."""
from types import SimpleNamespace
import os

import mlx.core as mx
import mlx.nn as nn

import vmlx_engine.vision_feature_cache as vfc


def test_identity_changes_when_weight_shard_is_replaced(tmp_path):
    model = SimpleNamespace(config=SimpleNamespace(vision_config={"d": 1}))
    (tmp_path / "config.json").write_text('{}')
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"old weights")
    before = vfc._model_identity(model, str(tmp_path))
    stat = shard.stat()
    shard.write_bytes(b"new weights")
    os.utime(shard, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    assert vfc._model_identity(model, str(tmp_path)) != before


def test_identity_changes_with_runtime_and_handles_dict_config(tmp_path, monkeypatch):
    import vmlx_engine.prefix_cache as prefix_cache

    model = SimpleNamespace(config={"vision_config": {"d": 1}})
    monkeypatch.setattr(prefix_cache, "runtime_cache_fingerprint", lambda: "runtime-a")
    before = vfc._model_identity(model, str(tmp_path))
    monkeypatch.setattr(prefix_cache, "runtime_cache_fingerprint", lambda: "runtime-b")
    assert vfc._model_identity(model, str(tmp_path)) != before
    before = vfc._model_identity(model, str(tmp_path))
    model.config["vision_config"]["d"] = 2
    assert vfc._model_identity(model, str(tmp_path)) != before


class FakeTower(nn.Module):
    calls = 0

    def __call__(self, pixel_values, grid_thw=None):
        FakeTower.calls += 1
        return pixel_values[:, :2] * 2, None


def test_per_item_cache_encodes_only_new_items(tmp_path, monkeypatch):
    monkeypatch.setattr(vfc, "_CONFIG", {})
    vfc.configure(root=str(tmp_path), max_size_bytes=1 << 30)
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen4_exp", vision_config={"d": 1}),
                            vision_tower=FakeTower())
    assert vfc.install(model, str(tmp_path))
    assert isinstance(model.vision_tower, FakeTower)          # still the same module class family
    a, b, c = mx.ones((4, 3)), mx.full((4, 3), 2.0), mx.full((4, 3), 3.0)
    grid = mx.array([[1, 2, 2], [1, 2, 2]])
    FakeTower.calls = 0
    first, _ = model.vision_tower(mx.concatenate([a, b]), grid)
    assert FakeTower.calls == 2                                # always per item
    again, _ = model.vision_tower(mx.concatenate([a, b]), grid)
    assert FakeTower.calls == 2                                # both from SSD
    assert mx.array_equal(first, again).item()
    three, _ = model.vision_tower(mx.concatenate([a, b, c]), mx.array([[1, 2, 2]] * 3))
    assert FakeTower.calls == 3                                # only the new item encoded
    assert mx.array_equal(three[:8], first).item()


def test_not_installed_without_ssd_or_for_other_families(tmp_path, monkeypatch):
    monkeypatch.setattr(vfc, "_CONFIG", {})
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen4_exp", vision_config={}), vision_tower=FakeTower())
    assert not vfc.install(model, str(tmp_path))
    vfc.configure(root=str(tmp_path), max_size_bytes=1 << 30)
    other = SimpleNamespace(config=SimpleNamespace(model_type="gemma4", vision_config={}), vision_tower=FakeTower())
    assert not vfc.install(other, str(tmp_path))


def test_multimodal_clear_removes_features_but_ram_clear_preserves_them(tmp_path, monkeypatch):
    import asyncio
    import weakref
    from vmlx_engine import server

    monkeypatch.setattr(vfc, "_CONFIG", {})
    monkeypatch.setattr(vfc, "_STORES", weakref.WeakSet())
    monkeypatch.setattr(server, "_get_scheduler", lambda: None)
    vfc.configure(root=str(tmp_path / "ssd"), max_size_bytes=1 << 30)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen4_exp", vision_config={}), vision_tower=FakeTower())
    assert vfc.install(model, str(bundle))
    pixels, grid = mx.ones((4, 3)), mx.array([[1, 2, 2]])
    FakeTower.calls = 0
    expected, _ = model.vision_tower(pixels, grid)
    store = model.vision_tower._vmlx_feature_cache
    assert list(store.directory.glob("*.safetensors"))
    # RAM clear must never invalidate this SSD tier.
    monkeypatch.setattr(server, "_df2_enabled", lambda: False, raising=False)
    asyncio.run(server.clear_cache("ram"))
    model.vision_tower(pixels, grid)
    assert FakeTower.calls == 1
    result = asyncio.run(server.clear_cache("multimodal"))
    assert "vision_feature_disk" in result["caches"]
    assert not list(store.directory.glob("*.safetensors"))
    actual, _ = model.vision_tower(pixels, grid)
    assert FakeTower.calls == 2
    assert mx.array_equal(expected, actual).item()
