"""Per-item SSD vision-feature cache (audit 2026-10-07)."""
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

import vmlx_engine.vision_feature_cache as vfc


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
