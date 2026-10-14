"""Processed MiniCPM crop identity must preserve earlier media in a growing chat."""
from types import SimpleNamespace

import mlx.core as mx
import pytest

from vmlx_engine.models.mllm import MLXMultimodalLM

IMAGE, VIDEO = 248056, 248057


@pytest.fixture
def plan(monkeypatch):
    import mlx_vlm.utils

    current = {}
    monkeypatch.setattr(mlx_vlm.utils, "prepare_inputs", lambda *args, **kwargs: current["inputs"])
    model = MLXMultimodalLM.__new__(MLXMultimodalLM)
    model.model_name = "MiniCPM-prefix-fixture"
    model.processor = SimpleNamespace()
    model.model = SimpleNamespace(
        config=SimpleNamespace(image_token_id=IMAGE, image_token_index=IMAGE,
                               video_token_id=VIDEO, downsample_mode="16x",
                               vision_config=SimpleNamespace(patch_size=14, num_channels=3)),
        language_model=SimpleNamespace(get_rope_index=lambda *args: None),
        get_input_embeddings=lambda *args, **kwargs: None,
    )
    model._guard_simple_image_prefill = lambda *args, **kwargs: None

    def get(ids, images=None, videos=None, sizes=None, video_sizes=None):
        inputs = {"input_ids": mx.array([ids])}
        if images is not None:
            inputs.update(pixel_values=images, target_sizes=mx.array(sizes))
        if videos is not None:
            inputs.update(pixel_values_videos=videos, target_sizes_videos=mx.array(video_sizes))
        current["inputs"] = inputs
        result = model._dspark_media_plan("prompt", ["image"] if images is not None else [], {},
                                         ["video"] if videos is not None else None)
        assert result["prompt_tokens"] == ids  # Key ids never reach the model.
        return result

    get.config = model.model.config
    return get


def _crop(value=0.0, dtype=mx.bfloat16):
    return mx.full((1, 3, 14, 256 * 14), value, dtype=dtype)


@pytest.mark.parametrize("mode", ["4x", "16x"])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_appending_image_keeps_prior_processed_media_prefix(plan, mode, dtype):
    plan.config.downsample_mode = mode
    pads = 64 if mode == "4x" else 16
    first = [1, 2] + [IMAGE] * pads + [3, 4]
    pixels = _crop(dtype=dtype)
    old = plan(first, pixels, sizes=[[16, 16]])
    new = plan(first + [IMAGE] * pads + [5], mx.concatenate([pixels, _crop(0.5, dtype)], axis=-1),
               sizes=[[16, 16], [16, 16]])
    assert new["key_tokens"][:len(first)] == old["key_tokens"]
    assert mx.array_equal(pixels, _crop(dtype=dtype))
    assert new["salt"] is None


def test_one_changed_pixel_diverges_at_its_crop_not_at_earlier_media(plan):
    ids = [1] + [IMAGE] * 16 + [2] + [IMAGE] * 16 + [3]
    pixels = mx.concatenate([_crop(), _crop()], axis=-1)
    old = plan(ids, pixels, sizes=[[16, 16], [16, 16]])
    changed = pixels.at[0, 0, 0, 256 * 14].add(1)
    new = plan(ids, changed, sizes=[[16, 16], [16, 16]])
    assert old["key_tokens"][:18] == new["key_tokens"][:18]
    assert old["key_tokens"][18:34] != new["key_tokens"][18:34]
    assert old["key_tokens"][34:] == new["key_tokens"][34:]


def test_appending_video_keeps_earlier_image_and_frame(plan):
    first = [1] + [IMAGE] * 16 + [2] + [VIDEO] * 16 + [3]
    old = plan(first, _crop(), _crop(0.25), [[16, 16]], [[16, 16]])
    new = plan(first + [VIDEO] * 16 + [4], _crop(),
               mx.concatenate([_crop(0.25), _crop(0.75)], axis=-1),
               [[16, 16]], [[16, 16], [16, 16]])
    assert new["key_tokens"][:len(first)] == old["key_tokens"]


def test_equal_pixel_bytes_different_crop_geometry_do_not_alias(plan):
    ids = [1] + [IMAGE] * 16 + [2]
    assert plan(ids, _crop(), sizes=[[16, 16]])["key_tokens"] != \
           plan(ids, _crop(), sizes=[[8, 32]])["key_tokens"]


def test_changed_first_crop_keeps_only_its_plain_text_prefix(plan):
    ids = [1, 2] + [IMAGE] * 16 + [3]
    original = plan(ids, _crop(), sizes=[[16, 16]])
    changed = plan(ids, _crop().at[0, 0, 0, 0].add(1), sizes=[[16, 16]])
    assert original["key_tokens"][:2] == changed["key_tokens"][:2] == [1, 2]
    assert original["key_tokens"][2:18] != changed["key_tokens"][2:18]


def test_short_crop_uses_legacy_identity_without_rejecting_media(plan):
    ids = [1] + [IMAGE] * 4 + [2]
    result = plan(ids, mx.zeros((1, 3, 14, 64 * 14)), sizes=[[8, 8]])
    assert result["salt"] is not None and result["prompt_tokens"] == ids


def test_unknown_supported_strip_rank_retains_legacy_identity(plan):
    # The native patch projection also supports rank3. It must remain usable
    # even when the per-crop metadata path does not attest that layout.
    ids = [1] + [IMAGE] * 16 + [2]
    result = plan(ids, _crop()[0], sizes=[[16, 16]])
    assert result["salt"] is not None and result["prompt_tokens"] == ids
