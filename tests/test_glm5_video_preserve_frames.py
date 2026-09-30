"""Opt-in sparse-video semantics, actual budgets and cache isolation."""
import numpy as np
import pytest
from vmlx_engine.models.glm5_next.processing import Glm5NextVideoProcessor
from vmlx_engine.video_controls import video_token_pixels
from test_glm5_video_processing import make_processor


def test_preservation_keeps_both_scene_cut_colors_in_separate_groups():
    frames = np.zeros((2, 28, 28, 3), dtype=np.uint8)
    frames[0, ..., 1] = 255
    frames[1, ..., 0] = 255
    p = Glm5NextVideoProcessor(temporal_mode='preserve_frames', min_image_tokens=1,
                              do_normalize=False, do_rescale=False)
    out = p([frames])
    assert out['video_grid_thw'].tolist() == [[2, 2, 2]]
    # Independent channel/temporal interpretation: every temporal slice in a
    # group must contain only that group's original color, never the next one.
    patches = out['pixel_values_videos'].reshape(2, 4, 3, 2, 14, 14)
    for t, color in enumerate(([0, 255, 0], [255, 0, 0])):
        for c, value in enumerate(color):
            np.testing.assert_array_equal(patches[t, :, c], value)
    assert video_token_pixels(p) == 28 * 28
    assert video_token_pixels(Glm5NextVideoProcessor()) == 2 * 28 * 28


@pytest.mark.parametrize('count', [1, 3, 8])
def test_preservation_budget_uses_encoded_groups(count):
    p = Glm5NextVideoProcessor(temporal_mode='preserve_frames', min_image_tokens=1,
                              max_image_tokens=count * 4)
    out = p([np.zeros((count, 256, 256, 3), dtype=np.uint8)])
    assert out['video_grid_thw'][0, 0] == count
    assert int(np.prod(out['video_grid_thw'][0])) // 4 <= count * 4


def test_unrepresentable_preservation_budget_rejected():
    p = Glm5NextVideoProcessor(temporal_mode='preserve_frames', min_image_tokens=1,
                              max_image_tokens=2)
    with pytest.raises(ValueError, match='cannot preserve'):
        p([np.zeros((3, 28, 28, 3), dtype=np.uint8)])


@pytest.mark.parametrize('explicit', [True, False])
def test_preservation_timestamps_keep_every_original_moment(explicit):
    p = make_processor()
    p.video_processor = Glm5NextVideoProcessor(temporal_mode='preserve_frames', min_image_tokens=1)
    kwargs = {'video_timestamps': [[0, 1.25, 3.5]]} if explicit else {'fps': 2}
    out = p(text='<|begin_of_video|><|video|><|end_of_video|>',
            videos=[np.zeros((3, 28, 28, 3), dtype=np.uint8)], **kwargs)
    assert out['video_grid_thw'].tolist() == [[3, 2, 2]]
    expected = ['0.0 seconds', '1.2 seconds', '3.5 seconds'] if explicit else ['0.0 seconds', '0.5 seconds', '1.0 seconds']
    for stamp in expected:
        assert stamp in p.tokenizer.text[0]
    assert np.count_nonzero(np.asarray(out['mm_token_type_ids']) == 2) == 3


def test_invalid_temporal_mode_rejected():
    with pytest.raises(ValueError, match='temporal_mode'):
        Glm5NextVideoProcessor(temporal_mode='automatic')


def test_mode_runtime_cache_namespace(monkeypatch):
    from vmlx_engine.prefix_cache import _resolve_runtime_cache_fingerprint
    monkeypatch.delenv('VMLX_GLM5_VIDEO_TEMPORAL_MODE', raising=False)
    native = _resolve_runtime_cache_fingerprint()
    monkeypatch.setenv('VMLX_GLM5_VIDEO_TEMPORAL_MODE', 'native_pairs')
    assert _resolve_runtime_cache_fingerprint() == native
    monkeypatch.setenv('VMLX_GLM5_VIDEO_TEMPORAL_MODE', 'preserve_frames')
    assert _resolve_runtime_cache_fingerprint() != native


@pytest.mark.parametrize('count,budget', [(3, 12), (8, 128), (8, 512)])
def test_request_token_budget_matches_actual_preserved_grid(count, budget):
    from types import SimpleNamespace
    from vmlx_engine.video_controls import VideoControls, clip_budget_factor
    from vmlx_engine.mllm_batch_generator import _apply_clip_pixel_budget
    vp = Glm5NextVideoProcessor(temporal_mode='preserve_frames', min_image_tokens=1)
    processor = SimpleNamespace(video_processor=vp)
    controls = VideoControls(token_budget=budget).with_processor(processor)
    assert clip_budget_factor(processor) == (28, 1)
    frames = np.zeros((count, 364, 644, 3), dtype=np.uint8)
    resized = _apply_clip_pixel_budget(frames, controls, processor, 'preserved-budget', strict=True)
    out = vp([resized])
    actual = int(np.prod(out['video_grid_thw'][0])) // 4
    assert actual <= budget
    assert out['video_grid_thw'][0, 0] == count
