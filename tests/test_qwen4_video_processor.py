"""Native video prompt geometry without loading model weights."""
from types import SimpleNamespace

import numpy as np
import pytest

from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLVideoProcessor
from vmlx_engine.models.qwen4_exp.processing import Qwen4Processor


class Tokenizer:
    vision_start_token = "<|vision_start|>"
    vision_end_token = "<|vision_end|>"

    def __call__(self, text, **kwargs):
        self.text = text
        self.kwargs = kwargs
        return {"input_ids": [[1]] * len(text)}


def processor():
    p = object.__new__(Qwen4Processor)
    p.tokenizer = Tokenizer()
    p.image_processor = SimpleNamespace(merge_size=2)
    p.video_processor = Qwen3VLVideoProcessor(min_pixels=32*32, max_pixels=32*32)
    p.video_token = "<|video_pad|>"
    p.image_token = "<|image_pad|>"
    return p


def test_timestamped_groups_keep_native_pixels_and_nonuniform_sample_times():
    p = processor()
    frames = np.arange(8*3*32*32, dtype=np.uint8).reshape(8, 3, 32, 32)
    native = p.video_processor(videos=[frames])
    out = p(text="A<|vision_start|><|video_pad|><|vision_end|>B", videos=[frames],
            video_timestamps=[[0, 1, 1.5, 2.5, 3, 4, 4.5, 5.5]], fps=[8/6])
    expected = "A" + "".join(
        f"<{t:.1f} seconds><|vision_start|><|video_pad|><|vision_end|>"
        for t in (.5, 2, 3.5, 5)
    ) + "B"
    assert p.tokenizer.text == [expected]
    assert "fps" not in p.tokenizer.kwargs
    assert "video_timestamps" not in p.tokenizer.kwargs
    np.testing.assert_array_equal(out["pixel_values_videos"], native["pixel_values_videos"])
    np.testing.assert_array_equal(out["video_grid_thw"], [[4, 2, 2]])


def test_multiple_clips_odd_frames_and_fps_fallback():
    p = processor()
    clips = [np.zeros((3, 3, 32, 32), np.uint8), np.zeros((2, 3, 32, 32), np.uint8)]
    p(text=["<|video_pad|>", "prefix <|video_pad|> suffix"], videos=clips, fps=[2, 4])
    assert "<0.2 seconds>" in p.tokenizer.text[0]
    assert "<1.0 seconds>" in p.tokenizer.text[0]
    assert "<0.1 seconds>" in p.tokenizer.text[1]
    assert p.tokenizer.text[1].endswith(" suffix")


@pytest.mark.parametrize("timestamps", [[[0]], [[1, 0]], [[0, float("nan")]], [[0, -1]], []])
def test_invalid_temporal_metadata_is_rejected(timestamps):
    p = processor()
    with pytest.raises(ValueError, match="timestamp"):
        p(text="<|video_pad|>", videos=[np.zeros((2, 3, 32, 32), np.uint8)],
          video_timestamps=timestamps)


def test_clip_placeholder_count_is_exact():
    p = processor()
    frames = np.zeros((2, 3, 32, 32), np.uint8)
    with pytest.raises(ValueError, match="no matching placeholder"):
        p(text="no video", videos=[frames])
    with pytest.raises(ValueError, match="no matching clip"):
        p(text="<|video_pad|><|video_pad|>", videos=[frames])


def test_text_only_path_keeps_base_behavior():
    p = processor()
    p(text="unchanged", add_special_tokens=False)
    assert p.tokenizer.text == ["unchanged"]
    assert p.tokenizer.kwargs["add_special_tokens"] is False
