"""SimpleEngine plans video frames like the batched frame fallback and never refuses on the image limit.

Audit 2026-10-07 (ISSUES I-24/I-33): a 20 s clip at the default 2 fps (40 frames) returned HTTP 400 "Total image count
(40, including video frames) exceeds limit of 20" on 27B (SimpleEngine) -- every pixel/budget control was unusable for
clips over 10 s -- while the batched engine planned the same clip to the per-video cap. SimpleEngine also kept frames at
full resolution (~882 tokens/frame vs ~292 batched).
"""
import inspect

import numpy as np

import vmlx_engine.models.mllm as mllm
from vmlx_engine.models.mllm import MLXMultimodalLM, _simple_video_frame_cap
from vmlx_engine.video_controls import VideoControls


def _lm(monkeypatch, n_frames=40, h=720, w=1280):
    lm = MLXMultimodalLM.__new__(MLXMultimodalLM)
    lm.processor = None
    frames = [np.full((h, w, 3), i, dtype=np.uint8) for i in range(n_frames)]
    monkeypatch.setattr(mllm, "process_video_input", lambda v: v)
    monkeypatch.setattr(mllm, "extract_video_frames_smart", lambda path, fps, max_frames: list(frames))
    monkeypatch.setattr(mllm, "save_frames_to_temp", lambda fr: fr)
    return lm


def test_frame_cap_splits_the_image_limit_across_videos():
    assert _simple_video_frame_cap({}, 1, 0) == 20
    assert _simple_video_frame_cap({}, 2, 2) == 9
    assert _simple_video_frame_cap({"max_images_per_request": 64}, 1, 0) == 64
    assert _simple_video_frame_cap({}, 1, 30) == 1


def test_default_clip_is_planned_to_the_cap_and_bounded_not_refused(monkeypatch):
    lm = _lm(monkeypatch)
    out = lm._prepare_video("clip.mp4", fps=2.0, max_frames=128, controls=None, frame_cap=20)
    assert len(out) == 20
    assert int(out[0][0, 0, 0]) == 0 and int(out[-1][0, 0, 0]) == 39  # evenly spread, first and last kept
    assert max(out[0].shape[:2]) <= 768  # batched default long edge


def test_explicit_max_frames_below_the_cap_is_honoured(monkeypatch):
    lm = _lm(monkeypatch)
    out = lm._prepare_video("clip.mp4", fps=2.0, max_frames=4, controls=VideoControls(max_frames=4), frame_cap=20)
    assert len(out) == 4


def test_explicit_resize_wins(monkeypatch):
    lm = _lm(monkeypatch)
    ctl = VideoControls(resized_height=224, resized_width=224)
    out = lm._prepare_video("clip.mp4", fps=2.0, max_frames=128, controls=ctl, frame_cap=20)
    assert out[0].shape[:2] == (224, 224)


def test_image_limit_is_advisory_on_every_simple_engine_path():
    src = inspect.getsource(MLXMultimodalLM)
    assert "including video frames) \"\n" not in src
    assert src.count("above the advisory") == 4
    assert "exceeds limit of" not in src
