"""Timestamps must label decoded frames, including an interrupted decoder."""
import numpy as np
import pytest


@pytest.mark.parametrize("loader", ["legacy", "utils"])
@pytest.mark.parametrize("decoded_count", [3, 8])
def test_sampled_timestamps_preserve_original_indices(monkeypatch, decoded_count, loader):
    cv2 = pytest.importorskip("cv2")
    from mlx_vlm.utils import load_video
    from vmlx_engine.mllm_batch_generator import _sampled_video_timestamps

    class Capture:
        def __init__(self):
            self.reads = 0
        def isOpened(self):
            return True
        def get(self, prop):
            return 80 if prop == cv2.CAP_PROP_FRAME_COUNT else 20.0
        def set(self, prop, value):
            self.index = int(value)
        def read(self):
            self.reads += 1
            if self.reads > decoded_count:
                return False, None
            return True, np.full((28, 28, 3), self.index, dtype=np.uint8)
        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda path: Capture())
    if loader == "legacy":
        from mlx_vlm.video_generate import load_video as legacy_load
        frames, sampled_fps = legacy_load({"video": "partial.mp4", "nframes": 8})
    else:
        frames, sampled_fps = load_video("partial.mp4", nframes=8)
    observed_indices = frames[:, 0, 0, 0]
    actual = _sampled_video_timestamps("partial.mp4", len(frames), sampled_fps)
    np.testing.assert_array_equal(actual, observed_indices / 20.0)
