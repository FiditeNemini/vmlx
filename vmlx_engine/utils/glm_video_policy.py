"""Shared GLM video serving policy and cache identity selection."""
import os


def glm_video_temporal_mode() -> str:
    """Preserve sampled moments by default; allow explicit checkpoint packing."""
    mode = os.environ.get("VMLX_GLM5_VIDEO_TEMPORAL_MODE") or "preserve_frames"
    if mode not in ("native_pairs", "preserve_frames"):
        raise ValueError("VMLX_GLM5_VIDEO_TEMPORAL_MODE must be native_pairs or preserve_frames")
    return mode
