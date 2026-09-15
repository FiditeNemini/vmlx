# SPDX-License-Identifier: Apache-2.0
"""Fail-closed identity for GLM's processed image/video checkpoint input.

This is deliberately a whole-media-set key. It cannot reuse a checkpoint
before a newly attached item, and it never treats URLs or tensor reprs as
pixel identity. The serving caller also checks the restored token boundary.
"""
from __future__ import annotations

import hashlib
import json

import mlx.core as mx
import numpy as np


GLM5_MEDIA_KEY = "glm5_native_media_input_v1"
_PAYLOADS = (
    ("pixel_values", "image_grid_thw"),
    ("video_pixel_values", "video_grid_thw"),
)
_AUDIO = ("audio", "audios", "audio_codes", "audio_embeds", "audio_features")
_CONTROLS = (
    "image_token_budget", "image_max_pixels", "image_min_pixels",
    "image_resized_height", "image_resized_width", "video_fps",
    "video_max_frames", "video_max_pixels", "video_min_pixels",
    "video_token_budget", "video_resized_height", "video_resized_width",
)


def glm5_media_input_key(request, tokens, media_ids):
    """Hash actual processor output or raise; a failed hash is never a key."""
    if any(getattr(request, name, None) is not None for name in _AUDIO):
        raise ValueError("native GLM media SSD does not support audio")
    if len(tokens) < 2 or not media_ids or not any(t in media_ids for t in tokens):
        raise ValueError("native GLM media SSD requires expanded media tokens")
    if tokens[-1] in media_ids:
        raise ValueError("native GLM checkpoint would leave unconsumed media")
    digest = hashlib.sha256(GLM5_MEDIA_KEY.encode())
    present = 0
    for pixel_name, grid_name in _PAYLOADS:
        pixels = getattr(request, pixel_name, None)
        grid = getattr(request, grid_name, None)
        if pixels is None and grid is None:
            continue
        if not isinstance(pixels, mx.array) or not isinstance(grid, mx.array):
            raise ValueError("native GLM media SSD requires pixels and grid tensors")
        if pixels.ndim < 2 or not pixels.size or grid.ndim != 2 or grid.shape[-1] != 3 or not grid.size:
            raise ValueError("native GLM media SSD received malformed processor tensors")
        present += 1
        for name, value in ((pixel_name, pixels), (grid_name, grid)):
            # Raw bytes preserve BF16, signed zero and every processor value.
            # Transfer bounded pieces instead of a second whole host payload.
            header = json.dumps([name, list(value.shape), str(value.dtype)], separators=(",", ":")).encode()
            digest.update(len(header).to_bytes(4, "little"))
            digest.update(header)
            raw = mx.contiguous(value).view(mx.uint8).reshape(-1)
            digest.update(int(raw.size).to_bytes(8, "little"))
            for start in range(0, raw.size, 1024 * 1024):
                digest.update(np.array(raw[start:start + 1024 * 1024]))
    if not present:
        raise ValueError("native GLM media SSD has no processed pixel payload")
    controls = {name: getattr(request, name, None) for name in _CONTROLS}
    # Unknown/non-JSON control values decline caching instead of using repr().
    digest.update(json.dumps(controls, sort_keys=True, allow_nan=False, separators=(",", ":")).encode())
    return digest.hexdigest()
