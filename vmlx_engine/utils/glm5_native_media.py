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


# Projection arithmetic is part of checkpoint identity, even with identical
# processor pixels. Never restore media state produced before fused patch bias.
GLM5_MEDIA_KEY = "glm5_native_media_input_v2_fused_patch"
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


def glm5_media_item_keys(request, tokens, config):
    """Bind each complete native media item to its own causal token span.

    GLM videos contain timestamped image runs, so generic run/source matching
    cannot identify clips. Parse native outer delimiters and validate every
    grid/patch/placeholder count before permitting partial restoration.
    """
    from types import SimpleNamespace
    from ..cache_key import scope_cache_extra_key

    def get(name):
        return config.get(name) if isinstance(config, dict) else getattr(config, name, None)

    image_id = get("image_token_id")
    starts = {get("image_start_token_id"): "image", get("video_start_token_id"): "video"}
    ends = {"image": get("image_end_token_id"), "video": get("video_end_token_id")}
    if image_id is None or None in starts or None in ends.values() or len(starts) != 2:
        raise ValueError("GLM media delimiters unavailable")
    vision = get("vision_config")
    merge = vision.get("spatial_merge_size") if isinstance(vision, dict) else getattr(vision, "spatial_merge_size", None)
    if not isinstance(merge, int) or merge <= 0:
        raise ValueError("GLM spatial merge unavailable")
    spans = []
    cursor = 0
    while cursor < len(tokens):
        modality = starts.get(tokens[cursor])
        if modality is None:
            if tokens[cursor] == image_id:
                raise ValueError("unwrapped GLM media placeholder")
            cursor += 1
            continue
        begin = cursor
        cursor += 1
        while cursor < len(tokens) and tokens[cursor] != ends[modality]:
            if tokens[cursor] == get("video_start_token_id"):
                raise ValueError("nested GLM video")
            cursor += 1
        if cursor == len(tokens):
            raise ValueError("unclosed GLM media item")
        spans.append((modality, begin, cursor + 1))
        cursor += 1
    if not spans:
        raise ValueError("no complete GLM media items")
    payloads = {}
    for modality, (pixel_name, grid_name) in zip(("image", "video"), _PAYLOADS):
        pixels, grid = getattr(request, pixel_name, None), getattr(request, grid_name, None)
        count = sum(m == modality for m, _, _ in spans)
        if not count:
            if pixels is not None or grid is not None:
                raise ValueError("unmapped GLM processor payload")
            continue
        if not isinstance(pixels, mx.array) or pixels.ndim != 2 or not isinstance(grid, mx.array) or grid.shape != (count, 3):
            raise ValueError("GLM processor item layout mismatch")
        rows = grid.tolist()
        sizes = [int(t) * int(h) * int(w) for t, h, w in rows]
        if any(n <= 0 or n % (merge * merge) for n in sizes) or sum(sizes) != pixels.shape[0]:
            raise ValueError("GLM processor patch count mismatch")
        payloads[modality] = (pixels, grid, sizes)
    keys, items = {}, []
    indices, offsets = {"image": 0, "video": 0}, {"image": 0, "video": 0}
    for index, (modality, start, end) in enumerate(spans):
        pixels, grid, sizes = payloads[modality]
        i, offset = indices[modality], offsets[modality]
        size = sizes[i]
        if sum(t == image_id for t in tokens[start:end]) != size // (merge * merge):
            raise ValueError("GLM item placeholder count mismatch")
        pixel_name, grid_name = _PAYLOADS[modality == "video"]
        controls = {name: getattr(request, name, None) for name in _CONTROLS if name.startswith(modality + "_")}
        item = SimpleNamespace(**controls, **{pixel_name: pixels[offset:offset + size], grid_name: grid[i:i + 1]})
        key = f"glm5_native_media_item_v3_{index:04d}"
        keys[key] = glm5_media_input_key(item, tokens[start:end], {image_id})
        keys = scope_cache_extra_key(keys, key, start)
        items.append({"modality": modality, "start": start, "end": end, "rows": size})
        indices[modality] += 1
        offsets[modality] += size
    return keys, items


def trim_glm5_cached_media(request, boundary):
    """Remove only whole processor items already represented by native state."""
    items = getattr(request, "_glm_native_media_items", None)
    if not items or any(item["start"] < boundary < item["end"] for item in items):
        return False
    updates = {}
    for modality, (pixel_name, grid_name) in zip(("image", "video"), _PAYLOADS):
        covered = [item for item in items if item["modality"] == modality and item["end"] <= boundary]
        if not covered:
            continue
        pixels, grid = getattr(request, pixel_name, None), getattr(request, grid_name, None)
        rows = sum(item["rows"] for item in covered)
        if pixels is None or grid is None or rows > pixels.shape[0] or len(covered) > grid.shape[0]:
            return False
        updates[pixel_name] = pixels[rows:] if rows < pixels.shape[0] else None
        updates[grid_name] = grid[len(covered):] if len(covered) < grid.shape[0] else None
    for name, value in updates.items():
        setattr(request, name, value)
    return True
