# SPDX-License-Identifier: Apache-2.0
"""Muse Glimmer image/video preprocessing.

The shipped bundles name ``MuseGlimmerProcessor`` / ``MuseGlimmerImageProcessor``
/ ``MuseGlimmerVideoProcessor`` in ``processor_config.json``. Those classes exist
in neither transformers nor mlx-vlm, so ``AutoProcessor`` silently degrades to a
TEXT-ONLY processor: it returns ``input_ids`` and nothing else, the chat
template's single ``<|patch|>`` is never expanded, and the model answers from the
text alone while confabulating a description of an image it never received.
That failure is completely silent — no exception, no warning.

Ground truth is the working Swift port,
``vmlx-swift/Libraries/MLXVLM/Models/MuseGlimmerProcessor.swift``.

Two values the bundle declares and Swift does NOT implement are honoured here,
because Swift's omission is a bug rather than a contract: ``num_frames`` (96)
caps sampled frames, and ``max_video_frame_tokens`` (144) sizes video frames.
Without them a long clip is sized by the 4096-token IMAGE budget and blows the
context.
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Patch grids are snapped to this multiple so a merge block is never split.
_MERGE_ALIGNED = "patch_size * merge_size"


class MuseGlimmerImageProcessor:
    """Resize -> normalize -> patchify, returning ``(patches, grid_thw)``."""

    def __init__(
        self,
        patch_size: int = 14,
        merge_size: int = 2,
        temporal_patch_size: int = 2,
        max_image_tokens: int = 4096,
        image_mean: Sequence[float] = (0.5, 0.5, 0.5),
        image_std: Sequence[float] = (0.5, 0.5, 0.5),
        rescale_factor: float = 1.0 / 255.0,
        **_: Any,
    ) -> None:
        self.patch_size = int(patch_size)
        self.merge_size = int(merge_size)
        self.temporal_patch_size = int(temporal_patch_size)
        self.max_image_tokens = int(max_image_tokens)
        self.image_mean = np.asarray(image_mean, dtype=np.float32).reshape(3, 1, 1)
        self.image_std = np.asarray(image_std, dtype=np.float32).reshape(3, 1, 1)
        self.rescale_factor = float(rescale_factor)

    # ---- geometry ---------------------------------------------------------

    @property
    def factor(self) -> int:
        return self.patch_size * self.merge_size

    def smart_resize(
        self, height: int, width: int, max_tokens: Optional[int] = None
    ) -> Tuple[int, int]:
        """Snap each side to a multiple of ``patch*merge`` under a token budget.

        Both sides are snapped INDEPENDENTLY, so aspect ratio is preserved only
        approximately — that is the reference behaviour, not an oversight.
        """
        factor = self.factor
        budget = self.max_image_tokens if max_tokens is None else int(max_tokens)
        max_pixels = budget * factor * factor
        min_pixels = factor * factor

        if height < factor or width < factor:
            raise ValueError(
                f"Muse Glimmer: image {height}x{width} is smaller than one merge "
                f"block ({factor}x{factor})."
            )
        if max(height, width) // min(height, width) > 200:
            raise ValueError(
                f"Muse Glimmer: aspect ratio of {height}x{width} exceeds 200:1."
            )

        h_bar = max(factor, int(round(height / factor)) * factor)
        w_bar = max(factor, int(round(width / factor)) * factor)

        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = int(math.floor(height / beta / factor)) * factor
            w_bar = int(math.floor(width / beta / factor)) * factor
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = int(math.ceil(height * beta / factor)) * factor
            w_bar = int(math.ceil(width * beta / factor)) * factor

        h_bar = max(factor, (h_bar // factor) * factor)
        w_bar = max(factor, (w_bar // factor) * factor)
        return h_bar, w_bar

    # ---- pixels -----------------------------------------------------------

    def _to_chw(self, image, size: Tuple[int, int]) -> np.ndarray:
        """RGB -> resize -> rescale -> normalize, as ``[3, H, W]`` float32."""
        from PIL import Image

        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = image.convert("RGB").resize(
            (size[1], size[0]), resample=Image.Resampling.BICUBIC
        )
        arr = np.asarray(image, dtype=np.float32) * self.rescale_factor
        arr = arr.transpose(2, 0, 1)
        return (arr - self.image_mean) / self.image_std

    def patchify(self, frames: List[np.ndarray]) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        """``[3,H,W]`` frames -> ``(L, C*T*p*p)`` rows, merge-block-major.

        A still image is DUPLICATED along time to fill the temporal patch; a
        video consumes two real frames per patch.
        """
        temporal = self.temporal_patch_size
        if len(frames) % temporal:
            frames = list(frames) + [frames[-1]] * (temporal - len(frames) % temporal)
        stack = np.stack(frames, axis=0)  # (N, 3, H, W)

        n, channels, height, width = stack.shape
        p, m = self.patch_size, self.merge_size
        grid_t, grid_h, grid_w = n // temporal, height // p, width // p

        patches = stack.reshape(
            grid_t, temporal, channels, grid_h // m, m, p, grid_w // m, m, p
        )
        # -> (grid_t, gh/m, gw/m, m, m, C, T, p, p): each merge block contiguous
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flat = patches.reshape(
            grid_t * grid_h * grid_w, channels * temporal * p * p
        )
        return np.ascontiguousarray(flat, dtype=np.float32), (grid_t, grid_h, grid_w)

    def __call__(self, images: Sequence[Any]) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        all_patches, grids = [], []
        for image in images:
            from PIL import Image

            pil = image if isinstance(image, Image.Image) else Image.fromarray(
                np.asarray(image)
            )
            size = self.smart_resize(pil.height, pil.width)
            patches, grid = self.patchify([self._to_chw(pil, size)])
            all_patches.append(patches)
            grids.append(grid)
        if not all_patches:
            return np.zeros((0, 0), dtype=np.float32), []
        return np.concatenate(all_patches, axis=0), grids


class MuseGlimmerVideoProcessor(MuseGlimmerImageProcessor):
    """Frame sampling plus the per-frame token budget the bundle declares."""

    def __init__(
        self,
        fps: float = 2.0,
        num_frames: int = 96,
        max_video_frame_tokens: int = 144,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.fps = float(fps)
        self.num_frames = int(num_frames)
        self.max_video_frame_tokens = int(max_video_frame_tokens)

    def __call__(  # type: ignore[override]
        self, frames: Sequence[Any]
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        from PIL import Image

        frames = list(frames)[: self.num_frames]
        if not frames:
            return np.zeros((0, 0), dtype=np.float32), []

        first = frames[0]
        if not isinstance(first, Image.Image):
            first = Image.fromarray(np.asarray(first))
        # Sized once, from the first frame, under the per-FRAME budget.
        size = self.smart_resize(
            first.height, first.width, max_tokens=self.max_video_frame_tokens
        )
        chw = [self._to_chw(f, size) for f in frames]
        patches, grid = self.patchify(chw)
        return patches, [grid]


def expand_media_placeholders(
    input_ids: List[int],
    grids: Sequence[Tuple[int, int, int]],
    token_id: int,
    merge_size: int,
) -> List[int]:
    """Replace each single placeholder with one id per MERGED cell.

    The count must equal the tower's output row count. If it does not, the
    scatter that writes vision features into the text stream misaligns silently
    and the model reads shifted features as if they were real.
    """
    positions = [i for i, t in enumerate(input_ids) if t == token_id]
    if not positions:
        return list(input_ids)
    if len(positions) != len(grids):
        raise ValueError(
            f"Muse Glimmer: {len(positions)} placeholder(s) for token {token_id} but "
            f"{len(grids)} media grid(s); the prompt and the media disagree."
        )

    out: List[int] = []
    cursor = 0
    for position, (grid_t, grid_h, grid_w) in zip(positions, grids):
        out.extend(input_ids[cursor:position])
        cells = grid_t * (grid_h // merge_size) * (grid_w // merge_size)
        out.extend([token_id] * cells)
        cursor = position + 1
    out.extend(input_ids[cursor:])
    return out


def merged_token_count(grid: Tuple[int, int, int], merge_size: int) -> int:
    grid_t, grid_h, grid_w = (int(v) for v in grid)
    return grid_t * (grid_h // merge_size) * (grid_w // merge_size)
