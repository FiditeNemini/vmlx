"""Processor loading for MiniCPM-V-4.7: transformers' native processor, unmodified.

``AutoProcessor`` resolves ``minicpmv4_7`` to ``MiniCPMV4_7Processor`` with the
MiniCPM-V-4.6 image/video processors (transformers >= 5.18, no remote code).
The only vMLX step is aligning ``downsample_mode`` with the bundle config: the
placeholder count the processor emits and the vision forward must use the same
mode (HF README: pass it to both ``apply_chat_template`` and ``generate``).
The bundle's ``config.json`` value is that single source of truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def bundle_downsample_mode(model_path: Any) -> Optional[str]:
    try:
        cfg = json.loads((Path(model_path) / "config.json").read_text())
    except Exception:
        return None
    mode = cfg.get("downsample_mode")
    return str(mode) if mode in ("16x", "4x") else None


def align_processor_downsample_mode(processor: Any, mode: Optional[str]) -> None:
    """Set the image/video processors (and the cached divisors) to ``mode``."""
    if mode not in ("16x", "4x"):
        return
    divisor = 4 if mode == "4x" else 16
    for attr in ("image_processor", "video_processor"):
        sub = getattr(processor, attr, None)
        if sub is not None and hasattr(sub, "downsample_mode"):
            sub.downsample_mode = mode
    if hasattr(processor, "image_token_divisor"):
        processor.image_token_divisor = divisor
    if hasattr(processor, "video_token_divisor"):
        processor.video_token_divisor = divisor


def configure_presampled_video(processor: Any) -> None:
    """vMLX decodes and samples video frames itself (SimpleEngine and batched
    engine both hand the processor frame arrays, never a file + metadata).

    * ``do_sample_frames=False``: the MiniCPM video processor otherwise
      raises without ``VideoMetadata`` (and would resample the frames).
    * ``max_slice_nums=1``: the model card's video recipe (no per-frame
      slicing); images keep the bundle's ``preprocessor_config.json`` value.
    * ``_vmlx_video_frames_presampled``: tells the batched engine not to pass
      its per-video ``fps`` list, which this processor's kwargs reject.
    """
    video = getattr(processor, "video_processor", None)
    if video is not None:
        if hasattr(video, "do_sample_frames"):
            video.do_sample_frames = False
        if hasattr(video, "max_slice_nums"):
            video.max_slice_nums = 1
    processor._vmlx_video_frames_presampled = True


class MiniCPMV47ProcessorLoader:
    """``AutoProcessor`` hook target: returns the native transformers processor."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers import MiniCPMV4_7Processor

        kwargs.pop("trust_remote_code", None)
        processor = MiniCPMV4_7Processor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        align_processor_downsample_mode(processor, bundle_downsample_mode(pretrained_model_name_or_path))
        configure_presampled_video(processor)
        return processor
