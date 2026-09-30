"""Qwen4 video prompts with the native timestamped temporal-group layout."""

import math

from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import (
    Qwen3VLProcessor as BaseProcessor,
)


class Qwen4Processor(BaseProcessor):
    supports_video_timestamps = True

    def __call__(self, images=None, text=None, videos=None,
                 video_timestamps=None, fps=None, **kwargs):
        if videos is None:
            return super().__call__(images=images, text=text, **kwargs)

        clips = videos if isinstance(videos, list) else [videos]
        video_processor = self.video_processor or self.image_processor
        # Keep native pixel packing and temporal pairs. Only the language-side
        # layout differs from the older MLX processor's single contiguous block.
        video_inputs = video_processor(
            videos=clips, **kwargs.pop("videos_kwargs", {})
        )
        grids = video_inputs["video_grid_thw"].tolist()
        if len(grids) != len(clips):
            raise ValueError("Qwen4 video grids must match supplied clips")
        if video_timestamps is not None and len(video_timestamps) != len(clips):
            raise ValueError("Qwen4 video timestamps must match supplied clips")
        temporal = int(video_processor.temporal_patch_size)
        merge = int(video_processor.merge_size)
        start = getattr(self.tokenizer, "vision_start_token", "<|vision_start|>")
        end = getattr(self.tokenizer, "vision_end_token", "<|vision_end|>")
        wrapper = start + self.video_token + end
        rows = [text] if isinstance(text, str) else list(text)
        clip_index = 0
        for row_index, row in enumerate(rows):
            # Split before expansion: emitted video_pad tokens must not be
            # interpreted as another input clip.
            parts = row.replace(wrapper, self.video_token).split(self.video_token)
            expanded = parts[0]
            for suffix in parts[1:]:
                if clip_index >= len(clips):
                    raise ValueError("Qwen4 video placeholder has no matching clip")
                count = int(clips[clip_index].shape[0])
                t, h, w = map(int, grids[clip_index])
                if temporal <= 0 or merge <= 0 or count <= 0 or t != math.ceil(count / temporal):
                    raise ValueError("Qwen4 video temporal grid does not match frames")
                if h % merge or w % merge:
                    raise ValueError("Qwen4 video spatial grid is not merge-aligned")
                if video_timestamps is None:
                    rate = fps[clip_index] if isinstance(fps, (list, tuple)) else fps
                    rate = 2.0 if rate is None else float(rate)
                    if not math.isfinite(rate) or rate <= 0:
                        raise ValueError("Qwen4 video fps must be finite and positive")
                    times = [i / rate for i in range(count)]
                else:
                    times = [float(x) for x in video_timestamps[clip_index]]
                if len(times) != count or any(not math.isfinite(x) or x < 0 for x in times):
                    raise ValueError("Qwen4 timestamps must cover every sampled frame")
                if any(a > b for a, b in zip(times, times[1:])):
                    raise ValueError("Qwen4 frame timestamps must be ordered")
                times.extend([times[-1]] * (t * temporal - count))
                frame_tokens = self.video_token * (h * w // merge**2)
                for frame in range(t):
                    stamp = (times[frame * temporal] + times[(frame + 1) * temporal - 1]) / 2
                    expanded += f"<{stamp:.1f} seconds>" + start + frame_tokens + end
                expanded += suffix
                clip_index += 1
            rows[row_index] = expanded
        if clip_index != len(clips):
            raise ValueError("Qwen4 supplied video has no matching placeholder")

        # Let the base processor retain its image/tokenizer and tensor contracts.
        # No videos are passed here, so the already expanded blocks stay intact.
        inputs = super().__call__(images=images, text=rows, **kwargs)
        inputs.update(video_inputs)
        return inputs
