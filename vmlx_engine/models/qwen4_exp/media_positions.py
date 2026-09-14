"""Qwen4 multimodal positions for contiguous and frame-wrapped processors.

Pinned mlx-vlm 0.5 emits one contiguous token block per video; newer processors
wrap each temporal group separately. Inspect the actual token span, not the
processor version. Reconstruct the same plan from full history on an SSD hit.
Only integer host work occurs here; there are no per-block GPU reductions.
"""
import mlx.core as mx
import numpy as np


def media_rope_index(config, input_ids, image_grid_thw, video_grid_thw, mask):
    tokens = np.asarray(input_ids)
    batch, length = tokens.shape
    keep = np.ones_like(tokens, dtype=bool) if mask is None else np.asarray(mask) == 1
    if keep.shape != tokens.shape:
        raise ValueError("Qwen4 media position mask must match input_ids")
    images = [] if image_grid_thw is None else image_grid_thw.tolist()
    videos = [] if video_grid_thw is None else video_grid_thw.tolist()
    image_index = video_index = video_frame = 0
    merge = int(config.vision_config.spatial_merge_size)
    positions = np.ones((3, batch, length), dtype=np.int32)
    deltas = np.zeros((batch, 1), dtype=np.int32)

    def grid_shape(grid):
        t, h, w = map(int, grid)
        if min(t, h, w, merge) <= 0 or h % merge or w % merge:
            raise ValueError("Qwen4 media grid is not spatially merge-aligned")
        return t, h // merge, w // merge

    for batch_index in range(batch):
        columns = np.flatnonzero(keep[batch_index])
        row = tokens[batch_index, columns].tolist()
        cursor = base = 0
        for start in range(len(row) - 1):
            if start < cursor or row[start] != config.vision_start_token_id:
                continue
            kind = row[start + 1]
            if kind not in (config.image_token_id, config.video_token_id):
                continue
            begin = start + 1
            end = begin
            while end < len(row) and row[end] == kind:
                end += 1
            if end == len(row) or row[end] != config.vision_end_token_id:
                raise ValueError("Qwen4 media token block has no matching vision end")
            count = end - begin
            if kind == config.image_token_id:
                if image_index >= len(images):
                    raise ValueError("Qwen4 image token block has no matching grid")
                t, h, w = grid_shape(images[image_index])
                image_index += 1
                if count != t * h * w:
                    raise ValueError("Qwen4 image token count does not match grid")
            else:
                if video_index >= len(videos):
                    raise ValueError("Qwen4 video token block has no matching grid")
                total_t, h, w = grid_shape(videos[video_index])
                if count == h * w:
                    # Timestamped processors reset temporal position in each
                    # frame block; intervening timestamp tokens advance base.
                    t = 1
                elif video_frame == 0 and count == total_t * h * w:
                    t = total_t
                else:
                    raise ValueError("Qwen4 video token count does not match grid")
                video_frame += t
                if video_frame == total_t:
                    video_index += 1
                    video_frame = 0

            text_length = begin - cursor
            positions[:, batch_index, columns[cursor:begin]] = (
                np.arange(text_length, dtype=np.int32) + base
            )
            base += text_length
            positions[:, batch_index, columns[begin:end]] = (
                np.indices((t, h, w), dtype=np.int32).reshape(3, -1) + base
            )
            base += max(t, h, w)
            cursor = end

        text_length = len(row) - cursor
        positions[:, batch_index, columns[cursor:]] = (
            np.arange(text_length, dtype=np.int32) + base
        )
        deltas[batch_index, 0] = base + text_length - len(row)

    if image_index != len(images) or video_index != len(videos) or video_frame:
        raise ValueError("Qwen4 media grids do not match complete token blocks")
    return mx.array(positions), mx.array(deltas)
