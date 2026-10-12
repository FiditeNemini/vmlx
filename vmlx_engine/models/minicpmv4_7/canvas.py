"""Canvas M-RoPE position ids for MiniCPM-V-4.7.

Exact port of ``MiniCPMV4_7Model._group_visual_frames`` / ``_frame_end_idx`` /
``get_vision_position_ids`` / ``get_rope_index`` from
``transformers/models/minicpmv4_7/modeling_minicpmv4_7.py`` (5.19.0, L796-1150).

The walk is pure host-side integer work over Python lists, kept structurally
identical to the reference so its edge semantics (including Python negative
indexing on ``input_ids[crop_start - 1]``) carry over unchanged. The only
floating-point step, ``torch.linspace(...).round()``, is replicated in float32
with torch's two-sided CPU formula and round-half-to-even (``np.round``); a
"mathematically clean" rational formula differs from torch on hundreds of
(canvas, grid) pairs.

Positions for the WHOLE prompt are computed once, before any chunked prefill;
decode positions are ``cache_len + arange + rope_delta`` on all three axes.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

Frame = Tuple[int, int, int, int, List[Tuple[int, int]]]
Group = Tuple[int, int, List[Frame]]


def merge_factor_for(downsample_mode: Optional[str]) -> int:
    """LLM tokens per patch side: 4 (16x: window merger + merger) or 2 (4x)."""
    return 2 if downsample_mode == "4x" else 4


def torch_linspace_round(end: int, steps: int) -> np.ndarray:
    """``torch.linspace(0, end, steps).round().long()`` on CPU, bit-for-bit.

    torch's CPU kernel computes ``step = (end - start) / (steps - 1)`` in the
    output dtype (float32) and fills the first half as ``start + step * i``
    and the second half as ``end - step * (steps - 1 - i)``. ``.round()`` is
    round-half-to-even, as is ``np.round``.
    """
    steps = int(steps)
    if steps <= 0:
        return np.zeros((0,), dtype=np.int64)
    if steps == 1:
        return np.zeros((1,), dtype=np.int64)
    start = np.float32(0.0)
    stop = np.float32(end)
    step = np.float32((stop - start) / np.float32(steps - 1))
    idx = np.arange(steps, dtype=np.int64)
    half = steps // 2
    out = np.empty((steps,), dtype=np.float32)
    out[:half] = start + step * idx[:half].astype(np.float32)
    out[half:] = stop - step * (steps - idx[half:] - 1).astype(np.float32)
    return np.round(out).astype(np.int64)


def vision_position_ids(
    start_position: int,
    grid_hw: Sequence[int],
    canvas_height: int = 0,
    canvas_width: int = 0,
    h_offset: int = 0,
    w_offset: int = 0,
    spatial_merge_size: int = 1,
) -> np.ndarray:
    """(3, h*w) [T, H, W] ids of one crop placed on a canvas (HF L890-946)."""
    llm_grid_h = int(grid_hw[0]) // spatial_merge_size
    llm_grid_w = int(grid_hw[1]) // spatial_merge_size
    canvas_height = canvas_height or llm_grid_h
    canvas_width = canvas_width or llm_grid_w
    h_coords = torch_linspace_round(canvas_height - 1, llm_grid_h) + h_offset
    w_coords = torch_linspace_round(canvas_width - 1, llm_grid_w) + w_offset
    h_grid, w_grid = np.meshgrid(h_coords, w_coords, indexing="ij")
    t_grid = np.zeros_like(h_grid)
    position_ids = np.stack([t_grid, h_grid, w_grid], axis=0).reshape(3, -1)
    return position_ids + int(start_position)


def frame_end_idx(
    last_crop_end_idx: int, input_ids: List[int], markers: Tuple[int, ...], limit: int
) -> int:
    """End of a frame: past every consecutive marker after its last crop (HF L796-802)."""
    end_idx = last_crop_end_idx
    while end_idx < limit and input_ids[end_idx] in markers:
        end_idx += 1
    return end_idx


def group_visual_frames(
    input_ids: List[int], mm_token_type_ids: List[int], markers: Tuple[int, ...]
) -> List[Group]:
    """Split one unpadded sequence into visual groups of frames (HF L804-888).

    ``markers`` is ``(slice_start, slice_end, image_start, image_end, newline)``
    in that order; ``<slice>`` must come first.
    """
    slice_start_id = markers[0]
    seq_len = len(input_ids)
    groups: list = []
    open_slices = None
    previous_end_idx = 0
    for modality, run in itertools.groupby(
        enumerate(mm_token_type_ids), lambda item: item[1]
    ):
        if modality == 0:
            continue
        run = list(run)
        crop = (run[0][0], run[-1][0] + 1)
        if open_slices is not None and input_ids[crop[0] - 1] == slice_start_id:
            open_slices.append(crop)
        else:
            open_slices = []
            frame = (modality, crop[0], crop[1], open_slices)
            gap = input_ids[previous_end_idx : crop[0] - 1]
            adjacent = bool(groups) and all(token_id in markers for token_id in gap)
            if adjacent:
                groups[-1][1].append(frame)
            else:
                groups.append((crop[0] - 1, [frame]))
        previous_end_idx = crop[1]

    visual_groups: List[Group] = []
    for group_start_idx, frames in groups:
        frames = [
            (
                modality,
                thumb_start_idx,
                thumb_end_idx,
                frame_end_idx(
                    slices[-1][1] if slices else thumb_end_idx,
                    input_ids,
                    markers,
                    seq_len,
                ),
                slices,
            )
            for modality, thumb_start_idx, thumb_end_idx, slices in frames
        ]
        group_end_idx = frames[-1][3]
        visual_groups.append((group_start_idx, group_end_idx, frames))
    return visual_groups


def mm_token_type_ids_from_input_ids(
    input_ids: np.ndarray, image_token_id: int, video_token_id: int
) -> np.ndarray:
    """Same labelling as ``ProcessorMixin.create_mm_token_type_ids``: 1 image pad, 2 video pad."""
    ids = np.asarray(input_ids)
    out = np.zeros(ids.shape, dtype=np.int64)
    out[ids == int(image_token_id)] = 1
    out[ids == int(video_token_id)] = 2
    return out


def _sequence_positions(
    ids: List[int],
    types: List[int],
    grid_iters: dict,
    markers: Tuple[int, ...],
    merge_factor: int,
) -> np.ndarray:
    seq_len = len(ids)
    curr = np.broadcast_to(np.arange(seq_len, dtype=np.int64), (3, seq_len)).copy()
    current_pos = 0
    current_idx = 0
    for group_start_idx, group_end_idx, frames in group_visual_frames(ids, types, markers):
        if group_start_idx > current_idx:
            text_len = group_start_idx - current_idx
            curr[:, current_idx:group_start_idx] = np.arange(text_len) + current_pos
            current_pos += text_len

        frame_cursor_idx = group_start_idx
        for modality, thumb_start_idx, thumb_end_idx, frame_end, slices in frames:
            grid_iter = grid_iters.get(modality)
            if grid_iter is None:
                raise ValueError(
                    "MiniCPM-V-4.7 canvas M-RoPE: media tokens of modality "
                    f"{modality} have no target sizes"
                )
            target_sizes_thumb = next(grid_iter)
            frame_start_idx = thumb_start_idx - 1

            if frame_start_idx > frame_cursor_idx:
                gap_len = frame_start_idx - frame_cursor_idx
                curr[:, frame_cursor_idx:frame_start_idx] = np.arange(gap_len) + current_pos
                current_pos += gap_len + 1

            frame_end = min(frame_end, group_end_idx)
            canvas_origin = current_pos
            halo_before_canvas = max(canvas_origin - 1, 0)

            llm_slice_h, llm_slice_w = 0, 0
            num_rows, num_cols = 0, 0
            canvas_height = target_sizes_thumb[0] // merge_factor
            canvas_width = target_sizes_thumb[1] // merge_factor
            target_sizes_first_slice = None
            if slices:
                num_cols = len(slices)
                for k in range(len(slices) - 1):
                    if slices[k + 1][0] - slices[k][1] > 2:
                        num_cols = k + 1
                        break
                num_rows = len(slices) // num_cols if num_cols > 0 else 1
                if num_rows * num_cols != len(slices):
                    num_rows, num_cols = 1, len(slices)
                target_sizes_first_slice = next(grid_iter)
                llm_slice_h = target_sizes_first_slice[0] // merge_factor
                llm_slice_w = target_sizes_first_slice[1] // merge_factor
                canvas_height = num_rows * llm_slice_h
                canvas_width = num_cols * llm_slice_w

            curr[:, frame_start_idx:frame_end] = canvas_origin
            curr[1:, frame_start_idx] = halo_before_canvas

            if thumb_end_idx < frame_end:
                curr[1, thumb_end_idx] = canvas_origin + canvas_height
                curr[2, thumb_end_idx] = canvas_origin + canvas_width

            curr[:, thumb_start_idx:thumb_end_idx] = vision_position_ids(
                start_position=canvas_origin,
                grid_hw=target_sizes_thumb,
                canvas_height=canvas_height,
                canvas_width=canvas_width,
                spatial_merge_size=merge_factor,
            )

            for k, (slice_start_idx, slice_end_idx) in enumerate(slices):
                h_off = (k // num_cols) * llm_slice_h
                w_off = (k % num_cols) * llm_slice_w
                target_sizes_slice = (
                    target_sizes_first_slice if k == 0 else next(grid_iter)
                )
                slice_h = target_sizes_slice[0] // merge_factor
                slice_w = target_sizes_slice[1] // merge_factor

                slice_start_marker_idx = slice_start_idx - 1
                if slice_start_marker_idx >= frame_start_idx:
                    curr[1, slice_start_marker_idx] = canvas_origin + h_off
                    curr[2, slice_start_marker_idx] = canvas_origin + w_off
                if slice_end_idx < frame_end:
                    curr[1, slice_end_idx] = canvas_origin + h_off + slice_h - 1
                    curr[2, slice_end_idx] = canvas_origin + w_off + slice_w - 1

                curr[:, slice_start_idx:slice_end_idx] = vision_position_ids(
                    start_position=canvas_origin,
                    grid_hw=target_sizes_slice,
                    h_offset=h_off,
                    w_offset=w_off,
                    spatial_merge_size=merge_factor,
                )

            for k in range(len(slices) - 1):
                gap_start_idx, gap_end_idx = slices[k][1], slices[k + 1][0]
                if gap_end_idx - gap_start_idx <= 2:
                    continue
                boundary_h = (k // num_cols + 1) * llm_slice_h - 1
                right_edge_w = num_cols * llm_slice_w
                for newline_idx in range(gap_start_idx + 1, gap_end_idx - 1):
                    curr[1, newline_idx] = canvas_origin + boundary_h
                    curr[2, newline_idx] = canvas_origin + right_edge_w

            current_pos = canvas_origin + max(canvas_height, canvas_width) + 1
            frame_cursor_idx = frame_end

        if frame_cursor_idx < group_end_idx:
            trail_len = group_end_idx - frame_cursor_idx
            curr[:, frame_cursor_idx:group_end_idx] = np.arange(trail_len) + current_pos
            current_pos += trail_len
        current_idx = group_end_idx

    if current_idx < seq_len:
        curr[:, current_idx:seq_len] = np.arange(seq_len - current_idx) + current_pos
    return curr


def _grid_rows(value) -> Optional[List[List[int]]]:
    if value is None:
        return None
    rows = np.asarray(value).reshape(-1, 2).astype(np.int64).tolist()
    return [[int(h), int(w)] for h, w in rows]


def canvas_rope_index(
    input_ids,
    mm_token_type_ids=None,
    target_sizes=None,
    target_sizes_videos=None,
    *,
    markers: Tuple[int, int, int, int, int],
    downsample_mode: Optional[str] = "16x",
    attention_mask=None,
    image_token_id: Optional[int] = None,
    video_token_id: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Canvas M-RoPE ``(3, B, S)`` position ids and ``(B, 1)`` rope deltas.

    Inputs may be numpy, MLX or torch arrays. ``mm_token_type_ids`` may be
    omitted when the pad ids are given: it is then derived exactly as the
    processor does (pad id -> 1/2). Padding (``attention_mask == 0``) keeps
    position 0, as in the reference.
    """
    ids = np.asarray(input_ids).astype(np.int64)
    if ids.ndim == 1:
        ids = ids[None, :]
    if mm_token_type_ids is None:
        if image_token_id is None or video_token_id is None:
            raise ValueError("canvas_rope_index needs mm_token_type_ids or the pad ids")
        types = mm_token_type_ids_from_input_ids(ids, image_token_id, video_token_id)
    else:
        types = np.asarray(mm_token_type_ids).astype(np.int64).reshape(ids.shape)
    mask = None if attention_mask is None else np.asarray(attention_mask).reshape(ids.shape)
    merge_factor = merge_factor_for(downsample_mode)

    image_rows = _grid_rows(target_sizes)
    video_rows = _grid_rows(target_sizes_videos)
    grid_iters = {
        1: iter(image_rows) if image_rows is not None else None,
        2: iter(video_rows) if video_rows is not None else None,
    }

    batch, length = ids.shape
    position_ids = np.zeros((3, batch, length), dtype=np.int64)
    for b in range(batch):
        if mask is not None:
            valid = mask[b].astype(bool)
        else:
            valid = np.ones((length,), dtype=bool)
        row_ids = ids[b][valid].tolist()
        row_types = types[b][valid].tolist()
        try:
            curr = _sequence_positions(row_ids, row_types, grid_iters, markers, merge_factor)
        except StopIteration as exc:
            raise ValueError(
                "MiniCPM-V-4.7 canvas M-RoPE: fewer target sizes than media crops"
            ) from exc
        position_ids[:, b, valid] = curr

    if mask is not None:
        seq_lens = mask.astype(np.int64).sum(-1)
    else:
        seq_lens = np.full((batch,), length, dtype=np.int64)
    rope_deltas = position_ids.max(axis=(0, 2))[:, None] + 1 - seq_lens[:, None]
    return position_ids, rope_deltas.astype(np.int64)


def crop_token_runs(mm_token_type_ids: Iterable[int]) -> List[Tuple[int, int, int]]:
    """Maximal non-text runs as ``(modality, start, end)`` (one per crop)."""
    runs = []
    for modality, run in itertools.groupby(enumerate(mm_token_type_ids), lambda item: item[1]):
        if modality == 0:
            continue
        run = list(run)
        runs.append((int(modality), run[0][0], run[-1][0] + 1))
    return runs
