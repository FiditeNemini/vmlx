"""Position geometry, not generated text, is the multimodal cache oracle."""
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from vmlx_engine.models.qwen4_exp.config import ModelConfig, VisionConfig
from vmlx_engine.models.qwen4_exp.language import LanguageModel


START, END, IMAGE, VIDEO = 248053, 248054, 248056, 248057


def _owner(**overrides):
    config = ModelConfig(text_config=None, vision_config=VisionConfig(), **overrides)
    return SimpleNamespace(config=config)


def _positions(rows, images=None, videos=None, mask=None, **overrides):
    return LanguageModel.get_rope_index(
        _owner(**overrides), mx.array(rows, dtype=mx.int32),
        None if images is None else mx.array(images, dtype=mx.int32),
        None if videos is None else mx.array(videos, dtype=mx.int32),
        None if mask is None else mx.array(mask, dtype=mx.int32),
    )


def test_qwen4_default_vision_delimiters_are_not_chat_delimiters():
    config = _owner().config
    assert (config.vision_start_token_id, config.vision_end_token_id) == (START, END)
    assert (config.image_token_index, config.video_token_index) == (IMAGE, VIDEO)


def test_qwen4_positions_enumerate_both_images_and_keep_suffix_origin():
    row = [10, START] + [IMAGE] * 4 + [END, 11, START] + [IMAGE] * 4 + [END, 12]
    positions, delta = _positions([row], images=[[1, 4, 4], [1, 4, 4]])
    expected = [
        [0, 1, 2, 2, 2, 2, 4, 5, 6, 7, 7, 7, 7, 9, 10],
        [0, 1, 2, 2, 3, 3, 4, 5, 6, 7, 7, 8, 8, 9, 10],
        [0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 7, 8, 9, 10],
    ]
    np.testing.assert_array_equal(np.asarray(positions)[:, 0], expected)
    assert delta.tolist() == [[-4]]
    # A longer connected prompt must not change positions of its cached prefix.
    extended, next_delta = _positions([row + [13, 14]], images=[[1, 4, 4], [1, 4, 4]])
    np.testing.assert_array_equal(np.asarray(extended)[:, :, :len(row)], positions)
    assert next_delta.tolist() == delta.tolist()
    np.testing.assert_array_equal(np.asarray(extended)[:, 0, -2:], [[11, 12]] * 3)


@pytest.mark.parametrize("frame_wrapped", [False, True])
def test_qwen4_positions_follow_actual_video_block_layout(frame_wrapped):
    block = [START] + [VIDEO] * 4 + [END]
    row = [9] + (block + [20] + block if frame_wrapped else [START] + [VIDEO] * 8 + [END]) + [30]
    positions, delta = _positions([row], videos=[[2, 4, 4]])
    assert positions.shape == (3, 1, len(row))
    assert delta.tolist() == [[-4 if frame_wrapped else -6]]
    if frame_wrapped:
        np.testing.assert_array_equal(np.asarray(positions)[:, 0, 2:6], [[2]*4, [2,2,3,3], [2,3,2,3]])
        np.testing.assert_array_equal(np.asarray(positions)[:, 0, 9:13], [[7]*4, [7,7,8,8], [7,8,7,8]])
    else:
        np.testing.assert_array_equal(np.asarray(positions)[0, 0, 2:10], [2]*4 + [3]*4)


def test_qwen4_positions_keep_image_video_grid_order_across_padded_rows():
    image = [START] + [IMAGE] * 4 + [END]
    video = [START] + [VIDEO] * 8 + [END]
    row = [9] + image + [10] + video + [11]
    rows = [[0, 0] + row, row + [0, 0], [0] * (len(row) + 2)]
    mask = [[0, 0] + [1]*len(row), [1]*len(row) + [0, 0], [0]*len(rows[0])]
    positions, delta = _positions(rows, images=[[1,4,4]] * 2, videos=[[2,4,4]] * 2, mask=mask)
    np.testing.assert_array_equal(np.asarray(positions)[:, 0, 2:], np.asarray(positions)[:, 1, :-2])
    np.testing.assert_array_equal(np.asarray(positions)[:, 2], 1)
    assert delta.tolist() == [[-8], [-8], [0]]


def test_qwen4_positions_honor_explicit_custom_bundle_ids():
    positions, delta = _positions([[10, 91, 93, 93, 93, 93, 92, 11]], images=[[1,4,4]],
                                  vision_start_token_id=91, vision_end_token_id=92,
                                  image_token_id=93, video_token_id=94)
    assert delta.tolist() == [[-2]]
    np.testing.assert_array_equal(np.asarray(positions)[1, 0], [0,1,2,2,3,3,4,5])


def test_qwen4_positions_text_only_remains_absolute():
    positions, delta = _positions([[10,11,12], [13,14,15]])
    np.testing.assert_array_equal(np.asarray(positions), [[[0,1,2], [0,1,2]]] * 3)
    assert delta.tolist() == [[0], [0]]


@pytest.mark.parametrize("row,images,videos", [
    ([START, IMAGE, END], [[1,4,4]], None),
    ([START, IMAGE, IMAGE, IMAGE, IMAGE], [[1,4,4]], None),
    ([9,10], [[1,4,4]], None),
    ([START, VIDEO, VIDEO, VIDEO, VIDEO, END], None, [[2,4,4]]),
])
def test_qwen4_positions_reject_incomplete_media_geometry(row, images, videos):
    with pytest.raises(ValueError, match="Qwen4"):
        _positions([row], images=images, videos=videos)
