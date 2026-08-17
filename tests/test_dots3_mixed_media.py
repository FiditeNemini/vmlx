# SPDX-License-Identifier: Apache-2.0
"""dots3_note: a video turn and an image turn in the SAME conversation.

Found 2026-08-17 by the variating-multiturn matrix (reasoning on/off x tools x
video x image): turn 2 showed a video, turn 4 showed an image, and turn 4 died
with "dots3_note does not support mixing a video with separate image/audio
inputs". Media is collected across the WHOLE message history, so any chat that
shows both eventually sends both.

The old guard was not paranoia — it was protecting against a real corruption:
both branches wrote `out["pixel_values"]`, so the image branch OVERWROTE the
video's patches while `expected_image_tokens` still counted both. That is the
silent-garbage class (row 148): the scatter reads shifted features as real
ones and the model confabulates fluently.

So the fix has to merge features in the order the PLACEHOLDERS appear in the
prompt, never in the order the branches happen to run. These tests pin the
ordering, because getting it wrong produces no error at all.
"""

import numpy as np
import pytest

from vmlx_engine.models.dots3_note import processor as proc_mod


class _FakeTokenizer:
    """Encodes a tiny marker language into placeholder ids."""

    def encode(self, text, add_special_tokens=False):
        ids = []
        for token in text.split():
            if token == "VID":
                ids.append(proc_mod._VIDEO_TOKEN_ID)
            elif token == "IMG":
                ids.append(proc_mod._IMAGE_TOKEN_ID)
            else:
                ids.append(1)
        return ids


def _rows(patches):
    return int(np.asarray(patches).shape[0])


@pytest.fixture
def processor(monkeypatch):
    """A processor with the heavy pixel work stubbed to identifiable rows.

    Video rows are filled with 1.0 and image rows with 2.0, so the merged
    tensor's row VALUES reveal the concatenation order — the thing that must
    match the prompt.
    """
    p = proc_mod.Dots3NoteProcessor.__new__(proc_mod.Dots3NoteProcessor)
    p.tokenizer = _FakeTokenizer()
    p.image_token_id = proc_mod._IMAGE_TOKEN_ID
    p.video_token_id = proc_mod._VIDEO_TOKEN_ID
    p.audio_token_id = proc_mod._AUDIO_TOKEN_ID

    class _ImgProc:
        merge_size = 2

        def __call__(self, images):
            # one grid [1,2,2] per image -> 1 merged token, 4 patch rows
            grids = [[1, 2, 2] for _ in images]
            patches = np.full((4 * len(images), 3), 2.0, dtype=np.float32)
            return patches, grids

    def _vid_proc(frames, fps=None, tokenizer=None):
        grids = [[1, 2, 2] for _ in frames]
        patches = np.full((4 * len(frames), 3), 1.0, dtype=np.float32)
        return patches, grids

    p.image_processor = _ImgProc()
    p.video_processor = _vid_proc
    monkeypatch.setattr(proc_mod, "_as_frame_sequence", lambda v: list(v))
    return p


def test_video_then_image_in_one_conversation_succeeds(processor):
    """The exact multiturn shape that failed: video earlier, image later."""
    out = processor(text="a VID b IMG c", videos=["f1", "f2"], images=["i1"])
    # 2 video frames (1 token each) + 1 image (1 token) = 3 image tokens
    assert _rows(out["pixel_values"]) == 12  # (2 + 1) items x 4 patch rows
    assert out["image_grid_thw"].shape == (3, 3)


def test_features_follow_prompt_order_not_branch_order(processor):
    """VIDEO first in the prompt => video rows first in pixel_values."""
    out = processor(text="a VID b IMG c", videos=["f1"], images=["i1"])
    rows = np.asarray(out["pixel_values"])
    assert rows[0, 0] == pytest.approx(1.0), "video rows must come first"
    assert rows[-1, 0] == pytest.approx(2.0), "image rows must come last"


def test_image_before_video_reverses_the_merge(processor):
    """IMAGE first in the prompt => image rows first.

    This is the case that a branch-ordered merge silently gets wrong: the
    video branch runs first regardless, so without prompt-order sorting the
    tower would read video features where image features belong.
    """
    out = processor(text="a IMG b VID c", videos=["f1"], images=["i1"])
    rows = np.asarray(out["pixel_values"])
    assert rows[0, 0] == pytest.approx(2.0), "image rows must come first"
    assert rows[-1, 0] == pytest.approx(1.0), "video rows must come last"


def test_single_modality_paths_are_unchanged(processor):
    video_only = processor(text="a VID b", videos=["f1", "f2"])
    assert _rows(video_only["pixel_values"]) == 8
    assert np.asarray(video_only["pixel_values"])[0, 0] == pytest.approx(1.0)

    image_only = processor(text="a IMG b", images=["i1"])
    assert _rows(image_only["pixel_values"]) == 4
    assert np.asarray(image_only["pixel_values"])[0, 0] == pytest.approx(2.0)


def test_video_plus_audio_still_refused(processor):
    """Audio has its own token id and no ordering evidence — keep refusing."""
    with pytest.raises(ValueError, match="audio"):
        processor(text="a VID b", videos=["f1"], audio=["sound.wav"])


def test_both_payloads_but_missing_a_placeholder_is_refused(processor):
    """Cannot place features we have no placeholder for — refuse, never guess."""
    with pytest.raises(ValueError, match="placeholder"):
        processor(text="a VID b", videos=["f1"], images=["i1"])


def test_count_mismatch_still_raises(processor):
    """The strict placeholder/row count check must survive the merge."""
    with pytest.raises(ValueError):
        # two IMG placeholders, one image supplied
        processor(text="a IMG b IMG c", images=["i1"])
