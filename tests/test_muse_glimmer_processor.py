# SPDX-License-Identifier: Apache-2.0
"""Geometry pins for Muse Glimmer image/video preprocessing.

Every quantity here is one a wrong port still produces a plausible answer for.
A scrambled patch order, a transposed position lookup or an off-by-one merged
cell count all yield a fluent caption of the wrong picture, so the numbers are
pinned rather than eyeballed.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw  # noqa: E402

_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "vmlx_engine/models/muse_glimmer/processor.py"
)
_spec = importlib.util.spec_from_file_location("_muse_processor", _PATH)
proc_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(proc_mod)

MuseGlimmerImageProcessor = proc_mod.MuseGlimmerImageProcessor
MuseGlimmerVideoProcessor = proc_mod.MuseGlimmerVideoProcessor
expand_media_placeholders = proc_mod.expand_media_placeholders
merged_token_count = proc_mod.merged_token_count

IMAGE_TOKEN = 200092
VIDEO_TOKEN = 200091


def _image(size=(448, 448)):
    img = Image.new("RGB", size, "white")
    inset_x, inset_y = max(1, size[0] // 8), max(1, size[1] // 8)
    ImageDraw.Draw(img).ellipse(
        [inset_x, inset_y, size[0] - inset_x, size[1] - inset_y], fill=(220, 20, 20)
    )
    return img


@pytest.fixture
def processor():
    return MuseGlimmerImageProcessor()


class TestSmartResize:
    def test_native_grid_is_untouched(self, processor):
        # 448 = 32 patches of 14 = exactly the learned position table's side.
        assert processor.smart_resize(448, 448) == (448, 448)

    def test_sides_snap_to_a_whole_merge_block(self, processor):
        h, w = processor.smart_resize(1000, 600)
        assert h % 28 == 0 and w % 28 == 0

    def test_large_image_is_clamped_under_the_token_budget(self, processor):
        h, w = processor.smart_resize(4000, 3000)
        merged = (h // 14 // 2) * (w // 14 // 2)
        assert merged <= processor.max_image_tokens
        assert h % 28 == 0 and w % 28 == 0

    def test_tiny_image_is_lifted_to_one_block(self, processor):
        assert processor.smart_resize(37, 37) == (28, 28)

    def test_below_one_block_is_rejected(self, processor):
        with pytest.raises(ValueError, match="smaller than one merge block"):
            processor.smart_resize(20, 20)

    def test_absurd_aspect_ratio_is_rejected(self, processor):
        with pytest.raises(ValueError, match="aspect ratio"):
            processor.smart_resize(28, 28 * 300)


class TestPatchify:
    def test_native_image_shape_and_grid(self, processor):
        patches, grids = processor([_image()])
        # 3 channels * 2 temporal * 14 * 14 = 1176
        assert patches.shape == (1024, 1176)
        assert grids == [(1, 32, 32)]

    def test_still_image_is_duplicated_along_time(self, processor):
        """temporal_patch_size is 2; a still image fills both slots itself."""
        chw = processor._to_chw(_image((28, 28)), (28, 28))
        patches, grid = processor.patchify([chw])
        assert grid[0] == 1
        spatial = 14 * 14
        row = patches[0].reshape(3, 2, spatial)
        assert np.allclose(row[:, 0, :], row[:, 1, :])

    def test_merged_cell_count(self, processor):
        _, grids = processor([_image()])
        assert merged_token_count(grids[0], 2) == 256

    def test_normalization_lands_in_minus_one_to_one(self, processor):
        chw = processor._to_chw(_image((28, 28)), (28, 28))
        assert chw.min() >= -1.0001 and chw.max() <= 1.0001
        white = processor._to_chw(Image.new("RGB", (28, 28), "white"), (28, 28))
        assert np.allclose(white, 1.0, atol=1e-3)
        black = processor._to_chw(Image.new("RGB", (28, 28), "black"), (28, 28))
        assert np.allclose(black, -1.0, atol=1e-3)


class TestPlaceholderExpansion:
    def test_one_placeholder_becomes_one_id_per_merged_cell(self, processor):
        _, grids = processor([_image()])
        ids = [1, 2, IMAGE_TOKEN, 3]
        out = expand_media_placeholders(ids, grids, IMAGE_TOKEN, 2)
        assert out.count(IMAGE_TOKEN) == 256
        assert len(out) == len(ids) - 1 + 256
        # surrounding text is preserved in order, with no sentinel tokens
        assert out[:2] == [1, 2] and out[-1] == 3

    def test_absent_placeholder_is_a_no_op(self):
        assert expand_media_placeholders([1, 2, 3], [(1, 4, 4)], IMAGE_TOKEN, 2) == [1, 2, 3]

    def test_count_mismatch_raises_rather_than_misaligning(self):
        """A silent mismatch shifts every vision feature in the scatter."""
        with pytest.raises(ValueError, match="disagree"):
            expand_media_placeholders(
                [IMAGE_TOKEN], [(1, 4, 4), (1, 4, 4)], IMAGE_TOKEN, 2
            )


class TestVideo:
    def test_frames_pair_into_temporal_patches(self):
        video = MuseGlimmerVideoProcessor()
        patches, grids = video([_image((224, 224))] * 5)
        # 5 frames padded to 6 -> grid_t 3
        assert grids[0][0] == 3
        assert patches.shape[1] == 1176

    def test_frame_budget_is_smaller_than_the_image_budget(self):
        """Sizing video by the 4096-token image budget blows the context."""
        video = MuseGlimmerVideoProcessor()
        _, grids = video([_image((896, 896))] * 2)
        grid_t, grid_h, grid_w = grids[0]
        assert (grid_h // 2) * (grid_w // 2) <= video.max_video_frame_tokens

    def test_frame_count_is_capped(self):
        video = MuseGlimmerVideoProcessor(num_frames=4)
        _, grids = video([_image((28, 28))] * 50)
        assert grids[0][0] == 2  # 4 frames -> 2 temporal patches
