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

from vmlx_engine.models.muse_glimmer_register import register_muse_glimmer_runtime  # noqa: E402

register_muse_glimmer_runtime()

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


class TestVisionAdapterContract:
    """Two structural rules that produce valid shapes and fluent nonsense.

    Both were live defects: the missing trailing activation put projected
    features at ~21x the text stream and made them unreadable, and the layer
    typing silently fell through to a cadence that mis-typed the final layer.
    """

    @staticmethod
    def _vision_config(layer_types=None):
        from mlx_vlm.models.muse_glimmer.config import VisionConfig

        return VisionConfig(
            model_type="muse_glimmer_vision", hidden_size=64, intermediate_size=128,
            num_hidden_layers=50, num_attention_heads=4, patch_size=14,
            patch_temporal=2, merge_size=2, pos_emb_height=4, pos_emb_width=4,
            max_position_embeddings=1024, layer_norm_eps=1e-5, hidden_act="gelu",
            rope_theta=10000.0, layer_types=layer_types or [],
        )

    def test_layer_typing_reads_the_checkpoint_not_the_cadence(self):
        """The shipped list breaks the x4 period at the tail: full at 47 THEN 49.

        A `(index+1) % 4` fallback yields 12 full layers and runs layer 49
        windowed. Inert at 448x448 (single window) and wrong for anything
        larger or any video.
        """
        from mlx_vlm.models.muse_glimmer.vision import VisionModel

        types = ["window_attention"] * 50
        for i in (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 49):
            types[i] = "full_attention"
        model = VisionModel(self._vision_config(types))

        full = [i for i in range(50) if not model._layer_is_windowed(i)]
        assert full == [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 49]
        assert 49 in full, "final layer must be full attention, not windowed"

    def test_adapter_activates_after_both_layers(self):
        """vision_projection was trained on the one-sided post-gelu output.

        Feeding it the raw symmetric pre-activation pushes the negative half —
        which training crushed to ~0 — through the projection at full
        magnitude, inflating the output and pointing each token vector
        somewhere the weights never saw.
        """
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_vlm.models.muse_glimmer.config import ModelConfig, TextConfig
        from mlx_vlm.models.muse_glimmer.vision import MuseVisionAdapter

        vision = self._vision_config()
        model_config = ModelConfig(
            model_type="muse_glimmer",
            text_config=TextConfig(model_type="muse_glimmer_text", hidden_size=64),
            vision_config=vision,
            projector_hidden_size=32,
        )
        adapter = MuseVisionAdapter(model_config)
        feats = mx.ones((4, vision.hidden_size)) * -5.0  # strongly negative
        out = adapter(feats, grid_thw=[(1, 2, 2)])

        # gelu_approx has a small negative lobe (~-0.17 min) but cannot emit
        # large negatives. A missing trailing activation would pass them through.
        assert float(mx.min(out)) > -1.0, "output looks pre-activation"

    def test_adapter_without_grid_raises_rather_than_guessing(self):
        import mlx.core as mx
        from mlx_vlm.models.muse_glimmer.config import ModelConfig, TextConfig
        from mlx_vlm.models.muse_glimmer.vision import MuseVisionAdapter

        vision = self._vision_config()
        adapter = MuseVisionAdapter(
            ModelConfig(
                model_type="muse_glimmer",
                text_config=TextConfig(model_type="muse_glimmer_text", hidden_size=64),
                vision_config=vision,
                projector_hidden_size=32,
            )
        )
        with pytest.raises(ValueError, match="grid_thw"):
            adapter(mx.ones((4, vision.hidden_size)))


class TestMultiMediaPrompts:
    """Two paths the panel advertised that could not actually run.

    Both failed structurally, not subtly: the tower and adapter consumed only
    ``grid_thw[0]``, so a second image got the FIRST image's geometry and a
    truncated gather; and the processor's ``if videos: ... elif images:``
    silently discarded every image in a mixed prompt while expanding only the
    video, leaving the placeholder count and the feature rows disagreeing.
    """

    IMAGE_TOKEN = 200092
    VIDEO_TOKEN = 200091

    def test_processor_keeps_both_images_and_video(self):
        proc = MuseGlimmerImageProcessor()
        video = MuseGlimmerVideoProcessor()
        images = [_image((224, 224)), _image((112, 112))]
        img_patches, img_grids = proc(images)
        vid_patches, vid_grids = video([_image((112, 112))] * 4)

        assert len(img_grids) == 2, "second image dropped"
        assert img_patches.shape[0] == sum(t * h * w for t, h, w in img_grids)
        assert vid_grids and vid_patches.shape[1] == img_patches.shape[1]

    def test_expansion_covers_every_image_independently(self):
        proc = MuseGlimmerImageProcessor()
        _, grids = proc([_image((224, 224)), _image((112, 112))])
        ids = [1, self.IMAGE_TOKEN, 2, self.IMAGE_TOKEN, 3]
        out = expand_media_placeholders(ids, grids, self.IMAGE_TOKEN, 2)
        expected = sum(merged_token_count(g, 2) for g in grids)
        assert out.count(self.IMAGE_TOKEN) == expected
        # each placeholder expands to ITS OWN grid, not the first one twice
        assert merged_token_count(grids[0], 2) != merged_token_count(grids[1], 2)

    def test_tower_and_adapter_encode_each_grid_separately(self):
        import mlx.core as mx
        import numpy as np
        from mlx_vlm.models.muse_glimmer.config import ModelConfig, TextConfig, VisionConfig
        from mlx_vlm.models.muse_glimmer.vision import MuseVisionAdapter, VisionModel

        vision = VisionConfig(
            model_type="muse_glimmer_vision", hidden_size=64, intermediate_size=128,
            num_hidden_layers=4, num_attention_heads=4, patch_size=14,
            patch_temporal=2, merge_size=2, pos_emb_height=4, pos_emb_width=4,
            max_position_embeddings=1024, layer_norm_eps=1e-5, hidden_act="gelu",
            rope_theta=10000.0,
            layer_types=["window_attention"] * 3 + ["full_attention"],
        )
        grids = [(1, 8, 8), (1, 4, 4)]
        total = sum(t * h * w for t, h, w in grids)
        patches = mx.array(
            np.random.RandomState(0).randn(total, 3 * 2 * 14 * 14).astype(np.float32)
        )

        tower = VisionModel(vision)
        feats = tower(patches, grid_thw=grids)
        assert feats.shape == (total, vision.hidden_size)

        adapter = MuseVisionAdapter(
            ModelConfig(
                model_type="muse_glimmer",
                text_config=TextConfig(model_type="muse_glimmer_text", hidden_size=64),
                vision_config=vision,
                projector_hidden_size=32,
            )
        )
        merged = adapter(feats, grid_thw=grids)
        assert merged.shape[0] == sum(
            t * (h // 2) * (w // 2) for t, h, w in grids
        ), "adapter merged under one grid instead of each"

    def test_batched_scatter_does_not_reuse_row_zero_features(self):
        """Per-row cumsum restarts at 0, so B>1 re-read row 0's features."""
        import mlx.core as mx

        from mlx_vlm.models.muse_glimmer.muse_glimmer import Model

        ids = mx.array([[1, self.IMAGE_TOKEN, 2], [3, self.IMAGE_TOKEN, 4]])
        embeds = mx.zeros((2, 3, 4))
        # two distinct feature rows, one per batch row
        features = mx.array([[1.0, 1, 1, 1], [2.0, 2, 2, 2]])

        model = Model.__new__(Model)
        model.config = type("C", (), {"image_token_id": self.IMAGE_TOKEN,
                                      "video_token_id": self.VIDEO_TOKEN})()
        out = Model._scatter_media(model, ids, embeds, features)
        assert float(out[0, 1, 0]) == 1.0
        assert float(out[1, 1, 0]) == 2.0, "row 1 re-read row 0's features"
