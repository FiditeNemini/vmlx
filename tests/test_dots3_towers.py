# SPDX-License-Identifier: Apache-2.0
"""dots3_note tower + processor structure pins (no real weights).

Every quantity here is one a wrong port still produces a plausible answer
for: a parameter path that misses the checkpoint's strict weight map, a bias
that should not exist, a video expanded to the WRONG placeholder id, or a
placeholder count that silently disagrees with the tower's row count. Tiny
synthetic configs only — numerics against torch were proven at capture time.
"""

import math

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from vmlx_engine.models.dots3_note.audio import AudioModel  # noqa: E402
from vmlx_engine.models.dots3_note.config import (  # noqa: E402
    AudioConfig,
    VisionConfig,
)
from vmlx_engine.models.dots3_note.processor import (  # noqa: E402
    Dots3NoteFeatureExtractor,
    Dots3NoteImageProcessor,
    Dots3NoteProcessor,
    Dots3NoteVideoProcessor,
    compute_audio_token_length,
    expand_media_placeholders,
    merged_token_count,
)
from vmlx_engine.models.dots3_note.vision import VisionModel  # noqa: E402
from vmlx_engine.models.dots3_note.dots3_note import Model as Dots3Model  # noqa: E402

IMG_ID = 151660
VID_ID = 151680
AUD_ID = 151720


# ----------------------------------------------------------------- fixtures


def _tiny_vision_config() -> VisionConfig:
    return VisionConfig(
        embed_dim=32,
        intermediate_size=48,
        moe_intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        pyramid_num_routed=[-1, 4],  # block 0 dense, block 1 MoE(4)
        adapter_in_dim=32,
        adapter_out_dim=16,
    )


def _tiny_audio_config() -> AudioConfig:
    config = AudioConfig(
        d_model=32,
        encoder_attention_heads=2,
        encoder_ffn_dim=48,
        encoder_layers=2,
        num_mel_bins=8,
        adapter_in_dim=32,
        adapter_out_dim=16,
    )
    # Not a dataclass field (BaseModelConfig.from_dict drops unknown keys);
    # the tower reads it via getattr with the real default 480.
    config.downsample_hidden_size = 4
    return config


class FakeTokenizer:
    """encode() returns canned ids per prompt; unknown text gets one filler
    token per whitespace word (enough for the video overhead arithmetic)."""

    model_max_length = 524_288

    def __init__(self, prompts=None):
        self.prompts = dict(prompts or {})

    def encode(self, text, add_special_tokens=False):
        if text in self.prompts:
            return list(self.prompts[text])
        return [0] * max(1, len(text.split()))


# ------------------------------------------------------------------- vision


class TestVisionStructure:
    def test_parameter_paths_match_checkpoint_contract(self):
        model = VisionModel(_tiny_vision_config())
        params = dict(tree_flatten(model.parameters()))
        for key in (
            "patch_embed.proj.weight",
            "patch_embed.proj.bias",
            "patch_embed.norm.weight",
            "blocks.0.norm_1.weight",
            "blocks.0.norm_2.weight",
            "blocks.0.attn.qkv.weight",
            "blocks.0.attn.proj.weight",
            "blocks.0.attn.q_norm.weight",
            "blocks.0.attn.k_norm.weight",
            "blocks.0.mlp.fc1.weight",
            "blocks.0.mlp.fc2.weight",
            "blocks.0.mlp.fc3.weight",
            "blocks.1.mlp.gate_weight",
            "blocks.1.mlp.router_bias",
            "blocks.1.mlp.experts.0.fc1.weight",
            "blocks.1.mlp.experts.3.fc2.weight",
            "post_trunk_norm.weight",
            "adapter.ln_q.weight",
            "adapter.ln_q.bias",
            "adapter.mlp.0.weight",
            "adapter.mlp.0.bias",
            "adapter.mlp.2.weight",
            "adapter.mlp.2.bias",
        ):
            assert key in params, f"missing checkpoint path {key}"
        # use_bias=False: fused qkv / proj / mlp have NO bias.
        for absent in (
            "blocks.0.attn.qkv.bias",
            "blocks.0.attn.proj.bias",
            "blocks.0.mlp.fc1.bias",
            "blocks.1.mlp.experts.0.fc1.bias",
        ):
            assert absent not in params, f"unexpected bias {absent}"
        # GELU placeholder at adapter.mlp.1 must be parameterless.
        assert not any(k.startswith("adapter.mlp.1") for k in params)

    def test_router_param_dtypes(self):
        model = VisionModel(_tiny_vision_config())
        params = dict(tree_flatten(model.parameters()))
        assert params["blocks.1.mlp.gate_weight"].dtype == mx.bfloat16
        assert params["blocks.1.mlp.gate_weight"].shape == (4, 32)
        assert params["blocks.1.mlp.router_bias"].dtype == mx.float32
        assert params["blocks.1.mlp.router_bias"].shape == (4,)

    def test_qkv_shapes(self):
        model = VisionModel(_tiny_vision_config())
        params = dict(tree_flatten(model.parameters()))
        assert params["blocks.0.attn.qkv.weight"].shape == (96, 32)
        assert params["blocks.0.mlp.fc1.weight"].shape == (48, 32)
        assert params["blocks.1.mlp.experts.0.fc1.weight"].shape == (24, 32)
        # adapter: (embed * merge^2) -> same -> out
        assert params["adapter.mlp.0.weight"].shape == (128, 128)
        assert params["adapter.mlp.2.weight"].shape == (16, 128)

    def test_forward_merges_four_to_one(self):
        model = VisionModel(_tiny_vision_config())
        pixel_values = np.random.default_rng(0).normal(
            size=(16, 3 * 14 * 14)
        ).astype(np.float32)
        out = model(pixel_values, np.array([[1, 4, 4]]))
        assert out.shape == (4, 16)  # 16 patches / merge^2 -> adapter_out_dim
        assert bool(mx.all(mx.isfinite(out)))

    def test_forward_multi_media_segments(self):
        model = VisionModel(_tiny_vision_config())
        pixel_values = np.random.default_rng(1).normal(
            size=(8, 3 * 14 * 14)
        ).astype(np.float32)
        out = model(pixel_values, np.array([[1, 2, 2], [1, 2, 2]]))
        assert out.shape == (2, 16)

    def test_scatter_accepts_native_image_and_video_tokens_in_order(self):
        input_ids = mx.array([[1, IMG_ID, VID_ID, 2, VID_ID]])
        embeds = mx.zeros((1, 5, 3))
        features = mx.array(
            [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]]
        )
        out = Dots3Model._scatter_at(
            None,
            input_ids,
            embeds,
            features,
            (IMG_ID, VID_ID),
            "image/video",
        )
        assert out[0, :, 0].tolist() == [0.0, 10.0, 20.0, 0.0, 30.0]


# -------------------------------------------------------------------- audio


class TestAudioStructure:
    def test_parameter_paths_match_checkpoint_contract(self):
        model = AudioModel(_tiny_audio_config())
        params = dict(tree_flatten(model.parameters()))
        stem = "dots_encoder.speech_encoder."
        for key in (
            stem + "conv2d1.weight",
            stem + "conv2d1.bias",
            stem + "conv2d2.weight",
            stem + "conv2d2.bias",
            stem + "conv2d3.weight",
            stem + "conv2d3.bias",
            stem + "conv_out.weight",
            stem + "layers.0.self_attn.q_proj.weight",
            stem + "layers.0.self_attn.q_proj.bias",
            stem + "layers.0.self_attn.k_proj.weight",
            stem + "layers.0.self_attn.v_proj.bias",
            stem + "layers.0.self_attn.out_proj.bias",
            stem + "layers.0.self_attn_layer_norm.weight",
            stem + "layers.0.final_layer_norm.weight",
            stem + "layers.1.fc1.weight",
            stem + "layers.1.fc1.bias",
            stem + "layers.1.fc2.weight",
            stem + "layer_norm.weight",
            "audio_adapter.proj.0.weight",
            "audio_adapter.proj.0.bias",
            "audio_adapter.proj.1.weight",
            "audio_adapter.proj.1.bias",
            "audio_adapter.proj.3.weight",
            "audio_adapter.proj.3.bias",
        ):
            assert key in params, f"missing checkpoint path {key}"
        # k_proj and conv_out have NO bias in the checkpoint.
        assert stem + "layers.0.self_attn.k_proj.bias" not in params
        assert stem + "conv_out.bias" not in params
        # GELU placeholder at proj.2 must be parameterless.
        assert not any(k.startswith("audio_adapter.proj.2") for k in params)

    def test_shapes(self):
        model = AudioModel(_tiny_audio_config())
        params = dict(tree_flatten(model.parameters()))
        stem = "dots_encoder.speech_encoder."
        # MLX Conv2d layout (O, kh, kw, I); mel bins 8 -> 4 -> 2 -> 1.
        assert params[stem + "conv2d1.weight"].shape == (4, 3, 3, 1)
        assert params[stem + "conv2d2.weight"].shape == (4, 3, 3, 4)
        assert params[stem + "conv_out.weight"].shape == (32, 4 * 1)
        # fc1 fuses (gate, value): d_model -> 2 * ffn.
        assert params[stem + "layers.0.fc1.weight"].shape == (96, 32)
        assert params["audio_adapter.proj.1.weight"].shape == (16, 32)
        assert params["audio_adapter.proj.3.weight"].shape == (16, 16)

    def test_forward_chunk_arithmetic(self):
        model = AudioModel(_tiny_audio_config())
        # token stride = hop(160) * 8: 3200 samples -> 3, 2560 -> 2.
        # conv valid-length chain checks: 3200//160=20 -> 10 -> 5 -> 3.
        mel = np.random.default_rng(2).normal(size=(2, 8, 40)).astype(np.float32)
        sample_lens = np.array([3200, 2560])
        token_lens = np.array(
            [math.ceil(3200 / 1280), math.ceil(2560 / 1280)]
        )
        assert token_lens.tolist() == [3, 2]
        out, lens = model(mel, sample_lens, token_lens, np.array([2]))
        assert out.shape == (5, 16)  # both chunks concatenated: one audio
        assert lens.tolist() == [5]
        assert bool(mx.all(mx.isfinite(out)))

    def test_forward_two_audios(self):
        model = AudioModel(_tiny_audio_config())
        mel = np.random.default_rng(3).normal(size=(2, 8, 40)).astype(np.float32)
        out, lens = model(
            mel, np.array([3200, 2560]), np.array([3, 2]), np.array([1, 1])
        )
        assert out.shape == (5, 16)
        assert lens.tolist() == [3, 2]


# ---------------------------------------------------------------- processor


class TestPlaceholderExpansion:
    def test_video_expands_to_native_video_tokens(self):
        # Video expansion preserves the native placeholder ID (151680).
        ids = expand_media_placeholders(
            [1, VID_ID, 2], VID_ID, [968], emit_id=VID_ID, modality="video"
        )
        assert ids.count(VID_ID) == 968
        assert IMG_ID not in ids
        assert ids[0] == 1 and ids[-1] == 2

    def test_image_expansion_per_occurrence(self):
        ids = expand_media_placeholders([IMG_ID, 5, IMG_ID], IMG_ID, [4, 9])
        assert ids.count(IMG_ID) == 13
        assert ids[4] == 5

    def test_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="disagree"):
            expand_media_placeholders([1, IMG_ID, 2], IMG_ID, [4, 4])

    def test_zero_placeholders_with_media_raises(self):
        # Silent drop = confabulation; this must never pass quietly.
        with pytest.raises(ValueError, match="disagree"):
            expand_media_placeholders([1, 2, 3], IMG_ID, [4])

    def test_merged_token_count(self):
        assert merged_token_count((1, 4, 4), 2) == 4
        assert merged_token_count((1, 22, 22), 2) == 121


class TestImageProcessor:
    def test_minimum_image_grid(self):
        from PIL import Image

        processor = Dots3NoteImageProcessor()
        patches, grids = processor([Image.new("RGB", (56, 56), "red")])
        assert patches.shape == (16, 3 * 14 * 14)
        assert grids == [(1, 4, 4)]

    def test_full_call_expands_image_placeholder(self):
        from PIL import Image

        tokenizer = FakeTokenizer({"img": [1, IMG_ID, 2]})
        processor = Dots3NoteProcessor(tokenizer)
        out = processor(text="img", images=[Image.new("RGB", (56, 56), "blue")])
        ids = out["input_ids"][0]
        assert ids.count(IMG_ID) == 4
        assert len(ids) == 2 + 4
        assert out["pixel_values"].shape == (16, 588)
        assert out["image_grid_thw"].tolist() == [[1, 4, 4]]

    def test_media_without_placeholder_raises(self):
        from PIL import Image

        tokenizer = FakeTokenizer({"no media": [1, 2, 3]})
        processor = Dots3NoteProcessor(tokenizer)
        with pytest.raises(ValueError, match="disagree"):
            processor(text="no media", images=[Image.new("RGB", (56, 56))])


class TestVideoProcessor:
    def test_eight_frame_224_clip_is_968_image_tokens(self):
        # Frames are UPSCALED to the 128-merged-block per-frame floor:
        # 224x224 -> 308x308 -> 22x22 patches -> 121 merged tokens; x8 = 968.
        frames = np.zeros((8, 224, 224, 3), dtype=np.uint8)
        tokenizer = FakeTokenizer({"vid": [1, VID_ID, 2]})
        processor = Dots3NoteProcessor(tokenizer)
        out = processor(text="vid", videos=frames)
        ids = out["input_ids"][0]
        assert ids.count(VID_ID) == 968
        assert IMG_ID not in ids
        assert out["pixel_values"].shape == (8 * 22 * 22, 588)
        assert out["image_grid_thw"].shape == (8, 3)
        assert out["image_grid_thw"].tolist()[0] == [1, 22, 22]

    def test_video_processor_grid_directly(self):
        frames = np.zeros((8, 224, 224, 3), dtype=np.uint8)
        patches, grids = Dots3NoteVideoProcessor()(frames, tokenizer=FakeTokenizer())
        assert len(grids) == 8
        assert grids[0] == (1, 22, 22)
        assert sum(merged_token_count(g, 2) for g in grids) == 968

    def test_image_payload_without_an_image_placeholder_is_refused(self):
        """An image we have nowhere to put must still be refused.

        2026-08-17: this used to assert a BLANKET ban on video+image, which
        broke real multiturn chats — media is collected across the whole
        history, so showing a video on one turn and an image on a later turn
        legitimately sends both (see tests/test_dots3_mixed_media.py). The ban
        was replaced by an ordered merge, but the underlying safety still
        holds and is now stated precisely: this prompt carries ONLY a video
        placeholder, so there is no position for the image's features and
        guessing would scatter shifted rows.
        """
        from PIL import Image

        tokenizer = FakeTokenizer({"vid": [1, VID_ID, 2]})
        processor = Dots3NoteProcessor(tokenizer)
        with pytest.raises(ValueError, match="placeholder"):
            processor(
                text="vid",
                videos=np.zeros((4, 224, 224, 3), dtype=np.uint8),
                images=[Image.new("RGB", (56, 56))],
            )


class TestAudioFeatures:
    def test_token_length_arithmetic(self):
        # 60 s chunks, stride 1280: 61 s = 750 + ceil(16000/1280)=13.
        assert compute_audio_token_length(960_000) == 750
        assert compute_audio_token_length(976_000) == 763
        assert compute_audio_token_length(1_280) == 1
        assert compute_audio_token_length(1_281) == 2

    def test_log_mel_shapes_and_metadata(self):
        extractor = Dots3NoteFeatureExtractor()
        waveform = np.sin(
            2 * np.pi * 440.0 * np.arange(96_000) / 16_000
        ).astype(np.float32)
        feats = extractor([waveform])
        # 60 s chunk pads to 960000 samples -> 6001 stft frames, last dropped.
        assert feats["input_features"].shape == (1, 128, 6000)
        assert feats["chunk_sample_lens"].tolist() == [96_000]
        assert feats["chunk_token_lens"].tolist() == [75]
        assert feats["audio_token_lengths"].tolist() == [75]
        assert feats["audio_chunk_counts"].tolist() == [1]
        assert np.isfinite(feats["input_features"]).all()

    def test_full_call_expands_audio_placeholder(self):
        tokenizer = FakeTokenizer({"aud": [1, AUD_ID, 2]})
        processor = Dots3NoteProcessor(tokenizer)
        waveform = np.zeros(12_800, dtype=np.float32)  # 10 tokens
        waveform[0] = 1.0  # avoid an all-silence degenerate spectrum
        out = processor(text="aud", audio=[waveform])
        ids = out["input_ids"][0]
        assert ids.count(AUD_ID) == 10
        assert out["input_features"].shape[0] == 1
        assert out["chunk_token_lens"].tolist() == [10]
        assert out["audio_chunk_counts"].tolist() == [1]

    def test_audio_count_mismatch_raises(self):
        tokenizer = FakeTokenizer({"aud": [1, 2, 3]})  # no placeholder
        processor = Dots3NoteProcessor(tokenizer)
        with pytest.raises(ValueError, match="disagree"):
            processor(text="aud", audio=[np.zeros(12_800, dtype=np.float32)])
