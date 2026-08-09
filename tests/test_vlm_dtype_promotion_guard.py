"""Guards against the VLM fp32 dtype-promotion defect class.

Image preprocessing emits fp32 pixels. Without an explicit cast at the vision
tower entry, MLX dtype promotion runs the whole tower in fp32; without a cast
at the text-embedding merge, the fp32 image embeddings promote the merged
prompt — and the prefilled KV cache — to fp32, after which every decode token
runs the full model off the bf16 fast path (~3x per-token cost, session-wide).
"""

import inspect

import mlx.core as mx
import pytest


class TestGemma4UnifiedVisionEmbedderDtype:
    def _make_embedder(self):
        from vmlx_engine.models import gemma4_unified_register

        gemma4_unified_register.register_gemma4_unified_runtime()
        from mlx_vlm.models.gemma4_unified.config import VisionConfig
        from mlx_vlm.models.gemma4_unified.gemma4_unified import VisionEmbedder

        config = VisionConfig(
            model_patch_size=4,
            mm_embed_dim=8,
            mm_posemb_size=6,
        )
        embedder = VisionEmbedder(config)
        embedder.set_dtype(mx.bfloat16)
        return embedder

    def test_fp32_pixels_do_not_promote_output(self):
        embedder = self._make_embedder()
        patch_dim = embedder.patch_dim
        pixels = mx.random.uniform(shape=(1, 3, patch_dim)).astype(mx.float32)

        out = embedder(pixels)
        assert out.dtype == mx.bfloat16

    def test_fp32_pixels_with_position_ids_do_not_promote_output(self):
        embedder = self._make_embedder()
        patch_dim = embedder.patch_dim
        pixels = mx.random.uniform(shape=(1, 3, patch_dim)).astype(mx.float32)
        position_ids = mx.array([[[0, 1], [2, 3], [-1, -1]]])

        out = embedder(pixels, position_ids)
        assert out.dtype == mx.bfloat16

    def test_bf16_pixels_pass_through_without_extra_cast(self):
        embedder = self._make_embedder()
        patch_dim = embedder.patch_dim
        pixels = mx.random.uniform(shape=(1, 3, patch_dim)).astype(mx.bfloat16)

        out = embedder(pixels)
        assert out.dtype == mx.bfloat16


class TestStep3p7DtypeCastGuards:
    """Source-level guards: step3p7's VisionModel needs a full checkpoint to
    construct, so pin the two cast sites structurally instead."""

    def test_patch_embed_casts_pixels_to_weight_dtype(self):
        from vmlx_engine.models.step3p7_mlx_vlm import VisionModel

        src = inspect.getsource(VisionModel.patch_embed)
        assert "conv1.weight.dtype" in src
        cast_idx = src.index("astype(weight_dtype)")
        conv_idx = src.index("self.conv1(pixel_values)")
        assert cast_idx < conv_idx, "cast must run before the first conv"

    def test_merge_casts_image_embeddings_to_text_dtype(self):
        from vmlx_engine.models.step3p7_mlx_vlm import Model

        src = inspect.getsource(Model.merge_multimodal_embeddings)
        assert "flat_multimodal.astype(inputs_embeds.dtype)" in src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
