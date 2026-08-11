# SPDX-License-Identifier: Apache-2.0
"""Numeric pins for the Muse Glimmer port.

Every assertion here corresponds to a divergence from Gemma that was ORIGINALLY
GOT WRONG, and each one failed the same way: the model loaded, ran, reported
healthy cache stats and emitted fluent-looking token soup. None of them crash.
That is why they are pinned numerically rather than left to live testing.

Ground truth is the working Swift port,
``vmlx-swift/Libraries/MLXLLM/Models/MuseGlimmerText.swift``.
"""

import mlx.core as mx
import numpy as np
import pytest

# The vendored package uses relative imports into mlx_vlm.models (``..base``),
# so it only resolves once installed under that namespace. Import it the same
# way the engine does rather than reaching into vmlx_engine.models directly.
from vmlx_engine.models.muse_glimmer_register import register_muse_glimmer_runtime

register_muse_glimmer_runtime()

from mlx_vlm.models.muse_glimmer.config import TextConfig  # noqa: E402
from mlx_vlm.models.muse_glimmer.language import (  # noqa: E402
    LanguageModel,
    MuseAttention,
    _scaleless_rms_norm,
)
from mlx_vlm.models.muse_glimmer.muse_glimmer import Model  # noqa: E402


def _text_config(**overrides):
    base = dict(
        model_type="muse_glimmer_text",
        hidden_size=64,
        num_hidden_layers=4,
        intermediate_size=128,
        num_attention_heads=4,
        head_dim=16,
        num_key_value_heads=2,
        vocab_size=128,
        rms_norm_eps=1e-5,
        post_norm_eps=1e-8,
        sliding_window=8,
        qk_scale_factor=3.87,
        output_multiplier=0.19611613513818404,
        final_logit_softcapping=20.0,
    )
    base.update(overrides)
    return TextConfig(**{k: v for k, v in base.items() if hasattr(TextConfig, k) or True})


class TestCenteredNormFold:
    """The checkpoint stores zero-centered RMSNorm gains: the real weight is 1+w.

    Without the fold every norm runs at gain ~0 instead of ~1, crushing each
    sublayer's contribution to the residual stream.
    """

    def test_all_four_layer_norms_and_final_norm_are_shifted(self):
        weights = {
            "language_model.model.layers.0.input_layernorm.weight": mx.zeros((4,)),
            "language_model.model.layers.0.post_attention_layernorm.weight": mx.zeros((4,)),
            "language_model.model.layers.0.pre_feedforward_layernorm.weight": mx.zeros((4,)),
            "language_model.model.layers.0.post_feedforward_layernorm.weight": mx.zeros((4,)),
            "language_model.model.norm.weight": mx.zeros((4,)),
        }
        out = Model.sanitize(Model, dict(weights))
        for key in weights:
            assert float(mx.mean(out[key]).item()) == pytest.approx(1.0), key

    def test_vision_layernorms_are_not_shifted(self):
        """The vision tower carries ordinary LayerNorms — shifting them is a bug."""
        weights = {
            "model.vision_tower.layers.0.norm1.weight": mx.zeros((4,)),
            "model.vision_tower.layers.0.norm2.weight": mx.zeros((4,)),
        }
        out = Model.sanitize(Model, dict(weights))
        for key in out:
            assert float(mx.mean(out[key]).item()) == pytest.approx(0.0), key

    def test_attention_and_mlp_weights_are_not_shifted(self):
        weights = {
            "language_model.model.layers.0.self_attn.q_proj.weight": mx.zeros((4, 4)),
            "language_model.model.layers.0.mlp.gate_proj.weight": mx.zeros((4, 4)),
        }
        out = Model.sanitize(Model, dict(weights))
        for key in out:
            assert float(mx.mean(out[key]).item()) == pytest.approx(0.0), key


class TestQKNorm:
    """Q and K get a WEIGHTLESS RMSNorm over head_dim; qk_scale_factor then
    multiplies Q only. The checkpoint has no q_norm/k_norm tensors because the
    norm has no learned gain — the absent tensor is not an absent operation.
    """

    def test_scaleless_norm_has_unit_rms(self):
        x = mx.array(np.random.RandomState(0).randn(2, 3, 16).astype(np.float32))
        out = _scaleless_rms_norm(x, 1e-5)
        rms = np.sqrt(np.mean(np.array(out) ** 2, axis=-1))
        assert np.allclose(rms, 1.0, atol=1e-3)

    def test_sdpa_scale_is_inverse_sqrt_head_dim_not_the_factor(self):
        config = _text_config()
        attn = MuseAttention(config, layer_idx=0)
        assert attn.scale == pytest.approx(config.head_dim**-0.5)
        assert attn.qk_scale_factor == pytest.approx(3.87)

    def test_queries_are_scaled_but_keys_are_not(self):
        """Asymmetry is the point: a symmetric scale would be absorbable into
        the SDPA scalar, which is why substituting constants never helped."""
        config = _text_config()
        attn = MuseAttention(config, layer_idx=0)
        captured = {}

        def fake_sdpa(q, k, v, scale, mask=None):
            captured["q_rms"] = float(mx.mean(mx.sqrt(mx.mean(q * q, axis=-1))).item())
            captured["k_rms"] = float(mx.mean(mx.sqrt(mx.mean(k * k, axis=-1))).item())
            captured["scale"] = scale
            return mx.zeros_like(q)

        original = mx.fast.scaled_dot_product_attention
        mx.fast.scaled_dot_product_attention = fake_sdpa
        try:
            attn(mx.zeros((1, 4, config.hidden_size)) + 0.1)
        finally:
            mx.fast.scaled_dot_product_attention = original

        assert captured["k_rms"] == pytest.approx(1.0, abs=1e-2)
        assert captured["q_rms"] == pytest.approx(3.87, abs=5e-2)
        assert captured["scale"] == pytest.approx(config.head_dim**-0.5)


class TestNormedEmbedding:
    """Muse has NO sqrt(hidden_size) embedding scale. A weightless RMSNorm over
    the raw lookup stands in its place; skipping it feeds layer 0 the wrong
    magnitude."""

    def test_embed_output_has_unit_rms(self):
        model = LanguageModel(_text_config())
        out = model.model.embed(mx.array([[1, 2, 3, 4]]))
        rms = np.sqrt(np.mean(np.array(out) ** 2, axis=-1))
        assert np.allclose(rms, 1.0, atol=1e-2)

    def test_embed_differs_from_raw_lookup(self):
        model = LanguageModel(_text_config())
        ids = mx.array([[1, 2, 3, 4]])
        assert not np.allclose(
            np.array(model.model.embed(ids)),
            np.array(model.model.embed_tokens(ids)),
        )


class TestLogitTail:
    def test_softcap_bounds_logits(self):
        model = LanguageModel(_text_config())
        logits = model(inputs=mx.array([[1, 2, 3]]))
        cap = 20.0
        assert float(mx.max(mx.abs(logits)).item()) <= cap


class TestMixedSlidingMasks:
    """Sliding and full layers cannot share one mask: reusing the full mask on
    the sliding layers lets them see past the window, and that only diverges
    once the context outgrows it."""

    def test_sliding_and_full_layers_get_different_masks(self):
        config = _text_config(
            num_hidden_layers=4,
            layer_types=["sliding_attention", "full_attention"] * 2,
            sliding_window=2,
        )
        model = LanguageModel(config)
        cache = model.make_cache()
        h = mx.zeros((1, 8, config.hidden_size))
        masks = model.model._make_masks(h, cache)

        sliding = [m for i, m in enumerate(masks) if config.layer_is_sliding(i)]
        full = [m for i, m in enumerate(masks) if not config.layer_is_sliding(i)]
        assert sliding and full
        # Same object reused within a kind, distinct between kinds.
        assert all(m is sliding[0] for m in sliding)
        assert all(m is full[0] for m in full)
        assert sliding[0] is not full[0]

    def test_make_cache_matches_layer_types(self):
        from mlx_lm.models.cache import KVCache, RotatingKVCache

        config = _text_config(
            num_hidden_layers=4,
            layer_types=["sliding_attention", "full_attention"] * 2,
            sliding_window=8,
        )
        cache = LanguageModel(config).make_cache()
        for index, slot in enumerate(cache):
            expected = RotatingKVCache if config.layer_is_sliding(index) else KVCache
            assert isinstance(slot, expected), index
            if expected is RotatingKVCache:
                # step must cover any prefill chunk the loop can emit.
                assert slot.step >= config.sliding_window
