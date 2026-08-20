"""vmlx#259: garbled Qwen3.8 output on already-converted bundles.

Qwen3.5-family checkpoints store zero-centered RMSNorm weights (effective
scale 1 + w), so sanitize adds 1.0. Bundles saved AFTER a conversion
pipeline ran sanitize are ALREADY shifted; the base family loaders applied
the shift unconditionally, roughly doubling the norm scale — fluent-looking
multilingual noise, empty visible content, mid-generation 502s.

Key naming cannot tell the cases apart (both ship language_model.model.*).
Measured means on real checkpoints, which these tests encode:
    JANG_4D-CRACK / JANG_2D  input_layernorm mean = -0.04  (needs shift)
    mlx-community 4bit       input_layernorm mean = +0.96  (already shifted)
"""

import mlx.core as mx
import pytest

from vmlx_engine.utils.zero_centered_norms import (
    qwen_norm_shard_looks_unshifted,
    zero_centered_norm_shift_needed,
)

NORM_KEYS = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)


def _shard(mean: float, key: str = "language_model.model.layers.0.input_layernorm.weight"):
    return {key: mx.full((1024,), mean, dtype=mx.bfloat16)}


class TestDetector:
    def test_zero_centered_jang_bundle_still_gets_the_shift(self):
        assert qwen_norm_shard_looks_unshifted(_shard(-0.04), NORM_KEYS) is True

    def test_already_shifted_conversion_is_left_alone(self):
        assert qwen_norm_shard_looks_unshifted(_shard(0.96), NORM_KEYS) is False

    def test_final_norm_uses_its_own_larger_threshold(self):
        # model.norm.weight carries a bigger scale than per-layer norms; a
        # value under 1.5 there still means unshifted.
        assert (
            qwen_norm_shard_looks_unshifted(
                _shard(1.1, "language_model.model.norm.weight"), NORM_KEYS
            )
            is True
        )
        assert (
            qwen_norm_shard_looks_unshifted(
                _shard(1.9, "language_model.model.norm.weight"), NORM_KEYS
            )
            is False
        )

    def test_no_readable_evidence_leaves_loads_unchanged(self):
        assert qwen_norm_shard_looks_unshifted({}, NORM_KEYS) is False
        assert qwen_norm_shard_looks_unshifted({"lm_head.weight": mx.zeros((4, 4))}, NORM_KEYS) is False

    def test_skip_is_logged_loudly(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            assert zero_centered_norm_shift_needed(_shard(0.96), NORM_KEYS, model_label="qwen3_5") is False
        assert any("already shifted" in r.getMessage() for r in caplog.records)


class TestEverySanitizeSiteIsGuarded:
    """No loader may reapply the shift unconditionally — six sites carried
    this idiom; the MTP adapter already had the detector, the base family
    loaders did not."""

    @pytest.mark.parametrize(
        "module_path",
        [
            "vmlx_engine/models/qwen3_5_family/qwen3_5/qwen3_5.py",
            "vmlx_engine/models/qwen3_5_family/qwen3_5_moe/qwen3_5_moe.py",
            "vmlx_engine/models/step3p7_mlx_vlm.py",
            "vmlx_engine/patches/mlx_vlm_mtp/qwen35_vl.py",
        ],
    )
    def test_norm_shift_is_conditional(self, module_path):
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parent.parent
        src = (root / module_path).read_text()
        for match in re.finditer(r"^.*(value \+= 1\.0|value \+ 1\.0).*$", src, re.M):
            line_no = src[: match.start()].count("\n") + 1
            window = "\n".join(src.split("\n")[max(0, line_no - 12) : line_no])
            assert (
                "apply_norm_shift" in window
                or "should_shift_norm" in window
                or "zero_centered_norm_shift_needed" in window
                or "_qwen_norm_shard_looks_unshifted" in window
            ), f"{module_path}:{line_no} applies the norm shift with no unshifted check"
