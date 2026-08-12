"""A generic "mlx" weight_format stamp names the loader, not the codec.

The 2026-08 Nemotron bundles declare weight_format "mlx" with the actual
codec in quantization.mode (mxfp8/mxfp4). The status display showed "mlx" —
uninformative — and inferred the codec via the generic q_cfg fallback.
"""
import json

from vmlx_engine.server import _model_quantization_status


def _write_bundle(tmp_path, config):
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


def test_generic_mlx_stamp_surfaces_the_quantization_mode(tmp_path):
    path = _write_bundle(
        tmp_path,
        {
            "model_type": "nemotron_h",
            "weight_format": "mlx",
            "quantization": {"mode": "mxfp8", "group_size": 32, "bits": 8},
        },
    )
    status = _model_quantization_status(path)
    assert status["weight_format"] == "mxfp8"


def test_concrete_weight_formats_are_untouched(tmp_path):
    path = _write_bundle(
        tmp_path,
        {
            "model_type": "nemotron_h",
            "weight_format": "affine",
            "quantization": {"mode": "affine"},
        },
    )
    status = _model_quantization_status(path)
    assert status["weight_format"] == "affine"


def test_mlx_stamp_with_no_codec_stays_mlx(tmp_path):
    path = _write_bundle(
        tmp_path, {"model_type": "llama", "weight_format": "mlx"}
    )
    status = _model_quantization_status(path)
    assert status["weight_format"] == "mlx"
