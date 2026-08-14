"""D-series capability-schema stamps must NOT route to the JANG codec loader.

Qwen3.6-27B JANG_{2D,4D,6D}/MXFP8 bundles carry a v3 jang_config that is pure
serving metadata (structured `chat` sampling presets, `capabilities`, `mtp`,
`reasoning`, `tools`, `vision`) with NO format/weight_format keys. Their
weights are stock MLX affine with per-module overrides in
config.json["quantization"] — stock mlx_vlm loads them with no custom code.

`is_jang_model`'s legacy fallback treated ANY stamp lacking format keys as a
legacy JANG bundle, so the codec loader was invoked and refused:
"Not a JANG VLM: format='None' weight_format='None' ... Application startup
failed." Every stamped D-series bundle was unservable, found live serving the
first one.
"""

import json

from vmlx_engine.utils.jang_loader import is_jang_model


def _write(tmp_path, cfg):
    (tmp_path / "jang_config.json").write_text(json.dumps(cfg))
    return str(tmp_path)


def test_v3_capability_stamp_is_not_jang(tmp_path):
    # The D-series shape: serving metadata only, no codec markers.
    cfg = {
        "capabilities": {"vision": True, "video": True},
        "chat": {"sampling_modes": {"thinking_general": {"temperature": 1.0}}},
        "mtp": {"tensors": 31},
        "quantization": {"imatrix_refit": {"modules": 580}},
        "reasoning": {"default": True, "parser": "qwen3"},
        "tools": {"parser": "qwen3_coder"},
        "vision": {"processor": "qwen"},
    }
    assert is_jang_model(_write(tmp_path, cfg)) is False


def test_legacy_empty_stamp_still_routes_to_jang(tmp_path):
    # Legacy JANG/JJQF bundles with an otherwise-empty stamp predate the
    # structured chat schema and must KEEP routing to the codec loader.
    assert is_jang_model(_write(tmp_path, {})) is True


def test_explicit_codec_format_still_wins(tmp_path):
    # A real JANG bundle that also grew a chat block must stay on the codec
    # path — format markers outrank the capability-schema discriminator.
    cfg = {"format": "jang", "chat": {"sampling_modes": {}}}
    assert is_jang_model(_write(tmp_path, cfg)) is True


def test_capability_only_mlx_stamp_still_not_jang(tmp_path):
    # The pre-existing case the docstring documents (Nemotron-Omni MXFP4):
    # explicit stock weight_format.
    assert is_jang_model(_write(tmp_path, {"weight_format": "mlx"})) is False
