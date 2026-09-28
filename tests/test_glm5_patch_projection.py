"""Native patch order and single-rounding GLM vision projection."""

import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm")

from vmlx_engine.models.glm5_next.config import VisionConfig
from vmlx_engine.models.glm5_next.vision import Glm5NextVisionPatchEmbed


@pytest.mark.parametrize("rows", [1, 7, 128])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_projection_matches_native_channel_temporal_order_and_bias(rows, dtype):
    if dtype == mx.float32 and os.environ.get("MLX_ENABLE_TF32") != "0":
        pytest.skip("Strict FP32 oracle requires launch-time MLX_ENABLE_TF32=0")
    rng = np.random.default_rng(1701)
    config = VisionConfig(hidden_size=16, patch_size=2, temporal_patch_size=2)
    layer = Glm5NextVisionPatchEmbed(config)
    native_weight = mx.array(rng.normal(size=(16, 3, 2, 2, 2)).astype(np.float32)).astype(dtype)
    bias = mx.array(rng.normal(size=16).astype(np.float32)).astype(dtype)
    patches = mx.array(rng.normal(size=(rows, 24)).astype(np.float32)).astype(dtype)
    # Checkpoint Conv3d is O,C,T,H,W; the loaded MLX parameter is O,T,H,W,C.
    layer.proj.weight = native_weight.transpose(0, 2, 3, 4, 1)
    layer.proj.bias = bias

    actual = layer(patches)
    mx.eval(actual)
    x = np.asarray(patches.astype(mx.float32)).astype(np.float64)
    w = np.asarray(native_weight.astype(mx.float32)).astype(np.float64)
    b = np.asarray(bias.astype(mx.float32)).astype(np.float64)
    reference = np.einsum("ik,jk->ij", x, w.reshape(16, -1), optimize=False) + b
    rounded = mx.array(reference.astype(np.float32)).astype(dtype).astype(mx.float32)
    expected = np.asarray(rounded)
    got = np.asarray(actual.astype(mx.float32))
    assert actual.dtype == dtype
    assert actual.shape == (rows, 16)
    relative_error = np.linalg.norm(got - expected) / np.linalg.norm(expected)
    assert relative_error < (2e-4 if dtype == mx.bfloat16 else 2e-5)
    # Keep serialized names/layout compatible with existing bundles.
    assert layer.proj.weight.shape == (16, 2, 2, 2, 3)
    assert set(layer.parameters()) == {"proj"}
