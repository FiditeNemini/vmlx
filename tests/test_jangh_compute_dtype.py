"""Compute casts must leave JANGH's typed FP16 scale payload unchanged."""
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from vmlx_engine.jangh.switch import TQSwitchGLU
from vmlx_engine.utils.jang_loader import _apply_large_expert_bfloat16_compute


@pytest.mark.parametrize("rows", [1, 32])
def test_mla_compute_cast_preserves_jangh_payload_and_kernels(rows):
    model = nn.Module()
    model.experts = TQSwitchGLU(64, 64, 4, 2, 3, rotation_gate_up="hadamard32",
                              rotation_down="hadamard32")
    model.dense = nn.Linear(64, 64, bias=False)
    saved = []
    for projection in (model.experts.gate_proj, model.experts.up_proj, model.experts.down_proj):
        projection.tq2_scales = (1 + mx.arange(256).reshape(4, 64) / 1024).astype(mx.float16)
        saved.append(projection.tq2_scales)
    x = (mx.random.normal((rows, 64)) * .01).astype(mx.bfloat16)
    indices = mx.broadcast_to(mx.array([0, 1], dtype=mx.uint32), (rows, 2))
    scores = mx.full((rows, 2), .5, dtype=mx.bfloat16)
    expected = model.experts.routed(x, indices, scores)
    mx.eval(expected)
    assert _apply_large_expert_bfloat16_compute(
        model, Path("unused"), {"model_type": "glm5_next", "kv_lora_rank": 64})
    assert model.dense.weight.dtype == mx.bfloat16
    for projection, original in zip(
        (model.experts.gate_proj, model.experts.up_proj, model.experts.down_proj), saved
    ):
        assert projection.tq2_scales.dtype == mx.float16
        assert mx.array_equal(projection.tq2_scales, original).item()
    actual = model.experts.routed(x, indices, scores)
    mx.eval(actual)
    assert mx.array_equal(actual, expected).item()
