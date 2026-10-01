"""Unqualified candidate's owning numerical gate; requires real supported Metal."""

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize("batch,tokens,strided", [(1, 33, False), (2, 37, True)])
def test_weighted_unsort_matches_native_fp16_bits(batch, tokens, strided):
    mx = pytest.importorskip("mlx.core")
    path = Path(__file__).parents[1] / "vmlx_engine/metal/qwen4_prefill_reduce.py"
    spec = importlib.util.spec_from_file_location("qwen_prefill_reduce_numeric", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not module._available():
        pytest.skip("candidate requires MLX 0.32.3 and Apple M5 Max GPU")
    rows = batch * tokens
    # Repeated experts, ragged routing, nonuniform signed weights and cancellation.
    ids = (mx.arange(rows * 10, dtype=mx.uint32) * 7) % 13
    inverse = mx.argsort(mx.argsort(ids))
    width = 5120 if strided else 2560
    y = mx.sin(mx.arange(rows * 10 * width, dtype=mx.float32) * 0.03125)
    y = y.reshape(rows * 10, width).astype(mx.float16)
    # Raw half bits preserve signed zeros and subnormals without host-float
    # conversion. Both layouts carry these values through the inverse gather.
    edge_bits = mx.array(
        [0x0000, 0x8000, 0x0001, 0x8001, 0x0002, 0x8002,
         0x03FF, 0x83FF, 0x0400, 0x8400, 0x0401, 0x8401,
         0x3800, 0xB800, 0x3C00, 0xBC00],
        dtype=mx.uint16,
    )
    edges = edge_bits.view(mx.float16)
    step = 2 if strided else 1
    y[:, :16 * step:step] = mx.broadcast_to(edges, (rows * 10, 16))
    if strided:
        y = y[:, ::2]
    scores = ((mx.arange(rows * 10, dtype=mx.float32) % 19) - 9) / 16
    scores = scores.astype(mx.float16).reshape(batch, tokens, 10)
    # First token has half weights: min-subnormal products underflow or tie,
    # while max-subnormal/min-normal products cross the normal boundary.
    # The next token also exercises subnormal and signed-zero score operands.
    scores = scores.reshape(rows, 10)
    scores[0] = mx.full((10,), 0.5, dtype=mx.float16)
    scores[1] = edges[:10]
    scores = scores.reshape(batch, tokens, 10)
    assert y.dtype == scores.dtype == mx.float16
    expected = (y[inverse].reshape(batch, tokens, 10, 2560) * scores[..., None]).sum(
        axis=-2
    )
    actual = module.weighted_unsort(y, inverse, scores)
    mx.eval(expected, actual)
    assert bool(mx.array_equal(expected.view(mx.uint16), actual.view(mx.uint16)))
