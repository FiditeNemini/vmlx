"""Affine shape repair must not reinterpret native MX weight encodings."""

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from vmlx_engine.utils.jang_loader import (
    _fix_quantized_bits,
    _pre_fix_bits_from_metadata,
    _pre_fix_bits_from_shard,
)


@pytest.mark.parametrize("mode,bits", [("mxfp8", 8), ("mxfp4", 4)])
@pytest.mark.parametrize("stage", ["metadata", "shard", "postload"])
def test_mx_projection_keeps_native_encoding_and_output(mode, bits, stage):
    # Router naming also exercises the post-load preference for affine g64.
    model = nn.Module()
    model.gate = nn.QuantizedLinear(128, 64, bias=False, group_size=32,
                                    bits=bits, mode=mode)
    x = mx.ones((2, 128), dtype=mx.bfloat16)
    expected = model.gate(x)
    mx.eval(expected)
    shard = {"gate.weight": model.gate.weight, "gate.scales": model.gate.scales}
    if stage == "metadata":
        _pre_fix_bits_from_metadata(model, {k: v.shape for k, v in shard.items()}, 64)
    elif stage == "shard":
        _pre_fix_bits_from_shard(model, shard, 64)
    else:
        _fix_quantized_bits(model)
    assert (model.gate.mode, model.gate.bits, model.gate.group_size) == (mode, bits, 32)
    actual = model.gate(x)
    mx.eval(actual)
    assert mx.array_equal(actual, expected).item()
