import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest


def _qlinear(input_dims: int, output_dims: int, *, bits: int = 4):
    dense = nn.Linear(input_dims, output_dims, bias=False)
    dense.weight = dense.weight.astype(mx.bfloat16)
    return dense.to_quantized(group_size=32, bits=bits)


def test_quantized_projection_group_is_bit_exact():
    from vmlx_engine.metal.quantized_projection_group import (
        QuantizedProjectionGroup,
    )

    linears = (_qlinear(64, 96), _qlinear(64, 32), _qlinear(64, 64))
    x = (mx.arange(128, dtype=mx.bfloat16) / 127.0).reshape(1, 2, 64)
    reference = tuple(linear(x) for linear in linears)
    group = QuantizedProjectionGroup(linears)
    candidate = group(x)
    mx.eval(*reference, *candidate)

    assert group.input_dims == 64
    assert group.output_dims == 192
    for expected, actual in zip(reference, candidate):
        np.testing.assert_array_equal(
            np.asarray(actual.astype(mx.float32)),
            np.asarray(expected.astype(mx.float32)),
        )


def test_quantized_projection_group_rejects_incompatible_quantization():
    from vmlx_engine.metal.quantized_projection_group import (
        QuantizedProjectionGroup,
    )

    with pytest.raises(ValueError, match="bit widths differ"):
        QuantizedProjectionGroup((_qlinear(64, 32, bits=4), _qlinear(64, 32, bits=8)))

    with pytest.raises(ValueError, match="input dimensions differ"):
        QuantizedProjectionGroup((_qlinear(64, 32), _qlinear(96, 32)))
