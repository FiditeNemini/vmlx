"""Load-boundary affine metadata dtype normalization.

The JANG affine ABI stores scales/biases as FP16. On a BF16-activation family
(GLM-5.3 Flash) MLX promotes FP16+BF16 quantized matmuls to FP32, which both
widens the whole activation path and disqualifies every FP16/BF16-only fused
decode kernel. The normalizer casts affine metadata to the resolved compute
dtype at load; these tests pin the promotion mechanism itself, the cast, and
the guards that keep non-affine and FP16-contract families untouched.
"""

import mlx.core as mx
import mlx.nn as nn

from vmlx_engine.utils.affine_metadata import normalize_affine_metadata_dtype


def _quantized_model(metadata_dtype=mx.float16):
    model = nn.Sequential(
        nn.Linear(128, 64, bias=False),
        nn.Linear(64, 128, bias=False),
    )
    nn.quantize(model, bits=4, group_size=32, mode="affine")
    for layer in model.layers:
        layer.scales = layer.scales.astype(metadata_dtype)
        layer.biases = layer.biases.astype(metadata_dtype)
    model.eval()
    return model


def test_fp16_metadata_with_bf16_activation_promotes_to_fp32():
    """The mechanism this normalizer exists for: without the cast, MLX
    promotes the quantized matmul output to FP32."""

    model = _quantized_model(mx.float16)
    x = mx.random.normal((1, 128)).astype(mx.bfloat16)
    assert model.layers[0](x).dtype == mx.float32


def test_normalize_casts_fp16_metadata_to_bf16_and_narrows_output():
    model = _quantized_model(mx.float16)
    x = mx.random.normal((1, 128)).astype(mx.bfloat16)
    reference = model(x.astype(mx.float32))

    changed = normalize_affine_metadata_dtype(model, mx.bfloat16)
    assert changed == 2
    for layer in model.layers:
        assert layer.scales.dtype == mx.bfloat16
        assert layer.biases.dtype == mx.bfloat16
        assert layer.weight.dtype == mx.uint32

    output = model(x)
    assert output.dtype == mx.bfloat16
    mx.eval(output, reference)
    max_ref = max(float(mx.max(mx.abs(reference)).item()), 1e-9)
    max_abs = float(
        mx.max(mx.abs(output.astype(mx.float32) - reference)).item()
    )
    # BF16 operand/storage rounding only; the promotion path is FP32-exact by
    # construction, so this bounds the accepted precision trade.
    assert max_abs / max_ref <= 3e-2


def test_normalize_is_a_noop_for_matching_or_fp16_contract():
    model = _quantized_model(mx.float16)
    assert normalize_affine_metadata_dtype(model, mx.float16) == 0
    assert model.layers[0].scales.dtype == mx.float16

    model_bf16 = _quantized_model(mx.bfloat16)
    assert normalize_affine_metadata_dtype(model_bf16, mx.bfloat16) == 0


def test_normalize_refuses_non_sixteen_bit_targets_and_non_affine_modes():
    model = _quantized_model(mx.float16)
    assert normalize_affine_metadata_dtype(model, mx.float32) == 0
    assert model.layers[0].scales.dtype == mx.float16

    model.layers[0].mode = "mxfp4"
    assert normalize_affine_metadata_dtype(model, mx.bfloat16) == 1
    assert model.layers[0].scales.dtype == mx.float16
    assert model.layers[1].scales.dtype == mx.bfloat16


def test_normalize_skips_integer_metadata():
    model = _quantized_model(mx.float16)
    model.layers[0].scales = mx.ones_like(model.layers[0].scales).astype(
        mx.uint8
    )
    assert normalize_affine_metadata_dtype(model, mx.bfloat16) == 1
    assert model.layers[0].scales.dtype == mx.uint8
    assert model.layers[1].scales.dtype == mx.bfloat16
