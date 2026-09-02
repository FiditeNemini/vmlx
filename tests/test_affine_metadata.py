"""Load-boundary affine metadata dtype harmonisation.

The JANG affine ABI stores scales/biases as FP16. On a BF16-activation family
(GLM-5.3 Flash) MLX promotes FP16+BF16 quantized matmuls to FP32, which both
widens the whole activation path and disqualifies every FP16/BF16-only fused
decode kernel. ``mlx_memory.harmonize_quant_metadata_dtypes`` is the single
owning correction, invoked on both production load routes (text
``utils/tokenizer.py`` and mllm ``models/mllm.py``). These tests pin the
promotion mechanism itself, the cast, and the anchor/declaration gates that
keep FP16-compute families (Step-3.7 shape) and undeclared bundles untouched.
"""

import mlx.core as mx
import mlx.nn as nn

from vmlx_engine.mlx_memory import harmonize_quant_metadata_dtypes


def _quantized_model(metadata_dtype=mx.float16, anchor_dtype=mx.bfloat16):
    model = nn.Sequential(
        nn.Linear(128, 64, bias=False),
        nn.Linear(64, 128, bias=False),
    )
    nn.quantize(model, bits=4, group_size=32, mode="affine")
    for layer in model.layers:
        layer.scales = layer.scales.astype(metadata_dtype)
        layer.biases = layer.biases.astype(metadata_dtype)
    # Non-metadata compute anchors deciding the harmonisation verdict.
    model.norm_a = mx.ones((128,), dtype=anchor_dtype)
    model.norm_b = mx.ones((128,), dtype=anchor_dtype)
    model.eval()
    return model


def test_fp16_metadata_with_bf16_activation_promotes_to_fp32():
    """The mechanism the harmoniser exists for: without the cast, MLX
    promotes the quantized matmul output to FP32."""

    model = _quantized_model(mx.float16)
    x = mx.random.normal((1, 128)).astype(mx.bfloat16)
    assert model.layers[0](x).dtype == mx.float32


def test_harmonize_casts_eligible_metadata_and_narrows_output():
    model = _quantized_model(mx.float16, anchor_dtype=mx.bfloat16)
    x = mx.random.normal((1, 128)).astype(mx.bfloat16)
    reference = model(x.astype(mx.float32))

    summary = harmonize_quant_metadata_dtypes(
        model, declared_dtype="bfloat16"
    )
    assert summary["eligible"] == 4
    assert summary["cast"] == 4
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


def test_harmonize_refuses_f16_anchor_dominant_models():
    """Step-3.7 shape: BF16 declared but F16 anchors dominate — casting the
    metadata would CREATE the promotion, so the gate must refuse."""

    model = _quantized_model(mx.float16, anchor_dtype=mx.float16)
    summary = harmonize_quant_metadata_dtypes(
        model, declared_dtype="bfloat16"
    )
    assert summary["cast"] == 0
    assert model.layers[0].scales.dtype == mx.float16


def test_harmonize_refuses_undeclared_bundle_dtype():
    model = _quantized_model(mx.float16, anchor_dtype=mx.bfloat16)
    summary = harmonize_quant_metadata_dtypes(model, declared_dtype=None)
    assert summary["cast"] == 0
    assert model.layers[0].scales.dtype == mx.float16


def test_harmonize_is_noop_when_metadata_already_matches():
    model = _quantized_model(mx.bfloat16, anchor_dtype=mx.bfloat16)
    summary = harmonize_quant_metadata_dtypes(
        model, declared_dtype="bfloat16"
    )
    assert summary["eligible"] == 0
    assert summary["cast"] == 0
