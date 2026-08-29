"""Exact grouping for compatible affine ``QuantizedLinear`` projections.

Affine quantization metadata is independent for every output row. Compatible
same-input projections can therefore be concatenated on the output-row axis
and evaluated by one ``mx.quantized_matmul`` without changing their math. The
grouped call retains MLX's normal device/shape dispatcher while removing
redundant command launches. Backend selection (including any M5 TensorOps/NAX
path) remains a separate live-measurement question.

This module does not choose model policy. Callers decide whether to retain the
original projections for a diagnostic cache or replace them at load time to
keep resident memory neutral.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as nn


def _quantized_linear_dimensions(linear: nn.QuantizedLinear) -> tuple[int, int]:
    """Derive logical dimensions from MLX's stored quantization tensors."""

    output_dims = int(linear.weight.shape[0])
    input_dims = int(linear.scales.shape[-1]) * int(linear.group_size)
    return input_dims, output_dims


class QuantizedProjectionGroup(nn.Module):
    """One packed affine projection that returns the original output tuple."""

    def __init__(self, linears: Sequence[nn.Module]):
        super().__init__()
        linears = tuple(linears)
        reason = quantized_projection_group_reason(linears)
        if reason is not None:
            raise ValueError(reason)

        first = linears[0]
        self.group_size = int(first.group_size)
        self.bits = int(first.bits)
        self.mode = str(first.mode)
        self.input_dims, _ = _quantized_linear_dimensions(first)
        self.output_dims = sum(
            _quantized_linear_dimensions(linear)[1] for linear in linears
        )
        self.weight = mx.concatenate([linear.weight for linear in linears], axis=0)
        self.scales = mx.concatenate([linear.scales for linear in linears], axis=0)
        self.biases = mx.concatenate([linear.biases for linear in linears], axis=0)

        splits: list[int] = []
        offset = 0
        for linear in linears[:-1]:
            offset += _quantized_linear_dimensions(linear)[1]
            splits.append(offset)
        self.split_indices = tuple(splits)
        self.freeze()

    def __call__(self, x: mx.array) -> tuple[mx.array, ...]:
        output = mx.quantized_matmul(
            x,
            self.weight,
            scales=self.scales,
            biases=self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        return tuple(mx.split(output, self.split_indices, axis=-1))


def quantized_projection_group_reason(
    linears: Sequence[nn.Module],
    *,
    activation_dtype: mx.Dtype | None = None,
) -> str | None:
    """Return why projections cannot share one exact affine dispatch."""

    linears = tuple(linears)
    if not linears:
        return "no projections"
    if not all(isinstance(linear, nn.QuantizedLinear) for linear in linears):
        return "projection is not QuantizedLinear"

    first = linears[0]
    first_input_dims, _ = _quantized_linear_dimensions(first)
    for linear in linears:
        if linear.weight.ndim != 2:
            return "packed weight is not rank two"
        if linear.scales.ndim != 2:
            return "scales are not rank two"
        if linear.biases is None:
            return "affine biases are missing"
        if linear.biases.shape != linear.scales.shape:
            return "affine metadata shapes differ"
        if (
            linear.weight.shape[0] != linear.scales.shape[0]
            or linear.weight.shape[0] != linear.biases.shape[0]
        ):
            return "output-row counts differ within a projection"
        input_dims, _ = _quantized_linear_dimensions(linear)
        if int(linear.weight.shape[-1]) * 32 != input_dims * int(linear.bits):
            return "packed weight geometry is inconsistent"
        if input_dims != first_input_dims:
            return "input dimensions differ"
        if int(linear.bits) != int(first.bits):
            return "bit widths differ"
        if int(linear.group_size) != int(first.group_size):
            return "group sizes differ"
        if str(linear.mode) != str(first.mode):
            return "quantization modes differ"
        if "bias" in linear:
            return "post-matmul bias is unsupported"
        if linear.weight.dtype != first.weight.dtype:
            return "packed weight dtypes differ"
        if linear.scales.dtype != first.scales.dtype:
            return "scale dtypes differ"
        if linear.biases.dtype != first.biases.dtype:
            return "affine bias dtypes differ"
    if activation_dtype is not None and (
        first.scales.dtype != activation_dtype
        or first.biases.dtype != activation_dtype
    ):
        return "activation and affine metadata dtypes differ"
    return None


def cached_quantized_projection_group(
    linears: Sequence[nn.Module],
    *,
    owner: object,
    cache_attr: str,
) -> QuantizedProjectionGroup:
    """Build and cache a group while retaining the caller's projections."""

    linears = tuple(linears)
    source_key = tuple(
        (id(linear.weight), id(linear.scales), id(linear.biases))
        for linear in linears
    )
    cached = getattr(owner, cache_attr, None)
    if cached is None or cached[0] != source_key:
        group = QuantizedProjectionGroup(linears)
        mx.eval(group.weight, group.scales, group.biases)
        cached = (source_key, group)
        setattr(owner, cache_attr, cached)
    return cached[1]


__all__ = [
    "QuantizedProjectionGroup",
    "cached_quantized_projection_group",
    "quantized_projection_group_reason",
]
