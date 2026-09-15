# SPDX-License-Identifier: Apache-2.0
"""Shape-preserving BF16 gate projection groups for experimental GLM AR.

Concatenating f_a/g_a/b output rows changes MLX's GEMV geometry and can change
BF16 rounding. Batch only equal-shaped f/g matrices, preserving the explicit
BF16 intermediate; beta remains separate. Packed views retain original-size
calls for prefill, multiple requests, other dtypes and speculative forwards.
"""

import logging

import mlx.core as mx
import mlx.nn as nn

_LOG = logging.getLogger(__name__)
_OBSERVED = False


class Glm5KDALowRankGroup(nn.Module):
    def __init__(self, f_a, g_a, beta, f_b, g_b):
        super().__init__()
        if not self.compatible(f_a, g_a, beta, f_b, g_b):
            raise ValueError("GLM KDA low-rank grouping requires matching unbiased BF16 linears")
        self.input_weights = mx.stack((f_a.weight, g_a.weight), axis=0)
        self.output_weights = mx.stack((f_b.weight, g_b.weight), axis=0)
        self.beta_weight = beta.weight
        self.enabled = True
        self.observed_calls = 0
        mx.eval(self.input_weights, self.output_weights, self.beta_weight)
        self.freeze()

    @staticmethod
    def compatible(f_a, g_a, beta, f_b, g_b) -> bool:
        linears = (f_a, g_a, beta, f_b, g_b)
        if any(type(l) is not nn.Linear or "bias" in l for l in linears):
            return False
        if any(l.weight.ndim != 2 or l.weight.dtype != mx.bfloat16 for l in linears):
            return False
        rank, hidden = f_a.weight.shape
        heads = beta.weight.shape[0]
        return (
            rank > 0 and hidden > 0 and heads > 0
            and g_a.weight.shape == (rank, hidden)
            and beta.weight.shape == (heads, hidden)
            and f_b.weight.shape == g_b.weight.shape == (heads * rank, rank)
        )

    def decode(self, x):
        if (
            not self.enabled or mx.default_device() != mx.gpu
            or x.ndim != 3 or x.shape[:2] != (1, 1)
            or x.dtype != mx.bfloat16
        ):
            return None
        low = x @ self.input_weights.transpose(0, 2, 1)
        high = low @ self.output_weights.transpose(0, 2, 1)
        beta = x @ self.beta_weight.T
        self.observed_calls += 1
        global _OBSERVED
        if not _OBSERVED:
            _OBSERVED = True
            _LOG.info(
                "GLM paired KDA projections scheduled: hidden=%d rank=%d heads=%d dtype=%s",
                self.input_weights.shape[-1], self.input_weights.shape[-2],
                self.beta_weight.shape[0], x.dtype,
            )
        return high[0:1], high[1:2], beta

    def decay(self, x):
        return (x @ self.input_weights[0].T) @ self.output_weights[0].T

    def output_gate(self, x):
        return (x @ self.input_weights[1].T) @ self.output_weights[1].T

    def beta(self, x):
        return x @ self.beta_weight.T
