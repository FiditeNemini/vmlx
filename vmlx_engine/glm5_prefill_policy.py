# SPDX-License-Identifier: Apache-2.0
"""Shared GLM prefill lifetime policy for execution and cache identity."""

import os


GLM5_REGISTER_SUM_MATH_ABI = "mlx0322_row128_v1"


def glm5_register_pairwise_sum_requested() -> bool:
    """Experimental same-tree reduction, off until serving qualification."""
    return os.environ.get("VMLX_GLM5_REGISTER_PAIRWISE_SUM", "0") == "1"


def glm5_prefill_layer_fence_enabled() -> bool:
    """Realize completed-layer state by default; retain an explicit off switch.

    GLM's mixed recurrent/latent caches retain lazy side outputs. Evaluating
    each layer's streams and native state bounds those graph lifetimes without
    changing token partitioning, projection shapes, arithmetic, or state ABI.
    The model applies this only to cached prefill longer than 64 tokens, never
    ordinary decode or short speculative verification.
    """
    return os.environ.get("VMLX_GLM5_PREFILL_LAYER_FENCE", "1") == "1"
