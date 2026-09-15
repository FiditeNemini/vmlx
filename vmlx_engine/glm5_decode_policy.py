# SPDX-License-Identifier: Apache-2.0
"""Experimental GLM AR arithmetic identity; no change to default decoding."""

import os

GLM5_EXACT_MOE_MATH_ABI = "mlx0322_q2g128_bf16_v1"


def glm5_exact_moe_requested() -> bool:
    return os.environ.get("VMLX_GLM5_EXACT_MOE_DECODE", "0") == "1"
