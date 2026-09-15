# SPDX-License-Identifier: Apache-2.0
"""Experimental GLM AR arithmetic identity; no change to default decoding."""

import os

GLM5_EXACT_MOE_MATH_ABI = "mlx0322_q2g128_bf16_v1"
GLM5_COMPILED_DSA_MATH_ABI = "mlx0322_bf16_f32sdpa_r512_v1"
GLM5_KDA_LOWRANK_MATH_ABI = "mlx0322_bf16_equal_shape_batch_v1"


def glm5_exact_moe_requested() -> bool:
    return os.environ.get("VMLX_GLM5_EXACT_MOE_DECODE", "0") == "1"


def glm5_compiled_dsa_requested() -> bool:
    return os.environ.get("VMLX_GLM5_COMPILED_DSA_DECODE", "0") == "1"


def glm5_kda_lowrank_requested() -> bool:
    return os.environ.get("VMLX_GLM5_KDA_LOWRANK_GROUP", "0") == "1"
