# SPDX-License-Identifier: Apache-2.0
"""Experimental Flash AR arithmetic identity; normal decoding is unchanged."""

import os

QWEN4_PRECISE_GDN_EPILOGUE_MATH_ABI = "mlx0322_m5max_fp16_post_rms_sigmoid_v1"


def precise_gdn_epilogue_requested() -> bool:
    return os.environ.get("VMLX_QWEN4_PRECISE_GDN_EPILOGUE", "0") == "1"
