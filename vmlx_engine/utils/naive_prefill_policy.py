# SPDX-License-Identifier: Apache-2.0
"""Experimental Naive attention arithmetic policy shared with cache identity."""
import os


def naive_padded_prefill_requested() -> bool:
    return os.environ.get("VMLX_NAIVE_PADDED_PREFILL", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }


NAIVE_PADDED_PREFILL_IDENTITY = "naive_padded_prefill_v1"
