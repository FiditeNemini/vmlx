# SPDX-License-Identifier: Apache-2.0
"""The prompt disk cache must not narrow bfloat16 to float16.

numpy has no bf16, so the store has to cast — but bf16 and f16 are NOT
interchangeable despite both being 16 bits. bf16 spends 8 bits on the exponent
(the same ~1e38 range as f32) and f16 spends 5 (max 65504). MEASURED with MLX:

    bf16 source        [70144.0, 9.98e-07, 3.5]
    cast via float16   [   inf, 1.013e-06, 3.5]     <- the old path
    cast via float32   [70144.0, 9.98e-07, 3.5]     <- exact

So a KV value over 65504 came back as inf and small values shifted, meaning a
restored prompt cache could differ from a fresh compute. That is the same
"a cache hit must equal a recompute" class as the mixed-SWA storage-TQ
divergence. The code carried "acceptable precision for prompt cache" as its
justification, with nothing measured behind it.

bf16 -> f32 is lossless in both directions — f32 has a wider mantissa AND the
same exponent range — and is what block_disk_store already does for exactly
this reason. It costs 2x the bytes for bf16 caches.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

mx = pytest.importorskip("mlx.core")


def test_float16_really_does_destroy_bf16_range():
    """Pins the premise with real MLX, so this file cannot rot into folklore."""
    src = mx.array([70000.0, 1e-6, 3.5], dtype=mx.float32).astype(mx.bfloat16)
    exact = src.astype(mx.float32).tolist()

    via_f16 = src.astype(mx.float16).astype(mx.float32).tolist()
    assert via_f16[0] == float("inf"), (
        "float16 no longer overflows here; re-check whether the cast still "
        "needs to be float32"
    )
    assert via_f16 != exact

    via_f32 = src.astype(mx.float32).tolist()
    assert via_f32 == exact, "bf16 -> f32 must be exact in both directions"


def test_store_casts_bfloat16_to_float32_not_float16():
    src = (ROOT / "vmlx_engine" / "disk_cache.py").read_text(encoding="utf-8")
    start = src.index("    def store(")
    body = src[start : src.index("\n    def ", start + 1)]

    assert "v.astype(mx.float32) if v.dtype == mx.bfloat16" in body, (
        "the prompt disk cache no longer widens bf16 losslessly"
    )
    assert not re.search(r"astype\(mx\.float16\)\s*if\s*v\.dtype\s*==\s*mx\.bfloat16", body), (
        "the lossy bf16 -> f16 cast is back; values above 65504 become inf"
    )
    assert "acceptable precision for prompt cache" not in body, (
        "the unmeasured justification is back"
    )


def test_block_disk_store_still_agrees():
    """The two tiers must not disagree about how a bf16 cache is persisted."""
    block = (ROOT / "vmlx_engine" / "block_disk_store.py").read_text(
        encoding="utf-8"
    )
    assert "astype(mx.float32)" in block
    assert "astype(mx.float16)" not in block, (
        "block_disk_store started narrowing to float16; the two disk tiers "
        "would then round-trip the same cache differently"
    )
