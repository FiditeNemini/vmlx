"""Bounded-memory MLX attention for bulk QSA prefill, not MTP/decode."""

import logging
import os
import sys

import mlx.core as mx

logger = logging.getLogger(__name__)
_logged_dispatch = False


def qwen4_prefill_sdpa(q, k, v, mask, *, scale, training=False):
    """Use MLX's supported fused kernel, or return None for normal dispatch.

    MLX 0.32.2 supports masked head-256 attention but its default heuristic
    still selects unfused prefill for an array mask. That materializes scores
    proportional to query_chunk * context * heads, even though QSA selected
    only a sparse subset. The Darwin dependency floor includes force_fused.

    Qualify from actual tensors, never the bundle name/quant label. Keep small
    suffixes and all decode/verification rows unchanged: the fused full kernel
    was slower for a 16-row control, while bulk rows bounded the transient.
    No mask, selected positions, cache layout, or model dtype is changed.
    """
    if (
        sys.platform != "darwin"
        or training
        or os.environ.get("VMLX_QWEN4_PREFILL_FUSED_SDPA", "1").lower()
        in {"0", "false", "no", "off"}
        or q.ndim != 4
        or k.ndim != 4
        or v.shape != k.shape
        or q.shape[0] != k.shape[0]
        or q.shape[2] < 256
        or q.shape[2] > k.shape[2]
        or q.shape[3] != 256
        or k.shape[3] != 256
        or q.dtype not in (mx.float16, mx.bfloat16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
        or mx.default_device() != mx.gpu
    ):
        return None

    out = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=mask, force_fused=True
    )
    global _logged_dispatch
    if not _logged_dispatch:
        logger.info(
            "QSA prefill dispatch path=mlx_fused_masked batch=%d heads=%d "
            "kv_heads=%d rows=%d context=%d head_dim=%d dtype=%s",
            q.shape[0], q.shape[1], k.shape[1], q.shape[2], k.shape[2],
            q.shape[3], q.dtype,
        )
        _logged_dispatch = True
    return out
