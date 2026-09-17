"""Fuse four-head QSA ReLU/reduction/scale without changing FP32 QK math.

The stock GEMM stays authoritative. This removes the full-size ReLU buffer
between GEMM and head reduction, not precision from the indexer's input.
"""
from functools import lru_cache
import logging

import mlx.core as mx

logger = logging.getLogger(__name__)
_observed = False
_failed = False


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="vmlx_qwen4_qsa_reduce4",
        input_names=["dots", "denominator"],
        output_names=["scores"],
        source="""
            // Runtime extents: growing context must not compile a new kernel
            // every time another four-token index block becomes complete.
            const uint ROWS = dots_shape[1];
            const uint POOLS = dots_shape[3];
            uint i = thread_position_in_grid.x;
            if (i >= ROWS * POOLS) return;
            uint row = i / POOLS;
            uint pool = i % POOLS;
            size_t offset = size_t(row) * 4 * POOLS + pool;
            float a = dots[offset];
            float b = dots[offset + POOLS];
            float c = dots[offset + 2 * POOLS];
            float d = dots[offset + 3 * POOLS];
            // Metal max discards a single NaN; MLX maximum propagates it.
            a = metal::isnan(a) ? a : metal::max(a, 0.0f);
            b = metal::isnan(b) ? b : metal::max(b, 0.0f);
            c = metal::isnan(c) ? c : metal::max(c, 0.0f);
            d = metal::isnan(d) ? d : metal::max(d, 0.0f);
            scores[i] = ((a + b) + c + d) / denominator[0];
        """,
    )


@lru_cache(maxsize=8)
def _denominator(head_dim):
    value = mx.array([head_dim**0.5], dtype=mx.float32)
    mx.eval(value)
    return value


def qsa_score_reduce(dots, *, head_dim, enabled):
    """Return [B,S,N] or None for the unchanged stock path."""
    if (not enabled or dots.ndim != 4 or dots.shape[0] != 1
            or dots.shape[2] != 4 or dots.dtype != mx.float32
            or head_dim <= 0 or dots.shape[1] < 1 or dots.shape[3] < 1):
        return None
    if (_failed or mx.default_device() != mx.gpu
            or mx.__version__ != "0.32.2"):
        return None
    rows, pools = int(dots.shape[1]), int(dots.shape[3])
    return _launch(dots, head_dim, rows, pools)


def _launch(dots, head_dim, rows, pools):
    global _observed, _failed
    try:
        result = _kernel()(
            inputs=[dots, _denominator(head_dim)],
            grid=(rows * pools, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(1, rows, pools)], output_dtypes=[mx.float32],
        )[0]
        if not _observed:
            mx.eval(result)
            _observed = True
            logger.info("Qwen4 QSA score reduction active: heads=4 dtype=float32")
        return result
    except Exception:
        _failed = True
        logger.warning("Qwen4 QSA score reduction disabled after launch failure; "
                       "using stock scoring", exc_info=True)
        return None
