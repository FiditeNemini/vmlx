"""Shared GLM DSA / Qwen QSA single-token pool-score fusion."""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

_ENV = {
    "glm5_next": "VMLX_GLM5_FUSED_DSA_SCORE",
    "qwen4_exp": "VMLX_QWEN4_FUSED_QSA_SCORE",
}


def fused_sparse_index_score_requested(family: str) -> bool:
    value = os.environ.get(_ENV[family], "0").strip().lower()
    return value not in {"", "0", "false", "off", "no"}


@lru_cache(maxsize=16)
def _kernel(heads: int, head_dim: int):
    source = f"""
        uint pool = thread_position_in_grid.x / 32u;
        uint lane = thread_index_in_simdgroup;
        if (pool >= POOLS) return;

        float total = 0.0f;
        for (uint head = 0u; head < {heads}u; ++head) {{
            float dot = 0.0f;
            size_t query_base = (size_t)head * {head_dim}u;
            size_t key_base = (size_t)pool * {head_dim}u;
            for (uint dim = lane; dim < {head_dim}u; dim += 32u) {{
                dot += float(query[query_base + dim]) *
                    float(pool_keys[key_base + dim]);
            }}
            dot = simd_sum(dot);
            if (WEIGHTED) {{
                total += float(head_weights[head]) *
                    metal::max(dot * float(scale[0]), 0.0f);
            }} else {{
                total += metal::max(dot, 0.0f);
            }}
        }}
        if (lane == 0u) {{
            scores[pool] = WEIGHTED ? total : total * float(scale[0]);
        }}
"""
    return mx.fast.metal_kernel(
        name=f"vmlx_sparse_index_score_h{heads}_d{head_dim}",
        input_names=["query", "pool_keys", "head_weights", "scale"],
        output_names=["scores"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=source,
    )


@lru_cache(maxsize=16)
def _scalar(value: float) -> mx.array:
    result = mx.array([value], dtype=mx.float32)
    mx.eval(result)
    return result


@lru_cache(maxsize=8)
def _ones(heads: int) -> mx.array:
    result = mx.ones((heads,), dtype=mx.float32)
    mx.eval(result)
    return result


def sparse_index_scores_decode(
    query: mx.array,
    pool_keys: mx.array,
    *,
    scale: float,
    head_weights: mx.array | None = None,
    enabled: bool,
) -> mx.array | None:
    """Return ``[1,1,N]`` pool scores or the stock-path sentinel."""

    if not enabled or query.ndim != 4 or pool_keys.ndim != 3:
        return None
    batch, rows, heads, head_dim = (int(value) for value in query.shape)
    if batch != 1 or rows != 1 or tuple(pool_keys.shape[:1]) != (1,):
        return None
    pools = int(pool_keys.shape[1])
    if pools < 1 or int(pool_keys.shape[2]) != head_dim:
        return None
    if query.dtype != mx.float32 or pool_keys.dtype != mx.float32:
        return None
    if head_weights is None:
        weights = _ones(heads)
        weighted = False
    else:
        if tuple(head_weights.shape) != (1, 1, heads):
            return None
        if head_weights.dtype != mx.float32:
            return None
        weights = head_weights.reshape(heads)
        weighted = True

    return _kernel(heads, head_dim)(
        inputs=[query, pool_keys, weights, _scalar(float(scale))],
        template=[("POOLS", pools), ("WEIGHTED", weighted)],
        grid=(32 * pools, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, 1, pools)],
        output_dtypes=[mx.float32],
    )[0]


__all__ = [
    "fused_sparse_index_score_requested",
    "sparse_index_scores_decode",
]
