"""Experimental FP32 GLM DSA prefill score tiling; disabled by default."""
import os
import hashlib
from pathlib import Path
from functools import lru_cache
import mlx.core as mx


def glm5_dsa_query_tile_requested():
    return os.environ.get("VMLX_GLM5_DSA_QUERY_TILE", "0").strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def glm5_dsa_query_tile_identity():
    return "fp32_q128_v1:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def try_glm5_dsa_query_tiles(q, pool_keys, head_weights, scale, *, enabled=False):
    """Realize reduced query tiles, avoiding a full head/query/pool tensor.

    Preserve FP32 score arithmetic and leave selection/visibility unchanged.
    Small prefills retain the incumbent path because component timing regressed.
    This opt-in requires full-model native-state qualification before promotion.
    """
    if not enabled or q.ndim != 4 or pool_keys.ndim != 3 or head_weights.ndim != 3:
        return None
    batch, queries, heads, dim = q.shape
    if batch != 1 or heads != 32 or dim != 128 or queries < 1024:
        return None
    if pool_keys.shape[0] != batch or pool_keys.shape[2] != dim:
        return None
    pools = pool_keys.shape[1]
    if tuple(head_weights.shape) != (batch, queries, heads):
        return None
    if any(a.dtype != mx.float32 for a in (q, pool_keys, head_weights)):
        return None
    if batch * queries * heads * pools * 4 < 512 * 1024**2:
        return None
    parts = []
    for begin in range(0, queries, 128):
        end = min(begin + 128, queries)
        scores = mx.einsum("bshd,bpd->bshp", q[:, begin:end], pool_keys)
        scores = mx.maximum(scores * scale, 0.0)
        part = mx.einsum("bsh,bshp->bsp", head_weights[:, begin:end], scores)
        mx.eval(part)
        del scores  # do not retain a realized head-wise tile across iterations
        parts.append(part)
    result = mx.concatenate(parts, axis=1)
    mx.eval(result)
    return result
