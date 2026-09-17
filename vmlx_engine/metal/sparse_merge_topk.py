# SPDX-License-Identifier: Apache-2.0
"""Experimental exact FP32 sparse selection with top-k-pruned merge rounds.

Sort adjacent tiles once, then merge their sorted candidate prefixes instead
of sorting each subsequent concatenation again. Discarding an element below
rank k in a subtree cannot change the global top k. Equal values retain input
order, using the packaged MLX comparator (including NaNs and signed zeros).

This changes neither scores nor model/cache state. The public caller is opt-in
and tied to the installed backend whose argpartition ordering is qualified.
"""

from functools import lru_cache
import importlib.metadata
import logging
from pathlib import Path

import mlx.core as mx

logger = logging.getLogger(__name__)
_FAILED = False
_OBSERVED = False
_VERIFIED_VARIANTS: set[str] = set()
_BLOCK = 1024
_K = 512
_THREADS = 256

_TILE_SOURCE = r"""
const uint lane = thread_index_in_threadgroup;
const uint chunk = threadgroup_position_in_grid.x;
const uint row = threadgroup_position_in_grid.y;
const uint n = values_shape[values_ndim - 1];
const uint out_n = (n / BLOCK) * K + min(uint(K), n % BLOCK);
const uint start = chunk * BLOCK;
const uint length = min(uint(BLOCK), n - start);
threadgroup float shared_values[BLOCK];
threadgroup uint shared_ids[BLOCK];
for (uint i = lane; i < BLOCK; i += THREADS) {
    shared_values[i] = i < length ? values[row * n + start + i] : LessThan<float>::init;
    shared_ids[i] = start + i;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
BlockMergeSort<float,uint,true,THREADS,4,LessThan<float>>::sort(
    shared_values, shared_ids, int(length), uint3(lane,0,0));
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = lane; i < min(uint(K), length); i += THREADS) {
    kept_values[row * out_n + chunk * K + i] = shared_values[i];
    kept_ids[row * out_n + chunk * K + i] = shared_ids[i];
}
"""

_MERGE_SOURCE = r"""
const uint lane = thread_index_in_threadgroup;
const uint pair = threadgroup_position_in_grid.x;
const uint row = threadgroup_position_in_grid.y;
const uint n = values_shape[values_ndim - 1];
const uint start = pair * (2 * K);
const uint len_a = min(uint(K), n - start);
const uint len_b = min(uint(K), n - start - len_a);
const uint length = min(uint(K), len_a + len_b);
const uint out_n = (n / (2 * K)) * K + min(uint(K), n % (2 * K));
threadgroup float shared_values[2 * K];
threadgroup uint shared_ids[2 * K];
for (uint i = lane; i < len_a + len_b; i += THREADS) {
    shared_values[i] = values[row * n + start + i];
    shared_ids[i] = ids[row * n + start + i];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
LessThan<float> less;
for (uint rank = lane; rank < length; rank += THREADS) {
    // Stable merge partition: the left run wins comparator-equivalent ties.
    int low = max(0, int(rank) - int(len_b));
    int high = min(int(rank), int(len_a));
    while (low < high) {
        int mid = low + (high - low) / 2;
        if (less(shared_values[len_a + rank - 1 - mid], shared_values[mid]))
            high = mid;
        else
            low = mid + 1;
    }
    uint a = uint(low), b = rank - a;
    bool take_b = b < len_b && (a >= len_a ||
        less(shared_values[len_a + b], shared_values[a]));
    uint source = take_b ? len_a + b : a;
    kept_values[row * out_n + pair * K + rank] = shared_values[source];
    kept_ids[row * out_n + pair * K + rank] = shared_ids[source];
}
"""


@lru_cache(maxsize=1)
def _compatible_sort_version() -> bool:
    try:
        return importlib.metadata.version("mlx") == "0.32.2"
    except importlib.metadata.PackageNotFoundError:
        return False


@lru_cache(maxsize=2)
def _kernel(stage: str):
    # Existing Apache-2.0 MLX sort helper and NOTICE are packaged together.
    header = Path(__file__).with_name("glm5_dsa_sort.metal").read_text()
    return mx.fast.metal_kernel(
        name=f"vmlx_sparse_topk_{stage}",
        input_names=["values"] if stage == "tile" else ["values", "ids"],
        output_names=["kept_values", "kept_ids"],
        header=header,
        source=_TILE_SOURCE if stage == "tile" else _MERGE_SOURCE,
        compile_options={"math_mode": "safe"},
    )


def sparse_merge_topk(negative_scores, *, k: int, enabled: bool):
    """Return stock-order indices, or None to retain the incumbent selector.

    Only single-batch FP32 top-512 with >2048 pools is admitted to this
    experimental path. Shapes do not enter the shader specialization key.
    No input, score, index, or cache arrays survive in Python global state.
    """
    global _FAILED, _OBSERVED
    if not enabled or _FAILED:
        return None
    if (
        negative_scores.ndim != 3
        or negative_scores.shape[0] != 1
        or negative_scores.shape[1] < 1
        or negative_scores.shape[-1] <= 2048
        or negative_scores.dtype != mx.float32
        or type(k) is not int or k != _K
        or negative_scores.size >= 2**32
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or not _compatible_sort_version()
    ):
        return None
    try:
        lead = negative_scores.shape[:-1]
        rows = negative_scores.shape[1]
        values, ids = negative_scores, None
        stage = "tile"
        while True:
            n = values.shape[-1]
            width = _BLOCK if stage == "tile" else 2 * _K
            full, tail = divmod(n, width)
            out_n = full * _K + min(_K, tail)
            inputs = [values] if stage == "tile" else [values, ids]
            values, ids = _kernel(stage)(
                inputs=inputs,
                template=[("BLOCK", _BLOCK), ("K", _K), ("THREADS", _THREADS)],
                grid=(_THREADS * (full + bool(tail)), rows, 1),
                threadgroup=(_THREADS, 1, 1),
                output_shapes=[(*lead, out_n), (*lead, out_n)],
                output_dtypes=[mx.float32, mx.uint32],
            )
            if stage not in _VERIFIED_VARIANTS:
                mx.eval(values, ids)
                _VERIFIED_VARIANTS.add(stage)
            if out_n == _K:
                break
            stage = "merge"
        if not _OBSERVED:
            mx.eval(ids)
            _OBSERVED = True
            logger.info(
                "Sparse pruned merge selection active: rows=%d pools=%d k=%d "
                "scores=fp32 stable_order=true cache_state=unchanged",
                rows, negative_scores.shape[-1], k,
            )
        return ids
    except (OSError, RuntimeError, ValueError) as exc:
        _FAILED = True
        logger.warning("Sparse pruned merge selection disabled after launch failure: %s", exc)
        return None
