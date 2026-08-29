import mlx.core as mx
import pytest

from vmlx_engine.metal.sparse_index_score_decode import (
    sparse_index_scores_decode,
)


@pytest.mark.parametrize(("heads", "weighted"), [(4, False), (32, True)])
def test_sparse_index_score_decode_matches_stock(heads, weighted):
    mx.random.seed(41 + heads)
    head_dim = 128
    pools = 257
    query = (mx.random.normal((1, 1, heads, head_dim)) * 0.1).astype(
        mx.float32
    )
    keys = (mx.random.normal((1, pools, head_dim)) * 0.1).astype(mx.float32)
    weights = (
        (mx.random.normal((1, 1, heads)) * 0.1).astype(mx.float32)
        if weighted
        else None
    )
    per_head = mx.einsum("bshd,bnd->bshn", query, keys)
    reference = (
        mx.einsum(
            "bsh,bshn->bsn",
            weights,
            mx.maximum(per_head * (head_dim**-0.5), 0.0),
        )
        if weights is not None
        else mx.maximum(per_head, 0.0).sum(axis=2) * (head_dim**-0.5)
    )
    candidate = sparse_index_scores_decode(
        query,
        keys,
        scale=head_dim**-0.5,
        head_weights=weights,
        enabled=True,
    )
    assert candidate is not None
    mx.eval(reference, candidate)
    max_abs = float(mx.max(mx.abs(candidate - reference)).item())
    max_ref = max(float(mx.max(mx.abs(reference)).item()), 1e-9)
    assert max_abs / max_ref <= 1e-3
    candidate_top = mx.sort(
        mx.argpartition(-candidate, kth=31, axis=-1)[..., :32], axis=-1
    )
    reference_top = mx.sort(
        mx.argpartition(-reference, kth=31, axis=-1)[..., :32], axis=-1
    )
    assert bool(mx.array_equal(candidate_top, reference_top).item())


def test_sparse_index_score_decode_refuses_non_decode_shape():
    query = mx.zeros((1, 2, 4, 128), dtype=mx.float32)
    keys = mx.zeros((1, 32, 128), dtype=mx.float32)
    assert sparse_index_scores_decode(
        query, keys, scale=128**-0.5, enabled=True
    ) is None
