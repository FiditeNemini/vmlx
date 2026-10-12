"""Fused DSA indexer scores (vmlx_engine/metal/dsa_index_scores.py) for Naive N0.5 prefill.

out[q, k] = sum_h w[q, h] * relu(Q[h, q] . K[k]). The stock path materialized the (heads, queries, keys) fp32
intermediate; the fused kernel keeps tiles in registers (2.1-2.5x faster per DSA layer). The stock GPU matmul on M5 is
~8e-4 relative off an fp64 reference; the kernel must stay at fp32 accuracy, give each row independently of how many
queries share the call (a warm turn's short tail == the same rows of the cold chunk), and fall back for decode.
"""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from vmlx_engine.metal.dsa_index_scores import dsa_index_scores  # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


def _inputs(L, Lk, H=16, D=128, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((1, H, L, D))
    k = mx.random.normal((1, 1, Lk, D))
    w = mx.random.normal((1, L, H)) * 0.25
    mx.eval(q, k, w)
    return q, k, w


def _ref(q, k, w):
    qn, kn, wn = (np.array(x).astype(np.float64) for x in (q, k, w))
    return np.einsum("hqk,qh->qk", np.maximum(np.einsum("hqd,kd->hqk", qn[0], kn[0, 0]), 0), wn[0])


@pytest.mark.parametrize("L,Lk", [(2048, 4099), (100, 333), (64, 32), (65, 97), (2, 40), (31, 4097)])
def test_matches_fp64_reference_at_fp32_accuracy(L, Lk):
    q, k, w = _inputs(L, Lk, seed=L + Lk)
    out = dsa_index_scores(q, k, w)
    assert out is not None and out.shape == (1, L, Lk) and out.dtype == mx.float32
    ref = _ref(q, k, w)
    assert np.abs(np.array(out)[0] - ref).max() <= 2e-6 * np.abs(ref).max()


def test_rows_do_not_depend_on_how_many_queries_share_the_call():
    """A warm turn prefills a short tail (zero-padded to one tile) — its rows equal the cold chunk's same rows."""
    q, k, w = _inputs(2048, 5000, seed=7)
    full = dsa_index_scores(q, k, w)
    for a, b in ((2038, 2048), (1000, 1033), (0, 64), (5, 9)):
        part = dsa_index_scores(q[:, :, a:b], k, w[:, a:b])
        assert mx.array_equal(part, full[:, a:b]).item(), (a, b)


def test_decode_and_unsupported_inputs_fall_back():
    q, k, w = _inputs(1, 5000)
    assert dsa_index_scores(q, k, w) is None                                   # decode: stock GEMV
    q, k, w = _inputs(64, 5000)
    assert dsa_index_scores(q.astype(mx.bfloat16), k, w) is None
    assert dsa_index_scores(q, mx.concatenate([k, k], axis=1), w) is None      # >1 key head
    assert dsa_index_scores(q, k, w, enabled=False) is None


def test_naive_indexer_uses_it_for_prefill_only(monkeypatch):
    import importlib
    import inspect

    from vmlx_engine.models.naive_n05_flash.register import register_naive_n05_flash_runtime
    register_naive_n05_flash_runtime()
    N = importlib.import_module("mlx_lm.models.naive_n05_flash")
    assert N.dsa_index_scores is dsa_index_scores
    src = inspect.getsource(N.Indexer.scores)
    assert "if L > 1:" in src and "dsa_index_scores(q, k, w" in src
    from vmlx_engine.utils.naive_prefill_policy import naive_padded_prefill_identity
    assert "dsa_fused_index_scores_v1" in (naive_padded_prefill_identity() or "")
    monkeypatch.setenv("VMLX_DSA_FUSED_INDEX_SCORES", "0")
    assert "dsa_fused_index_scores_v1" not in (naive_padded_prefill_identity() or "")


def test_glm_pool_variant_matches_fp64_and_the_model_uses_it_for_prefill_chunks():
    from vmlx_engine.metal.dsa_index_scores import dsa_pool_index_scores
    H, D, scale = 32, 128, 128 ** -0.5
    for S, P in ((100, 999), (64, 32), (2047, 600)):
        mx.random.seed(S + P)
        q = mx.random.normal((1, S, H, D)); pk = mx.random.normal((1, P, D)); hw = mx.random.normal((1, S, H)) * 0.2
        out = dsa_pool_index_scores(q, pk, hw, scale)
        qn, kn, wn = (np.array(x).astype(np.float64) for x in (q, pk, hw))
        ref = np.einsum("sh,shp->sp", wn[0], np.maximum(np.einsum("shd,pd->shp", qn[0], kn[0]) * scale, 0))
        assert out.shape == (1, S, P)
        assert np.abs(np.array(out)[0] - ref).max() <= 2e-6 * np.abs(ref).max()
    import inspect
    import vmlx_engine.models.glm5_next.glm5_next as G
    src = inspect.getsource(G.Glm5NextIndexer.topk_indices)
    assert "if S >= 64:" in src and "dsa_pool_index_scores(" in src
    from vmlx_engine.prefix_cache import runtime_cache_fingerprint
    assert "dsa_fused_index_scores_v1" in runtime_cache_fingerprint()


def test_short_calls_use_16_row_tiles_with_identical_rows(monkeypatch):
    """Calls under 32 rows run 16-row tiles (1.3-2.8x faster per short tail); every row equals the 64-row tile's."""
    import vmlx_engine.metal.dsa_index_scores as m
    assert [m._rows_frags(L) for L in (2, 16, 31, 32, 63, 64, 2048)] == [1, 1, 1, 4, 4, 4, 4]
    q, k, w = _inputs(31, 4099, seed=3)
    sized = dsa_index_scores(q, k, w)
    monkeypatch.setattr(m, "_rows_frags", lambda L: 4)
    assert mx.array_equal(sized, dsa_index_scores(q, k, w)).item()
