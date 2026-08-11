"""Parity tests for the DSV4 fused indexed-decode attention kernel (D1).

Real DSV4-Flash decode shapes: batch=1, q_len=1, H=64 query heads over the
single shared MQA KV latent, head_dim=512 (64 rope dims folded inside — the
kernel treats the full 512-dim latent row as both K and V, exactly like the
stock decode branch). The reference below is the stock decode math inline
with plain mx ops: gather the selected pool rows, concatenate with the local
sliding-window rows, fp32 softmax(qK^T)V with the attention sink folded into
the denominator (zero-value sink logit) — the same semantics as
``mx.fast.scaled_dot_product_attention(..., sinks=...)`` over the gathered KV.

Split-K coverage: the kernel partitions the R+K row space into 256-row
chunks, so cases below cross the tile boundary (multiple chunks), land
mid-tile (N not a multiple of 256), and stay within a single tile.
"""

import mlx.core as mx
import pytest

from jang_tools.dsv4 import indexed_decode_attention as ida

H = 64
D = 512
SCALE = float(D) ** -0.5
ATOL = 1e-2  # bf16 output ulp ~ 2^-8 on O(1) values; fp32 accumulation inside
RTOL = 2.5e-2  # matches the module's own self-test bar


def _reference(q, kv2d, pool2d, topk1d, sinks32, scale=SCALE):
    """Stock decode branch, inline: gather + SDPA-with-sinks in fp32."""
    q32 = q.astype(mx.float32)
    gathered = pool2d.astype(mx.float32)[topk1d]
    keys = mx.concatenate([kv2d.astype(mx.float32), gathered], axis=0)
    scores = (q32 @ keys.T[None, None]) * scale  # (1, H, 1, R+K)
    sink = sinks32.reshape(1, H, 1, 1)
    m = mx.maximum(scores.max(axis=-1, keepdims=True), sink)
    p = mx.exp(scores - m)
    denom = p.sum(axis=-1, keepdims=True) + mx.exp(sink - m)
    return ((p @ keys[None, None]) / denom).astype(q.dtype)


def _inputs(rows, pool_rows, k, dtype, seed, *, sorted_idx=False, duplicates=False):
    mx.random.seed(seed)
    q = (mx.random.normal((1, H, 1, D)) * 0.3).astype(dtype)
    kv2d = (mx.random.normal((rows, D)) * 0.3).astype(dtype)
    pool2d = (mx.random.normal((pool_rows, D)) * 0.3).astype(dtype)
    if duplicates:
        idx = mx.random.randint(0, pool_rows, (k,)).astype(mx.int32)
    else:
        perm = mx.argsort(mx.random.uniform(shape=(pool_rows,)))
        idx = perm[:k].astype(mx.int32)
    if sorted_idx:
        idx = mx.sort(idx)
    sinks32 = (mx.random.normal((H,)) * 0.5).astype(mx.float32)
    return q, kv2d, pool2d, idx, sinks32


def _assert_parity(got, ref):
    got32 = got.astype(mx.float32)
    ref32 = ref.astype(mx.float32)
    mx.eval(got32, ref32)
    max_abs = float(mx.abs(got32 - ref32).max())
    rel = max_abs / max(float(mx.abs(ref32).max()), 1e-6)
    assert got.shape == ref.shape == (1, H, 1, D)
    assert max_abs < ATOL, f"max abs diff {max_abs:.3e} >= {ATOL:.0e}"
    assert rel < RTOL, f"rel diff {rel:.3e} >= {RTOL:.0e}"


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize(
    "rows,pool_rows,k,sorted_idx,duplicates",
    [
        # Real decode shape: full index_topk=2048 selection over a bigger
        # pool + 128 local window rows; N=2176 -> 9 split-K chunks, last
        # chunk partial (2176 = 8*256 + 128). Unsorted argpartition order.
        (128, 4096, 2048, False, False),
        # Same shape, sorted selection.
        (128, 4096, 2048, True, False),
        # Fewer selected than index_topk, well under one split-K tile.
        (5, 64, 37, False, False),
        # Selection length not a multiple of the split-K tile (N=428 -> 2
        # chunks of 256 + 172).
        (128, 1024, 300, False, False),
        # No local rows at all; N=513 -> chunk boundary + 1.
        (0, 700, 513, True, False),
        # Duplicate indices: stock gather attends the row twice; so must we.
        (16, 256, 96, False, True),
    ],
)
def test_kernel_matches_reference(dtype, rows, pool_rows, k, sorted_idx, duplicates):
    q, kv2d, pool2d, idx, sinks32 = _inputs(
        rows, pool_rows, k, dtype, seed=1234 + rows + k,
        sorted_idx=sorted_idx, duplicates=duplicates,
    )
    got = ida._run_kernel(q, kv2d, pool2d, idx, sinks32, scale=SCALE)
    ref = _reference(q, kv2d, pool2d, idx, sinks32)
    _assert_parity(got, ref)


def test_wrapper_enabled_matches_reference(monkeypatch):
    monkeypatch.setenv("VMLX_DSV4_INDEXED_DECODE", "1")
    rows, pool_rows, k = 128, 2048, 600
    q, kv2d, pool2d, idx, sinks32 = _inputs(rows, pool_rows, k, mx.bfloat16, seed=77)
    local_kv = kv2d.reshape(1, 1, rows, D)
    pooled = pool2d.reshape(1, pool_rows, D)
    topk = idx.reshape(1, 1, k)
    sinks = sinks32.astype(mx.bfloat16)  # module param dtype; wrapper re-casts
    got = ida.dsv4_indexed_decode_attention(
        q, local_kv, pooled, topk, scale=SCALE, sinks=sinks
    )
    assert got is not None, ida.dsv4_indexed_decode_status()
    status = ida.dsv4_indexed_decode_status()
    assert status["self_test"] == "passed"
    ref = _reference(q, kv2d, pool2d, idx, sinks.astype(mx.float32))
    _assert_parity(got, ref)


def test_wrapper_default_off(monkeypatch):
    monkeypatch.delenv("VMLX_DSV4_INDEXED_DECODE", raising=False)
    rows, pool_rows, k = 8, 32, 16
    q, kv2d, pool2d, idx, sinks32 = _inputs(rows, pool_rows, k, mx.bfloat16, seed=3)
    out = ida.dsv4_indexed_decode_attention(
        q,
        kv2d.reshape(1, 1, rows, D),
        pool2d.reshape(1, pool_rows, D),
        idx.reshape(1, 1, k),
        scale=SCALE,
        sinks=sinks32,
    )
    assert out is None


def test_wrapper_rejects_unsupported_layouts(monkeypatch):
    monkeypatch.setenv("VMLX_DSV4_INDEXED_DECODE", "1")
    rows, pool_rows, k = 8, 32, 16
    q, kv2d, pool2d, idx, sinks32 = _inputs(rows, pool_rows, k, mx.bfloat16, seed=5)
    local_kv = kv2d.reshape(1, 1, rows, D)
    pooled = pool2d.reshape(1, pool_rows, D)
    topk = idx.reshape(1, 1, k)

    # fp32 activations -> stock path.
    assert (
        ida.dsv4_indexed_decode_attention(
            q.astype(mx.float32), local_kv, pooled, topk,
            scale=SCALE, sinks=sinks32,
        )
        is None
    )
    # q_len != 1 -> stock path (prefill owns multi-token shapes).
    q2 = mx.concatenate([q, q], axis=2)
    topk2 = mx.concatenate([topk, topk], axis=1)
    assert (
        ida.dsv4_indexed_decode_attention(
            q2, local_kv, pooled, topk2, scale=SCALE, sinks=sinks32
        )
        is None
    )
    # Empty selection -> stock path.
    assert (
        ida.dsv4_indexed_decode_attention(
            q, local_kv, pooled, topk[:, :, :0], scale=SCALE, sinks=sinks32
        )
        is None
    )
    # Quantized pool views must stay on the tiled q8 decode path.
    class _QuantView:
        is_dsv4_quantized_pool_view = True
        ndim = 3
        shape = (1, pool_rows, D)

    assert (
        ida.dsv4_indexed_decode_attention(
            q, local_kv, _QuantView(), topk, scale=SCALE, sinks=sinks32
        )
        is None
    )


def test_out_of_range_indices_are_skipped(monkeypatch):
    """Defensive guard: invalid selections drop out instead of faulting."""
    monkeypatch.setenv("VMLX_DSV4_INDEXED_DECODE", "1")
    rows, pool_rows, k = 12, 40, 24
    q, kv2d, pool2d, idx, sinks32 = _inputs(rows, pool_rows, k, mx.bfloat16, seed=9)
    bad = mx.concatenate(
        [idx, mx.array([-1, pool_rows, pool_rows + 7], dtype=mx.int32)]
    )
    got = ida._run_kernel(q, kv2d, pool2d, bad, sinks32, scale=SCALE)
    # Reference over the valid prefix only: invalid rows contribute nothing.
    ref = _reference(q, kv2d, pool2d, idx, sinks32)
    _assert_parity(got, ref)


def test_model_wiring_imports_kernel():
    from jang_tools.dsv4 import mlx_model

    assert (
        mlx_model.dsv4_indexed_decode_attention
        is ida.dsv4_indexed_decode_attention
    )
