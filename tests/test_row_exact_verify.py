"""Row-exact native-MTP verification.

A verify forward over S rows must give every row the logits that the same
token gets from an ordinary single-token decode step, bit for bit, so greedy
native MTP reproduces AR output exactly. MLX switches projection, dense
matmul, rotary and SDPA kernels between one row and several rows; the verify
scope keeps every row on the single-row arithmetic.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from vmlx_engine.metal import row_exact_qmv as R


def _quantized(out_dim, in_dim, bits, dtype, group_size=64):
    weight = mx.random.normal((out_dim, in_dim), key=mx.random.key(out_dim * 31 + in_dim)).astype(dtype)
    w, scales, biases = mx.quantize(weight, group_size=group_size, bits=bits)
    return w, scales.astype(dtype), biases.astype(dtype)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("shape", [(1024, 2048), (2560, 640), (1028, 1344), (12, 640), (96, 64), (997, 128), (5, 64), (4, 256), (1, 512)])
@pytest.mark.parametrize("rows", [2, 3, 4])
def test_each_row_equals_its_single_row_quantized_matmul(bits, dtype, shape, rows):
    out_dim, in_dim = shape
    group_size = 64 if in_dim % 64 == 0 else 32
    w, scales, biases = _quantized(out_dim, in_dim, bits, dtype, group_size)
    x = mx.random.normal((1, rows, in_dim), key=mx.random.key(7)).astype(dtype)
    got = R.row_exact_qmv(x, w, scales, biases, group_size=group_size, bits=bits)
    if got is None:
        # Only K=64/128 packs without a transcribed MLX M=1 kernel may decline (stock MLX keeps them).
        assert in_dim in (64, 128) and bits not in (2, 4, 8)
        pytest.skip("no row-exact transcription for this MLX M=1 kernel")
    want = mx.concatenate([
        mx.quantized_matmul(x[:, r:r + 1], w, scales, biases, transpose=True, group_size=group_size, bits=bits)
        for r in range(rows)
    ], axis=1)
    assert got.shape == want.shape
    assert bool(mx.array_equal(got, want).item())


def test_unsupported_inputs_fall_back_to_stock():
    w, scales, biases = _quantized(256, 512, 4, mx.float16)
    one_row = mx.zeros((1, 1, 512), dtype=mx.float16)
    too_many = mx.zeros((1, R.MAX_ROWS + 1, 512), dtype=mx.float16)
    assert R.row_exact_qmv(one_row, w, scales, biases, group_size=64, bits=4) is None
    assert R.row_exact_qmv(too_many, w, scales, biases, group_size=64, bits=4) is None
    assert R.row_exact_qmv(mx.zeros((1, 2, 512), dtype=mx.float16), w, scales, None,
                           group_size=64, bits=4, mode="mxfp4") is None


def test_missing_mlx_header_disables_the_kernel(monkeypatch):
    monkeypatch.setattr(R, "_mlx_qmv_header", lambda: None)
    w, scales, biases = _quantized(256, 512, 4, mx.float16)
    assert R.row_exact_qmv(mx.zeros((1, 2, 512), dtype=mx.float16), w, scales, biases,
                           group_size=64, bits=4) is None


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("shape", [(36, 10240), (10240, 32), (512, 2560), (4, 1024)])
def test_dense_linear_rows_equal_single_row_gemv(dtype, shape):
    out_dim, in_dim = shape
    weight = (mx.random.normal((out_dim, in_dim), key=mx.random.key(3)) * 0.02).astype(dtype)
    x = mx.random.normal((1, 4, in_dim), key=mx.random.key(5)).astype(dtype)
    got = R.row_exact_linear(x, weight)
    assert got is not None
    want = mx.concatenate([x[:, r:r + 1] @ weight.T for r in range(4)], axis=1)
    assert bool(mx.array_equal(got, want).item())


def test_scope_routes_quantized_and_dense_linears_only_inside_verify(monkeypatch):
    monkeypatch.delenv("VMLX_ROW_EXACT_VERIFY_QMV", raising=False)
    q = nn.QuantizedLinear(512, 256, bias=False, group_size=64, bits=4)
    q.set_dtype(mx.float16)  # the kernel serves fp16/bf16 activations
    d = nn.Linear(512, 256, bias=False)
    x = mx.random.normal((1, 3, 512), key=mx.random.key(9))
    xq = x.astype(mx.float16)
    calls = dict(R.STATS)
    with R.row_exact_verify_scope() as active:
        assert active and R.row_exact_scope_active()
        q(xq), d(x)
    assert not R.row_exact_scope_active()
    assert R.STATS["calls"] == calls.get("calls", 0) + 1
    assert R.STATS.get("dense_calls", 0) == calls.get("dense_calls", 0) + 1
    q(xq), d(x)  # outside the scope: stock MLX, no new kernel calls
    assert R.STATS["calls"] == calls.get("calls", 0) + 1


def test_scope_opt_out(monkeypatch):
    monkeypatch.setenv("VMLX_ROW_EXACT_VERIFY_QMV", "0")
    with R.row_exact_verify_scope() as active:
        assert active is False and not R.row_exact_scope_active()


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("width", [2, 3, 4])
def test_tiny_qwen4_verify_rows_equal_decode_steps(bits, width):
    from tests.test_qwen4_exp_runtime import _randomize, _tiny_args
    from vmlx_engine.models.qwen4_exp.language import LanguageModel

    model = LanguageModel(_tiny_args())
    _randomize(model)
    nn.quantize(model, group_size=32, bits=bits,
                class_predicate=lambda _p, m: isinstance(m, nn.Linear) and m.weight.shape[-1] % 32 == 0)
    model.set_dtype(mx.float16)
    mx.eval(model.parameters())
    prompt = mx.array([[5, 17, 300, 41, 2, 999 % 997, 63, 128, 7, 11, 501, 33, 90]])

    def prefill():
        cache = model.make_cache()
        logits = model(prompt, cache=cache).logits
        nxt = int(mx.argmax(logits[:, -1], -1).item())
        return cache, nxt

    cache, token = prefill()
    chain, steps = [token], []
    for _ in range(width):
        logits = model(mx.array([[chain[-1]]]), cache=cache).logits[:, -1]
        mx.eval(logits)
        steps.append(logits)
        chain.append(int(mx.argmax(logits, -1).item()))

    cache, again = prefill()
    assert again == token
    with R.row_exact_verify_scope():
        verify = model(mx.array([chain[:width]]), cache=cache, n_confirmed=1).logits
    mx.eval(verify)
    for r in range(width):
        assert bool(mx.array_equal(verify[:, r], steps[r]).item()), f"row {r} differs from its decode step"
