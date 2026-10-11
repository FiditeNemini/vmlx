"""KDA prefill chunks above 2048 tokens (NT > 32) keep the qualified WY substitution kernel by slicing to <= 32 chunks
per launch, bit-identical to the stock 63-step loop. Before, NT > 32 sent the whole call to the stock Python loop
(14.4 vs 2.6 ms per layer per 2048 tokens), the KDA half of GLM's 4096-step prefill slowdown (2026-10-10)."""
import pytest

mx = pytest.importorskip("mlx.core")

import vmlx_engine.models.glm5_next.kda as KD  # noqa: E402
import vmlx_engine.metal.glm5_kda_substitution as SUB  # noqa: E402


def _inputs(T, seed, H=64, K=128):
    mx.random.seed(seed)
    q = mx.random.normal((1, T, H, K)).astype(mx.bfloat16)
    k = (mx.random.normal((1, T, H, K)) * 0.1).astype(mx.bfloat16)
    v = mx.random.normal((1, T, H, K)).astype(mx.bfloat16)
    g = -mx.abs(mx.random.normal((1, T, H, K))) * 0.05
    beta = mx.sigmoid(mx.random.normal((1, T, H)))
    return q, k, v, g, beta, None


@pytest.mark.parametrize("T", [2112, 4096])
def test_sliced_substitution_matches_stock_loop(monkeypatch, T):
    if not SUB.kda_substitution_status()["installed"]:
        pytest.skip("qualified KDA substitution kernel not available on this runtime")
    args = _inputs(T, T)
    calls0 = SUB._CALLS
    o1, s1 = KD.kda_chunked(*args); mx.eval(o1, s1)
    assert SUB._CALLS - calls0 == (T // 64 + 31) // 32, "the kernel must run once per <= 32-chunk slice"
    monkeypatch.setattr(KD, "_substitution_sliced", lambda A: None)          # stock 63-step loop
    o0, s0 = KD.kda_chunked(*args); mx.eval(o0, s0)
    assert bool(mx.array_equal(o0, o1)) and bool(mx.array_equal(s0, s1))
