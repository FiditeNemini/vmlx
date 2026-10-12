"""One-dispatch KDA decode (metal/glm5_kda_fused_decode) == the multi-dispatch stock decode path.

Real head size (K=128), fewer heads; random weights incl. conv / A_log / dt_bias (zero at init would hide conv and
gate errors). A stock prefill builds real conv + recurrent states; then several decode steps run through both paths
from identical cache copies. Compared: o_proj output, recurrent state, three conv states.
"""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from vmlx_engine.models.glm5_next.glm5_next import Glm5KDACache, KDAAttention, ModelArgs

H, K, D = 4, 128, 256


def _layer(seed=0):
    mx.random.seed(seed)
    args = ModelArgs(hidden_size=D, linear_num_heads=H, linear_head_dim=K, num_hidden_layers=1, vocab_size=64,
                     layer_types=["linear_attention"])
    a = KDAAttention(args)
    for n in ("q_conv1d", "k_conv1d", "v_conv1d"):
        setattr(a, n, (mx.random.normal(getattr(a, n).shape) * 0.5).astype(mx.bfloat16))
    a.A_log = mx.random.normal(a.A_log.shape) * 0.5
    a.dt_bias = mx.random.normal(a.dt_bias.shape) * 0.5
    a.o_norm = (1 + 0.2 * mx.random.normal(a.o_norm.shape)).astype(mx.bfloat16)
    for m in ("q_proj", "k_proj", "v_proj", "o_proj", "b_proj", "f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj"):
        lin = getattr(a, m)
        lin.weight = (mx.random.normal(lin.weight.shape) * lin.weight.shape[1] ** -0.5).astype(mx.bfloat16)
    mx.eval(a.parameters())
    return a


def _copy(c):
    n = Glm5KDACache()
    n.cache = [None if x is None else mx.array(x) for x in c.cache]
    return n


def _run(a, x, cache, fused, monkeypatch):
    monkeypatch.setenv("VMLX_GLM5_FUSED_KDA_DECODE", "1" if fused else "0")
    y = a(x, cache=cache)
    mx.eval(y, cache.cache)
    return y


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fused_decode_matches_stock(monkeypatch, seed):
    a = _layer(seed)
    xs = (mx.random.normal((1, 9, D)) * 2).astype(mx.bfloat16)
    c0 = Glm5KDACache()
    _run(a, xs[:, :5], c0, False, monkeypatch)                 # stock prefill -> real states
    cs, cf = _copy(c0), _copy(c0)
    for t in range(5, 9):
        ys = _run(a, xs[:, t:t + 1], cs, False, monkeypatch)
        yf = _run(a, xs[:, t:t + 1], cf, True, monkeypatch)
        ys_, yf_ = np.array(ys.astype(mx.float32)), np.array(yf.astype(mx.float32))
        assert np.max(np.abs(ys_ - yf_)) <= 8e-3 * np.max(np.abs(ys_)), (t, np.max(np.abs(ys_ - yf_)))   # <= 1 bf16 ULP
        st, sf = np.array(cs.cache[3]), np.array(cf.cache[3])
        assert np.max(np.abs(st - sf)) <= 1e-5 * np.max(np.abs(st))   # fp32 summation order (measured 1.5e-7)
        for i in range(3):                                      # conv tails are exact copies of inputs
            assert np.array_equal(np.array(cs.cache[i].astype(mx.float32)), np.array(cf.cache[i].astype(mx.float32)))


def test_fused_path_is_taken(monkeypatch):
    """Fail closed: the comparison is vacuous if the fused kernel never runs."""
    from vmlx_engine.metal import glm5_kda_fused_decode as F
    a = _layer(3)
    c = Glm5KDACache()
    x = mx.random.normal((1, 4, D)).astype(mx.bfloat16)
    _run(a, x[:, :3], c, False, monkeypatch)
    n0 = F._OBSERVED
    _run(a, x[:, 3:], c, True, monkeypatch)
    assert F._OBSERVED == n0 + 1


@pytest.mark.parametrize("T", [4, 7])
def test_fused_verify_slab_matches_vectorized(monkeypatch, T):
    """Verify slab (n_confirmed=1): output, final state and EVERY recorded rollback state match the vectorized path."""
    from vmlx_engine.metal import glm5_kda_fused_decode as F
    a = _layer(10 + T)
    xs = (mx.random.normal((1, 6 + T, D)) * 2).astype(mx.bfloat16)
    c0 = Glm5KDACache()
    _run(a, xs[:, :6], c0, False, monkeypatch)
    cs, cf = _copy(c0), _copy(c0)
    monkeypatch.setenv("VMLX_GLM5_FUSED_KDA_DECODE", "0")
    ys = a(xs[:, 6:], cache=cs, n_confirmed=1); mx.eval(ys, cs.cache)
    monkeypatch.setenv("VMLX_GLM5_FUSED_KDA_DECODE", "1")
    n0 = F._OBSERVED
    yf = a(xs[:, 6:], cache=cf, n_confirmed=1); mx.eval(yf, cf.cache)
    assert F._OBSERVED == n0 + 1                              # fail closed: the fused verify kernel ran
    ys_, yf_ = np.array(ys.astype(mx.float32)), np.array(yf.astype(mx.float32))
    assert np.max(np.abs(ys_ - yf_)) <= 8e-3 * np.max(np.abs(ys_)), np.max(np.abs(ys_ - yf_))
    for i in range(4):
        a_, b_ = np.array(cs.cache[i].astype(mx.float32)), np.array(cf.cache[i].astype(mx.float32))
        assert np.max(np.abs(a_ - b_)) <= 1e-5 * max(np.max(np.abs(a_)), 1e-6)
    ss, sf = cs._speculative_states, cf._speculative_states
    assert len(ss) == len(sf) == T
    for t in range(T):
        for i in range(4):
            a_, b_ = np.array(ss[t][i].astype(mx.float32)), np.array(sf[t][i].astype(mx.float32))
            assert np.max(np.abs(a_ - b_)) <= 1e-5 * max(np.max(np.abs(a_)), 1e-6), (t, i)
    # a partial rejection restores the same boundary in both
    assert cs.rollback_speculative(2) and cf.rollback_speculative(2)
    for i in range(4):
        assert np.allclose(np.array(cs.cache[i].astype(mx.float32)), np.array(cf.cache[i].astype(mx.float32)),
                           rtol=1e-5, atol=1e-6)


def _raw_inputs(T, Hh, seed, W=4):
    mx.random.seed(seed)
    C, bf = Hh * K, mx.bfloat16
    q, k, v = (mx.random.normal((1, T, C)).astype(bf) for _ in range(3))
    cq, ck, cv = (mx.random.normal((1, W - 1, C)).astype(bf) for _ in range(3))
    wq, wk, wv = ((mx.random.normal((C, W)) * 0.5).astype(bf) for _ in range(3))
    f_raw = mx.random.normal((1, T, C)).astype(bf)
    dt_bias, A_log = mx.random.normal((C,)) * 0.1, mx.random.normal((Hh,)) * 0.3
    b_raw, gate_raw = mx.random.normal((1, T, Hh)).astype(bf), mx.random.normal((1, T, C)).astype(bf)
    o_norm = (1 + 0.1 * mx.random.normal((K,))).astype(bf)
    state = mx.random.normal((1, Hh, K, K)) * 0.05
    arrs = [q, k, v, cq, ck, cv, wq, wk, wv, f_raw, dt_bias, A_log, b_raw, gate_raw, o_norm, state]
    mx.eval(arrs)
    return arrs


@pytest.mark.parametrize("T,Hh,seed", [(8, 4, 1), (16, 4, 2), (11, 4, 3), (16, 64, 4)])
def test_replay_records_give_bit_identical_rows(T, Hh, seed):
    """Records variant (no per-row state stores) == per-row-state kernel: gated, final state, and the state replayed
    after EVERY row prefix, bit for bit."""
    from vmlx_engine.metal import glm5_kda_fused_decode as F
    arrs = _raw_inputs(T, Hh, seed)
    kw = dict(heads=Hh, key_dim=K, lower_bound=-5.0, rms_eps=1e-6)
    gated, states = F.glm5_kda_fused_verify(*arrs, **kw)
    g2, records, final = F.glm5_kda_fused_verify_records(*arrs, **kw)
    assert mx.array_equal(g2, gated).item() and mx.array_equal(final, states[T - 1]).item()
    for rows in range(1, T + 1):
        assert mx.array_equal(F.glm5_kda_replay(arrs[15], records, rows), states[rows - 1]).item(), rows


def test_per_row_state_kernel_source_is_unchanged():
    """The records variant is generated from the same text; the per-row kernel's source must stay byte-identical (its
    bits are what the AR decode kernel and earlier evidence were qualified against)."""
    import hashlib
    from vmlx_engine.metal import glm5_kda_fused_decode as F
    src = F._verify_source(64, 128, 4, -5.0, 1e-6, records=False)
    assert hashlib.sha256(src.encode()).hexdigest()[:16] == "0ecd5350eb15b373"


@pytest.mark.parametrize("T", [8, 10])
def test_wide_verify_slab_replays_bit_identical_rollback_states(monkeypatch, T):
    from vmlx_engine.metal import glm5_kda_fused_decode as F
    a = _layer(30 + T)
    xs = (mx.random.normal((1, 6 + T, D)) * 2).astype(mx.bfloat16)
    c0 = Glm5KDACache()
    _run(a, xs[:, :6], c0, False, monkeypatch)
    monkeypatch.setenv("VMLX_GLM5_FUSED_KDA_DECODE", "1")
    outs = {}
    for mode in ("0", "1"):
        monkeypatch.setenv("VMLX_GLM5_KDA_VERIFY_REPLAY", mode)
        c = _copy(c0)
        n0 = F._OBSERVED
        y = a(xs[:, 6:], cache=c, n_confirmed=1)
        mx.eval(y, c.cache)
        assert F._OBSERVED == n0 + 1                                   # fail closed: a fused verify kernel ran
        assert isinstance(c._speculative_states, F.ReplayedKDAStates) == (mode == "1")
        outs[mode] = (y, c)
    (y0, c0_), (y1, c1_) = outs["0"], outs["1"]
    assert mx.array_equal(y0, y1).item()
    for i in range(4):
        assert mx.array_equal(c0_.cache[i], c1_.cache[i]).item()
    for t in range(T):
        for i in range(4):
            assert mx.array_equal(c0_._speculative_states[t][i], c1_._speculative_states[t][i]).item(), (t, i)
    for rejected in (0, 3, T - 1):
        r0, r1 = _copy(c0_), _copy(c1_)
        r0._speculative_states, r1._speculative_states = c0_._speculative_states, c1_._speculative_states
        assert r0.rollback_speculative(rejected) and r1.rollback_speculative(rejected)
        mx.eval(r0.cache, r1.cache)
        for i in range(4):
            assert mx.array_equal(r0.cache[i], r1.cache[i]).item(), (rejected, i)


def test_narrow_slabs_keep_per_row_states(monkeypatch):
    from vmlx_engine.metal import glm5_kda_fused_decode as F
    a = _layer(41)
    xs = (mx.random.normal((1, 6 + 5, D)) * 2).astype(mx.bfloat16)
    c = Glm5KDACache()
    _run(a, xs[:, :6], c, False, monkeypatch)
    monkeypatch.setenv("VMLX_GLM5_FUSED_KDA_DECODE", "1")
    monkeypatch.setenv("VMLX_GLM5_KDA_VERIFY_REPLAY", "1")
    a(xs[:, 6:], cache=c, n_confirmed=1)
    assert isinstance(c._speculative_states, list) and F.REPLAY_MIN_ROWS > 5
