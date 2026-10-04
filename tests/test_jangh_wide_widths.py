"""6/8-bit JANGH: the contract accepts declared wide codebooks, and every routed path (decode, sorted prefill,
weighted prefill, fused weighted unsort, H32 rows) matches a dense float32 reference at qwen4_exp geometry
(D=2560, I=640 -> K=640 exercises the tail-split path; 640 * bits / 32 is not a multiple of the tile at 6 bits)."""
import copy
import importlib.util
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from vmlx_engine.jangh import switch as S
from vmlx_engine.jangh.format import pack_bitstream, codebook

_PATH = Path(__file__).resolve().parents[1] / "vmlx_engine/jangh/contract.py"
_SPEC = importlib.util.spec_from_file_location("jangh_contract_wide", _PATH)
contract = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(contract)


def _books(widths):
    out = {}
    for b in widths:
        a, be = contract.CUBIC_PARAMS[b]
        lv = [contract._f32((i - ((1 << b) - 1) / 2) * (a + be * (i - ((1 << b) - 1) / 2) ** 2)) for i in range(1 << b)]
        out[str(b)] = dict(alpha=a, beta=be, levels=lv)
    return out


def _cfg(widths, gu, dn):
    return {"jangtq": dict(version=2, packing="lsb-bitstream", scale_dtype="float16", codebook_family="odd-cubic",
                           rotation="hadamard32", codebooks=_books(widths)),
            "quantization": {"model.layers.0.mlp.switch_mlp.gate_proj": dict(mode="jangtq2", bits=gu),
                             "model.layers.0.mlp.switch_mlp.up_proj": dict(mode="jangtq2", bits=gu),
                             "model.layers.0.mlp.switch_mlp.down_proj": dict(mode="jangtq2", bits=dn)}}


def test_contract_wide_widths():
    assert contract.projection_contract(_cfg((2, 3, 4, 6), 6, 4))["model.layers.0.mlp.switch_mlp"]["gate_proj"]["bits"] == 6
    assert contract.validate_format(_cfg((2, 3, 4, 6, 8), 8, 8))
    with pytest.raises(ValueError):            # 6-bit projection without a declared 6-bit codebook
        contract.projection_contract(_cfg((2, 3, 4), 6, 6))
    with pytest.raises(ValueError):            # the base widths stay mandatory
        contract.validate_format(_cfg((4, 6), 4, 4))
    with pytest.raises(ValueError):            # 5-bit does not exist
        contract.validate_format({**_cfg((2, 3, 4), 4, 4), "jangtq": {**_cfg((2, 3, 4), 4, 4)["jangtq"],
                                  "codebooks": {**_books((2, 3, 4)), "5": dict(alpha=0.1, beta=0.0, levels=[0.0] * 32)}}})
    cfg = _cfg((2, 3, 4, 6), 6, 6)
    cfg["jangtq"]["codebooks"]["6"]["levels"][7] += 1e-3
    with pytest.raises(ValueError):
        contract.validate_format(cfg)
    with pytest.raises(ValueError):            # a declared width is still checked for bits outside the table
        contract.projection_contract(_cfg((2, 3, 4), 5, 4))


D, I, E, KK = 2560, 640, 12, 10


def _module(gu, dn, seed):
    rng = np.random.default_rng(seed)
    m = S.TQSwitchGLU(D, I, E, gu, dn, 0.0, rotation_gate_up="hadamard32", rotation_down="hadamard32")
    dense = {}
    for name, (N, K) in (("gate_proj", (I, D)), ("up_proj", (I, D)), ("down_proj", (D, I))):
        lin = getattr(m, name)
        codes = rng.integers(0, 1 << lin.bits, (E, N, K)).astype(np.uint8)
        scale = (rng.random((E, N)) * 0.02 + 0.005).astype(np.float16)
        lin.tq2_packed = pack_bitstream(mx.array(codes), lin.bits)
        lin.tq2_scales = mx.array(scale)
        dense[name] = codebook(lin.bits)[codes] * scale.astype(np.float32)[..., None]
    mx.eval(m.parameters())
    return m, dense


def _h32(x):
    x = np.asarray(x, np.float32)
    return np.array(mx.hadamard_transform(mx.array(x).reshape(*x.shape[:-1], x.shape[-1] // 32, 32))).reshape(x.shape)


def _reference(dense, x, inds, scores):
    xr = _h32(x)
    out = np.zeros((x.shape[0], D), np.float32)
    for t in range(x.shape[0]):
        for j in range(KK):
            e = inds[t, j]
            g, u = dense["gate_proj"][e] @ xr[t], dense["up_proj"][e] @ xr[t]
            h = (g / (1 + np.exp(-g))) * u
            out[t] += scores[t, j] * (dense["down_proj"][e] @ _h32(h[None])[0])
    return out


@pytest.mark.parametrize("gu,dn", [(4, 4), (4, 6), (6, 6), (6, 8), (8, 8), (6, 4)])
@pytest.mark.parametrize("T", [1, 3, 40])
@pytest.mark.parametrize("fused", [False, True])
def test_routed_paths_match_dense_reference(gu, dn, T, fused):
    m, dense = _module(gu, dn, seed=gu * 100 + dn * 10 + T)
    if fused:
        for lin in (m, m.gate_proj, m.up_proj, m.down_proj):
            lin.use_weighted_unsort = lin.use_h32_rows = True
    rng = np.random.default_rng(T)
    x = (rng.standard_normal((T, D)) * 0.5).astype(np.float16)
    inds = np.stack([rng.choice(E, KK, replace=False) for _ in range(T)]).astype(np.int32)
    scores = rng.random((T, KK)).astype(np.float32); scores /= scores.sum(-1, keepdims=True)
    ref = _reference(dense, x.astype(np.float32), inds, scores)
    xm = mx.array(x)[None]
    got = np.array(m.routed(xm, mx.array(inds)[None], mx.array(scores.astype(np.float16))[None]).astype(mx.float32))[0]
    err = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert np.isfinite(got).all() and err < 2e-3, (gu, dn, T, fused, err)
    unweighted = np.array(m(xm, mx.array(inds)[None]).astype(mx.float32))[0]           # (T, k, D)
    err2 = np.linalg.norm((unweighted * scores[..., None]).sum(1) - ref) / np.linalg.norm(ref)
    assert err2 < 2e-3, (gu, dn, T, fused, err2)


class TestDecodeVsSortedPathSelection:
    """Routed-path choice by TOKENS (measured crossover) must be honoured at both call sites
    (routed() and the unweighted __call__), and the legacy row rule must survive for modules
    without a loader-set limit (GLM/Naive geometry stays exactly as qualified)."""

    @staticmethod
    def _paths_taken(m, T, monkeypatch):
        taken = []
        real_prefill, real_decode = m._prefill, m._decode_h

        def spy_prefill(*args, **kwargs):
            taken.append("sorted"); return real_prefill(*args, **kwargs)

        def spy_decode(*args, **kwargs):
            taken.append("decode"); return real_decode(*args, **kwargs)

        monkeypatch.setattr(m, "_prefill", spy_prefill)
        monkeypatch.setattr(m, "_decode_h", spy_decode)
        rng = np.random.default_rng(T)
        x = mx.array((rng.standard_normal((T, D)) * 0.5).astype(np.float16))[None]
        inds = mx.array(np.stack([rng.choice(E, KK, replace=False) for _ in range(T)]).astype(np.int32))[None]
        scores = mx.array((np.ones((T, KK)) / KK).astype(np.float16))[None]
        mx.eval(m.routed(x, inds, scores)); routed = taken[-1]
        taken.clear()
        mx.eval(m(x, inds)); unweighted = taken[-1]
        return routed, unweighted

    def test_legacy_row_rule_without_limit(self, monkeypatch):
        m, _ = _module(4, 6, seed=1)
        assert not hasattr(m, "decode_max_tokens")
        assert self._paths_taken(m, 6, monkeypatch) == ("decode", "decode")   # 60 rows < 64
        assert self._paths_taken(m, 7, monkeypatch) == ("sorted", "sorted")   # 70 rows >= 64

    @pytest.mark.parametrize("limit,T,expected", [(96, 7, "decode"), (96, 96, "decode"), (96, 97, "sorted"), (8, 8, "decode"), (8, 9, "sorted"), (0, 1, "sorted")])
    def test_token_limit_rules_both_call_sites(self, monkeypatch, limit, T, expected):
        m, _ = _module(4, 6, seed=2)
        m.decode_max_tokens = limit
        assert self._paths_taken(m, T, monkeypatch) == (expected, expected)

    def test_selector_arithmetic(self):
        m, _ = _module(4, 4, seed=3)
        assert m._use_sorted(63, KK) is False and m._use_sorted(64, KK) is True
        m.decode_max_tokens = 96
        assert m._use_sorted(96 * KK, KK) is False and m._use_sorted(97 * KK, KK) is True
        assert m._use_sorted(96 * KK + 9, KK) is False   # partial token rows round down

    def test_qwen4_loader_default_matches_measured_crossover(self):
        from vmlx_engine.jangh import runtime_identity as R
        assert R.QWEN4_DECODE_MAX_TOKENS == 96
