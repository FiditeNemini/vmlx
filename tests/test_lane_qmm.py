"""Lane matmul (metal/lane_qmm.py): row-independent, accurate at every JANG_4D width, tiled == untiled, safe routing."""
import mlx.core as mx
import mlx.nn as nn
import pytest

from vmlx_engine.metal import lane_qmm as L

pytestmark = pytest.mark.skipif(not L.available(), reason="needs Metal 4 tensor ops (M5-class GPU)")


def _weight(n, k, bits):
    w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=64, bits=bits)
    return q, s.astype(mx.bfloat16), b.astype(mx.bfloat16)


@pytest.mark.parametrize("bits", [4, 5, 6, 8])
def test_rows_match_solo_calls_and_reference(bits):
    mx.random.seed(bits)
    q, s, b = _weight(256, 512, bits)
    sbt = L.pack_scales(s, b)
    x = mx.random.normal((16, 512)).astype(mx.bfloat16)
    many = L.lane_matmul(x, q, sbt, bits=bits)
    solo = mx.concatenate([L.lane_matmul(x[i:i + 1], q, sbt, bits=bits) for i in range(16)], axis=0)
    assert mx.array_equal(many, solo)                       # bits never depend on the row count
    for m in (5, 8, 12):
        assert mx.array_equal(L.lane_matmul(x[:m], q, sbt, bits=bits), many[:m])
    ref = x.astype(mx.float32) @ mx.dequantize(q, s, b, group_size=64, bits=bits).astype(mx.float32).T
    mlx = mx.quantized_matmul(x, q, s, b, transpose=True, group_size=64, bits=bits).astype(mx.float32)
    lane_err = mx.abs(many.astype(mx.float32) - ref).max().item()
    mlx_err = mx.abs(mlx - ref).max().item()
    assert lane_err <= 2 * mlx_err + 1e-3


@pytest.mark.parametrize("bits", [4, 8])
def test_tiled_layout_gives_identical_bits(bits):
    q, s, b = _weight(128, 256, bits)
    sbt = L.pack_scales(s, b)
    x = mx.random.normal((7, 256)).astype(mx.bfloat16)
    tiled = L.tile_weight(q, bits=bits)
    assert mx.array_equal(L.untile_weight(tiled, bits=bits), q)
    assert mx.array_equal(L.lane_matmul(x, tiled, sbt, bits=bits, tiled=True), L.lane_matmul(x, q, sbt, bits=bits))


def test_install_routes_only_plain_affine_layers_and_wide_calls_stay_mlx_exact():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(256, 128, bias=True)
            self.f = nn.Linear(256, 64, bias=False)            # left unquantized
    m = M()
    nn.quantize(m, group_size=64, bits=4, class_predicate=lambda p, mod: p == "a")
    m.a.scales = m.a.scales.astype(mx.bfloat16)
    m.a.biases = m.a.biases.astype(mx.bfloat16)
    wide = mx.random.normal((1, 300, 256)).astype(mx.bfloat16)  # > MAX_ROWS: prefill chunk
    ref_wide = m.a(wide)
    x = mx.random.normal((1, 6, 256)).astype(mx.bfloat16)
    counts = L.install(m)
    assert counts["lane"] == 1 and counts["tiled"] == 1
    assert isinstance(m.a, L.LaneQuantizedLinear) and type(m.f) is nn.Linear
    assert mx.array_equal(m.a(wide), ref_wide)                 # untiled on the fly, MLX's own kernel
    y = m.a(x)
    assert y.shape == (1, 6, 128)
    assert L.install(m)["lane"] == 0                           # idempotent


def test_install_keeps_vocabulary_heads_on_mlx(monkeypatch):
    """lm_head-sized projections (N >= VOCAB_HEAD_MIN_N) stay on MLX: the lane kernel measured slower at 1/5/16 rows
    (N=248,320 K=5,120 8-bit: 3.46 vs 2.54 ms at 1 row). VMLX_LANE_VOCAB_HEAD=1 restores the lane for A/B."""
    def model():
        m = nn.Sequential(nn.Linear(256, 256, bias=False), nn.Linear(256, L.VOCAB_HEAD_MIN_N, bias=False))
        nn.quantize(m, group_size=64, bits=8)
        for layer in m.layers:
            layer.scales = layer.scales.astype(mx.bfloat16)
            layer.biases = layer.biases.astype(mx.bfloat16)
        return m

    m = model()
    counts = L.install(m)
    assert counts["lane"] == 1 and counts["skipped"].get("vocab-sized head") == 1
    assert type(m.layers[1]) is nn.QuantizedLinear and isinstance(m.layers[0], L.LaneQuantizedLinear)
    monkeypatch.setenv("VMLX_LANE_VOCAB_HEAD", "1")
    m2 = model()
    assert L.install(m2)["lane"] == 2


def test_untiled_single_row_takes_mlx_qmv_exactly():
    """VMLX_LANE_QMM_TILED=0 keeps MLX layout; then 1-row calls (AR steps) use MLX qmv bit-for-bit, wider calls the lane."""
    m = nn.Sequential(nn.Linear(256, 128, bias=False))
    nn.quantize(m, group_size=64, bits=4)
    m.layers[0].scales = m.layers[0].scales.astype(mx.bfloat16)
    m.layers[0].biases = m.layers[0].biases.astype(mx.bfloat16)
    q = m.layers[0]
    x1 = mx.random.normal((1, 256)).astype(mx.bfloat16)
    ref = mx.quantized_matmul(x1, q.weight, q.scales, q.biases, transpose=True, group_size=64, bits=4)
    assert L.install(m, tile=False)["tiled"] == 0
    assert mx.array_equal(m.layers[0](x1), ref)
    assert m.layers[0](mx.random.normal((6, 256)).astype(mx.bfloat16)).shape == (6, 128)
