"""Owning opt-in sorted-output epilogue contract; not full-model proof."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_identity_default_frozen_and_validated(monkeypatch):
    path = Path(__file__).parents[1] / 'vmlx_engine/jangh/runtime_identity.py'
    def load():
        spec = importlib.util.spec_from_file_location('prefill_reduce_identity', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    monkeypatch.delenv('JANGH_PREFILL_REDUCE', raising=False)
    baseline = load()
    identity = baseline.runtime_identity()
    assert baseline.PREFILL_REDUCE == '0'
    monkeypatch.setenv('JANGH_PREFILL_REDUCE', '1')
    assert baseline.runtime_identity() == identity
    assert ';prefill_reduce=1' in load().runtime_identity()
    assert load().runtime_identity() != identity
    monkeypatch.setenv('JANGH_PREFILL_REDUCE', 'yes')
    with pytest.raises(ValueError, match='JANGH_PREFILL_REDUCE'):
        load()


@pytest.mark.parametrize("version,metal,device,expected", [
    ("0.32.2", True, "Apple M5 Max", True),
    ("0.32.3", True, "Apple M5 Max", True),
    ("0.32.4", True, "Apple M5 Max", False),
    ("0.33.0", True, "Apple M5 Max", False),
    ("0.32.3", True, "Apple M4 Max", False),
    ("0.32.3", True, "Apple M5 Ultra", False),
    ("0.32.3", False, "Apple M5 Max", False),
])
def test_availability_requires_qualified_version_and_device(
    monkeypatch, version, metal, device, expected
):
    import importlib.metadata
    pytest.importorskip("mlx.core")
    from vmlx_engine.jangh import kernels as K

    monkeypatch.setattr(importlib.metadata, "version", lambda name: version)
    monkeypatch.setattr(K.mx.metal, "is_available", lambda: metal)
    monkeypatch.setattr(K.mx, "device_info", lambda: {"device_name": device})
    K.prefill_weighted_unsort_available.cache_clear()
    try:
        assert K.prefill_weighted_unsort_available() is expected
    finally:
        # Never leave a synthetic device/version result cached for numeric tests.
        K.prefill_weighted_unsort_available.cache_clear()


@pytest.mark.parametrize('rows,strided', [(8, False), (9, True), (33, False)])
def test_fused_matches_native_words(rows, strided):
    mx = pytest.importorskip('mlx.core')
    from vmlx_engine.jangh import kernels as K
    if not K.prefill_weighted_unsort_available():
        pytest.skip('Qualified M5 Max / MLX 0.32.2 or 0.32.3 only')
    mx.random.seed(73017 + rows)
    y = mx.random.normal((rows * 8, 4096)).astype(mx.bfloat16)
    order = mx.argsort(mx.random.randint(0, 256, (rows * 8,)))
    inv = mx.argsort(order).astype(mx.uint32)
    s = mx.random.normal((rows, 16 if strided else 8))
    if strided:
        s = s[:, ::2]
    expected = (y[inv].reshape(rows, 8, 4096) * s[..., None].astype(y.dtype)).sum(axis=-2)
    actual = K.prefill_weighted_unsort(y, inv, s, enabled=True)
    assert actual is not None
    mx.eval(expected, actual)
    assert bool(mx.array_equal(expected.view(mx.uint16), actual.view(mx.uint16)))


def test_unsupported_shapes_and_disabled_do_not_dispatch(monkeypatch):
    mx = pytest.importorskip('mlx.core')
    from vmlx_engine.jangh import kernels as K
    monkeypatch.setattr(K, 'prefill_weighted_unsort_available', lambda: True)
    def forbidden():
        raise AssertionError('unsupported geometry dispatched')
    monkeypatch.setattr(K, '_prefill_weighted_unsort_kernel', forbidden)
    y = mx.zeros((64, 4096), dtype=mx.bfloat16)
    inv = mx.arange(64, dtype=mx.uint32)
    s = mx.zeros((8, 8), dtype=mx.float32)
    assert K.prefill_weighted_unsort(y, inv, s, enabled=False) is None
    for yy, ii, ss in [(y[:63], inv, s), (y, inv[:63], s), (y, inv, s[:, :7]),
                       (y.astype(mx.float16), inv, s), (y[:, :4095], inv, s),
                       (y, inv.astype(mx.int32), s), (y, inv, s.reshape(-1)),
                       (y[:56], inv[:56], s[:7])]:
        assert K.prefill_weighted_unsort(yy, ii, ss, enabled=True) is None
    monkeypatch.setattr(K, 'prefill_weighted_unsort_available', lambda: False)
    assert K.prefill_weighted_unsort(y, inv, s, enabled=True) is None


def test_invalid_inverse_does_not_read_out_of_bounds():
    mx = pytest.importorskip('mlx.core')
    from vmlx_engine.jangh import kernels as K
    if not K.prefill_weighted_unsort_available():
        pytest.skip('Qualified M5 Max / MLX 0.32.2 or 0.32.3 only')
    y = mx.zeros((64, 4096), dtype=mx.bfloat16)
    inv = mx.full((64,), 2**32-1, dtype=mx.uint32)
    actual = K.prefill_weighted_unsort(y, inv, mx.ones((8, 8)), enabled=True)
    mx.eval(actual)
    assert bool(mx.all(mx.isnan(actual)))


def test_routed_prefill_optin_and_stock_fallback(monkeypatch):
    mx = pytest.importorskip('mlx.core')
    from vmlx_engine.jangh import switch
    calls = []
    def prefill(x, idx, kk, scores=None):
        calls.append((tuple(x.shape), tuple(idx.shape), kk, tuple(scores.shape)))
        return mx.full((8, 4096), 17, dtype=mx.bfloat16)
    def experts(x, indices):
        return mx.ones((1, 8, 8, 4096), dtype=mx.bfloat16)
    owner = SimpleNamespace(down_proj=None, _prefill=prefill, _experts=experts,
                            _use_sorted=lambda rows, kk: rows >= 64)  # TQSwitchGLU legacy rule
    x = mx.zeros((1, 8, 4096), dtype=mx.bfloat16)
    idx = mx.zeros((1, 8, 8), dtype=mx.uint32)
    s = mx.ones((1, 8, 8))
    monkeypatch.setattr(switch, 'PREFILL_REDUCE', '0')
    old = switch.TQSwitchGLU.routed(owner, x, idx, s)
    assert bool(mx.all(old == 8)) and calls == []
    monkeypatch.setattr(switch, 'PREFILL_REDUCE', '1')
    new = switch.TQSwitchGLU.routed(owner, x, idx, s)
    assert bool(mx.all(new == 17))
    assert calls == [((8, 4096), (64,), 8, (8, 8))]
    with pytest.raises(ValueError, match='indices/scores shape'):
        switch.TQSwitchGLU.routed(owner, x, idx, s[:, :, :7])


def test_nonfinite_classification_and_finite_words():
    mx = pytest.importorskip('mlx.core')
    from vmlx_engine.jangh import kernels as K
    if not K.prefill_weighted_unsort_available():
        pytest.skip('Qualified M5 Max / MLX 0.32.2 or 0.32.3 only')
    values = mx.array([0.0, -0.0, float('inf'), -float('inf'), float('nan'),
                       1.0, -1.0, 0.00390625], dtype=mx.bfloat16)
    y = mx.tile(values, (64, 512))
    inv = mx.arange(64, dtype=mx.uint32)[::-1]
    s = mx.tile(mx.array([[1., -1., 0., 0.5, 1., -0.5, 1., 1.]]), (8, 1))
    expected = (y[inv].reshape(8, 8, 4096) * s[..., None].astype(y.dtype)).sum(axis=-2)
    actual = K.prefill_weighted_unsort(y, inv, s, enabled=True)
    mx.eval(expected, actual)
    assert bool(mx.array_equal(mx.isnan(expected), mx.isnan(actual)))
    assert bool(mx.array_equal(mx.isinf(expected), mx.isinf(actual)))
    # NaN payload identity is intentionally not part of this contract.
    finite = mx.isfinite(expected)
    assert bool(mx.all(mx.where(finite, expected.view(mx.uint16) == actual.view(mx.uint16), True)))


def test_unweighted_entry_remains_independent(monkeypatch):
    pytest.importorskip('mlx.core')
    from vmlx_engine.jangh import switch
    sentinel = object()
    owner = SimpleNamespace(_experts=lambda x, indices: sentinel, _use_sorted=lambda rows, kk: rows >= 64)
    monkeypatch.setattr(switch, 'PREFILL_REDUCE', '1')
    assert switch.TQSwitchGLU.__call__(owner, object(), object()) is sentinel


@pytest.mark.parametrize("shape", [(8, 1, 4096), (2, 8, 4096), (8, 4096)])
def test_batch_decode_and_unqualified_layout_keep_stock(monkeypatch, shape):
    mx = pytest.importorskip("mlx.core")
    from vmlx_engine.jangh import switch
    def forbidden(*args, **kwargs):
        raise AssertionError("non-single-sequence prefill entered candidate")
    def experts(x, ids):
        return mx.ones((*x.shape[:-1], 8, 4096), dtype=mx.bfloat16)
    owner = SimpleNamespace(down_proj=None, _prefill=forbidden, _experts=experts,
                            _use_sorted=lambda rows, kk: rows >= 64)
    x = mx.zeros(shape, dtype=mx.bfloat16)
    ids = mx.zeros((*shape[:-1], 8), dtype=mx.uint32)
    scores = mx.ones(ids.shape)
    monkeypatch.setattr(switch, "PREFILL_REDUCE", "1")
    result = switch.TQSwitchGLU.routed(owner, x, ids, scores)
    assert result.shape == shape and bool(mx.all(result == 8))
