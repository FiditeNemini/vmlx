"""Rotation fusion preserves native Hadamard rounding, shapes and dispatch."""
import importlib.util
from pathlib import Path

import pytest


def test_h32_rows_identity_is_frozen_and_validated(monkeypatch):
    path = Path(__file__).parents[1] / 'vmlx_engine/jangh/runtime_identity.py'
    def load():
        spec = importlib.util.spec_from_file_location('h32_identity', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    monkeypatch.delenv('JANGH_H32_ROWS', raising=False)
    baseline = load()
    identity = baseline.runtime_identity()
    assert baseline.H32_ROWS == '0'
    monkeypatch.setenv('JANGH_H32_ROWS', '1')
    assert baseline.runtime_identity() == identity
    assert load().runtime_identity() != identity
    monkeypatch.setenv('JANGH_H32_ROWS', 'invalid')
    with pytest.raises(ValueError, match='JANGH_H32_ROWS'):
        load()


@pytest.mark.parametrize('dtype', ['bfloat16', 'float16', 'float32'])
@pytest.mark.parametrize('shape,strided', [((1,32),False), ((7,96),True), ((2048,4096),False)])
def test_h32_rows_matches_native_hadamard(dtype, shape, strided):
    mx = pytest.importorskip('mlx.core')
    from vmlx_engine.jangh.kernels import h32_rows
    mx.random.seed(928)
    source_shape = (*shape[:-1],shape[-1]*2) if strided else shape
    x = mx.random.normal(source_shape).astype(getattr(mx,dtype))
    if strided:
        x = x[...,::2]
    expected = mx.hadamard_transform(
        x.astype(mx.float32).reshape(*shape[:-1],shape[-1]//32,32)
    ).reshape(shape).astype(x.dtype)
    actual = h32_rows(x)
    mx.eval(actual,expected)
    assert actual.dtype == x.dtype
    assert actual.shape == x.shape
    assert bool(mx.array_equal(actual,expected))


def test_h32_rows_dispatch_preserves_dtype_and_nonrotated_path(monkeypatch):
    mx = pytest.importorskip('mlx.core')
    from vmlx_engine.jangh import switch
    x = mx.arange(96).reshape(3,32).astype(mx.bfloat16)
    rotated = switch.TQSwitchLinear(32,32,2,2,'hadamard32')
    plain = switch.TQSwitchLinear(32,32,2,2,'none')
    monkeypatch.setattr(switch,'H32_ROWS','0')
    reference = switch.rotate_rows(x,rotated)
    monkeypatch.setattr(switch,'H32_ROWS','1')
    actual = switch.rotate_rows(x,rotated)
    mx.eval(reference,actual)
    assert actual.dtype == mx.bfloat16
    assert bool(mx.array_equal(actual,reference))
    assert switch.rotate_rows(x,plain) is x


def test_h32_rows_empty_and_invalid_shapes():
    mx = pytest.importorskip('mlx.core')
    from vmlx_engine.jangh.kernels import h32_rows
    assert h32_rows(mx.zeros((0,32),dtype=mx.bfloat16)).shape == (0,32)
    for value in [mx.zeros((1,31)),mx.zeros((1,0)),mx.array(1.0),mx.zeros((1,32),dtype=mx.int32)]:
        with pytest.raises(ValueError):
            h32_rows(value)
