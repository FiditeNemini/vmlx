"""CPU-only configuration contract; no numerical or live-runtime claim."""
import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / 'vmlx_engine/jangh/runtime_identity.py'


def load():
    spec = importlib.util.spec_from_file_location('isolated_jangh_identity', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_h32_default_is_off(monkeypatch):
    monkeypatch.delenv('JANGH_GATEUP_H32', raising=False)
    module = load()
    assert module.GATEUP_H32 == '0'
    assert ';gateup_h32=0' in module.runtime_identity()


def test_h32_identity_is_separate_and_startup_frozen(monkeypatch):
    monkeypatch.setenv('JANGH_GATEUP_H32', '0')
    baseline = load()
    first = baseline.runtime_identity()
    monkeypatch.setenv('JANGH_GATEUP_H32', '1')
    assert baseline.GATEUP_H32 == '0'
    assert baseline.runtime_identity() == first
    enabled = load()
    assert enabled.GATEUP_H32 == '1'
    assert enabled.runtime_identity() != first
    assert ';gateup_h32=1' in enabled.runtime_identity()


@pytest.mark.parametrize('value', ['auto', 'true', '2', ''])
def test_h32_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv('JANGH_GATEUP_H32', value)
    with pytest.raises(ValueError, match='JANGH_GATEUP_H32 must be 0 or 1'):
        load()
