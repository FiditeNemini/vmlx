"""Dispatch contract only. Sentinels do not establish kernel numerical equality."""
from types import SimpleNamespace

import pytest

mx = pytest.importorskip('mlx.core')
from vmlx_engine.jangh import kernels, switch


class Rows:
    """No tensor evaluation: preserve identity through route gathers."""
    dtype = mx.bfloat16
    shape = (8, 4096)

    def __init__(self, name):
        self.name = name

    def __getitem__(self, key):
        return self


class Order:
    def __floordiv__(self, divisor):
        assert divisor == 8
        return self


def projection(i, o):
    return SimpleNamespace(input_dims=i, output_dims=o, num_experts=288,
                           bits=2, rotated=True, tq2_packed=object(),
                           tq2_scales=object(), _cb=object())


@pytest.mark.parametrize('expert_flag,h32_flag', [('0', '0'), ('0', '1'), ('1', '0'), ('1', '1')])
def test_h32_requires_both_flags_and_skips_only_down_rotation(monkeypatch, expert_flag, h32_flag):
    module = SimpleNamespace(gate_proj=projection(4096, 2048),
                             up_proj=projection(4096, 2048),
                             down_proj=projection(2048, 4096), limit=0.0)
    module._use_expert_tiles = lambda x, kk: switch.TQSwitchGLU._use_expert_tiles(module, x, kk)
    monkeypatch.setattr(switch, 'EXPERT_TILES', expert_flag)
    monkeypatch.setattr(switch, 'GATEUP_H32', h32_flag)
    monkeypatch.setattr(kernels, 'nax_available', lambda: True)
    monkeypatch.setattr(switch.mx, 'argsort', lambda x: Order())
    x, idx = Rows('input'), Rows('indices')
    rotated_x, h, rotated_h, y = (Rows(n) for n in ('rotated_x', 'h', 'rotated_h', 'output'))
    rotations, calls = [], []

    def rotate(value, proj):
        rotations.append((value, proj))
        if proj is module.gate_proj:
            assert value is x
            return rotated_x
        assert proj is module.down_proj and value is h
        return rotated_h

    monkeypatch.setattr(switch, 'rotate_rows', rotate)
    plan = object()
    monkeypatch.setattr(kernels, 'expert_tile_plan', lambda indices, experts: plan)

    def gather_expert(value, packed, scales, indices, bits, received_plan, **kwargs):
        assert expert_flag == '1' and received_plan is plan
        calls.append(('expert', value, packed, kwargs))
        if packed is module.gate_proj.tq2_packed:
            assert value is rotated_x
            assert kwargs['rotate_output'] is (h32_flag == '1')
            assert kwargs['packed_u'] is module.up_proj.tq2_packed
            return h
        assert packed is module.down_proj.tq2_packed
        assert value is (h if h32_flag == '1' else rotated_h)
        assert not kwargs
        return y

    def gather_sorted(value, packed, scales, cb, indices, bits, **kwargs):
        assert expert_flag == '0'
        calls.append(('sorted', value, packed, kwargs))
        if packed is module.gate_proj.tq2_packed:
            assert value is rotated_x
            assert 'rotate_output' not in kwargs
            return h
        assert packed is module.down_proj.tq2_packed and value is rotated_h
        return y

    monkeypatch.setattr(kernels, 'gather_qmm_expert_sorted', gather_expert)
    monkeypatch.setattr(kernels, 'gather_qmm_sorted', gather_sorted)
    result = switch.TQSwitchGLU._prefill(module, x, idx, 8)
    assert result is y and len(calls) == 2
    assert rotations == ([(x, module.gate_proj)] if expert_flag == h32_flag == '1'
                         else [(x, module.gate_proj), (h, module.down_proj)])
