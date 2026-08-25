"""Focused regression for MLX 0.32.2's scalar ``repeat`` contract."""

import mlx.core as mx

from vmlx_engine.utils import mlx_vlm_compat


def _old_mlx_vlm_vision_call(self, hidden_states, grid_thw, **kwargs):
    del self, hidden_states, kwargs
    cu_seqlens = []
    for i in range(grid_thw.shape[0]):
        seq_len = grid_thw[i, 1] * grid_thw[i, 2]
        cu_seqlens.append(mx.repeat(seq_len, grid_thw[i, 0]))
    return mx.concatenate(cu_seqlens)


def test_qwen3_vl_repeat_count_backport_converts_scalar_array_to_int():
    patched, changed = mlx_vlm_compat._qwen3_vl_repeat_count_compat(
        _old_mlx_vlm_vision_call,
        globals(),
        __file__,
    )

    assert changed is True
    output = patched(None, None, mx.array([[2, 3, 4]], dtype=mx.int32))
    assert output.tolist() == [12, 12]


def test_installed_qwen3_vl_call_is_scalar_repeat_safe():
    mlx_vlm_compat.apply()

    from mlx_vlm.models.qwen3_vl.vision import VisionModel

    assert (
        getattr(VisionModel.__call__, "_vmlx_repeat_count_scalar_safe", False) is True
    )
