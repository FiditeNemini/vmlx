"""BlockDiskStore._detach_safetensors_tensors fast path (buffer protocol) == the general contiguous+view+eval path.

The store path freezes every block tensor for the SSD writer on the request's terminal path (~36k tensors for a 46k
Naive prompt). Contiguous MLX buffers are exposed to NumPy directly (BF16 as uint16 bits); non-contiguous inputs keep
the general path (a copy). Bytes, dtypes, shapes, read-only flags and buffer aliasing must match the general path.
"""
import mlx.core as mx
import numpy as np
import pytest

from vmlx_engine.block_disk_store import BlockDiskStore


def _general(value):
    bits = mx.contiguous(value).view(mx.uint16) if value.dtype == mx.bfloat16 else mx.contiguous(value)
    mx.eval(bits)
    return np.asarray(bits)


def _cases():
    base = (mx.arange(4 * 128 * 8).reshape(1, 4, 128, 8) % 251).astype(mx.bfloat16)
    mx.eval(base)
    return {
        "bf16": mx.array(np.arange(2 * 4 * 64 * 8, dtype=np.uint16).reshape(2, 4, 64, 8)).view(mx.bfloat16),
        "zero_size": mx.zeros((1, 1, 64, 0), dtype=mx.bfloat16),
        "meta_u8": mx.array(np.frombuffer(b'{"k": 1}' * 9, dtype=np.uint8)),
        "i32": mx.array([7, 8], dtype=mx.int32),
        "f16": mx.ones((3, 5), dtype=mx.float16),
        "f32": mx.ones((3, 5), dtype=mx.float32) * 2.5,
        "non_contiguous_slice": base[:, :, 10:74],
        "lazy": mx.ones((2, 64), dtype=mx.bfloat16) * 3,
    }


def test_fast_path_matches_the_general_path():
    cases = _cases()
    out = BlockDiskStore._detach_safetensors_tensors(dict(cases))
    assert set(out) == set(cases)
    for name, value in cases.items():
        ref = _general(value)
        got = out[name]
        assert got.dtype == ref.dtype and got.shape == ref.shape, name
        assert np.array_equal(got, ref), name
        assert not got.flags.writeable and got.flags.c_contiguous, name


def test_contiguous_inputs_are_not_copied_and_numpy_inputs_are():
    arr = mx.array(np.arange(64, dtype=np.uint16).reshape(1, 64)).view(mx.bfloat16)
    src = np.arange(16, dtype=np.float32)
    out = BlockDiskStore._detach_safetensors_tensors({"a": arr, "n": src})
    # zero-copy: the frozen view aliases the evaluated MLX buffer (as the general path's contiguous() did)
    assert np.shares_memory(out["a"], np.asarray(arr.view(mx.uint16)))
    assert not np.shares_memory(out["n"], src)
    src[0] = 99.0
    assert out["n"][0] == 0.0


def test_non_array_values_still_raise():
    with pytest.raises(TypeError):
        BlockDiskStore._detach_safetensors_tensors({"x": [1, 2, 3]})
