"""Processor conversion must preserve existing dtype, shape and raw values."""
import mlx.core as mx
import numpy as np
import pytest
from vmlx_engine.mllm_batch_generator import _mllm_processor_array


@pytest.mark.parametrize('kind', ['finite','strided','reverse','special','nan_payload','scalar','empty','float64','int64'])
def test_processor_array_matches_legacy_conversion(kind):
    source = np.arange(24,dtype=np.float32).reshape(4,6)
    if kind == 'strided':source=source[::2,::2]
    elif kind == 'reverse':source=source[::-1,::-1]
    elif kind == 'special':source=np.array([0.,-0.,np.inf,-np.inf,np.nan],dtype=np.float32)
    elif kind == 'nan_payload':source=np.array([0x7fc00001,0xffc01234,0x7f800001,0xff800123],dtype=np.uint32).view(np.float32)
    elif kind == 'scalar':source=np.array(-0.,dtype=np.float32)
    elif kind == 'empty':source=np.empty((0,2),dtype=np.float32)
    elif kind == 'float64':source=source.astype(np.float64)
    elif kind == 'int64':source=source.astype(np.int64)
    expected=mx.array(source.tolist())
    actual=_mllm_processor_array(source)
    mx.eval(actual,expected)
    assert actual.shape==expected.shape and actual.dtype==expected.dtype
    assert np.array(actual.reshape(-1).view(mx.uint8)).tobytes()==np.array(expected.reshape(-1).view(mx.uint8)).tobytes()


def test_processor_array_target_dtype_and_existing_array():
    source=np.array([[1.,2.]],dtype=np.float32)
    actual=_mllm_processor_array(source,mx.float16)
    assert actual.dtype==mx.float16 and actual.tolist()==[[1.,2.]]
    assert _mllm_processor_array(actual) is actual
    assert _mllm_processor_array(None) is None
