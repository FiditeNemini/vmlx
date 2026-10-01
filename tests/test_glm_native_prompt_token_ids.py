"""Native GLM host IDs retain first-row semantics without an extra MLX op."""
import mlx.core as mx
import pytest
from vmlx_engine.mllm_batch_generator import _glm_native_prompt_token_ids


@pytest.mark.parametrize('shape', [(4,), (1,4), (2,4), (1,2,2)])
def test_native_prompt_ids_match_existing_first_row(shape):
    values=mx.arange(4 if shape!=(2,4) else 8,dtype=mx.int32).reshape(shape)
    expected=values.tolist() if values.ndim==1 else values[0].tolist()
    assert _glm_native_prompt_token_ids(values)==expected


@pytest.mark.parametrize('shape', [(), (1,0), (0,4)])
def test_native_prompt_ids_preserve_invalid_empty_or_scalar_behavior(shape):
    values=mx.zeros(shape,dtype=mx.int32)
    try:
        expected=values[0].tolist()
    except ValueError as old:
        with pytest.raises(type(old)) as new:
            _glm_native_prompt_token_ids(values)
        assert str(new.value)==str(old)
    else:
        assert _glm_native_prompt_token_ids(values)==expected


def test_single_row_host_read_does_not_index_device_array():
    class AlreadyRealizedPrompt:
        ndim=2
        shape=(1,4)
        def tolist(self):return [[9,7,5,3]]
        def __getitem__(self,key):raise AssertionError('new device indexing operation')
    assert _glm_native_prompt_token_ids(AlreadyRealizedPrompt())==[9,7,5,3]
    assert _glm_native_prompt_token_ids(None)==[]


def test_strided_single_row_preserves_token_order():
    values=mx.arange(8,dtype=mx.int32).reshape(1,8)[:,::2]
    assert _glm_native_prompt_token_ids(values)==values[0].tolist()==[0,2,4,6]
