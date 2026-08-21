"""A buffer extent is not a token count, and must never become an offset.

`KVCache` allocates keys/values in `step` (256) token chunks, so
`keys.shape[seq] > offset` is the NORMAL state of a live cache. Two code
shapes keep re-deriving a length from the buffer instead of from `offset`:

  safe_target = min(target_len, keys.shape[-2])   # missing `offset`
  layer_cache.offset = prefix_len                 # unconditional

Both turn allocation slack into tokens. In a truncation path that feeds
`store_cache`, that persists zero rows into the prefix cache as though a user
had sent them. dots3 shipped the model-side version of this in v1.6.34
(empty answers on any restore where cached_tokens % 256 != 0).
"""

import mlx.core as mx
import pytest

from vmlx_engine.utils.cache_extent import cache_offset, logical_truncate_target
from vmlx_engine.utils.mlx_vlm_compat import _vmlx_trim_prompt_cache


class _Cache:
    def __init__(self, offset, physical, ndim=4, dim=8):
        self.offset = offset
        shape = (1, 1, physical, dim) if ndim == 4 else (1, physical, dim)
        seq_axis = 2 if ndim == 4 else 1
        real = list(shape)
        real[seq_axis] = offset
        pad = list(shape)
        pad[seq_axis] = physical - offset
        self.keys = mx.concatenate(
            [mx.ones(tuple(real)), mx.zeros(tuple(pad))], axis=seq_axis
        )
        self.values = mx.concatenate(
            [mx.ones(tuple(real)), mx.zeros(tuple(pad))], axis=seq_axis
        )


def test_clamp_never_exceeds_the_logical_offset():
    c = _Cache(offset=100, physical=256)
    assert logical_truncate_target(c, 199, 256) == 100, (
        "a target_len between offset and the buffer size must clamp to offset "
        "— slicing to 199 would hand 99 ZERO rows downstream as real tokens"
    )


@pytest.mark.parametrize("target", [1, 50, 99, 100])
def test_clamp_passes_through_legitimate_targets(target):
    c = _Cache(offset=100, physical=256)
    assert logical_truncate_target(c, target, 256) == target


def test_clamp_still_honours_the_physical_bound():
    """An offset larger than the buffer is itself corrupt; the physical bound
    must still win so we slice something valid rather than raise."""

    class _OverstatedOffset:
        offset = 300

    assert logical_truncate_target(_OverstatedOffset(), 1000, 256) == 256


def test_offset_zero_is_unknown_not_empty():
    """Some cache types never populate offset. Clamping those to 0 would
    delete a valid cache — advise, never refuse."""

    class _NoOffset:
        offset = 0

    assert logical_truncate_target(_NoOffset(), 128, 256) == 128


def test_cache_offset_tolerates_garbage():
    class _Bad:
        offset = "not-a-number"

    assert cache_offset(_Bad()) == 0
    assert cache_offset(object()) == 0


@pytest.mark.parametrize("ndim", [4, 3])
def test_vlm_prefix_truncate_cannot_raise_the_offset(ndim):
    """The regression this guards: prefix_len sits between offset and the
    buffer size, and the old code assigned offset = prefix_len."""
    c = _Cache(offset=100, physical=256, ndim=ndim)
    _vmlx_trim_prompt_cache([c], 199)
    seq_axis = 2 if ndim == 4 else 1
    assert c.offset == 100, "offset was raised into allocation slack"
    assert c.keys.shape[seq_axis] <= 100
    assert float(mx.min(c.keys)) == 1.0, "a zero pad row survived as a token"


@pytest.mark.parametrize("ndim", [4, 3])
def test_vlm_prefix_truncate_still_truncates(ndim):
    c = _Cache(offset=200, physical=256, ndim=ndim)
    _vmlx_trim_prompt_cache([c], 120)
    seq_axis = 2 if ndim == 4 else 1
    assert c.offset == 120
    assert c.keys.shape[seq_axis] == 120
