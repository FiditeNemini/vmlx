"""The disk/paged cache is only safe because every store path reads `.state`.

`KVCache.state` slices to `:offset`; `.keys`/`.values` do not. A live cache
allocates in 256-token chunks, so its buffers routinely carry trailing rows
that are not tokens. Every extractor that feeds `store_cache()` currently goes
through `.state`, which is the ONLY reason padding never reaches disk.

That invariant is load-bearing and otherwise unguarded: plain-KV layers persist
no offset (`KVCache.meta_state` is empty), so on restore the length is taken
from the on-disk tensor shape. If an extractor is ever "optimised" to read
`.keys` directly, zero rows would be written as though a user had sent them,
and every later restore of that block would silently inherit them.

dots3 shipped the in-memory version of this in v1.6.34 (empty answers on any
restore where cached_tokens % 256 != 0). This pins the disk side so the same
class of change fails here instead of in production.
"""

import mlx.core as mx
import pytest

from vmlx_engine.mllm_scheduler import MLLMScheduler

PAD_SENTINEL = 7.0
REAL_VALUE = 1.0


class _PaddedKVCache:
    """A live cache: 100 real tokens in a 256-row buffer."""

    step = 256

    def __init__(self, offset=100, physical=256, heads=2, dim=8):
        self.offset = offset
        real = mx.full((1, heads, offset, dim), REAL_VALUE)
        pad = mx.full((1, heads, physical - offset, dim), PAD_SENTINEL)
        self.keys = mx.concatenate([real, pad], axis=2)
        self.values = mx.concatenate([real, pad], axis=2)

    @property
    def state(self):
        return (
            self.keys[..., : self.offset, :],
            self.values[..., : self.offset, :],
        )

    @property
    def meta_state(self):
        # Plain KVCache persists no offset — this is why the on-disk extent
        # must already BE the logical length.
        return ""


class _Stub(MLLMScheduler):
    """Only the extraction logic is under test."""

    def __init__(self):  # noqa: D107 - deliberately skips MLLMScheduler.__init__
        pass

    def _detect_n_kv_heads(self):
        return 2

    def _detect_allowed_n_kv_heads(self):
        return {2}

    def _normalize_gqa_state(self, state, n_kv, allowed_n_kv_heads=None):
        return state


def _extract(cache):
    return _Stub()._extract_cache_states([cache])


def test_store_extraction_drops_the_allocation_slack():
    out = _extract(_PaddedKVCache(offset=100, physical=256))
    assert len(out) == 1
    keys, values = out[0]["state"][0], out[0]["state"][1]
    assert keys.shape[2] == 100, (
        f"extracted {keys.shape[2]} rows for a 100-token cache — the extra "
        f"rows are allocation slack and would be stored as real tokens"
    )
    assert values.shape[2] == 100


def test_no_pad_row_ever_reaches_the_store():
    out = _extract(_PaddedKVCache(offset=100, physical=256))
    keys = out[0]["state"][0]
    assert float(mx.max(keys)) == REAL_VALUE, (
        "a pad row reached the store path; every later restore of this block "
        "would inherit zeros as though they were user tokens"
    )


@pytest.mark.parametrize("offset", [1, 99, 100, 128, 255, 256])
def test_extraction_is_correct_at_every_residue(offset):
    """256 and 512 are the ONLY lengths where padded == logical. Testing only
    those is what hid the dots3 defect, so sweep off-boundary values here."""
    physical = 256 if offset <= 256 else 512
    out = _extract(_PaddedKVCache(offset=offset, physical=physical))
    assert out[0]["state"][0].shape[2] == offset


def test_an_unpadded_cache_is_unaffected():
    out = _extract(_PaddedKVCache(offset=256, physical=256))
    assert out[0]["state"][0].shape[2] == 256
    assert float(mx.max(out[0]["state"][0])) == REAL_VALUE
