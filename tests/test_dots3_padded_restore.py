"""dots3 must adopt the LOGICAL extent of a restored KV layer, not the buffer.

Restored plain-KV layers are zero-padded up to KVCache.step (256) while
`offset` stays logical. dots3's _adopt_restored_caches took `.keys`/`.values`
(padded) instead of `.state` (sliced to offset), so it attended over zero
rows: DSA counts latent.shape[2] as real keys while `past` is the logical
offset, putting the causal frontier (physical - logical) tokens behind and
making pad rows legal top-k candidates. Generation never converged — it ran
to max_tokens with no visible answer.

MEASURED on dots3-note: a partial cache hit of 2176 tokens (2176 % 256 = 128)
returned an EMPTY answer after 1500 tokens and 5.8k chars of reasoning; with
the fix the same request answers 'CACHE-OK-42' in 78 tokens. Any restore
whose cached length is not a multiple of 256 hit this — alignment is a
1-in-256 accident, so this was NOT limited to divergent prefixes.

Existing dots3 restore tests only used boundaries 256 and 512 — both exact
multiples of the step — which is why none of them could see it.
"""

import mlx.core as mx
import pytest

from vmlx_engine.models.dots3_note.language import (
    _logical_kv_extent,
    _logical_offset,
)


class _PaddedKV:
    """A restored KVCache: physically padded, logically shorter."""

    def __init__(self, logical, physical, dim=8, heads=1):
        self.offset = logical
        real = mx.ones((1, heads, logical, dim))
        pad = mx.zeros((1, heads, physical - logical, dim))
        self.keys = mx.concatenate([real, pad], axis=2)
        self.values = mx.concatenate([real, pad], axis=2)


@pytest.mark.parametrize("logical", [256, 320, 448, 512, 576, 2176, 2245])
def test_logical_extent_drops_the_padding(logical):
    step = 256
    physical = ((logical + step - 1) // step) * step
    c = _PaddedKV(logical, physical)
    keys, values = _logical_kv_extent(c)
    assert keys.shape[2] == logical, (
        "adopted %d rows for a %d-token cache (%d padded) — the extra rows "
        "are zeros and would be attended as real keys"
        % (keys.shape[2], logical, physical)
    )
    assert values.shape[2] == logical
    # every surviving row must be real, never a pad row
    assert float(mx.min(keys)) == 1.0


def test_aligned_restore_is_untouched():
    c = _PaddedKV(512, 512)
    keys, _ = _logical_kv_extent(c)
    assert keys.shape[2] == 512
    assert keys is c.keys, "an unpadded cache must not be copied or sliced"


def test_offset_never_comes_from_a_padded_length():
    c = _PaddedKV(2176, 2304)
    keys, _ = _logical_kv_extent(c)
    assert _logical_offset(c, keys) == 2176, (
        "a padded physical length must never become the logical offset"
    )


def test_missing_offset_falls_back_to_the_given_buffer():
    class _NoOffset:
        offset = 0
        keys = mx.ones((1, 1, 128, 8))
        values = mx.ones((1, 1, 128, 8))

    c = _NoOffset()
    keys, values = _logical_kv_extent(c)
    assert _logical_offset(c, keys) == 128


def test_degenerate_inputs_do_not_crash():
    class _Empty:
        offset = 10
        keys = None
        values = None

    keys, values = _logical_kv_extent(_Empty())
    assert keys is None and values is None
    assert _logical_offset(_Empty(), None) == 10


def test_the_boundaries_the_old_tests_used_would_have_passed_anyway():
    """Documents the blind spot: 256 and 512 are exact multiples of the step,
    so the padded and logical extents coincide and the bug is invisible."""
    for aligned in (256, 512):
        c = _PaddedKV(aligned, aligned)
        keys, _ = _logical_kv_extent(c)
        assert keys.shape[2] == aligned
