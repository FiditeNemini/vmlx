# SPDX-License-Identifier: Apache-2.0
"""GitHub #225: continuous-batching multi-client engine abort.

Under --continuous-batching, two concurrently-arriving requests are
coalesced into one generation batch. The vMLX single-sequence fast path in
``_merge_caches`` (kept for Bailing/Ling numerical fidelity) leaves NATIVE
single-sequence cache objects in the batch, so mlx-lm's ``_extend_cache``
zip then crashed the whole engine loop:

  * plain KVCache + plain KVCache  -> AttributeError: no attribute 'extend'
  * TurboQuant  + plain KVCache    -> AttributeError: no attribute '_batch_state'

The fix promotes both sides to batch-native caches at extend time. These
tests prove the promotion is faithful (row contents byte-identical to the
originals) and that the fidelity fast paths are untouched.
"""

import pytest

mx = pytest.importorskip("mlx.core")

import importlib

gen_module = importlib.import_module("mlx_lm.generate")
from mlx_lm.models.cache import (
    BatchKVCache,
    BatchRotatingKVCache,
    KVCache,
    RotatingKVCache,
)

from vmlx_engine.utils.mamba_cache import ensure_mamba_support

ensure_mamba_support()


def _fill_kv(cache, n, base=1.0, heads=2, dim=8):
    """Advance a single-sequence cache by n distinct tokens."""
    for i in range(n):
        k = mx.full((1, heads, 1, dim), base + i)
        v = mx.full((1, heads, 1, dim), -(base + i))
        cache.update_and_fetch(k, v)
    mx.eval(cache.keys, cache.values)
    return cache


def _row_contents(batch_cache, idx):
    """Extract row idx of a batch cache back to a single KVCache."""
    row = batch_cache.extract(idx)
    mx.eval(row.keys, row.values)
    return row


class TestExtendPromotion:
    """Plain single-sequence caches must batch, not abort the engine."""

    def test_plain_kvcache_pair_promotes_to_batch(self):
        """Issue #225 mode 2: KVCache + KVCache (no paged cache)."""
        a = _fill_kv(KVCache(), 5, base=1.0)
        b = _fill_kv(KVCache(), 3, base=100.0)
        a_keys = mx.array(a.keys[..., : a.offset, :])
        b_keys = mx.array(b.keys[..., : b.offset, :])

        merged = gen_module._extend_cache([a], [b])

        assert len(merged) == 1
        out = merged[0]
        assert type(out).__name__ == "BatchKVCache"
        assert out.keys.shape[0] == 2

        row_a = _row_contents(out, 0)
        row_b = _row_contents(out, 1)
        assert row_a.offset == 5
        assert row_b.offset == 3
        assert bool(mx.all(row_a.keys == a_keys))
        assert bool(mx.all(row_b.keys == b_keys))

    def test_rotating_pair_promotes_to_batch_rotating(self):
        """gemma-4-style sliding layers must batch the same way."""
        a = _fill_kv(RotatingKVCache(max_size=64, keep=4), 6, base=1.0)
        b = _fill_kv(RotatingKVCache(max_size=64, keep=4), 4, base=50.0)

        merged = gen_module._extend_cache([a], [b])

        out = merged[0]
        assert type(out).__name__ == "BatchRotatingKVCache"
        assert out.keys.shape[0] == 2
        assert out.max_size == 64

    def test_batch_plus_plain_asymmetric_join(self):
        """A 3rd single-seq request joining an already-promoted batch."""
        a = BatchKVCache.merge(
            [_fill_kv(KVCache(), 5, base=1.0), _fill_kv(KVCache(), 3, base=9.0)]
        )
        c = _fill_kv(KVCache(), 4, base=200.0)
        c_keys = mx.array(c.keys[..., : c.offset, :])

        merged = gen_module._extend_cache([a], [c])

        out = merged[0]
        assert type(out).__name__ == "BatchKVCache"
        assert out.keys.shape[0] == 3
        row_c = _row_contents(out, 2)
        assert row_c.offset == 4
        assert bool(mx.all(row_c.keys == c_keys))

    def test_compatible_batch_pair_extends_in_place(self):
        """Batch-native pairs keep the original zero-copy extend path."""
        a = BatchKVCache.merge([_fill_kv(KVCache(), 5)])
        b = BatchKVCache.merge([_fill_kv(KVCache(), 3)])

        merged = gen_module._extend_cache([a], [b])

        assert merged[0] is a  # in-place, no promotion detour
        assert a.keys.shape[0] == 2

    def test_single_sequence_merge_fast_path_preserved(self):
        """Bailing/Ling fidelity: _merge_caches(len==1) stays NATIVE."""
        a = _fill_kv(KVCache(), 5)
        merged = gen_module._merge_caches([[a]])
        assert merged[0] is a

    def test_mixed_swa_layer_list_promotes_layerwise(self):
        """Mixed full+sliding per-layer lists (gemma-4 shape)."""
        a = [
            _fill_kv(KVCache(), 5, base=1.0),
            _fill_kv(RotatingKVCache(max_size=32, keep=0), 5, base=2.0),
        ]
        b = [
            _fill_kv(KVCache(), 3, base=10.0),
            _fill_kv(RotatingKVCache(max_size=32, keep=0), 3, base=20.0),
        ]
        merged = gen_module._extend_cache(a, b)
        assert type(merged[0]).__name__ == "BatchKVCache"
        assert type(merged[1]).__name__ == "BatchRotatingKVCache"
        assert merged[0].keys.shape[0] == 2
        assert merged[1].keys.shape[0] == 2

    def test_prefix_hit_shapes_survive_prompt_batch_merge(self):
        """Co-prefilling two cache-hit requests (multi-cache _merge_caches)."""
        a = _fill_kv(KVCache(), 7, base=1.0)
        b = _fill_kv(KVCache(), 2, base=30.0)
        merged = gen_module._merge_caches([[a], [b]])
        assert type(merged[0]).__name__ == "BatchKVCache"
        assert merged[0].keys.shape[0] == 2


class TestPreFixBehaviorDocumented:
    """Bidirectional proof: the UNPATCHED extend reproduces issue #225."""

    def test_unpatched_extend_raises_attributeerror(self):
        a = _fill_kv(KVCache(), 5)
        b = _fill_kv(KVCache(), 3)
        with pytest.raises(AttributeError, match="extend"):
            for ca, cb in zip([a], [b]):
                ca.extend(cb)


_tq_mod = None
try:  # TurboQuant is an optional jang_tools dependency
    import jang_tools.turboquant.cache as _tq_mod
except Exception:
    _tq_mod = None


@pytest.mark.skipif(_tq_mod is None, reason="jang_tools TurboQuant unavailable")
class TestTurboQuantMixedPromotion:
    """Issue #225 mode 1: paged-cache-hit TQ cache + fresh plain KVCache."""

    def _make_tq(self, n=6):
        tq = _tq_mod.TurboQuantKVCache(8, 8)
        _fill_kv(tq, n, base=1.0)
        return tq

    def test_tq_plus_plain_decodes_to_batch_kv(self):
        tq = self._make_tq(6)
        b = _fill_kv(KVCache(), 3, base=100.0)
        b_keys = mx.array(b.keys[..., : b.offset, :])

        merged = gen_module._extend_cache([tq], [b])

        out = merged[0]
        assert type(out).__name__ == "BatchKVCache"
        assert out.keys.shape[0] == 2
        row_b = _row_contents(out, 1)
        assert bool(mx.all(row_b.keys == b_keys))

    def test_plain_plus_tq_reverse_order(self):
        a = _fill_kv(KVCache(), 3, base=100.0)
        tq = self._make_tq(6)

        merged = gen_module._extend_cache([a], [tq])

        out = merged[0]
        assert type(out).__name__ == "BatchKVCache"
        assert out.keys.shape[0] == 2

    def test_tq_pair_stays_quantized(self):
        """TQ<->TQ keeps the native quantized extend (no decode detour)."""
        t1, t2 = self._make_tq(5), self._make_tq(3)
        if not all(
            getattr(t, "_vmlx_batch_api", None) == "turboquant_kv_v1"
            for t in (t1, t2)
        ):
            pytest.skip("TQ build without v1 batch API")
        merged = gen_module._extend_cache([t1], [t2])
        assert merged[0] is t1

    def test_tq_empty_pair_extend_keeps_rows(self):
        """Two FRESH TQ caches co-prefilling (rope offset shape-(0) bug).

        jang_tools _batch_state() reported 0-length offsets for empty
        caches, so the merged batch lost both rows' identity and the
        prefill crashed with '[rope] offset ... has shape (0)'.
        """
        t1 = _tq_mod.TurboQuantKVCache(8, 8)
        t2 = _tq_mod.TurboQuantKVCache(8, 8)

        t1.extend(t2)

        assert int(t1.offset.size) == 2, "both rows must survive the merge"
        assert int(t1.left_padding.size) == 2
        assert t1.keys is None  # still empty — pure bookkeeping, stays TQ
        assert t1._is_batched

    def test_tq_empty_plus_nonempty_keeps_row_count(self):
        """Empty + filled TQ extend must NOT drop the empty side's row."""
        t1 = _tq_mod.TurboQuantKVCache(8, 8)
        t2 = self._make_tq(6)

        t1.extend(t2)

        assert t1.keys is not None
        assert int(t1.keys.shape[0]) == 2, "empty row silently dropped"
        assert int(t1.offset.size) == 2

    def test_tq_nonempty_plus_empty_reverse(self):
        t1 = self._make_tq(6)
        t2 = _tq_mod.TurboQuantKVCache(8, 8)

        t1.extend(t2)

        assert t1.keys is not None
        assert int(t1.keys.shape[0]) == 2
        assert int(t1.offset.size) == 2
