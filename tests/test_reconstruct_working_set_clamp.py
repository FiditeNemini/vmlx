# SPDX-License-Identifier: Apache-2.0
"""Clamping a paged prefix so its reconstruction cannot exhaust Metal.

Reconstructing a long chain materialises every block's KV at once and
dequantises it — one allocation burst. On a weight-heavy box that burst can
exceed the free working set, and the failure is NOT catchable: it comes from
Metal's command-buffer completion handler, off the Python thread, so it reaches
std::terminate rather than raising. Measured on MiniMax-M2.7-JANG_K (80GB
weights, 128GB box): a 451-block / ~28.9k-token / 62-layer reconstruct aborted
the process every time, at ~38.5k context — barely half the ~69,690 the engine
advertised at startup.

The cache is an optimisation, so the correct response is to restore a shorter
prefix and re-prefill the rest.
"""

from types import SimpleNamespace

import pytest

from vmlx_engine.paged_cache import BlockTable
from vmlx_engine.prefix_cache import BlockAwarePrefixCache


class _FakePaged:
    def __init__(self, per_block_bytes, block_size=64, n_blocks=100):
        self.block_size = block_size
        self.released = []
        self.allocated_blocks = {
            i: SimpleNamespace(cache_data=object()) for i in range(n_blocks)
        }
        self._per_block = per_block_bytes

    @staticmethod
    def estimate_block_nbytes(cache_data):
        return _FakePaged._current_per_block

    def release_request_refs(self, table):
        self.released.append(list(table.block_ids))
        return len(table.block_ids)


def _cache(per_block_bytes, *, bits=0, n_blocks=100, block_size=64):
    paged = _FakePaged(per_block_bytes, block_size=block_size, n_blocks=n_blocks)
    _FakePaged._current_per_block = per_block_bytes
    obj = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
    obj.paged_cache = paged
    obj.block_size = block_size
    obj.kv_quant_bits = bits
    obj.model = SimpleNamespace()  # no config -> exercises the sampling fallback
    return obj, paged


def _table(n_blocks, block_size=64):
    return BlockTable(
        request_id="req-1",
        block_ids=list(range(n_blocks)),
        num_tokens=n_blocks * block_size,
        matched_tokens=n_blocks * block_size,
        checkpoint_tokens=n_blocks * block_size,
    )


@pytest.fixture
def big_free(monkeypatch):
    def _set(free_gb, active_gb=80.0):
        import vmlx_engine.utils.memory_limits as ml

        total = (active_gb + free_gb) * 1024**3
        monkeypatch.setattr(
            ml,
            "get_effective_metal_working_set_bytes",
            lambda mx_module=None: (int(active_gb * 1024**3), int(total)),
        )
    return _set


def test_no_clamp_when_it_comfortably_fits(big_free):
    big_free(27.5)
    cache, paged = _cache(1 * 1024**2)  # 1MB/block, 100 blocks = 100MB
    table = _table(100)

    assert cache.clamp_block_table_to_working_set(table) == 0
    assert len(table.block_ids) == 100
    assert paged.released == []


def test_clamps_and_releases_only_the_dropped_blocks(big_free):
    big_free(1.0)  # 1GB free -> 0.6GB budget
    cache, paged = _cache(100 * 1024**2)  # 100MB/block -> ~6 affordable
    table = _table(100)

    dropped = cache.clamp_block_table_to_working_set(table)

    assert dropped > 0
    kept = len(table.block_ids)
    assert kept + dropped == 100
    assert paged.released == [list(range(kept, 100))], "released the wrong blocks"
    assert table.num_tokens == kept * 64
    assert table.matched_tokens == table.num_tokens
    assert table.checkpoint_tokens == table.num_tokens


def test_quantised_storage_expands_by_the_codec_width(big_free):
    """q8 -> 16-bit compute doubles the materialised size, so fewer blocks fit."""
    big_free(1.0)
    unquant, _ = _cache(100 * 1024**2, bits=0)
    quant, _ = _cache(100 * 1024**2, bits=8)

    t1, t2 = _table(100), _table(100)
    unquant.clamp_block_table_to_working_set(t1)
    quant.clamp_block_table_to_working_set(t2)

    assert len(t2.block_ids) < len(t1.block_ids)
    assert len(t2.block_ids) == pytest.approx(len(t1.block_ids) / 2, abs=1)


def test_always_keeps_at_least_one_block(big_free):
    big_free(0.001)
    cache, paged = _cache(4 * 1024**3)  # 4GB/block, nothing affordable
    table = _table(10)

    cache.clamp_block_table_to_working_set(table)

    assert len(table.block_ids) == 1, "a hit must not degrade to a bare miss"
    assert paged.released == [list(range(1, 10))]


def test_geometry_is_used_when_available_even_with_nothing_resident(monkeypatch, big_free):
    """The OOM case is a DISK-backed chain, where no block is resident yet.

    An earlier version of this guard sampled only resident payloads and so
    declined on exactly the restores it was meant to protect - MM2.7 died at
    turn 3 with the clamp never firing. Model geometry is exact and available
    regardless of residency.
    """
    big_free(1.0)
    cache, paged = _cache(0)
    for block in cache.paged_cache.allocated_blocks.values():
        block.cache_data = None
    cache.model = SimpleNamespace(config=SimpleNamespace(_marker=True))

    import vmlx_engine.utils.memory_limits as ml

    monkeypatch.setattr(
        ml, "estimate_kv_bytes_per_token_from_config",
        lambda cfg: 2 * 1024 * 1024,  # 2MB/token -> 128MB per 64-token block
    )
    table = _table(100)

    dropped = cache.clamp_block_table_to_working_set(table)

    assert dropped > 0, "geometry-based projection did not fire on a disk-backed chain"
    assert len(table.block_ids) < 100
    assert paged.released == [list(range(len(table.block_ids), 100))]


def test_no_resident_blocks_and_no_geometry_means_no_guessing(big_free):
    big_free(1.0)
    cache, paged = _cache(100 * 1024**2)
    for block in cache.paged_cache.allocated_blocks.values():
        block.cache_data = None
    table = _table(100)

    assert cache.clamp_block_table_to_working_set(table) == 0
    assert len(table.block_ids) == 100
    assert paged.released == []


def test_single_block_and_empty_tables_are_untouched(big_free):
    big_free(0.001)
    cache, _ = _cache(4 * 1024**3)

    assert cache.clamp_block_table_to_working_set(None) == 0
    assert cache.clamp_block_table_to_working_set(_table(1)) == 0


def test_release_failure_abandons_the_clamp(big_free):
    """Never leave the table inconsistent with the ref counts."""
    big_free(1.0)
    cache, paged = _cache(100 * 1024**2)

    def _boom(_table):
        raise RuntimeError("ref release failed")

    paged.release_request_refs = _boom
    table = _table(100)

    assert cache.clamp_block_table_to_working_set(table) == 0
    assert len(table.block_ids) == 100, "table was truncated without releasing refs"


def test_scheduler_recomputes_remaining_after_a_clamp():
    """remaining must be re-derived from the clamped num_tokens."""
    import inspect

    import vmlx_engine.scheduler as sched

    src = inspect.getsource(sched.Scheduler.add_request)
    assert "clamp_block_table_to_working_set" in src
    assert "remaining = list(_fetch_tokens[int(block_table.num_tokens or 0):])" in src
