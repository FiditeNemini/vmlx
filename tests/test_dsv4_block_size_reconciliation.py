"""DSV4's 256-token native delta granularity must survive every launcher.

The 256 force lived only in cli.serve_command, inside a try whose except logs
at DEBUG — while generator selection is a SEPARATE, class-based detection. A
registry lookup that threw therefore left a 64-block cutter driving native-256
records: interval (0, 64) never matches, and 100% of stores abort with a
generic ValueError naming no cause.
"""

from types import SimpleNamespace

import pytest

from vmlx_engine.utils.dsv4_batch_generator import DSV4_NATIVE_BLOCK_SIZE


def test_native_block_size_is_still_256():
    assert DSV4_NATIVE_BLOCK_SIZE == 256


def test_scheduler_reconciles_a_wrong_block_size_instead_of_shipping_it():
    import inspect

    from vmlx_engine.scheduler import Scheduler

    src = inspect.getsource(Scheduler.__init__)
    assert "DSV4 paged block size reconciled" in src, (
        "the reconciliation must live in the Scheduler, where every launcher "
        "converges — not in one CLI command function"
    )
    assert "self.config.paged_cache_block_size = DSV4_NATIVE_BLOCK_SIZE" in src

    # Ordering: reconcile BEFORE the manager is built, or it is inert.
    reconcile_at = src.index("DSV4 paged block size reconciled")
    construct_at = src.index("PagedCacheManager(")
    assert reconcile_at < construct_at, (
        "reconciling after PagedCacheManager is constructed changes nothing"
    )


def test_store_raises_a_NAMED_error_on_a_block_size_mismatch():
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
    cache.block_size = 64
    cache.paged_cache = SimpleNamespace(_disk_store=None)

    # is_tensor_data requires a list whose first entry is a dict with
    # "state"; _cache_data_has_dsv4_deltas requires every entry to carry both
    # dsv4_block_records and dsv4_record_intervals.
    cache_data = [
        {
            "state": ("k", "v"),
            "dsv4_block_records": [{"r": 1}],
            "dsv4_record_intervals": [(0, 256)],
        }
    ]
    with pytest.raises(ValueError) as excinfo:
        cache._store_cache_impl(
            request_id="r",
            tokens=list(range(300)),
            cache_data=cache_data,
            _write_fence=None,
        )
    msg = str(excinfo.value)
    assert "block_size=64" in msg and "256" in msg, (
        "the error must name the mismatch, not just say an interval is "
        "missing: %s" % msg
    )
