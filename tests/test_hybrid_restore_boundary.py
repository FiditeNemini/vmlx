"""Hybrid restore invariant: restored KV offset must equal the SSM companion
boundary, else the hit is rejected (never paired split-brained)."""

from types import SimpleNamespace

from vmlx_engine.utils.cache_extent import hybrid_kv_boundary_mismatch


def _kv(offset):
    return SimpleNamespace(offset=offset, keys=object(), values=object())


def _ssm():
    return SimpleNamespace()  # recurrent slot: no offset attribute


def test_aligned_restore_passes():
    caches = [_kv(5951), _ssm(), _kv(5951), _ssm()]
    assert hybrid_kv_boundary_mismatch(caches, 5951) is None


def test_off_by_one_kv_is_rejected_with_layer_and_offset():
    caches = [_kv(5952), _ssm(), _kv(5952)]
    assert hybrid_kv_boundary_mismatch(caches, 5951) == (0, 5952)


def test_single_divergent_layer_is_caught():
    caches = [_kv(1000), _ssm(), _kv(999), _kv(1000)]
    assert hybrid_kv_boundary_mismatch(caches, 1000) == (2, 999)


def test_unknown_offsets_are_not_mismatches():
    # Recurrent slots and wrappers that never populate offset report 0.
    caches = [_ssm(), SimpleNamespace(offset=0), _ssm()]
    assert hybrid_kv_boundary_mismatch(caches, 4096) is None


def test_guards():
    assert hybrid_kv_boundary_mismatch([], 10) is None
    assert hybrid_kv_boundary_mismatch([_kv(5)], 0) is None
    assert hybrid_kv_boundary_mismatch([_kv(5)], "x") is None
