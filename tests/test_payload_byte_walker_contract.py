"""Shared payload byte-walker — per-tier semantics pinned.

Paged residency and disk admission historically counted payloads with
DIFFERENT walkers; divergence is the byte-budget-hole class. The shared
walker preserves each tier's exact semantics via flags — these pins keep
the flags honest.
"""

from __future__ import annotations

import numpy as np

from vmlx_engine.cache.byte_estimators import walk_payload_bytes


def _paged(value):
    return walk_payload_bytes(
        value,
        require_shape_for_arrays=False,
        dedupe_arrays=False,
        count_raw_bytes=False,
        walk_object_dicts=False,
        max_depth=None,
    )


def _disk(value):
    return walk_payload_bytes(
        value,
        require_shape_for_arrays=True,
        dedupe_arrays=True,
        count_raw_bytes=True,
        walk_object_dicts=True,
        max_depth=16,
    )


def test_paged_counts_aliased_arrays_every_occurrence():
    arr = np.zeros(16, dtype=np.uint8)
    total, tensors = _paged([arr, arr])
    assert total == 32
    assert tensors == 2


def test_disk_counts_aliased_arrays_once():
    arr = np.zeros(16, dtype=np.uint8)
    total, tensors = _disk([arr, arr])
    assert total == 16
    assert tensors == 1


def test_disk_requires_shape_paged_does_not():
    class FakeNbytes:
        nbytes = 64  # no .shape — a stray attribute, not an array

    total_paged, _ = _paged([FakeNbytes()])
    total_disk, _ = _disk([FakeNbytes()])
    assert total_paged == 64
    assert total_disk == 0


def test_disk_counts_raw_bytes_and_object_dicts():
    class Holder:
        def __init__(self):
            self.payload = np.zeros(8, dtype=np.uint8)

    total, tensors = _disk({"blob": b"12345", "obj": Holder()})
    assert total == 5 + 8
    assert tensors == 1
    total_paged, _ = _paged({"blob": b"12345", "obj": Holder()})
    assert total_paged == 0


def test_zero_byte_tensor_still_counts_toward_header_reservation():
    arr = np.zeros(0, dtype=np.uint8)
    _total, tensors = _disk([arr])
    assert tensors == 1


def test_self_referential_trees_terminate():
    d: dict = {}
    d["self"] = d
    assert _paged(d) == (0, 0)
    assert _disk(d) == (0, 0)


def test_dict_nested_composite_counts_in_both_tiers():
    arr = np.zeros(32, dtype=np.uint8)
    composite = ("deepseek_v4", {"state": {"kv": arr}}, {"meta": 1})
    assert _paged(composite)[0] == 32
    assert _disk(composite)[0] == 32
