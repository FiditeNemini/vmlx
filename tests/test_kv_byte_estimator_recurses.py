# SPDX-License-Identifier: Apache-2.0
"""The KV byte estimator must see nested cache states.

``_estimate_state_memory`` was a flat one-level scan, so anything whose elements
are themselves containers counted as ZERO: QuantizedKVCache extracted state
(tuples of packed/scales/biases), CacheList sub-caches (a list of per-layer
dicts), and DSV4 transport dicts. Every ``--kv-cache-quantization`` entry and
every MoE-hybrid entry was therefore invisible to the byte budgets that call
this — a cache "bounded by bytes" that could still grow without limit.
"""

from __future__ import annotations

import numpy as np

from vmlx_engine.memory_cache import _estimate_state_memory

BLOCK = 1000


def _arr():
    return np.zeros(BLOCK, dtype=np.uint8)


def test_flat_state_still_counted():
    assert _estimate_state_memory([_arr(), _arr()]) == 2 * BLOCK


def test_quantized_tuple_of_tuples_is_not_zero():
    """packed/scales/biases nested one level down used to vanish."""
    state = [(_arr(), _arr(), _arr())]
    assert _estimate_state_memory(state) == 3 * BLOCK


def test_nested_dict_states_are_not_zero():
    """CacheList sub-caches and DSV4 transport dicts used to vanish."""
    state = {"layer0": {"state": (_arr(), _arr())}, "layer1": {"state": (_arr(),)}}
    assert _estimate_state_memory(state) == 3 * BLOCK


def test_aliased_arrays_are_counted_once():
    """A slice or COW copy aliases the same buffer; counting twice over-reports."""
    shared = _arr()
    assert _estimate_state_memory([shared, shared, (shared,)]) == BLOCK
