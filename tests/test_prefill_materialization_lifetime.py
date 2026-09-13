# SPDX-License-Identifier: Apache-2.0
"""Prefill realizes live cache arrays without retaining superseded chunks."""

import gc
import weakref
from types import SimpleNamespace

import pytest

import vmlx_engine.mllm_batch_generator as mllm


class ArrayMarker:
    """Weak-referenceable stand-in; collection must not depend on array math."""


@pytest.fixture
def cyclic_gc_disabled():
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        gc.collect()
        if enabled:
            gc.enable()


@pytest.mark.parametrize("nested", [False, True])
def test_collector_does_not_keep_arrays_after_caller_releases_them(
    cyclic_gc_disabled, nested
):
    array = ArrayMarker()
    ref = weakref.ref(array)
    entry = SimpleNamespace(cache=[array, None])
    cache = [SimpleNamespace(caches=(entry,))] if nested else [entry]
    items = mllm._prefill_cache_materialization_items(cache)
    assert len(items) == 1 and items[0] is array
    del array, entry, cache, items
    assert ref() is None, "collector retained a chunk until cyclic GC"


def test_materializer_releases_every_replaced_chunk(cyclic_gc_disabled, monkeypatch):
    calls = []

    def evaluate(*arrays):
        calls.append(tuple(id(array) for array in arrays))

    monkeypatch.setattr(mllm.mx, "eval", evaluate)
    cache = SimpleNamespace(cache=[])
    refs = []
    for _ in range(12):
        array = ArrayMarker()
        refs.append(weakref.ref(array))
        cache.cache = [array]
        mllm._materialize_prefill_cache_state([cache])
        del array
        assert all(ref() is None for ref in refs[:-1])
        assert refs[-1]() is cache.cache[0]
    cache.cache = []
    assert all(ref() is None for ref in refs)
    assert len(calls) == 12


def test_nested_cache_traversal_preserves_order_identity_and_duplicates():
    arrays = [ArrayMarker() for _ in range(8)]
    keys_values = SimpleNamespace(
        keys=[arrays[0], None, arrays[1]], values=(arrays[2],),
        cache=[arrays[7]],  # KV attributes keep precedence over other schemas.
    )
    nested = SimpleNamespace(caches=(
        None,
        SimpleNamespace(cache=[None, arrays[3]]),
        SimpleNamespace(caches=[SimpleNamespace(state=(arrays[4], None))]),
    ))
    state_scalar = SimpleNamespace(state=arrays[5])
    state_list = SimpleNamespace(state=[arrays[6]])
    items = mllm._prefill_cache_materialization_items(
        [None, keys_values, nested, state_scalar, state_list, keys_values]
    )
    expected = arrays[:7] + arrays[:3]
    assert len(items) == len(expected)
    assert all(actual is expected_item for actual, expected_item in zip(items, expected))


@pytest.mark.parametrize("cache", [None, [], [None], [SimpleNamespace(cache=[])]])
def test_empty_cache_never_evaluates(cache, monkeypatch):
    def unexpected_eval(*arrays):
        raise AssertionError("empty cache must not call mx.eval")

    monkeypatch.setattr(mllm.mx, "eval", unexpected_eval)
    mllm._materialize_prefill_cache_state(cache)
