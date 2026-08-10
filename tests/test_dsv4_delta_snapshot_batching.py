# SPDX-License-Identifier: Apache-2.0
"""Correctness of the batched DSV4 block-delta snapshot materialization.

The DSV4 extended-store boundary capture copies ~7 tensors per composite layer
at every 256-token boundary. Materializing each copy with its own blocking
mx.eval serialized hundreds of GPU round-trips onto the decode thread (the
long-output per-256-token stall). The fix batches every copy into one
async_eval. This test proves the batched path is byte-identical to the
per-leaf-eval path AND that the copy stays detached when the source array is
rebound between the copy and its eval (the decode-step cache update pattern).
"""
import platform
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon MLX",
)

import mlx.core as mx

from jang_tools.dsv4.mlx_model import DeepseekV4Cache


def test_collector_defers_eval_but_matches_eager_values():
    src = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    eager = DeepseekV4Cache._copy_delta_tree(src)  # per-leaf eval path

    collector = []
    batched = DeepseekV4Cache._copy_delta_tree(src, collector)
    assert len(collector) == 1  # detached leaf queued, not yet materialized
    mx.async_eval(*collector)

    assert mx.array_equal(eager, batched)
    assert mx.array_equal(batched, src)


def test_batched_copy_stays_detached_when_source_is_rebound():
    # The cache rebinds its buffer arrays each decode step. A deferred copy must
    # snapshot the value at copy-build time, not read the rebound array.
    src = mx.array([10.0, 20.0, 30.0])
    collector = []
    snapshot = DeepseekV4Cache._copy_delta_tree(src, collector)

    # Rebind src as the cache would (functional update -> new array object).
    src = src * 0.0 + 999.0
    mx.eval(src)

    mx.eval(*collector)
    assert mx.array_equal(snapshot, mx.array([10.0, 20.0, 30.0]))


def test_collector_threads_through_nested_trees():
    tree = {
        "a": mx.array([1.0, 2.0]),
        "b": (mx.array([3.0]), mx.array([4.0, 5.0])),
        "c": [mx.array([6.0]), None],
        "meta": ("not", "an", "array"),
    }
    collector = []
    copied = DeepseekV4Cache._copy_delta_tree(tree, collector)
    # 4 array leaves queued; string/None leaves are pass-through.
    assert len(collector) == 4
    mx.async_eval(*collector)
    assert mx.array_equal(copied["a"], tree["a"])
    assert mx.array_equal(copied["b"][1], tree["b"][1])
    assert copied["meta"] == ("not", "an", "array")
    assert copied["c"][1] is None


def test_none_collector_preserves_eager_eval_default():
    src = mx.array([7.0, 8.0])
    out = DeepseekV4Cache._copy_delta_tree(src)  # default path, no collector
    assert mx.array_equal(out, src)
