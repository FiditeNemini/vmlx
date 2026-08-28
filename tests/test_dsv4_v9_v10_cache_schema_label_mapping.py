# SPDX-License-Identifier: Apache-2.0
"""Resolve whether the DSV4 v9 bundle schema and the engine's internal
``deepseek_v4_v10_delta`` schema describe the SAME retained state under
renamed labels, or a genuine functional gap.

Background: DeepSeek-V4-Flash-0731-JANG-CRACK's own ``jang_config.json``
declares ``cache.schema = "deepseek_v4_v9"`` with
``cache.components = ["swa", "csa", "hca", "compressor", "indexer"]``
(5 entries). The engine never reads that field (grepped: zero references to
``native_cache_schema``/``dsv4_runtime_requirements`` anywhere in
vmlx_engine/). Internally the engine labels its own schema
``deepseek_v4_v10_delta`` in two DIFFERENT places that disagree with each
other:

  * ``_dsv4_native_state_memory_status`` (server.py, the memory-estimate
    "includes" list) enumerates 5 components: swa_local,
    csa_compressed_pool, csa_indexer_pool, hca_compressed_pool,
    incomplete_tail_state.
  * ``_native_cache_status`` (server.py, the health/status "schema"/
    "components" attestation actually reported as
    ``"schema": "deepseek_v4_v10_delta"``) used to enumerate only 4:
    swa_local, csa_compressed_pool, hca_compressed_pool,
    incomplete_tail_state -- silently dropping the indexer pool from the
    public component list, even though ``layer_cache_roles.ratio_4`` still
    described it in prose ("csa_overlap_compressed_pool_plus_indexer").
    FIXED below: both lists now derive from one module-level constant,
    ``_DSV4_NATIVE_CACHE_COMPONENTS``.

This file proves two things with real objects, not string comparison:

1. The REAL cache class (jang_tools.dsv4.pool_quant_cache.PoolQuantizedV4Cache)
   always carries both ``compressor_state`` and ``indexer_state`` as
   first-class attributes regardless of which server.py label list is used
   to describe it, and both round-trip losslessly through ``storage_state``
   (the native q8-preserving path) independent of the health-endpoint label
   drift. So the label drift below is COSMETIC, not a serialization gap --
   see tests/test_dsv4_paged_cache.py::test_dsv4_pool_q8_disk_tree_and_reconstruction_are_lossless
   and ::test_dsv4_block_disk_serialization_round_trips_nested_state for the
   pre-existing, independent proof of that losslessness this file leans on.

2. The two server.py label lists genuinely disagree with each other (a real
   bug in the health/telemetry reporting layer, independent of the bundle's
   v9 declaration): the "includes" list has 5 entries, the "components"
   list has 4, and the missing one is the indexer pool. This is a
   regression lock -- it fails the moment someone "fixes" one list without
   the other, which is exactly the kind of silent drift that produced this
   investigation.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

mx = pytest.importorskip("mlx.core")


# ---------------------------------------------------------------------------
# Part 1: the real cache object -- compressor_state/indexer_state are the
# SAME two attributes regardless of whether the layer plays the "CSA"
# (ratio 4) or "HCA" (ratio 128) role.  server.py's csa_compressed_pool and
# hca_compressed_pool labels both refer to `.compressor_state` on different
# per-layer cache instances; only ratio-4 (CSA) layers populate
# `.indexer_state`.  This is read directly off jang_tools, not asserted.
# ---------------------------------------------------------------------------


def _csa_layer_cache():
    """A ratio-4 layer cache: CSA role, uses BOTH compressor and indexer."""
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache

    cache = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)
    cache.local.state = (
        mx.ones((1, 1, 7, 64), dtype=mx.float16),
        mx.ones((1, 1, 7, 64), dtype=mx.float16) * 2,
    )
    cache.meta_state = ("0", "128", "7", "7")
    cache.update_pool(mx.ones((1, 5, 64), dtype=mx.float16) * 3, "compressor_state")
    cache.update_pool(mx.ones((1, 5, 64), dtype=mx.float16) * 4, "indexer_state")
    # buffer_kv/buffer_gate are the "incomplete_tail_state" subfields --
    # populate them directly since update_pool() only ever touches "pooled".
    cache.compressor_state["buffer_kv"] = mx.ones((1, 3, 64), dtype=mx.float16) * 6
    cache.compressor_state["buffer_gate"] = mx.ones((1, 3, 64), dtype=mx.float16) * 7
    cache.indexer_state["buffer_kv"] = mx.ones((1, 3, 64), dtype=mx.float16) * 8
    cache.indexer_state["buffer_gate"] = mx.ones((1, 3, 64), dtype=mx.float16) * 9
    return cache


def _hca_layer_cache():
    """A ratio-128 layer cache: HCA role, per the estimator only retains a
    compressor pool (no indexer pool -- see memory_limits.py ratio!=4
    branch, which never touches csa_indexer_bytes)."""
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache

    cache = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=128)
    cache.local.state = (
        mx.ones((1, 1, 7, 64), dtype=mx.float16),
        mx.ones((1, 1, 7, 64), dtype=mx.float16) * 2,
    )
    cache.meta_state = ("0", "128", "7", "7")
    cache.update_pool(mx.ones((1, 1, 64), dtype=mx.float16) * 5, "compressor_state")
    # indexer_state deliberately left empty: the real runtime never routes
    # an Indexer branch into it for a non-CSA (ratio != 4) layer.
    return cache


def test_csa_and_hca_layers_share_the_same_two_attribute_names():
    """csa_compressed_pool and hca_compressed_pool (v10_delta labels) both
    resolve to `.compressor_state` on the underlying object -- there is no
    separate "hca" attribute. This is the concrete answer to "is
    csa_compressed_pool the same state as compressor_state, just relabeled":
    yes, and hca_compressed_pool is *also* the same `.compressor_state`
    attribute, on a different (ratio-128) instance.
    """
    csa = _csa_layer_cache()
    hca = _hca_layer_cache()

    assert hasattr(csa, "compressor_state") and hasattr(csa, "indexer_state")
    assert hasattr(hca, "compressor_state") and hasattr(hca, "indexer_state")

    # CSA (ratio 4) populates both pools.
    assert csa.compressor_state._pooled_bf16 is not None
    assert csa.indexer_state._pooled_bf16 is not None

    # HCA (ratio 128) only ever gets a compressor pool in real usage; the
    # indexer branch stays empty -- this is *why* v10_delta's public
    # "components" list only names hca_compressed_pool for ratio_128
    # (see layer_cache_roles.ratio_128 == "hca_compressed_pool" at
    # server.py:11315) and not an "hca indexer" concept that does not exist.
    assert hca.compressor_state._pooled_bf16 is not None
    assert hca.indexer_state._pooled_bf16 is None


def test_compressor_and_indexer_state_round_trip_losslessly_via_storage_state():
    """Direct, no-model round trip of the SAME object graph server.py's
    v10_delta label list claims to describe, using the real lossless
    ``storage_state`` transport (not the lossy compat ``.state`` getter).
    This is the concrete answer to "is this a real functional gap": no --
    both compressor_state and indexer_state (v9's "compressor"/"indexer",
    v10_delta's csa_compressed_pool/csa_indexer_pool) survive an export/
    import cycle bit-for-bit, independent of what server.py's health
    endpoint chooses to call them.
    """
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache

    source = _csa_layer_cache()
    exported = source.storage_state

    restored = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)
    restored.storage_state = exported

    for attr in ("compressor_state", "indexer_state"):
        src_state = getattr(source, attr)
        dst_state = getattr(restored, attr)
        assert mx.array_equal(
            src_state["pooled"], dst_state["pooled"]
        ).item(), f"{attr} pooled rows did not round-trip losslessly"
        assert mx.array_equal(
            src_state["buffer_kv"], dst_state["buffer_kv"]
        ).item(), f"{attr} buffer_kv (incomplete_tail_state) did not round-trip"
        assert mx.array_equal(
            src_state["buffer_gate"], dst_state["buffer_gate"]
        ).item(), f"{attr} buffer_gate (incomplete_tail_state) did not round-trip"


def test_incomplete_tail_state_is_a_subfield_not_a_separate_cache_object():
    """v10_delta's 'incomplete_tail_state' has no analog in v9's 5-item
    component list (swa, csa, hca, compressor, indexer). Confirm it is not
    a missing FIFTH storage object v9 anticipated and v10 dropped -- it is
    the buffer_kv/buffer_gate slots that already live INSIDE compressor_state
    and indexer_state (jang_tools mlx_model.py's _STATE_KEYS =
    ("buffer_kv", "buffer_gate", "pooled")). v9 never named it separately
    because it was never a separate object; v10_delta's label just surfaces
    a subfield that was always implicitly part of "compressor"/"indexer".
    """
    from jang_tools.dsv4.pool_quant_cache import _STATE_KEYS

    assert _STATE_KEYS == ("buffer_kv", "buffer_gate", "pooled")
    # Both "tail" fields (buffer_kv, buffer_gate) and the "pool" field live
    # in the exact same state container -- there is no independent
    # incomplete_tail_state object anywhere in the class.
    csa = _csa_layer_cache()
    assert set(csa.compressor_state.keys()) == set(_STATE_KEYS)
    assert set(csa.indexer_state.keys()) == set(_STATE_KEYS)


# ---------------------------------------------------------------------------
# Part 2: server.py's two label lists used to disagree with each other (v10
# vs v10, not v9 vs v10) -- a reporting/attestation bug, not a
# state-preservation bug (proved above). Fixed by deriving both from one
# module-level constant, ``_DSV4_NATIVE_CACHE_COMPONENTS``; these tests now
# prove that single source of truth stays authoritative and complete.
# ---------------------------------------------------------------------------


def test_native_cache_components_constant_names_all_five_pool_concepts():
    """The single source of truth both server.py label lists now derive
    from enumerates all 5 real per-layer pool concepts, matching v9's
    declared cardinality of 5 (even though the engine never reads v9's
    declaration -- see the module docstring)."""
    from vmlx_engine.server import _DSV4_NATIVE_CACHE_COMPONENTS

    assert _DSV4_NATIVE_CACHE_COMPONENTS == [
        "swa_local",
        "csa_compressed_pool",
        "csa_indexer_pool",
        "hca_compressed_pool",
        "incomplete_tail_state",
    ]


def test_memory_estimate_and_native_cache_status_report_the_same_components():
    """`_dsv4_native_state_memory_status`'s "includes" list and
    `_native_cache_status`'s "components" list -- the one actually reported
    as ``"schema": "deepseek_v4_v10_delta"`` at the health/status endpoint
    -- used to disagree (4 vs 5 entries, missing csa_indexer_pool from the
    public attestation) even though `layer_cache_roles.ratio_4` in the same
    function still described the indexer in prose. Both now derive from
    `_DSV4_NATIVE_CACHE_COMPONENTS`, so a caller reading either surface sees
    the complete, identical component list -- including the indexer pool,
    which was proven above to already serialize and restore losslessly
    regardless of this label's history.
    """
    from vmlx_engine.server import _DSV4_NATIVE_CACHE_COMPONENTS

    memory_estimate_source = textwrap.dedent(
        inspect.getsource(
            __import__(
                "vmlx_engine.server", fromlist=["_dsv4_native_state_memory_status"]
            )._dsv4_native_state_memory_status
        )
    )
    assert "_DSV4_NATIVE_CACHE_COMPONENTS" in memory_estimate_source

    module_source = inspect.getsource(__import__("vmlx_engine.server", fromlist=["*"]))
    tree = ast.parse(module_source)
    status_dict_uses_constant = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = {
            key.value: value
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant)
        }
        schema_value = keys.get("schema")
        components_value = keys.get("components")
        if (
            isinstance(schema_value, ast.Constant)
            and schema_value.value == "deepseek_v4_v10_delta"
            and isinstance(components_value, ast.Call)
            and isinstance(components_value.func, ast.Name)
            and components_value.func.id == "list"
            and len(components_value.args) == 1
            and isinstance(components_value.args[0], ast.Name)
            and components_value.args[0].id == "_DSV4_NATIVE_CACHE_COMPONENTS"
        ):
            status_dict_uses_constant = True
            break
    assert status_dict_uses_constant, (
        "the deepseek_v4_v10_delta status dict's 'components' key must be "
        "list(_DSV4_NATIVE_CACHE_COMPONENTS), not a separately hand-written "
        "literal -- that duplication is exactly what caused the original drift"
    )
    assert len(_DSV4_NATIVE_CACHE_COMPONENTS) == 5
    assert "csa_indexer_pool" in _DSV4_NATIVE_CACHE_COMPONENTS
