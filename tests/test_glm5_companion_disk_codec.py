"""Typed GLM native state over the aggregate companion disk transport.

These are codec/ownership tests, not proof of MLLM scheduler admission.
"""

import json
from dataclasses import replace

import mlx.core as mx
import pytest

from vmlx_engine.models.glm5_next.glm5_next import Glm5KDACache, Glm5MLACache
from vmlx_engine.utils.ssm_companion_cache import SSMCompanionCache
from vmlx_engine.utils.ssm_companion_disk_store import SSMCompanionDiskStore
from vmlx_engine.utils.glm5_native_blocks import split_native_sequence, restore_native_sequence


@pytest.mark.parametrize("length", [1, 4, 9, 17])
@pytest.mark.parametrize("absorbed", [False, True])
@pytest.mark.parametrize("pooled", [False, True])
def test_native_sequence_blocks_preserve_exact_typed_state(length, absorbed, pooled):
    original = native_state(length, absorbed=absorbed, pooled=pooled)
    boundary, blocks = split_native_sequence(original, 4)
    assert len(blocks) == (length + 3) // 4
    assert all(block.layers[0] is None for block in blocks)
    same_state(original, restore_native_sequence(boundary, blocks))


@pytest.mark.parametrize("damage", ["missing", "reverse", "duplicate", "length", "dtype", "pool", "kda"])
def test_native_sequence_blocks_reject_incomplete_or_mixed_state(damage):
    boundary, blocks = split_native_sequence(native_state(9), 4)
    if damage == "missing": blocks.pop(1)
    elif damage == "reverse": blocks.reverse()
    elif damage == "duplicate": blocks.insert(1, blocks[0])
    elif damage == "kda": boundary = replace(boundary, recurrent=(None, None))
    else:
        arrays = list(blocks[1].layers[1])
        if damage == "length": arrays[0] = arrays[0][:, :, :3]
        elif damage == "dtype": arrays[0] = arrays[0].astype(mx.float32)
        else: arrays[2] = arrays[2][:, :0]
        blocks[1] = replace(blocks[1], layers=(None, tuple(arrays)))
    with pytest.raises(ValueError):
        restore_native_sequence(boundary, blocks)


def test_native_sequence_blocks_keep_stable_full_blocks_and_partial_tail():
    _, old = split_native_sequence(native_state(9), 4)
    _, new = split_native_sequence(native_state(13), 4)
    for a, b in zip(old[:2], new[:2]):
        for x, y in zip(a.layers[1], b.layers[1]):
            assert mx.array_equal(x, y).item()
    assert (old[-1].start, old[-1].end) == (8, 9)
    assert (new[-1].start, new[-1].end) == (12, 13)


def test_native_sequence_blocks_refuse_unaligned_pool_slicing():
    with pytest.raises(ValueError, match="align"):
        split_native_sequence(native_state(9), 3)


def native_block_store(path):
    from vmlx_engine.block_disk_store import BlockDiskStore
    from vmlx_engine.utils.glm5_native_block_store import Glm5NativeBlockStore
    blocks = BlockDiskStore(str(path / "model"), max_size_gb=0.02,
                            global_cache_root=str(path), expected_num_layers=2)
    checkpoints = SSMCompanionDiskStore(
        directory=path / "model" / "ssm_companion", budget_bytes=20 * 1024**2,
        global_budget=blocks.global_budget,
    )
    return Glm5NativeBlockStore(checkpoints, blocks, block_size=4)


@pytest.mark.parametrize("absorbed", [False, True])
def test_native_block_ssd_extension_reuses_full_blocks_and_survives_restart(tmp_path, absorbed):
    store = native_block_store(tmp_path)
    try:
        first = native_state(9, absorbed=absorbed)
        assert store.store("a" * 64, first, True, list(range(9)), 9)
        assert store.wait_for_write("a" * 64)
        same_state(first, store.fetch("a" * 64)[0])
        extended = native_state(13, absorbed=absorbed)
        assert store.store("b" * 64, extended, True, list(range(13)), 13)
        assert store.wait_for_write("b" * 64)
        assert store.last_block_write["reused_blocks"] == 2
        assert store.last_block_write["new_blocks"] == 2
        same_state(extended, store.fetch("b" * 64)[0])
        same_state(first, store.fetch("a" * 64)[0])
    finally:
        store.shutdown()
    restored = native_block_store(tmp_path)
    try:
        same_state(first, restored.fetch("a" * 64)[0])
        same_state(extended, restored.fetch("b" * 64)[0])
    finally:
        restored.shutdown()


def test_native_block_ssd_missing_dependency_is_a_miss(tmp_path):
    store = native_block_store(tmp_path)
    try:
        assert store.store("c" * 64, native_state(9), True, list(range(9)), 9)
        assert store.wait_for_write("c" * 64)
        payload = next((tmp_path / "model" / "blocks").rglob("*.safetensors"))
        payload.unlink()
        assert store.fetch("c" * 64) is None
        assert not store.has_complete("c" * 64)
    finally:
        store.shutdown()


def test_native_block_ssd_aggregate_eviction_is_safe_and_recoverable(tmp_path):
    from vmlx_engine.global_disk_cache_budget import get_global_disk_cache_budget

    store = native_block_store(tmp_path)
    pressure = None
    key = "e" * 64
    original = native_state(13)
    try:
        assert store.store(key, original, True, list(range(13)), 13)
        assert store.wait_for_write(key)
        same_state(original, store.fetch(key)[0])
        # A second root lease imposes genuine aggregate pressure on both
        # transports; do not simulate eviction by unlinking a payload.
        pressure = get_global_disk_cache_budget(tmp_path, 1)
        result = pressure.enforce(force=True)
        assert result.evicted_entries > 0
        assert store.fetch(key) is None
        assert not store.has_complete(key)
        pressure.close()
        pressure = None
        # Capacity is available again: publish a complete dependency chain,
        # then verify exact native reconstruction rather than stale metadata.
        assert store.store(key, original, True, list(range(13)), 13)
        assert store.wait_for_write(key)
        same_state(original, store.fetch(key)[0])
    finally:
        if pressure is not None:
            pressure.close()
        store.shutdown()


def test_native_block_facade_restores_contiguous_prefix_after_restart(tmp_path):
    from vmlx_engine.utils.glm5_native_prefix_cache import Glm5NativePrefixCache, glm5_native_layout
    def open_cache():
        return Glm5NativePrefixCache(root=tmp_path, max_size_bytes=20 * 1024**2,
            model_key="block-facade", layout=glm5_native_layout(native_state(9)),
            sequence_block_size=4)
    cache = open_cache()
    try:
        assert cache.store(list(range(10)), 9, native_state(9))["durable"]
        assert cache.store(list(range(14)), 13, native_state(13))["durable"]
        assert cache.disk.last_block_write["reused_blocks"] == 2
    finally:
        cache.close()

    cache = open_cache()
    try:
        n, state = cache.fetch(list(range(15)))
        assert n == 13
        same_state(native_state(13), state)
        n, state = cache.fetch(list(range(10)))
        assert n == 9
        same_state(native_state(9), state)
        assert cache.fetch([99] + list(range(1, 15))) is None
    finally:
        cache.close()


def test_native_block_ssd_clear_owns_manifest_and_blocks(tmp_path):
    store = native_block_store(tmp_path)
    try:
        assert store.store("d" * 64, native_state(9), True, list(range(9)), 9)
        assert store.wait_for_write("d" * 64)
        stats = store.stats()
        assert stats["bytes"] > stats["checkpoint_bytes"] > 0
        assert stats["sequence_blocks"]["blocks_on_disk"] == 3
        store.clear()
        assert store.fetch("d" * 64) is None
        assert store.stats()["bytes"] == 0
        assert store.stats()["sequence_blocks"]["blocks_on_disk"] == 0
    finally:
        store.shutdown()
    assert not store.store("e" * 64, native_state(9), True, list(range(9)), 9)


def test_native_block_ssd_rejects_and_repairs_changed_payload(tmp_path):
    store = native_block_store(tmp_path)
    key = "f" * 64
    original = native_state(9)
    try:
        assert store.store(key, original, True, list(range(9)), 9)
        assert store.wait_for_write(key)
        path = next((tmp_path / "model" / "blocks").rglob("*.safetensors"))
        data, meta = mx.load(str(path), return_metadata=True)
        name = next(k for k in data if k.endswith("cumulative_1"))
        data[name] = data[name] + 1
        replacement = path.with_suffix(".replacement.safetensors")
        mx.save_safetensors(str(replacement), data, meta)
        replacement.replace(path)
        assert store.fetch(key) is None
        assert store.store(key, original, True, list(range(9)), 9)
        assert store.wait_for_write(key)
        same_state(original, store.fetch(key)[0])
    finally:
        store.shutdown()


def native_state(length, *, absorbed=True, pooled=True):
    kda = Glm5KDACache()
    kda.cache = [mx.full((1, 3, 8), i + 0.25, mx.bfloat16) for i in range(3)]
    kda.cache.append(mx.full((1, 2, 4, 4), 1e10, mx.float32))
    mla = Glm5MLACache(4, absorbed=absorbed)
    if absorbed:
        mla.update_latent(mx.full((1, 1, length, 8), 1e10, mx.bfloat16))
    else:
        mla.update_kv(mx.full((1, 2, length, 4), 1e10, mx.bfloat16),
                      mx.full((1, 2, length, 4), -1e10, mx.bfloat16))
    mla.update_packed(mx.arange(length * 8, dtype=mx.float32).reshape(1, length, 8))
    if pooled and length >= 4:
        mla.update_pool_keys(mx.full((1, length // 4, 4), 3.125, mx.float32))
    mx.eval(kda.state, mla.state)
    return [kda, mla]


def same_state(expected, actual):
    assert [type(x) for x in actual] == [type(x) for x in expected]
    for left, right in zip(expected, actual):
        assert left.meta_state == right.meta_state
        assert len(left.state) == len(right.state) == 4
        for a, b in zip(left.state, right.state):
            if a is None:
                assert b is None
            else:
                assert b.dtype == a.dtype
                assert b.shape == a.shape
                assert mx.array_equal(a, b).item()


@pytest.mark.parametrize("length", [1, 3, 4, 5, 8, 9])
@pytest.mark.parametrize("absorbed", [False, True])
@pytest.mark.parametrize("pooled", [False, True])
def test_exact_typed_round_trip_and_restart(tmp_path, length, absorbed, pooled):
    key = "1" * 64
    original = native_state(length, absorbed=absorbed, pooled=pooled)
    disk = SSMCompanionDiskStore(directory=tmp_path, budget_bytes=8 * 1024**2)
    try:
        assert disk.store(key, original, True, list(range(length)), length)
        assert disk.wait_for_write(key)
        restored, complete = disk.fetch(key)
        assert complete is True
        same_state(original, restored)
    finally:
        assert disk.shutdown()
    second = SSMCompanionDiskStore(directory=tmp_path, budget_bytes=8 * 1024**2)
    try:
        restored, complete = second.fetch(key)
        assert complete is True
        same_state(original, restored)
    finally:
        assert second.shutdown()


def test_freeze_and_refault_do_not_alias_live_arrays(tmp_path):
    key = "2" * 64
    original = native_state(5)
    expected = native_state(5)
    disk = SSMCompanionDiskStore(directory=tmp_path, budget_bytes=8 * 1024**2)
    try:
        assert disk.store(key, original, True, [1, 2, 3, 4, 5], 5)
        # store() freezes bytes before returning; the background writer must
        # never evaluate model state after the next token has mutated it.
        original[0].cache[-1] = mx.zeros_like(original[0].cache[-1])
        original[1].update_latent(mx.full((1, 1, 1, 8), 42, mx.bfloat16))
        original[1].update_packed(mx.full((1, 1, 8), 42, mx.float32))
        mx.eval(original[0].state, original[1].state)
        assert disk.wait_for_write(key)
        restored, _ = disk.fetch(key)
        same_state(expected, restored)
        restored[0].cache[-1] = mx.zeros_like(restored[0].cache[-1])
        restored[1].update_latent(mx.zeros((1, 1, 1, 8), mx.bfloat16))
        restored[1].update_packed(mx.zeros((1, 1, 8), mx.float32))
        mx.eval(restored[0].state, restored[1].state)
        again, _ = disk.fetch(key)
        same_state(expected, again)
    finally:
        assert disk.shutdown()


@pytest.mark.parametrize("change", [
    "unknown_class", "generic_kind", "missing_schema", "wrong_schema",
    "wrong_offset", "bad_presence", "missing_array", "wrong_length",
])
def test_corrupt_typed_descriptor_is_a_miss(tmp_path, change):
    key = "3" * 64
    disk = SSMCompanionDiskStore(directory=tmp_path, budget_bytes=8 * 1024**2)
    try:
        assert disk.store(key, native_state(5), True, [1, 2, 3, 4, 5], 5)
        assert disk.wait_for_write(key)
        _, side_path = disk._entry_paths(key)
        side = json.loads(side_path.read_text())
        meta = side["layer_metas"][1]
        if change == "unknown_class":
            meta["class"] = "UserSuppliedCache"
        elif change == "generic_kind":
            meta["kind"] = "ArraysCache"
        elif change == "missing_schema":
            meta.pop("native_meta_state")
        elif change == "wrong_schema":
            meta["native_meta_state"][0] = "unknown"
        elif change == "wrong_offset":
            meta["native_meta_state"][3] = "6"
        elif change == "bad_presence":
            meta["state_present"][0] = "yes"
        elif change == "missing_array":
            meta["state_present"][0] = False
        else:
            meta["state_len"] = 5
        side_path.write_text(json.dumps(side))
        assert disk.fetch(key) is None
    finally:
        assert disk.shutdown()


def test_invalid_native_state_is_refused_before_enqueue(tmp_path):
    bad = native_state(5)
    bad[0].cache[1] = None
    disk = SSMCompanionDiskStore(directory=tmp_path, budget_bytes=8 * 1024**2)
    try:
        assert disk.store("4" * 64, bad, True, [1, 2, 3, 4, 5], 5) is False
        assert disk.stats()["write_pipeline"]["pending_jobs"] == 0
    finally:
        assert disk.shutdown()


def test_companion_clone_keeps_native_capacity_ownership():
    original = native_state(5)
    expected = native_state(5)
    companion = SSMCompanionCache(max_entries=1, disk_store=False)
    companion.store([1, 2, 3, 4, 5], 5, original)
    original[1].update_latent(mx.zeros((1, 1, 1, 8), mx.bfloat16))
    original[1].update_packed(mx.zeros((1, 1, 8), mx.float32))
    mx.eval(original[1].state)
    restored, complete = companion.fetch([1, 2, 3, 4, 5], 5)
    assert complete is True
    same_state(expected, restored)
    restored[1].update_latent(mx.zeros((1, 1, 1, 8), mx.bfloat16))
    restored[1].update_packed(mx.zeros((1, 1, 8), mx.float32))
    mx.eval(restored[1].state)
    again, _ = companion.fetch([1, 2, 3, 4, 5], 5)
    same_state(expected, again)


def native_facade(root, *, limit=8 * 1024**2, model_key="test-glm"):
    from vmlx_engine.utils.glm5_native_prefix_cache import (
        Glm5NativePrefixCache, glm5_native_layout,
    )
    return Glm5NativePrefixCache(
        root=root, max_size_bytes=limit, model_key=model_key,
        layout=glm5_native_layout([Glm5KDACache(), Glm5MLACache(4, absorbed=True)]),
    )


def test_native_facade_longest_boundary_restart_and_clear(tmp_path):
    tokens = list(range(20))
    cache = native_facade(tmp_path)
    try:
        for boundary in (3, 5, 8):
            result = cache.store(tokens, boundary, native_state(boundary))
            assert result["outcome"] == "stored"
            assert result["durable"] is True
            assert result["retained_tokens"] == boundary
        assert cache.lookup.ram_enabled is False
        assert cache.lookup.total_nbytes == 0
        assert cache.lookup.size == 0
        assert cache.fetch(tokens)[0] == 8
        assert cache.fetch(tokens[:7])[0] == 5
        assert cache.fetch([99] + tokens[1:]) is None
        health = cache.budget.enforce(force=True)
        assert health.accounted and health.compliant
        assert 0 < health.bytes_after <= 8 * 1024**2
    finally:
        cache.close()
    after = native_facade(tmp_path)
    try:
        boundary, state = after.fetch(tokens)
        assert boundary == 8
        same_state(native_state(8), state)
        assert after.disk.stats()["global_budget"]["root"] == str(tmp_path)
        after.disk.clear()
        assert after.fetch(tokens) is None
    finally:
        after.close()


def test_native_facade_dedup_and_model_side_key_isolation(tmp_path):
    tokens = list(range(6))
    cache = native_facade(tmp_path)
    try:
        first = cache.store(tokens, 5, native_state(5), extra_keys={"schema": "A"})
        second = cache.store(tokens, 5, native_state(5), extra_keys={"schema": "A"})
        assert first["key"] == second["key"]
        assert second["outcome"] == "already_durable"
        assert cache.disk.stats()["stores"] == 1
        assert cache.fetch(tokens, extra_keys={"schema": "A"})[0] == 5
        assert cache.fetch(tokens, extra_keys={"schema": "B"}) is None
        assert cache.fetch(tokens) is None
    finally:
        cache.close()
    other = native_facade(tmp_path, model_key="different-bundle")
    try:
        assert other.fetch(tokens, extra_keys={"schema": "A"}) is None
    finally:
        other.close()


@pytest.mark.parametrize("invalid", ["offset", "pool", "layout", "empty_kda"])
def test_native_facade_rejects_wrong_model_boundary(tmp_path, invalid):
    state = native_state(5)
    if invalid == "offset":
        boundary = 4
    else:
        boundary = 5
    if invalid == "pool":
        state[1].kpool = 5  # same floor(5/k), but wrong native model geometry
    elif invalid == "layout":
        state = list(reversed(state))
    elif invalid == "empty_kda":
        state[0] = Glm5KDACache()
    cache = native_facade(tmp_path)
    try:
        result = cache.store(list(range(6)), boundary, state)
        assert result["outcome"] == "refused"
        assert result["durable"] is False
        assert result["retained_tokens"] == 0
        assert cache.disk.stats()["stores"] == 0
    finally:
        cache.close()


def test_native_facade_pool_capacity_refusal_is_not_durable(tmp_path):
    cache = native_facade(tmp_path, limit=4096)
    try:
        result = cache.store(list(range(4097)), 4096, native_state(4096))
        assert result["outcome"] == "refused"
        assert result["durable"] is False
        assert result["retained_tokens"] == 0
        assert cache.disk.stats()["stores"] == 0
        assert cache.lookup.total_nbytes == 0
    finally:
        cache.close()


def test_native_block_facade_duplicate_fetches_once(tmp_path, monkeypatch):
    from vmlx_engine.utils.glm5_native_prefix_cache import Glm5NativePrefixCache, glm5_native_layout
    original = native_state(9)
    cache = Glm5NativePrefixCache(
        root=tmp_path, max_size_bytes=20 * 1024**2,
        model_key="single-fetch", layout=glm5_native_layout(original),
        sequence_block_size=4,
    )
    try:
        first = cache.store(list(range(10)), 9, original)
        assert first["durable"]
        fetch = cache.disk.fetch
        observed = []
        def counted(key):
            found = fetch(key)
            observed.append(found)
            return found
        monkeypatch.setattr(cache.disk, "fetch", counted)
        second = cache.store(list(range(10)), 9, original)
        assert second["outcome"] == "already_durable"
        assert second["durable"] and second["retained_tokens"] == 9
        assert len(observed) == 1
        same_state(original, observed[0][0])
    finally:
        cache.close()


def test_native_plain_facade_keeps_incomplete_probe(tmp_path, monkeypatch):
    cache = native_facade(tmp_path)
    try:
        monkeypatch.setattr(cache.disk, "has_complete", lambda key: False)
        def unexpected_fetch(key):
            pytest.fail("plain incomplete record must not be fetched")
        monkeypatch.setattr(cache.disk, "fetch", unexpected_fetch)
        monkeypatch.setattr(cache.disk, "store", lambda *args, **kwargs: False)
        receipt = cache.store(list(range(6)), 5, native_state(5))
        assert not receipt["durable"]
        assert receipt["outcome"] == "refused"
    finally:
        cache.close()


@pytest.mark.parametrize("absorbed", [False, True])
def test_full_block_digests_are_memoized_per_live_cache_and_trim_clears_them(tmp_path, monkeypatch, absorbed):
    """Per-chunk GLM checkpoints must not re-hash the whole prefix (audit 2026-10-07: a 32k prefill spent 22.9 s in
    100 synchronous stores whose cost grew 0.05 -> 0.34 s with history). Full blocks of a live cache are immutable, so
    their digests are computed once per cache object; trim() drops the memo because positions past it may change."""
    import vmlx_engine.utils.glm5_native_block_store as nbs

    calls = []
    real = nbs._block_hash
    monkeypatch.setattr(nbs, "_block_hash", lambda block, parent: calls.append((block.start, block.end)) or real(block, parent))
    store = native_block_store(tmp_path)
    try:
        live = native_state(9, absorbed=absorbed)          # blocks [0,4) [4,8) full + [8,9) partial
        assert store.store("a" * 64, live, True, list(range(9)), 9)
        assert store.wait_for_write("a" * 64)
        assert calls == [(0, 4), (4, 8), (8, 9)]
        calls.clear()
        assert store.store("b" * 64, live, True, list(range(9)), 9)  # same live object, later checkpoint
        assert store.wait_for_write("b" * 64)
        assert calls == [(8, 9)]                             # only the partial tail is re-hashed
        same_state(live, store.fetch("b" * 64)[0])           # the memoized checkpoint restores exactly
        memo = live[1]._vmlx_native_block_digests
        parent = None
        for fragment in split_native_sequence(native_state(9, absorbed=absorbed), 4)[1]:
            digest = real(fragment, parent)                   # fresh cache object, recomputed from bytes
            if fragment.end - fragment.start == 4:
                assert memo[(fragment.start, fragment.end, parent, nbs._fragment_layout(fragment))] == digest
            parent = digest
        mla = live[1]
        assert getattr(mla, "_vmlx_native_block_digests", None)
        mla.trim(1)
        assert getattr(mla, "_vmlx_native_block_digests", None) is None
    finally:
        store.shutdown()


@pytest.mark.parametrize("absorbed", [False, True])
def test_memoized_digest_follows_dsa_pool_materialization(tmp_path, absorbed):
    """Live regression (audit 2026-10-07): a dense-only prefix (within index_topk) keeps a zero-length DSA pool; once
    the indexer engages, the SAME (start, end, parent) full block gains pool rows. A memo keyed without the fragment
    layout returned the pool-less digest, the block was counted as reused and never written, and a 32k prompt
    restored only 2,048 tokens. The memo key must change with the fragment layout."""
    store = native_block_store(tmp_path)
    try:
        live = native_state(9, absorbed=absorbed, pooled=False)
        assert store.store("a" * 64, live, True, list(range(9)), 9)
        assert store.wait_for_write("a" * 64)
        live[1].update_pool_keys(mx.full((1, 9 // 4, 4), 3.125, mx.float32))  # indexer engages
        assert store.store("b" * 64, live, True, list(range(9)), 9)
        assert store.wait_for_write("b" * 64)
        restored = store.fetch("b" * 64)
        assert restored is not None
        same_state(native_state(9, absorbed=absorbed, pooled=True), restored[0])
    finally:
        store.shutdown()
