"""Typed GLM native state over the aggregate companion disk transport.

These are codec/ownership tests, not proof of MLLM scheduler admission.
"""

import json

import mlx.core as mx
import pytest

from vmlx_engine.models.glm5_next.glm5_next import Glm5KDACache, Glm5MLACache
from vmlx_engine.utils.ssm_companion_cache import SSMCompanionCache
from vmlx_engine.utils.ssm_companion_disk_store import SSMCompanionDiskStore


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
