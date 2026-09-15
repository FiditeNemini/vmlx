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
