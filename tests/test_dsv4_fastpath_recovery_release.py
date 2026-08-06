# SPDX-License-Identifier: Apache-2.0
"""Hardening gates for the DSV4 exact fastpaths after PR #250.

Pins the review findings from the integration pass: cache-error recovery
must release the pinned FP32 head so retries run under lower pressure, the
runtime env gate must both bypass and release, and the shared RoPE tables
must be bit-identical to the real jang_tools rotation including inverse and
positions handling.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest


class _RecordingMx:
    def __init__(self):
        self.limits = []

    def set_cache_limit(self, value):
        self.limits.append(value)


def _fresh_lm_head_module(monkeypatch, mx_stub):
    import vmlx_engine.models.dsv4_lm_head_fastpath as mod

    monkeypatch.setattr(mod, "mx", mx_stub)
    monkeypatch.setattr(mod, "_CONFIGS", mod._WeakIdentityMap())
    monkeypatch.setattr(mod, "_CLASS_PATCHES", {})
    monkeypatch.setattr(mod, "_RESERVATIONS", {})
    monkeypatch.setattr(mod, "_RESERVATION_LOCK", threading.RLock())
    monkeypatch.setattr(mod, "_CACHE_LIMIT_BASE_BYTES", None)
    monkeypatch.setattr(mod, "_CACHE_LIMIT_EFFECTIVE_BYTES", None)
    return mod


def _install_real_lm_head_model(monkeypatch, mod, mx):
    monkeypatch.setattr(mod, "_cache_admitted", lambda *_a: True)
    monkeypatch.setenv("VMLX_DSV4_LM_HEAD_MODE", "exact-cache")

    vocab, dims, group = 64, 128, 32
    source = mx.random.normal((vocab, dims)).astype(mx.float16)
    weight_q, scales, biases = mx.quantize(source, group_size=group, bits=8)

    class _Head:
        def __init__(self):
            self.weight = weight_q
            self.scales = scales
            self.biases = biases
            self.bits = 8
            self.group_size = group
            self.mode = "affine"

        def __contains__(self, _key):
            return False

    hidden = mx.random.normal((1, 3, dims)).astype(mx.float16)

    class _Model:
        model_type = "deepseek_v4"

        def __init__(self):
            self.lm_head = _Head()

        def model(self, _input_ids, cache=None, mask=None):
            del cache, mask
            return hidden

        def __call__(self, input_ids, cache=None, mask=None):
            return mod._reference_logits(
                self.model(input_ids, cache=cache, mask=mask), self.lm_head
            )

    model = _Model()
    return model


def test_release_all_releases_reservations_and_restores_ceiling(monkeypatch):
    mx_stub = _RecordingMx()
    mod = _fresh_lm_head_module(monkeypatch, mx_stub)
    monkeypatch.setenv("VMLX_DSV4_MLX_CACHE_MIN_GB", "1")
    gib = 1024**3

    class _Model:
        pass

    model = _Model()
    assert mod.configure_dsv4_lm_head_cache_budget(8 * gib) == 8 * gib
    assert mod._reserve_allocator_cache(model, 2 * gib)
    mod._CONFIGS[model] = mod._HeadConfig(
        signature=("sig",), weight=object(), reservation_bytes=2 * gib
    )
    assert mx_stub.limits[-1] == 6 * gib

    released = mod.release_all_dsv4_lm_head_caches("cache-error recovery")

    assert released == 1
    assert not mod._RESERVATIONS
    assert mx_stub.limits[-1] == 8 * gib
    status = mod.dsv4_lm_head_fastpath_status(model)
    assert status["disabled_reason"] == "cache-error recovery"
    assert status["reservation_bytes"] == 0


def test_release_all_ignores_foreign_live_model_on_id_guard(monkeypatch):
    mx_stub = _RecordingMx()
    mod = _fresh_lm_head_module(monkeypatch, mx_stub)
    monkeypatch.setenv("VMLX_DSV4_MLX_CACHE_MIN_GB", "1")
    gib = 1024**3

    class _Model:
        pass

    owner = _Model()
    other = _Model()
    mod.configure_dsv4_lm_head_cache_budget(8 * gib)
    assert mod._reserve_allocator_cache(owner, 2 * gib)

    # A release keyed to the owner's id but attributed to a different live
    # model must not drop the owner's reservation.
    mod._release_cache_reservation(id(owner), owner=other)
    assert mod._RESERVATIONS

    mod._release_cache_reservation(id(owner), owner=owner)
    assert not mod._RESERVATIONS
    assert mx_stub.limits[-1] == 8 * gib


def test_cache_error_recovery_releases_lm_head_caches(monkeypatch):
    import vmlx_engine.models.dsv4_lm_head_fastpath as mod
    from vmlx_engine.scheduler import Scheduler

    calls = []
    monkeypatch.setattr(
        mod,
        "release_all_dsv4_lm_head_caches",
        lambda reason: calls.append(reason) or 1,
    )

    scheduler = object.__new__(Scheduler)
    scheduler.batch_generator = SimpleNamespace(close=lambda: None)
    scheduler._current_sampler_params = None
    scheduler.block_aware_cache = None
    scheduler.memory_aware_cache = None
    scheduler.prefix_cache = None
    scheduler._ssm_state_cache = None
    scheduler.request_id_to_uid = {}
    scheduler.uid_to_request_id = {}

    scheduler._recover_from_cache_error()

    assert calls == ["cache-error recovery"]


def test_env_gate_flip_releases_reservation_and_routes_reference(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    import vmlx_engine.models.dsv4_lm_head_fastpath as mod

    monkeypatch.setattr(mod, "_CONFIGS", mod._WeakIdentityMap())
    monkeypatch.setattr(mod, "_CLASS_PATCHES", {})
    monkeypatch.setattr(mod, "_RESERVATIONS", {})
    monkeypatch.setattr(mod, "_reserve_allocator_cache", lambda *_a: True)
    released = []
    monkeypatch.setattr(
        mod,
        "_release_cache_reservation",
        lambda identity, reference=None, owner=None: released.append(identity),
    )

    model = _install_real_lm_head_model(monkeypatch, mod, mx)
    reference = model("ids")
    mx.eval(reference)

    assert mod.install_dsv4_lm_head_fastpath(model)
    monkeypatch.setenv("VMLX_DSV4_LM_HEAD_MODE", "native")
    flipped = model("ids")
    mx.eval(flipped)

    assert bool(mx.array_equal(flipped, reference))
    assert released == [id(model)]
    status = mod.dsv4_lm_head_fastpath_status(model)
    assert status["disabled_reason"] is not None
    assert status["cache_bytes"] == 0


def _fresh_rope_module(monkeypatch, mx):
    import vmlx_engine.models.dsv4_rope_cache as mod

    monkeypatch.setattr(mod, "mx", mx)
    monkeypatch.setattr(mod, "_CONFIGS", mod._WeakIdentityMap())
    monkeypatch.setattr(mod, "_GROUPS", {})
    monkeypatch.setattr(mod, "_GROUP_LOCK", threading.RLock())
    monkeypatch.setattr(mod, "_CLASS_PATCHES", {})
    monkeypatch.setenv("VMLX_DSV4_ROPE_CACHE", "1")
    return mod


@pytest.mark.parametrize("yarn", [False, True])
def test_rope_tables_bit_identical_to_real_jang_rope(monkeypatch, yarn):
    mx = pytest.importorskip("mlx.core")
    jang_model = pytest.importorskip("jang_tools.dsv4.mlx_model")
    mod = _fresh_rope_module(monkeypatch, mx)

    scaling = (
        {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 4096}
        if yarn
        else None
    )
    rope = jang_model.DeepseekV4RoPE(8, 10000.0, scaling_config=scaling)
    rope_class = jang_model.DeepseekV4RoPE
    original_call = rope_class.__call__
    cases = [
        (0, 1, False),
        (0, 5, False),
        (7, 5, False),
        (1000, 1, False),
        (7, 5, True),
        (1000, 1, True),
    ]
    try:
        expected = {}
        for dtype in (mx.float16, mx.bfloat16):
            for offset, length, inverse in cases:
                x = mx.random.normal((1, 2, length, 8)).astype(dtype)
                out = rope(x, offset=offset, inverse=inverse)
                mx.eval(out)
                expected[(str(dtype), offset, length, inverse)] = (x, out)

        holder = SimpleNamespace(named_modules=lambda: [("rope", rope)])
        assert mod._install_for_class(holder, rope_class) == 1

        for (dtype_name, offset, length, inverse), (x, out) in expected.items():
            for _repeat in range(2):  # miss then hit must both be exact
                cached = rope(x, offset=offset, inverse=inverse)
                mx.eval(cached)
                assert cached.dtype == out.dtype, (dtype_name, offset, length)
                assert bool(mx.array_equal(cached, out)), (
                    dtype_name,
                    offset,
                    length,
                    inverse,
                )
    finally:
        rope_class.__call__ = original_call
        for attr in ("_vmlx_dsv4_rope_original_call", "_vmlx_dsv4_rope_cache"):
            if hasattr(rope_class, attr):
                delattr(rope_class, attr)


def test_rope_positions_bypasses_cache_and_matches_reference(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    jang_model = pytest.importorskip("jang_tools.dsv4.mlx_model")
    mod = _fresh_rope_module(monkeypatch, mx)

    rope = jang_model.DeepseekV4RoPE(8, 10000.0)
    rope_class = jang_model.DeepseekV4RoPE
    original_call = rope_class.__call__
    try:
        x = mx.random.normal((1, 2, 3, 8)).astype(mx.float16)
        positions = mx.array([5, 1, 9])
        out = rope(x, positions=positions)
        mx.eval(out)

        holder = SimpleNamespace(named_modules=lambda: [("rope", rope)])
        assert mod._install_for_class(holder, rope_class) == 1
        group = mod._CONFIGS.get(rope)

        cached = rope(x, positions=positions)
        mx.eval(cached)
        assert bool(mx.array_equal(cached, out))
        assert len(group.tables) == 0
    finally:
        rope_class.__call__ = original_call
        for attr in ("_vmlx_dsv4_rope_original_call", "_vmlx_dsv4_rope_cache"):
            if hasattr(rope_class, attr):
                delattr(rope_class, attr)


def test_release_group_clears_tables_under_group_lock(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    mod = _fresh_rope_module(monkeypatch, mx)

    group = mod._TableGroup(signature=(1.0,))
    mod._GROUPS[(1.0,)] = group
    group.registrations = 1
    group.tables[(0, 1, "float16")] = (object(), object())

    entered = threading.Event()
    release = threading.Event()

    def _hold_lock():
        with group.lock:
            entered.set()
            release.wait(timeout=5)

    holder = threading.Thread(target=_hold_lock)
    holder.start()
    assert entered.wait(timeout=5)

    cleared = []
    clearer = threading.Thread(
        target=lambda: cleared.append(mod._release_group(group))
    )
    clearer.start()
    clearer.join(timeout=0.5)
    # The clear must block until the in-flight table user releases the lock.
    assert clearer.is_alive()
    assert group.tables

    release.set()
    clearer.join(timeout=5)
    assert not clearer.is_alive()
    assert not group.tables
    holder.join(timeout=5)
