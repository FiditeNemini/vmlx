# SPDX-License-Identifier: Apache-2.0
"""The prompt disk cache must not narrow bfloat16 to float16.

numpy has no bf16, so the store has to cast — but bf16 and f16 are NOT
interchangeable despite both being 16 bits. bf16 spends 8 bits on the exponent
(the same ~1e38 range as f32) and f16 spends 5 (max 65504). MEASURED with MLX:

    bf16 source        [70144.0, 9.98e-07, 3.5]
    cast via float16   [   inf, 1.013e-06, 3.5]     <- the old path
    cast via float32   [70144.0, 9.98e-07, 3.5]     <- exact

So a KV value over 65504 came back as inf and small values shifted, meaning a
restored prompt cache could differ from a fresh compute. That is the same
"a cache hit must equal a recompute" class as the mixed-SWA storage-TQ
divergence. The code carried "acceptable precision for prompt cache" as its
justification, with nothing measured behind it.

bf16 -> f32 is lossless in both directions — f32 has a wider mantissa AND the
same exponent range — and is what block_disk_store already does for exactly
this reason. It costs 2x the bytes for bf16 caches.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

mx = pytest.importorskip("mlx.core")


def test_float16_really_does_destroy_bf16_range():
    """Pins the premise with real MLX, so this file cannot rot into folklore."""
    src = mx.array([70000.0, 1e-6, 3.5], dtype=mx.float32).astype(mx.bfloat16)
    exact = src.astype(mx.float32).tolist()

    via_f16 = src.astype(mx.float16).astype(mx.float32).tolist()
    assert via_f16[0] == float("inf"), (
        "float16 no longer overflows here; re-check whether the cast still "
        "needs to be float32"
    )
    assert via_f16 != exact

    via_f32 = src.astype(mx.float32).tolist()
    assert via_f32 == exact, "bf16 -> f32 must be exact in both directions"


def test_store_casts_bfloat16_to_float32_not_float16():
    src = (ROOT / "vmlx_engine" / "disk_cache.py").read_text(encoding="utf-8")
    start = src.index("    def store(")
    body = src[start : src.index("\n    def ", start + 1)]

    assert "v.astype(mx.float32) if v.dtype == mx.bfloat16" in body, (
        "the prompt disk cache no longer widens bf16 losslessly"
    )
    assert not re.search(r"astype\(mx\.float16\)\s*if\s*v\.dtype\s*==\s*mx\.bfloat16", body), (
        "the lossy bf16 -> f16 cast is back; values above 65504 become inf"
    )
    assert "acceptable precision for prompt cache" not in body, (
        "the unmeasured justification is back"
    )


def test_block_disk_store_still_agrees():
    """The two tiers must not disagree about how a bf16 cache is persisted.

    The invariant is LOSSLESSNESS, not a particular cast. This used to assert
    the literal string "astype(mx.float32)", which pinned an implementation
    rather than the property: the block store now carries bf16 as its raw
    uint16 bits (1x bytes instead of 2x) and reinterprets on restore, which is
    equally lossless and half the IO on what is now the default prefix tier.

    So test the round trip instead of the source text.
    """
    block = (ROOT / "vmlx_engine" / "block_disk_store.py").read_text(
        encoding="utf-8"
    )
    assert "astype(mx.float16)" not in block, (
        "block_disk_store started narrowing to float16; values above 65504 "
        "become inf and the two disk tiers would round-trip differently"
    )

    mx = pytest.importorskip("mlx.core")
    from vmlx_engine.block_disk_store import (
        BlockDiskStore,
        _restore_serialized_dtype,
    )

    original = mx.array(
        [[1.0, -2.5, 65504.0 * 4, 1e30, 6e-8]], dtype=mx.bfloat16
    )
    detached = BlockDiskStore._detach_safetensors_tensors({"x": original})

    # 1x: the payload must be the same byte count as the bf16 source, not
    # double it. This is the whole point of the change.
    assert detached["x"].nbytes == original.size * 2, (
        f"expected 1x bf16 bytes, got {detached['x'].nbytes} for "
        f"{original.size} elements"
    )

    restored = _restore_serialized_dtype(mx.array(detached["x"]), mx.bfloat16)
    assert restored.dtype == mx.bfloat16
    assert bool(mx.all(restored == original).item()), (
        "bf16 did not survive the block-store round trip bit-exactly"
    )

    # Backward compatibility: caches written by older builds hold f32 payloads
    # and must still restore by casting.
    legacy = original.astype(mx.float32)
    legacy_restored = _restore_serialized_dtype(legacy, mx.bfloat16)
    assert legacy_restored.dtype == mx.bfloat16
    assert bool(mx.all(legacy_restored == original).item()), (
        "an f32 block written by an older build no longer restores"
    )


# ---------------------------------------------------------------------------
# Runtime round-trip: dtype must come BACK, not just survive widened.
#
# Widening bf16 -> f32 for storage is correct (numpy has no bf16, f16 clips),
# but the restore must cast back: mlx-lm caches extend via mx.concatenate,
# which promotes bf16+f32 to f32, so an un-restored cache runs f32 for its
# whole continuation — 2x live KV bytes and attention numerics off the
# fresh-compute bf16 path (both measured 2026-08-12 with MLX 0.31.2).
# ---------------------------------------------------------------------------


def _bf16_kv_cache_layers(num_layers=2, tokens=16):
    from mlx_lm.models.cache import KVCache

    layers = []
    for i in range(num_layers):
        c = KVCache()
        mx.random.seed(i)
        k = mx.random.normal((1, 2, tokens, 8)).astype(mx.bfloat16)
        v = mx.random.normal((1, 2, tokens, 8)).astype(mx.bfloat16)
        # Values only bf16's 8-bit exponent can hold: the old f16 storage
        # cast sent these to inf. They must survive the round trip finite
        # AND exact, proving the dtype restore did not resurrect that bug.
        k[..., 0, 0] = mx.array(1e30, dtype=mx.bfloat16)
        v[..., 0, 0] = mx.array(3e38, dtype=mx.bfloat16)
        c.update_and_fetch(k, v)
        layers.append(c)
    mx.eval([(c.keys, c.values) for c in layers])
    return layers


def test_roundtrip_restores_bfloat16_and_keeps_large_values_finite(tmp_path):
    """bf16 in -> bf16 out through the REAL store()/fetch() path."""
    from vmlx_engine.disk_cache import DiskCacheManager

    layers = _bf16_kv_cache_layers()
    tokens = list(range(200, 217))
    mgr = DiskCacheManager(cache_dir=str(tmp_path), max_size_gb=1.0)
    try:
        assert mgr.store(tokens, layers), "store was not enqueued"
        mgr._write_queue.join()
        restored = mgr.fetch(tokens)
        assert restored is not None, "disk cache fetch missed its own store"
        assert len(restored) == len(layers)
        for i, (orig, rest) in enumerate(zip(layers, restored)):
            assert rest.keys.dtype == mx.bfloat16, (
                f"layer {i}: restored keys are {rest.keys.dtype}, not bf16 — "
                "the continuation would extend f32 state forever "
                "(mx.concatenate promotes) at 2x live KV bytes"
            )
            assert rest.values.dtype == mx.bfloat16
            assert rest.offset == orig.offset
            for name, o, r in (
                ("keys", orig.keys, rest.keys),
                ("values", orig.values, rest.values),
            ):
                o32 = o[..., : orig.offset, :].astype(mx.float32)
                r32 = r[..., : rest.offset, :].astype(mx.float32)
                assert mx.all(mx.isfinite(r32)).item(), (
                    f"layer {i} {name}: non-finite values after restore — "
                    "the bf16->f16 inf bug is back"
                )
                assert mx.array_equal(o32, r32).item(), (
                    f"layer {i} {name}: restored values differ from stored"
                )
            # The 3e38 sentinel specifically must still be there, finite.
            vmax = mx.max(mx.abs(rest.values.astype(mx.float32))).item()
            assert 2.9e38 < vmax < 3.1e38, (
                f"layer {i}: large-magnitude sentinel lost (max={vmax!r})"
            )
    finally:
        mgr.shutdown()


def test_legacy_file_without_dtype_record_still_loads(tmp_path):
    """Old-format files (no widening record) must load, not crash.

    Writers before the record stored bf16 state widened to f32 with nothing
    written down, so those files cannot come back as bf16 — they must load
    exactly as before: widened-but-exact f32.
    """
    import time as _time

    from mlx_lm.models.cache import KVCache, save_prompt_cache

    from vmlx_engine.disk_cache import (
        DiskCacheManager,
        _hash_tokens,
        _runtime_cache_fingerprint,
    )

    # Old writer output: f32 tensors (the widened form), old metadata keys,
    # no widened_dtypes record.
    c = KVCache()
    mx.random.seed(3)
    k = mx.random.normal((1, 2, 8, 4))  # float32
    v = mx.random.normal((1, 2, 8, 4))
    c.update_and_fetch(k, v)
    mx.eval(c.keys, c.values)

    tokens = list(range(300, 309))
    mgr = DiskCacheManager(cache_dir=str(tmp_path), max_size_gb=1.0)
    try:
        token_hash = _hash_tokens(tokens)
        file_name = f"cache_{token_hash[:16]}_{len(tokens)}tok.safetensors"
        save_prompt_cache(
            str(tmp_path / file_name),
            [c],
            metadata={
                "num_tokens": str(len(tokens)),
                "created_at": str(_time.time()),
                "runtime_cache_fingerprint": _runtime_cache_fingerprint(),
            },
        )
        conn = mgr._pool.get()
        try:
            now = _time.time()
            conn.execute(
                "INSERT OR REPLACE INTO cache_entries "
                "(token_hash, file_name, num_tokens, file_size, created_at, "
                "last_accessed, access_count) VALUES (?, ?, ?, ?, ?, ?, 1)",
                (
                    token_hash,
                    file_name,
                    len(tokens),
                    (tmp_path / file_name).stat().st_size,
                    now,
                    now,
                ),
            )
            conn.commit()
        finally:
            mgr._pool.put(conn)

        restored = mgr.fetch(tokens)
        assert restored is not None, "legacy-format file failed to load"
        assert restored[0].keys.dtype == mx.float32, (
            "legacy files carry no original-dtype record; they must keep "
            "loading as the same widened f32 as before, not be guessed at"
        )
        assert mx.array_equal(
            restored[0].keys, c.keys[..., : c.offset, :]
        ).item()
        assert mx.array_equal(
            restored[0].values, c.values[..., : c.offset, :]
        ).item()
    finally:
        mgr.shutdown()
