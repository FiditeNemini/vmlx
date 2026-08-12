# SPDX-License-Identifier: Apache-2.0
"""An explicit --cache-memory-mb must not be silently reduced by a made-up cap.

`compute_memory_limit` clamped every value to a flat 32GB with the comment
"Metal GPU doesn't get 100% of system RAM". The premise is true; the number was
invented, and the clamp was silent.

On the 128GB box this engine is developed on, MLX reports a Metal working-set
ceiling of ~115GB (the owner has raised `iogpu.wired_limit_mb`). So
`--cache-memory-mb 65536` returned 32GB — less than HALF what was asked for —
and nothing said so. That also contradicts a rule this repo already records in
cli.py `_install_jangtq_wired_limit_from_sysctl`: the user's sysctl is
authoritative, and a hardcoded default that clips below it is a bug rather than
a safety feature.

The ceiling now comes from the OS. The flat constant survives only as the
fallback for when MLX cannot be asked (non-Darwin, MLX missing).
"""

from __future__ import annotations

import logging

import vmlx_engine.memory_cache as mc
from vmlx_engine.memory_cache import MemoryCacheConfig

_GB = 1024 * 1024 * 1024


def test_explicit_request_under_the_os_ceiling_is_honoured_exactly(monkeypatch):
    monkeypatch.setattr(
        mc, "_resolve_cache_memory_ceiling_bytes", lambda: 115 * _GB
    )
    # 64 GiB — the value that used to come back as 32GB.
    got = MemoryCacheConfig(max_memory_mb=65536).compute_memory_limit()
    assert got == 65536 * mc._BYTES_PER_MB, (
        "an explicit request below the machine ceiling must be honoured exactly"
    )
    assert got > 32 * _GB, "the old flat 32GB clamp is back"


def test_over_ceiling_request_is_capped_but_says_so(monkeypatch, caplog):
    monkeypatch.setattr(
        mc, "_resolve_cache_memory_ceiling_bytes", lambda: 16 * _GB
    )
    with caplog.at_level(logging.WARNING, logger=mc.logger.name):
        got = MemoryCacheConfig(max_memory_mb=64 * 1024).compute_memory_limit()
    assert got == 16 * _GB, "an over-ceiling request must still be capped"
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "exceeds" in text and "ceiling" in text, (
        "the cap must be reported; silently returning a different number than "
        f"the user asked for is the defect. Got: {text!r}"
    )
    assert "iogpu.wired_limit_mb" in text, (
        "the warning should say how to raise the ceiling"
    )


def test_small_explicit_values_keep_working(monkeypatch):
    """No minimum floor — a test or a small machine may want a tiny cache."""
    monkeypatch.setattr(
        mc, "_resolve_cache_memory_ceiling_bytes", lambda: 115 * _GB
    )
    assert MemoryCacheConfig(max_memory_mb=8).compute_memory_limit() == (
        8 * mc._BYTES_PER_MB
    )


def test_ceiling_comes_from_mlx_and_falls_back_when_it_cannot(monkeypatch):
    resolved = mc._resolve_cache_memory_ceiling_bytes()
    assert resolved > 0

    # With MLX unavailable the old constant is the documented fallback.
    import builtins

    real_import = builtins.__import__

    def _no_mlx(name, *a, **kw):
        if name == "mlx.core":
            raise ImportError("simulated: no MLX")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _no_mlx)
    assert (
        mc._resolve_cache_memory_ceiling_bytes()
        == mc._FALLBACK_CACHE_CEILING_BYTES
    )
