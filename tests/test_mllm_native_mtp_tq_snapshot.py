"""Native-MTP verify rollback across the TurboQuant live-encode crossing.

TurboQuantKVCache.trim() rewinds offset only. If the one-time compress()
fires inside a verify advance that later REJECTS, the rolled-back draft KV
stays baked into the decoded/compressed buffers and every later joined-buffer
grow re-copies it back. The snapshot path must therefore force a __dict__
snapshot for TQ layers whose compress threshold falls inside the advance.
"""

import mlx.core as mx
import pytest

from vmlx_engine.mllm_batch_generator import (
    _native_mtp_restore_replay_cache,
    _native_mtp_should_snapshot_layer,
    _native_mtp_snapshot_replay_cache,
)

try:
    from jang_tools.turboquant.cache import TurboQuantKVCache
except ImportError:  # pragma: no cover
    TurboQuantKVCache = None

requires_tq = pytest.mark.skipif(
    TurboQuantKVCache is None, reason="jang_tools not installed"
)


def _make_tq(compress_after: int = 8) -> "TurboQuantKVCache":
    return TurboQuantKVCache(
        key_dim=16,
        value_dim=16,
        key_bits=4,
        value_bits=4,
        seed=7,
        compress_after=compress_after,
        sink_tokens=0,
    )


def _feed(cache, tokens: int, seed: int):
    mx.random.seed(seed)
    k = mx.random.normal((1, 2, tokens, 16))
    v = mx.random.normal((1, 2, tokens, 16))
    out = cache.update_and_fetch(k, v)
    mx.eval(*out)
    return out


@requires_tq
class TestTQSnapshotDecision:
    def test_no_advance_keeps_trim_path(self):
        tq = _make_tq()
        _feed(tq, 6, seed=1)
        assert _native_mtp_should_snapshot_layer(tq, 0) is False

    def test_advance_short_of_threshold_keeps_trim_path(self):
        tq = _make_tq()
        _feed(tq, 6, seed=1)
        assert _native_mtp_should_snapshot_layer(tq, 1) is False

    def test_advance_crossing_threshold_forces_snapshot(self):
        tq = _make_tq()
        _feed(tq, 6, seed=1)
        assert _native_mtp_should_snapshot_layer(tq, 3) is True

    def test_already_compressed_keeps_trim_path(self):
        tq = _make_tq()
        _feed(tq, 9, seed=1)
        assert tq._compressed_tokens > 0
        assert _native_mtp_should_snapshot_layer(tq, 3) is False

    def test_live_encode_disabled_keeps_trim_path(self):
        tq = _make_tq(compress_after=0)
        _feed(tq, 6, seed=1)
        assert _native_mtp_should_snapshot_layer(tq, 3) is False


@requires_tq
class TestTQSnapshotRestoreAcrossCompressCrossing:
    def test_rejected_verify_crossing_compress_matches_control(self):
        tq = _make_tq()
        ctrl = _make_tq()
        _feed(tq, 6, seed=1)
        _feed(ctrl, 6, seed=1)

        snapshot = _native_mtp_snapshot_replay_cache([tq], 3)
        assert snapshot[0] is not None

        # Rejected verify advance: 3 draft tokens cross compress_after=8.
        _feed(tq, 3, seed=2)
        assert tq._compressed_tokens > 0
        assert _native_mtp_restore_replay_cache([tq], snapshot, 3) is True
        assert tq.offset == 6
        assert tq._compressed_tokens == 0

        # Replay confirmed tokens on both; each crosses compress identically.
        k1, v1 = _feed(tq, 3, seed=3)
        k2, v2 = _feed(ctrl, 3, seed=3)
        assert tq._compressed_tokens == ctrl._compressed_tokens
        assert mx.array_equal(k1, k2)
        assert mx.array_equal(v1, v2)
