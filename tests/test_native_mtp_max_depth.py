"""Native-MTP depth ceiling is parameterizable for measurement.

The product default is 3 (unchanged); VMLX_NATIVE_MTP_MAX_DEPTH raises it so a
depth sweep can probe D4/D5 on lanes that can verify them. Raising the ceiling
must also grow the adaptive value-state arrays so a deeper probe cannot
index-error.
"""

import mlx.core as mx  # noqa: F401  (ensures engine import graph is intact)

from vmlx_engine.native_mtp import native_mtp_max_depth


def test_default_ceiling_is_three(monkeypatch):
    monkeypatch.delenv("VMLX_NATIVE_MTP_MAX_DEPTH", raising=False)
    monkeypatch.delenv("VMLINUX_NATIVE_MTP_MAX_DEPTH", raising=False)
    assert native_mtp_max_depth() == 3


def test_env_raises_and_clamps(monkeypatch):
    monkeypatch.setenv("VMLX_NATIVE_MTP_MAX_DEPTH", "5")
    assert native_mtp_max_depth() == 5
    monkeypatch.setenv("VMLX_NATIVE_MTP_MAX_DEPTH", "99")
    assert native_mtp_max_depth() == 8  # hard ceiling
    monkeypatch.setenv("VMLX_NATIVE_MTP_MAX_DEPTH", "0")
    assert native_mtp_max_depth() == 1
    monkeypatch.setenv("VMLX_NATIVE_MTP_MAX_DEPTH", "not-a-number")
    assert native_mtp_max_depth() == 3


def test_adaptive_state_arrays_scale_with_ceiling(monkeypatch):
    monkeypatch.setenv("VMLX_NATIVE_MTP_MAX_DEPTH", "5")
    from vmlx_engine.native_mtp_adaptive import NativeMTPAdaptiveValueState

    state = NativeMTPAdaptiveValueState()
    assert len(state.samples_by_depth) == 5
    assert len(state.probe_revert_counts) == 5
    assert len(state.last_sample_cycle) == 5
    assert len(state.last_probe_cycle) == 5


def test_coerce_depth_honors_ceiling(monkeypatch):
    from vmlx_engine import native_mtp

    monkeypatch.setenv("VMLX_NATIVE_MTP_MAX_DEPTH", "5")
    assert native_mtp._coerce_native_mtp_depth("5") == 5
    assert native_mtp._coerce_native_mtp_depth("9") == 5
    monkeypatch.setenv("VMLX_NATIVE_MTP_MAX_DEPTH", "3")
    assert native_mtp._coerce_native_mtp_depth("5") == 3


def test_stats_arrays_scale_with_ceiling(monkeypatch):
    monkeypatch.setenv("VMLX_NATIVE_MTP_MAX_DEPTH", "5")
    from vmlx_engine.mllm_batch_generator import MLLMNativeMTPStats

    stats = MLLMNativeMTPStats()
    assert len(stats.accepted_by_depth) == 5
    assert len(stats.drafted_by_depth) == 5
