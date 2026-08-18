"""Sample size required before MTP demotes a request to autoregressive.

Demotion is irreversible for the remainder of the request, so it must not be
decided on the handful of cold cycles that open a request.  Measured on
Qwen3.8-27B-JANG_4D-CRACK, same bundle and same prompt: the first request after
load read 7/12 = 58.3% with a cold MTP cache and demoted the whole request,
while the next request scored 2033/2104 = 96.6% and held MTP for 34.1 t/s.
"""

import mlx.core as mx  # noqa: F401  (import parity with the generator module)
import pytest

from vmlx_engine import mllm_batch_generator as gen


class _Stats:
    def __init__(self, drafted, accepted):
        self.cycles = drafted
        self.drafted_by_depth = [drafted, 0, 0]
        self.accepted_by_depth = [accepted, 0, 0]
        self.mtp_forwards = drafted


class _State:
    def __init__(self, drafted, accepted, depth=1):
        self.depth = depth
        self.stats = _Stats(drafted, accepted)
        self.ar_fallback_pending = False
        self.ar_fallback_reason = None


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for name in (
        "VMLINUX_NATIVE_MTP_ADAPTIVE_WARMUP_CYCLES",
        "VMLX_NATIVE_MTP_ADAPTIVE_WARMUP_CYCLES",
        "VMLINUX_NATIVE_MTP_AR_FALLBACK_MIN_SAMPLE",
        "VMLX_NATIVE_MTP_AR_FALLBACK_MIN_SAMPLE",
        "VMLINUX_NATIVE_MTP_D1_MIN_ACCEPT",
        "VMLX_NATIVE_MTP_D1_MIN_ACCEPT",
        "VMLINUX_NATIVE_MTP_COST_FALLBACK",
        "VMLX_NATIVE_MTP_COST_FALLBACK",
    ):
        monkeypatch.delenv(name, raising=False)


def test_cold_twelve_cycle_window_no_longer_demotes():
    """The exact live case: 7 of 12 accepted must NOT kill the request."""
    state = _State(drafted=12, accepted=7)
    gen._native_mtp_maybe_adapt_depth("req-cold", state)
    assert state.ar_fallback_pending is False


def test_sustained_sub_breakeven_head_still_demotes():
    """The case the gate exists for: 58.6% over a real sample must demote."""
    state = _State(drafted=483, accepted=283)
    gen._native_mtp_maybe_adapt_depth("req-weak", state)
    assert state.ar_fallback_pending is True
    assert "d1_acceptance" in (state.ar_fallback_reason or "")


def test_healthy_head_is_never_demoted():
    state = _State(drafted=2104, accepted=2033)  # 96.6%, the warm live run
    gen._native_mtp_maybe_adapt_depth("req-healthy", state)
    assert state.ar_fallback_pending is False


def test_demotion_needs_the_full_sample_not_just_the_warmup():
    """Just past warmup but short of the sample floor must not demote."""
    state = _State(drafted=20, accepted=6)  # 30%, far below the 0.65 floor
    gen._native_mtp_maybe_adapt_depth("req-small", state)
    assert state.ar_fallback_pending is False


def test_sample_floor_is_tunable(monkeypatch):
    monkeypatch.setenv("VMLX_NATIVE_MTP_AR_FALLBACK_MIN_SAMPLE", "16")
    state = _State(drafted=20, accepted=6)
    gen._native_mtp_maybe_adapt_depth("req-tuned", state)
    assert state.ar_fallback_pending is True


def test_boundary_at_the_sample_floor_demotes():
    state = _State(drafted=64, accepted=20)  # 31.3% at exactly the floor
    gen._native_mtp_maybe_adapt_depth("req-boundary", state)
    assert state.ar_fallback_pending is True


def test_one_short_of_the_floor_does_not_demote():
    state = _State(drafted=63, accepted=20)
    gen._native_mtp_maybe_adapt_depth("req-just-under", state)
    assert state.ar_fallback_pending is False
