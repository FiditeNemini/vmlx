# SPDX-License-Identifier: Apache-2.0
"""No-heavy contracts for prefill-loop perf cleanup intake.

These tests pin the source-level behavior from PR #163 without loading a model.
They deliberately avoid approving output quality or cache equivalence; those
remain live-model gates.
"""

from __future__ import annotations

import inspect
import os


def _mllm_source() -> str:
    import vmlx_engine.mllm_batch_generator as mod

    return inspect.getsource(mod)


def test_chunk_loop_uses_sorted_boundary_pointer():
    src = _mllm_source()
    assert "_sorted_boundaries" in src
    assert "_boundary_idx" in src
    assert "for b in ssm_boundaries" not in src


def test_chunk_loop_precomputes_state_layers():
    src = _mllm_source()
    assert "_materialize_prefill_cache_state(cache)" in src
    assert "mx.eval(*items)" in src


def test_chunk_loop_hoists_all_tokens_tolist():
    src = _mllm_source()
    assert "_hoisted_all_tokens" in src


def test_chunk_loop_env_gates_clear_cache():
    # Both MLLM sites must route through the shared prefill_admission helper
    # and never read the env directly again: when they read it themselves,
    # they read only VMLX_PREFILL_KEEP_ALLOC, so the VMLINUX_ spelling toggled
    # the text generator but silently not the MLLM one.
    src = _mllm_source()
    assert "prefill_keep_alloc_enabled()" in src
    assert "PREFILL_KEEP_ALLOC" not in src
    assert "if not _prefill_keep_alloc:" in src


def test_single_batch_prefill_loop_env_gates_clear_cache():
    import vmlx_engine.utils.single_batch_generator as mod

    src = inspect.getsource(mod.SingleBatchGenerator._prefill)
    assert "_prefill_keep_alloc_enabled()" in src
    assert "PREFILL_KEEP_ALLOC" not in src
    assert "if not _prefill_keep_alloc" in src


def test_cli_flag_propagates_to_env():
    import vmlx_engine.cli as cli_mod

    src = inspect.getsource(cli_mod)
    assert '"--prefill-keep-alloc"' in src
    assert "VMLX_PREFILL_KEEP_ALLOC" in src
    assert "prefill_keep_alloc" in src


def test_prefill_keep_alloc_env_off_by_default(monkeypatch):
    from vmlx_engine.utils.prefill_admission import prefill_keep_alloc_enabled

    monkeypatch.delenv("VMLINUX_PREFILL_KEEP_ALLOC", raising=False)
    monkeypatch.delenv("VMLX_PREFILL_KEEP_ALLOC", raising=False)
    assert prefill_keep_alloc_enabled() is False


def test_prefill_keep_alloc_helper_accepts_both_env_spellings(monkeypatch):
    from vmlx_engine.utils.prefill_admission import prefill_keep_alloc_enabled

    monkeypatch.setenv("VMLINUX_PREFILL_KEEP_ALLOC", "1")
    monkeypatch.delenv("VMLX_PREFILL_KEEP_ALLOC", raising=False)
    assert prefill_keep_alloc_enabled() is True

    monkeypatch.delenv("VMLINUX_PREFILL_KEEP_ALLOC", raising=False)
    monkeypatch.setenv("VMLX_PREFILL_KEEP_ALLOC", "true")
    assert prefill_keep_alloc_enabled() is True

    # VMLINUX_ wins when both are set (the original text-path precedence).
    monkeypatch.setenv("VMLINUX_PREFILL_KEEP_ALLOC", "0")
    monkeypatch.setenv("VMLX_PREFILL_KEEP_ALLOC", "1")
    assert prefill_keep_alloc_enabled() is False


def test_prefill_keep_alloc_helper_parses_booleans_not_truthiness(monkeypatch):
    # One MLLM site used raw string truthiness, so "0" KEPT allocations there.
    from vmlx_engine.utils.prefill_admission import prefill_keep_alloc_enabled

    monkeypatch.delenv("VMLINUX_PREFILL_KEEP_ALLOC", raising=False)
    monkeypatch.setenv("VMLX_PREFILL_KEEP_ALLOC", "0")
    assert prefill_keep_alloc_enabled() is False


def test_boundary_pointer_advances_past_captured():
    sorted_boundaries = [100, 250, 400, 800]
    captured: set[int] = {100, 400}
    processed = 50

    idx = 0
    while idx < len(sorted_boundaries) and (
        sorted_boundaries[idx] <= processed
        or sorted_boundaries[idx] in captured
    ):
        idx += 1
    assert idx == 1
    assert sorted_boundaries[idx] == 250

    processed = 300
    captured.add(250)
    while idx < len(sorted_boundaries) and (
        sorted_boundaries[idx] <= processed
        or sorted_boundaries[idx] in captured
    ):
        idx += 1
    assert idx == 3
    assert sorted_boundaries[idx] == 800
