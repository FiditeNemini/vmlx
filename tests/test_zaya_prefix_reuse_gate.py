"""ZAYA CCA prefix reuse can be turned off for answer stability.

MEASURED 2026-08-13 on Zaya-8B-JANG_4M at temperature 0 (ISSUE-LEDGER L63):
same prompt, prefix cache OFF gives a 228-token reply (sha 335824b8, 2/2
reproducible); prefix cache ON, reusing a mere 24-token prefix, gives 119 tokens
(sha 74b2e5c7, 5/5 reproducible). Both arms are deterministic internally, so the
difference is the reuse itself -- and it is NOT the stored q8 codec, because it
reproduces under --kv-cache-quantization none.

The restore-side guards already reject a chain that lacks terminal CCA
conv_state/prev_hs, and neither fired: the chain was judged complete and the
answer still changed. They verify PRESENCE of terminal state, never EQUIVALENCE
to a one-pass cold prefill.

This gate is the operator-facing lever until a chunk-exact restore exists. It is
default OFF because flipping it removes a working cache tier.
"""

import os
from types import SimpleNamespace

import pytest

from vmlx_engine.cli import _apply_zaya_cca_cache_policy


class _Logger:
    def __init__(self):
        self.warnings = []

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)

    def info(self, msg, *args):
        pass


def _args():
    return SimpleNamespace(
        enable_prefix_cache=True,
        enable_block_disk_cache=True,
        use_paged_cache=False,
        kv_cache_quantization="q8",
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("VMLX_ZAYA_DISABLE_PREFIX_REUSE", raising=False)
    yield


def test_default_leaves_prefix_reuse_enabled():
    # Default OFF: the gate must not change behaviour for anyone who has not
    # opted in. A gate that silently removes a cache tier is worse than the bug.
    args, log = _args(), _Logger()
    _apply_zaya_cca_cache_policy(args, log)

    assert args.enable_prefix_cache is True
    assert args.enable_block_disk_cache is True
    # The family's existing contract still applies.
    assert args.use_paged_cache is True
    assert args.kv_cache_quantization == "none"


def test_gate_disables_both_reuse_tiers(monkeypatch):
    monkeypatch.setenv("VMLX_ZAYA_DISABLE_PREFIX_REUSE", "1")
    args, log = _args(), _Logger()
    _, changed = _apply_zaya_cca_cache_policy(args, log)

    assert args.enable_prefix_cache is False
    # L2 must go too: a disk chain would re-introduce the same reuse.
    assert args.enable_block_disk_cache is False
    assert "prefix_reuse=disabled_for_answer_stability" in changed


def test_gate_announces_itself(monkeypatch):
    # A silent policy change is exactly the class this campaign keeps finding.
    monkeypatch.setenv("VMLX_ZAYA_DISABLE_PREFIX_REUSE", "1")
    args, log = _args(), _Logger()
    _apply_zaya_cca_cache_policy(args, log)

    assert any("VMLX_ZAYA_DISABLE_PREFIX_REUSE" in w for w in log.warnings)


def test_gate_does_not_re_enable_paged_after_disabling_prefix(monkeypatch):
    # The paged upgrade is conditional on prefix cache being on; with the gate
    # set it must not fire, or the session would carry a paged tier it cannot use.
    monkeypatch.setenv("VMLX_ZAYA_DISABLE_PREFIX_REUSE", "1")
    args, log = _args(), _Logger()
    _apply_zaya_cca_cache_policy(args, log)

    assert args.use_paged_cache is False


@pytest.mark.parametrize("value", ["0", "", "true", "yes"])
def test_only_exactly_1_arms_the_gate(monkeypatch, value):
    # Match the codebase's existing env convention; a typo must not silently
    # disable a cache tier.
    monkeypatch.setenv("VMLX_ZAYA_DISABLE_PREFIX_REUSE", value)
    args, log = _args(), _Logger()
    _apply_zaya_cca_cache_policy(args, log)

    assert args.enable_prefix_cache is True


class TestRecurrentClassGate:
    """L70: the drift is a CLASS, not one family.

    Nemotron-3.5-Lightning-30B-A3B (`nemotron_h_ssm_attention`) shows the same
    signature as ZAYA at temperature 0 -- a 23-token reuse turned a 902-token
    reply (sha 236724e7) into 781 (sha 1321ab79), with the cache-off arm
    deterministic across runs. DSV4 (1,792 tokens reused) and gemma-4-E4B are
    byte-identical under the same harness, so it is neither the harness nor
    path-dependence in general: both drifting families carry RECURRENT/SSM state.

    `VMLX_DISABLE_RECURRENT_PREFIX_REUSE` is the class-level lever so an operator
    does not need to know which families are affected.
    """

    def test_class_gate_default_off(self):
        args, log = _args(), _Logger()
        from vmlx_engine.cli import _disable_recurrent_prefix_reuse

        assert _disable_recurrent_prefix_reuse(args, log, "X") is False
        assert args.enable_prefix_cache is True

    def test_class_gate_disables_both_tiers(self, monkeypatch):
        monkeypatch.setenv("VMLX_DISABLE_RECURRENT_PREFIX_REUSE", "1")
        args, log = _args(), _Logger()
        from vmlx_engine.cli import _disable_recurrent_prefix_reuse

        assert _disable_recurrent_prefix_reuse(args, log, "Nemotron-H SSM") is True
        assert args.enable_prefix_cache is False
        # L2 must go too or a disk chain reintroduces the same reuse.
        assert args.enable_block_disk_cache is False
        assert any("Nemotron-H SSM" in w for w in log.warnings)

    @pytest.mark.parametrize("value", ["0", "", "true", "yes"])
    def test_class_gate_only_exact_1(self, monkeypatch, value):
        monkeypatch.setenv("VMLX_DISABLE_RECURRENT_PREFIX_REUSE", value)
        args, log = _args(), _Logger()
        from vmlx_engine.cli import _disable_recurrent_prefix_reuse

        assert _disable_recurrent_prefix_reuse(args, log, "X") is False

    def test_class_gate_also_covers_zaya(self, monkeypatch):
        # Set ONLY the class gate; the ZAYA policy must still disable reuse so an
        # operator does not have to set two variables.
        monkeypatch.delenv("VMLX_ZAYA_DISABLE_PREFIX_REUSE", raising=False)
        monkeypatch.setenv("VMLX_DISABLE_RECURRENT_PREFIX_REUSE", "1")
        args, log = _args(), _Logger()
        _apply_zaya_cca_cache_policy(args, log)

        assert args.enable_prefix_cache is False
        assert args.enable_block_disk_cache is False

    def test_nemotron_is_wired_into_the_family_chain(self):
        # A gate nothing calls is the dead-code shape this campaign keeps finding.
        from pathlib import Path

        src = Path(__file__).resolve().parents[1].joinpath("vmlx_engine/cli.py").read_text()
        # Target the WIRING site, not the first textual match -- the docstring
        # above the helper also names the subtype, and matching that would make
        # this test pass while nothing called the gate.
        marker = '"nemotron_h_ssm_attention", "lfm2_moe_hybrid_ssm"'
        assert marker in src, "drifting families are not wired into the policy chain"
        i = src.index(marker)
        assert "_disable_recurrent_prefix_reuse" in src[i : i + 900]

    def test_lfm2_is_covered(self):
        # L71a: LFM2.5-8B-A1B drifts on a 19-token paged+ssm hit (469 -> 398).
        from pathlib import Path

        src = Path(__file__).resolve().parents[1].joinpath("vmlx_engine/cli.py").read_text()
        assert "lfm2_moe_hybrid_ssm" in src
