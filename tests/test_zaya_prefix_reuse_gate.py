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
    # Clear every accepted name: a real one exported in the developer's shell
    # would otherwise make the default-OFF tests pass for the wrong reason.
    for _name in (
        "VMLX_ZAYA_DISABLE_PREFIX_REUSE",
        "VMLX_DISABLE_DRIFTING_PREFIX_REUSE",
        "VMLX_DISABLE_RECURRENT_PREFIX_REUSE",
    ):
        monkeypatch.delenv(_name, raising=False)
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
        from vmlx_engine.cli import _disable_drifting_prefix_reuse

        assert _disable_drifting_prefix_reuse(args, log, "X") is False
        assert args.enable_prefix_cache is True

    def test_class_gate_disables_both_tiers(self, monkeypatch):
        monkeypatch.setenv("VMLX_DISABLE_RECURRENT_PREFIX_REUSE", "1")
        args, log = _args(), _Logger()
        from vmlx_engine.cli import _disable_drifting_prefix_reuse

        assert _disable_drifting_prefix_reuse(args, log, "Nemotron-H SSM") is True
        assert args.enable_prefix_cache is False
        # L2 must go too or a disk chain reintroduces the same reuse.
        assert args.enable_block_disk_cache is False
        assert any("Nemotron-H SSM" in w for w in log.warnings)

    @pytest.mark.parametrize("value", ["0", "", "true", "yes"])
    def test_class_gate_only_exact_1(self, monkeypatch, value):
        monkeypatch.setenv("VMLX_DISABLE_RECURRENT_PREFIX_REUSE", value)
        args, log = _args(), _Logger()
        from vmlx_engine.cli import _disable_drifting_prefix_reuse

        assert _disable_drifting_prefix_reuse(args, log, "X") is False

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
        # Window sized generously on purpose: a fixed 900 chars broke as soon as
        # the branch grew an explanatory comment, which is a documentation change,
        # not a wiring regression. What matters is that the CALL follows the
        # match, before the next elif arm.
        window = src[i : i + 2500]
        nxt = window.find("\n        elif ")
        if nxt != -1:
            window = window[:nxt]
        assert "_disable_drifting_prefix_reuse" in window

    def test_lfm2_is_covered(self):
        # L71a: LFM2.5-8B-A1B drifts on a 19-token paged+ssm hit (469 -> 398).
        from pathlib import Path

        src = Path(__file__).resolve().parents[1].joinpath("vmlx_engine/cli.py").read_text()
        assert "lfm2_moe_hybrid_ssm" in src

    def test_legacy_env_var_still_arms_the_gate(self, monkeypatch):
        # The RECURRENT name shipped before the mechanism was refuted. An
        # operator who set it must not silently lose the protection.
        monkeypatch.delenv("VMLX_DISABLE_DRIFTING_PREFIX_REUSE", raising=False)
        monkeypatch.setenv("VMLX_DISABLE_RECURRENT_PREFIX_REUSE", "1")
        args, log = _args(), _Logger()
        from vmlx_engine.cli import _disable_drifting_prefix_reuse

        assert _disable_drifting_prefix_reuse(args, log, "X") is True
        assert args.enable_prefix_cache is False

    def test_log_names_the_variable_actually_set(self, monkeypatch):
        # With two accepted names, a message hardcoding one of them tells the
        # operator to check a variable they never set.
        monkeypatch.delenv("VMLX_DISABLE_RECURRENT_PREFIX_REUSE", raising=False)
        monkeypatch.setenv("VMLX_DISABLE_DRIFTING_PREFIX_REUSE", "1")
        args, log = _args(), _Logger()
        from vmlx_engine.cli import _disable_drifting_prefix_reuse

        _disable_drifting_prefix_reuse(args, log, "X")
        assert any("VMLX_DISABLE_DRIFTING_PREFIX_REUSE=1" in w for w in log.warnings)
        assert not any("RECURRENT" in w for w in log.warnings)


class TestDriftSetMembership:
    """The measured-drift set is DATA, and every member cites a measurement.

    Six families are measured to change their answer on a cache hit; three of
    them (minimax, muse_glimmer, step3p7) have no branch in the family policy
    chain, which is why the gate is applied ahead of the chain rather than as
    another arm of it.
    """

    @pytest.mark.parametrize(
        "family",
        ["zaya", "nemotron_h", "lfm2", "minimax", "muse_glimmer", "step3p7",
         "nanbeige"],
    )
    def test_measured_family_is_in_the_set(self, family):
        from vmlx_engine.cli import _family_drifts_on_cache_hit

        assert _family_drifts_on_cache_hit(SimpleNamespace(family_name=family)) is True

    @pytest.mark.parametrize(
        "family",
        # NOT "families that are exact" — no family is. These are families the
        # gate does not name, so the switch must leave them alone. Names
        # resolved by running the registry against the real bundles, not
        # recalled: "openpangu_v2" (not "openpangu"), "qwen3_5" (not "qwen3.6").
        ["deepseek_v4", "gemma4", "qwen3_5", "laguna", "openpangu_v2"],
    )
    def test_unnamed_family_is_left_alone(self, family):
        from vmlx_engine.cli import _family_drifts_on_cache_hit

        assert _family_drifts_on_cache_hit(SimpleNamespace(family_name=family)) is False

    def test_the_list_is_not_a_claim_of_exemption(self):
        """The retraction, pinned so it cannot quietly regress.

        An earlier per-family "EXACT vs DIVERGES" map was built from ONE prompt
        per family and treated as an architectural property. Over six prompts
        the classification collapses: **Nanbeige is 5/6 — the worst measured —
        and it is NOT in this list**, while ZAYA, which is, comes in at 1/6.

        A cache hit can change the answer on every family measured; only the
        rate differs. The mechanism (a tail recompute at a different batch
        shape flipping an argmax on a near-tie) is prompt- and
        generation-dependent, not architectural. Today's Muse run reproduced
        exactly that: byte-exact on a short prompt, divergent at ~1.4k tokens.

        So the source must not describe absence from the list as exemption.
        """
        from pathlib import Path

        src = Path(__file__).resolve().parents[1].joinpath("vmlx_engine/cli.py").read_text()
        # Normalise comment reflow: these phrases live in a wrapped block, so a
        # raw substring match breaks every time the comment is re-wrapped. That
        # brittleness already failed this test once for no real reason.
        flat = " ".join(src.replace("#", " ").split())
        assert "ABSENCE FROM THIS LIST IS NOT EXEMPTION" in flat
        # The measured rates are the evidence; keep them next to the list.
        assert "Nanbeige4.2-3B 5/6" in flat
        assert "Report a RATE, never a label." in flat

        from vmlx_engine.cli import _family_drifts_on_cache_hit

        # Nanbeige is 5/6 — the worst measured — and was MISSING from the list
        # until review caught it, which made the switch an exemption for exactly
        # the family most likely to need it. It is named now; this pins that.
        assert (
            _family_drifts_on_cache_hit(SimpleNamespace(family_name="nanbeige"))
            is True
        )

    def test_subtype_alone_is_enough(self):
        # Nemotron-Omni-Nano-JANGTQ is a DIFFERENT bundle from
        # Nemotron-3.5-Lightning and was measured to drift the same way
        # (cold 09e50462 x3, hit be6c858a, 23 reused). Matching on subtype as
        # well as family is what covers bundles nobody has enumerated.
        from vmlx_engine.cli import _family_drifts_on_cache_hit

        mc = SimpleNamespace(family_name="something_new", cache_subtype="zaya_cca")
        assert _family_drifts_on_cache_hit(mc) is True

    def test_missing_config_is_not_a_match(self):
        from vmlx_engine.cli import _family_drifts_on_cache_hit

        assert _family_drifts_on_cache_hit(None) is False
        assert _family_drifts_on_cache_hit(SimpleNamespace()) is False

    def test_gate_is_applied_ahead_of_the_family_chain(self):
        # A correctness switch that only fires for families holding a branch in
        # the chain would silently skip half the measured set. Pin the ordering.
        from pathlib import Path

        src = Path(__file__).resolve().parents[1].joinpath("vmlx_engine/cli.py").read_text()
        gate = src.index("if _family_drifts_on_cache_hit(_mc):")
        chain = src.index('if _mc.family_name == "deepseek_v4":')
        assert gate < chain, "drift gate must run before the family policy chain"


class TestGateFailureIsAnnounced:
    """An armed correctness switch must not fail closed in silence.

    The gate is applied inside `serve_command`'s registry `try`, whose `except`
    logs at DEBUG. Registry lookup raises on a malformed or incomplete JANG
    stamp, so an operator who exported the variable would get NO protection and
    NO indication — the "enabled != engaged" shape this campaign keeps finding.
    """

    def test_source_warns_when_armed_and_lookup_failed(self):
        from pathlib import Path

        src = Path(__file__).resolve().parents[1].joinpath("vmlx_engine/cli.py").read_text()
        i = src.index('logger.debug(f"Registry auto-apply skipped: {e}")')
        window = src[i : i + 2600]
        assert "Prefix-reuse drift gate NOT applied" in window, (
            "a registry failure must announce that the armed gate did not apply"
        )
        # It must check BOTH accepted names, or the legacy variable fails silent.
        assert "VMLX_DISABLE_DRIFTING_PREFIX_REUSE" in window
        assert "VMLX_DISABLE_RECURRENT_PREFIX_REUSE" in window
        # And it must be a warning, not another debug line.
        assert "logger.warning" in window
        # Review caught the first version claiming "lookup failed" and "reuse
        # remains ENABLED" unconditionally — the except wraps the ENTIRE
        # auto-apply block, so a post-gate failure made both claims false. The
        # warning must be gated on what actually happened.
        assert "_drifting_prefix_reuse_announced" in window, (
            "the failure warning must not fire when the gate already applied"
        )
        assert 'getattr(args, "enable_prefix_cache", True)' in window, (
            "the warning must not claim reuse is enabled without checking"
        )
