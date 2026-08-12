# SPDX-License-Identifier: Apache-2.0
"""The CLI and the app must not disagree about defaults on the same build.

Every entry here is a MEASURED divergence, not a style preference: the same
engine, same model, behaved differently depending on whether it was started by
the Electron app or from a terminal. Those are the worst kind of bug to chase,
because a reproduction from one surface does not reproduce from the other.

The tests read BOTH sources so a change to either side breaks the pair.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "vmlx_engine" / "cli.py"
SESSIONS = ROOT / "panel" / "src" / "main" / "sessions.ts"

pytestmark = pytest.mark.skipif(
    not SESSIONS.exists(), reason="panel sources not present in this checkout"
)


def test_cache_index_target_tokens_match():
    """Both surfaces size the cache index by TOKENS, to the same target.

    A flat block COUNT indexes a different amount of context at a different
    block size: the old CLI default of 1000 blocks at 64 tokens indexed 63,936
    tokens, while the app indexed 262,144. Measured symptom recorded in
    sessions.ts: a 77k prompt got ZERO reuse on an exact repeat (82.5s vs 55.7s
    cold) from a CLI-started engine.
    """
    from vmlx_engine.cli import GENERIC_INDEX_TARGET_TOKENS

    panel = re.search(
        r"const GENERIC_INDEX_TARGET_TOKENS\s*=\s*(\d+)",
        SESSIONS.read_text(encoding="utf-8"),
    )
    assert panel, "panel no longer defines GENERIC_INDEX_TARGET_TOKENS"
    assert int(panel.group(1)) == GENERIC_INDEX_TARGET_TOKENS


def test_cli_sizes_the_index_the_same_way_the_panel_does():
    """indexBlocksForCapacity: ceil(target / blockSize) + 1, the +1 being null."""
    from vmlx_engine.cli import GENERIC_INDEX_TARGET_TOKENS

    def panel_rule(block_size: int) -> int:
        return -(-GENERIC_INDEX_TARGET_TOKENS // block_size) + 1

    # 64 is the shipped default block size; the app sends exactly 4097 there.
    assert panel_rule(64) == 4097
    # And the DSV4 lane's own constant stays consistent with its 256 block.
    from vmlx_engine.cli import DSV4_MAX_CACHE_BLOCKS, DSV4_PAGED_CACHE_BLOCK_SIZE

    assert DSV4_MAX_CACHE_BLOCKS == 4097
    assert DSV4_PAGED_CACHE_BLOCK_SIZE == 256


def test_step3p7_is_paged_exempt_like_the_panel_requires():
    """step3p7 is is_mllm, but the app treats its cache subtype as paged-REQUIRED.

    Without the exemption a bare CLI launch ran it unpaged and re-prefilled every
    repeated long prompt, while the app reused blocks — same build, same model.
    """
    from vmlx_engine.cli import _PAGED_MLLM_EXEMPT_FAMILIES

    assert "step3p7" in _PAGED_MLLM_EXEMPT_FAMILIES
    # The panel's cache-shape rules used to be byte-identical copies in the argv
    # BUILDER (sessions.ts) and the argv PREVIEW (SessionSettings.tsx). They now
    # live in one shared module, so assert the RULE, and separately that both
    # consumers read it — grepping sessions.ts for the literal would pin the
    # duplication this consolidation removed.
    shared = ROOT / "panel/src/shared/cacheTypeCapabilities.ts"
    assert shared.exists(), "the shared cache-shape capability module is gone"
    assert "step3p7_full_sliding_kv" in shared.read_text(encoding="utf-8"), (
        "panel no longer declares the step3p7 paged-required subtype; "
        "re-check whether the CLI exemption is still correct"
    )
    for consumer in (
        SESSIONS,
        ROOT / "panel/src/renderer/src/components/sessions/SessionSettings.tsx",
    ):
        assert "cacheTypeCapabilities" in consumer.read_text(encoding="utf-8"), (
            f"{consumer.name} stopped reading the shared cache-shape rules"
        )


def _serve_arg_default(flag: str):
    """Read a serve-subparser default straight out of cli.py source."""
    src = CLI.read_text(encoding="utf-8")
    i = src.index(f'"{flag}"')
    window = src[i : i + 600]
    m = re.search(r"default=([0-9.]+)", window)
    assert m, f"no default found for {flag}"
    return float(m.group(1))


def test_cache_memory_percent_matches_the_app():
    """CLI 0.20 vs app 15 meant app engines evicted ~25% earlier on the same model."""
    assert _serve_arg_default("--cache-memory-percent") == pytest.approx(0.15)
    panel = (
        ROOT / "panel/src/renderer/src/components/sessions/SessionConfigForm.tsx"
    ).read_text(encoding="utf-8")
    m = re.search(r"cacheMemoryPercent:\s*(\d+)", panel)
    assert m and int(m.group(1)) == 15


def test_flash_moe_slot_bank_matches_the_app_and_itself():
    """CLI 64 vs app 256 is a 4x smaller expert cache on a CLI launch.

    The panel also disagreed with ITSELF: the slider's reset marker was a
    hardcoded 64 while DEFAULT_CONFIG was 256, so clicking reset on a fresh
    session silently quartered it.
    """
    assert _serve_arg_default("--flash-moe-slot-bank") == 256
    form = (
        ROOT / "panel/src/renderer/src/components/sessions/SessionConfigForm.tsx"
    ).read_text(encoding="utf-8")
    m = re.search(r"flashMoeSlotBank:\s*(\d+)", form)
    assert m and int(m.group(1)) == 256
    assert "defaultValue={64}" not in form, (
        "the slot-bank slider reset marker is hardcoded again; it must track "
        "DEFAULT_CONFIG or reset will disagree with the session default"
    )


def test_slow_families_get_the_app_timeout():
    """900s for DSV4/M3/openPangu existed only panel-side; CLI killed at 300s."""
    from vmlx_engine.cli import _SLOW_FAMILY_TIMEOUTS

    # The three the panel declares, plus the hybrid SSM families whose chunked
    # prefill takes minutes on a long prompt. The hybrid entries came from a
    # LIVE app failure: a 101,502-token prompt to Qwen3.6-27B rendered "Message
    # failed - Request timed out after 300s" in the chat while the engine served
    # it in ~230s of prefill. API probes pass their own long timeout and never
    # see it.
    assert set(_SLOW_FAMILY_TIMEOUTS) >= {
        "deepseek_v4", "minimax_m3", "openpangu_v2",
    }
    # qwen3_5_moe was missing here while every panel copy had it, so a bare CLI
    # launch of a Qwen3.6-MoE was still cut off at 300s. The old assertion did
    # not require it.
    assert {
        "qwen3_5", "qwen3_5_moe", "qwen3_next", "nemotron_h",
    } <= set(_SLOW_FAMILY_TIMEOUTS)
    assert all(v == 900 for v in _SLOW_FAMILY_TIMEOUTS.values())


def test_every_panel_surface_carries_the_same_slow_families():
    """The rule must exist ONCE, and every surface must read that one copy.

    MEASURED: it existed in SEVEN places and four had diverged. The in-app chat
    table and the gateway table are the two that actually abort a request.

    This assertion originally grepped each surface for the family literals —
    which pinned the DUPLICATION, exactly the mistake the consolidation removes.
    Now it checks the shared table for coverage and each consumer for the
    import, so a surface that grows its own copy back fails here.

    Keys are PANEL REGISTRY names, not engine family_name: the registry maps
    qwen3_5 -> qwen3.5, qwen3_next -> qwen3-next, nemotron_h -> nemotron-h. A
    previous fix asserted the engine spellings and therefore matched nothing.
    """
    shared = ROOT / "panel/src/shared/slowFamilyTimeouts.ts"
    assert shared.exists(), "the shared slow-family timeout table is gone"
    shared_src = shared.read_text(encoding="utf-8")
    for family in (
        "deepseek-v4", "minimax_m3", "openpangu_v2",
        "qwen3.5", "qwen3.5-moe", "qwen3-next", "nemotron-h",
    ):
        assert family in shared_src, (
            f"shared timeout table does not cover {family!r}"
        )

    consumers = {
        "chat IPC": ROOT / "panel/src/main/ipc/chat.ts",
        "api gateway": ROOT / "panel/src/main/api-gateway.ts",
        "sessions": SESSIONS,
        "CLI preview": ROOT
        / "panel/src/renderer/src/components/sessions/SessionSettings.tsx",
    }
    for label, path in consumers.items():
        src = path.read_text(encoding="utf-8")
        assert "slowFamilyTimeouts" in src, (
            f"{label} ({path.name}) does not read the shared timeout table — "
            f"it will drift and abort requests at the generic 300s"
        )


# Engine family_name -> panel REGISTRY name. Taken from
# panel/src/main/model-config-registry.ts MODEL_TYPE_TO_FAMILY individually, not
# transliterated: the registry is not internally consistent about underscores
# vs dots vs hyphens, and two earlier fixes in this project were silent no-ops
# because they assumed it was.
_ENGINE_TO_PANEL_FAMILY = {
    "gemma4": "gemma4",
    "gemma4_text": "gemma4-text",
    "hy_v3": "hy3",
    "laguna": "laguna",
    "minimax": "minimax",
    # Engine-only alias: "minimax_m2" is the M2.x REASONING PARSER name, and the
    # engine deliberately keeps it in the set so a future bundle that does report
    # it as a family stays covered. No bundle resolves to it today, so it shares
    # the panel row of the family that does.
    "minimax_m2": "minimax",
    "nanbeige": "nanbeige",
    "nemotron": "nemotron",
    "nemotron_h": "nemotron-h",
    "openpangu_v2": "openpangu_v2",
    "qwen3": "qwen3",
    "qwen3_5": "qwen3.5",
    "qwen3_5_moe": "qwen3.5-moe",
    "qwen3_next": "qwen3-next",
}


def test_thinking_budget_families_match_between_engine_and_app():
    """The app must offer a budget control for every family the engine caps.

    This is not cosmetic. The engine's never-empty answer pass is ARMED by a
    CAPPED first pass — without a cap the model can spend the whole budget
    inside <think> and return EMPTY content at finish=length, and the salvage
    never runs. The only way a user supplies that cap from the app is the "Max
    Thinking Tokens" field, which renders only when the family's
    supportsThinkingBudget is true.

    MEASURED 2026-08-12: nemotron/nemotron_h had been in the engine's cap set
    since 2026-08-11, but no panel row declared supportsThinkingBudget, so the
    Chat Settings drawer for a LIVE Nemotron session showed Enable Thinking and
    no budget field — the fix existed and the user could not reach it. Nanbeige
    was in the same state, with a measured empty answer to show for it
    (max_tokens=220/max_thinking_tokens=160: reasoning 1016 chars, content 0).
    """
    from vmlx_engine.server import _THINKING_BUDGET_CAP_FAMILIES

    shared = ROOT / "panel/src/shared/thinkingBudgetFamilies.ts"
    assert shared.exists(), "the shared thinking-budget family list is gone"
    panel_families = set(
        re.findall(r"^\s*'([^']+)',$", shared.read_text(encoding="utf-8"), re.M)
    )
    assert panel_families, "could not parse THINKING_BUDGET_FAMILIES"

    unmapped = set(_THINKING_BUDGET_CAP_FAMILIES) - set(_ENGINE_TO_PANEL_FAMILY)
    assert not unmapped, (
        f"engine caps {sorted(unmapped)} but this test has no panel registry "
        "spelling for them — add the mapping AND the panel entry, or the app "
        "silently hides the control that arms their answer pass"
    )

    expected = {_ENGINE_TO_PANEL_FAMILY[f] for f in _THINKING_BUDGET_CAP_FAMILIES}
    missing = expected - panel_families
    assert not missing, (
        f"panel does not offer a thinking budget for {sorted(missing)}, which "
        "the engine caps — their never-empty answer pass is unreachable in-app"
    )
    # minimax_m3 runs its own dedicated reasoning-only answer path rather than
    # the shared cap set, so it is legitimately panel-only.
    assert panel_families - expected <= {"minimax_m3"}, (
        f"panel offers a thinking budget for {sorted(panel_families - expected)}"
        " that the engine does not cap — the control would do nothing"
    )


def test_registry_rows_no_longer_carry_their_own_thinking_budget_flag():
    """The flag must come from the one shared list, not from per-family rows.

    Eleven rows each declared it by hand, and the ones added later simply forgot
    — which is how nemotron shipped a capped engine and an app with no control.
    """
    registry = ROOT / "panel/src/main/model-config-registry.ts"
    src = registry.read_text(encoding="utf-8")
    assert "thinkingBudgetFamilies" in src, (
        "model-config-registry.ts no longer reads the shared family list"
    )
    assert "familySupportsThinkingBudget(familyName)" in src, (
        "registerFamily no longer derives supportsThinkingBudget from the list"
    )
    assert not re.search(r"supportsThinkingBudget:\s*(?:true|false)", src), (
        "a registerFamily row grew its own supportsThinkingBudget literal back"
    )
