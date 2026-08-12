"""DSV4 keeps its native direct answer pass; Step3p7 does not fake one.

Regression guard for the DSV4/Step-3.7 empty-content-after-reasoning fix:
- DSV4 must be in _REASONING_ANSWER_PASS_FAMILIES because its source encoder
  owns a real direct rail.
- Step3p7 must be absent because its official template always opens <think>.
- Both remain outside _THINKING_BUDGET_CAP_FAMILIES: they key on
  reasoning_effort, not a thinking-token budget, so max_thinking_tokens is an
  inert field and must not cap max_tokens for them (#89).
"""
import vmlx_engine.server as server_mod


def test_dsv4_has_native_answer_pass_but_step37_does_not_fake_one():
    fams = server_mod._REASONING_ANSWER_PASS_FAMILIES
    assert "deepseek_v4" in fams
    assert "step3p7" not in fams
    # existing families still covered
    for f in ("qwen3", "qwen3_5", "qwen3_5_moe", "gemma4", "hy_v3", "laguna",
              "minimax_m2", "openpangu_v2"):
        assert f in fams


def test_dsv4_and_step37_excluded_from_thinking_budget_cap():
    cap = server_mod._THINKING_BUDGET_CAP_FAMILIES
    assert "deepseek_v4" not in cap
    assert "step3p7" not in cap
    # families that DO honor a thinking-token budget remain capped
    for f in ("qwen3", "qwen3_5", "qwen3_5_moe", "gemma4", "hy_v3", "laguna",
              "minimax_m2", "openpangu_v2"):
        assert f in cap


def test_answer_pass_labels_for_new_families():
    label = server_mod._reasoning_answer_pass_family_label
    assert label("deepseek_v4") == "DeepSeek-V4"
    assert label("step3p7") == "Step-3.7"
    assert label("qwen3") == "Qwen3"
    # unchanged defaults
    assert label("hy_v3") == "Hy3"
    assert label("qwen3_5") == "Qwen3.5"


_MSGS = [{"role": "user", "content": "Remember codeword BLUE-FALCON."}]
_TRUNC = "We are given the task. Interpretation: We need BLUE-F"


def test_answer_pass_fresh_context_families():
    """Malformed/double assistant templates and Qwen's live-proven planning
    continuation must re-run the ORIGINAL messages with nothing appended."""
    for fam in (
        "deepseek_v4", "gemma4", "laguna", "step3p7", "minimax", "minimax_m2",
        "qwen3", "qwen3_5", "qwen3_5_moe",
    ):
        out = server_mod._answer_pass_messages(_MSGS, fam, _TRUNC)
        assert out == _MSGS
        assert out is not _MSGS  # fresh copy, caller list not aliased


def test_answer_pass_appends_reasoning_turn_for_legacy_families():
    """Other legacy families keep the truncated
    reasoning rides along as an assistant turn."""
    for fam in ("hy_v3", "openpangu_v2", None,
                "reasoning model"):
        out = server_mod._answer_pass_messages(_MSGS, fam, _TRUNC)
        assert out[:-1] == _MSGS
        assert out[-1] == {
            "role": "assistant",
            "content": "",
            "reasoning_content": _TRUNC,
        }


def test_no_answer_family_keeps_full_pass_buffer_by_name():
    """All families use the dynamic control-prefix guard for stream safety."""
    guard = server_mod._ANSWER_PASS_LEAK_GUARD_FAMILIES
    assert guard == frozenset()
    for fam in (
        "step3p7", "minimax", "minimax_m2", "qwen3", "qwen3_5", "qwen3_5_moe",
        "gemma4", "hy_v3", "laguna", "openpangu_v2",
    ):
        assert fam not in guard


def test_thinking_reentry_matches_tag_variants():
    """DSV4 live-emitted "<thinking>..." (deterministic 3/3). A literal
    "<think>" needle does NOT match it — the closing ">" never aligns with
    "ing>" — which is exactly the miss the open-prefix helper fixes."""
    reentry = server_mod._answer_pass_thinking_reentry
    assert "<think>" not in "<thinking>Let's parse the user's request."
    assert reentry("<thinking>Let's parse the user's request.")
    assert reentry("<think>\nplanning\n</think>")
    assert reentry("prefix text <think>")
    assert not reentry("A clean answer: BLUE-FALCON 37 Paris.")
    assert not reentry("")
    assert not reentry(None)


def test_answer_pass_buffers_degraded_gemma_thought_prefix():
    """The Gemma detokenizer may emit ``thought`` before its newline.

    Streaming that first chunk is irreversible; wait until the native channel
    either closes (then expose only the answer) or ends unclosed (hide it).
    """
    visible = server_mod._answer_pass_safe_visible_raw
    for partial in ("t", "thoug", "thought"):
        assert visible(partial, finished=False) is None
    assert visible("thought\nprivate", finished=False) is None
    assert visible("thought\nprivate", finished=True) == ""
    assert (
        visible("thought\nprivate<channel|>VISIBLE", finished=False)
        == "VISIBLE"
    )
    assert visible("thoughtful response", finished=False) == "thoughtful response"


def test_nemotron_families_armed_for_answer_pass():
    """Nemotron reasons in a plain <think> block with no stop pressure, so a
    hard prompt can spend the whole budget reasoning and return EMPTY content
    at finish=length — it was simply missing from the never-empty set. Both
    registry spellings ("nemotron" dense, "nemotron_h" hybrid incl. the
    nemotron_h_v2 model_type alias) must arm the rail, cap the first pass on a
    client max_thinking_tokens (token-budget keyed like qwen3, not
    effort-keyed), and use FRESH context: the template renders an appended
    reasoning turn as a completed assistant turn followed by a second
    assistant open (render-probed 2026-08-11)."""
    for fam in ("nemotron", "nemotron_h"):
        assert fam in server_mod._REASONING_ANSWER_PASS_FAMILIES
        assert fam in server_mod._THINKING_BUDGET_CAP_FAMILIES
        assert fam in server_mod._ANSWER_PASS_FRESH_CONTEXT_FAMILIES
        out = server_mod._answer_pass_messages(_MSGS, fam, _TRUNC)
        assert out == _MSGS
        assert out is not _MSGS
        assert server_mod._reasoning_answer_pass_family_label(fam) == "Nemotron"


def test_minimax_family_armed_for_answer_pass():
    """MiniMax-M2.x bundles report family_name "minimax" — the parser name
    "minimax_m2" alone left M2.7 reasoning-only turns EMPTY (live-proven
    2026-07-12). Both spellings must arm the rail and label as MiniMax-M2."""
    assert "minimax" in server_mod._REASONING_ANSWER_PASS_FAMILIES
    assert "minimax" in server_mod._THINKING_BUDGET_CAP_FAMILIES
    label = server_mod._reasoning_answer_pass_family_label
    assert label("minimax") == "MiniMax-M2"
    assert label("minimax_m2") == "MiniMax-M2"


def test_nanbeige_armed_for_answer_pass():
    """Nanbeige 4.2 runs the qwen3 reasoning parser with think_in_template, so a
    hard prompt spends the whole budget inside <think> and returns EMPTY content
    at finish=length — it was simply missing from the never-empty set.

    Live-measured on the box 2026-08-12 (Nanbeige4.2-3B-JANG_4M, max_tokens=220,
    max_thinking_tokens=160, temp 0): finish=length, reasoning 1016 chars,
    content 0 chars. The IDENTICAL request against nemotron_h — already a member
    — answered in 259 chars, which isolates the difference to this set. After
    adding it: content 250 chars, reasoning capped 1016 -> 740.

    FRESH context, not an appended reasoning turn: the bundle's own template
    renders the appended turn as "</think>\\n\\n<|im_end|>\\n<|im_start|>
    assistant\\n<think>..." — a completed assistant turn followed by a second
    assistant open, the same back-to-back shape as nemotron/step3p7/minimax
    (render-probed 2026-08-12).

    The rail is native rather than coercion — enable_thinking=False prefills
    "<think>\\n\\n</think>\\n\\n" — so it does not belong in the step3p7 carve-out.
    """
    assert "nanbeige" in server_mod._REASONING_ANSWER_PASS_FAMILIES
    assert "nanbeige" in server_mod._THINKING_BUDGET_CAP_FAMILIES
    assert "nanbeige" in server_mod._ANSWER_PASS_FRESH_CONTEXT_FAMILIES
    out = server_mod._answer_pass_messages(_MSGS, "nanbeige", _TRUNC)
    assert out == _MSGS
    assert out is not _MSGS


def test_qwen3_next_and_gemma4_text_cover_their_twins():
    """Parity additions reasoned from the registry rows, NOT live-proven — no
    bundle on the test box resolves to either family, so they are guarding the
    twin rather than a measured repro.

    qwen3_next declares reasoning_parser="qwen3", think_in_template=True,
    supports_thinking=True — the same reasoning contract as qwen3/qwen3_5/
    qwen3_5_moe, which are all members.

    gemma4_text is the text-only twin of gemma4: identical tool_parser,
    reasoning_parser, eos_tokens and special_tokens_to_clean, differing only by
    is_mllm/architecture_hints. Note the shipping Gemma 4 bundles (26B-A4B, E4B,
    31B) carry text_config.model_type "gemma4_text" yet still resolve to family
    "gemma4" through registry.lookup() (verified 2026-08-12), so this row is
    reached only by a bundle whose TOP-LEVEL model_type is gemma4_text.
    """
    for fam in ("qwen3_next", "gemma4_text"):
        assert fam in server_mod._REASONING_ANSWER_PASS_FAMILIES
        assert fam in server_mod._THINKING_BUDGET_CAP_FAMILIES
        assert fam in server_mod._ANSWER_PASS_FRESH_CONTEXT_FAMILIES


def test_every_answer_pass_family_has_an_explicit_label():
    """_reasoning_answer_pass_family_label falls back to "Qwen3.5", and the
    result is printed in the engine log the user reads in the Logs tab. An
    unmapped member therefore names the WRONG model in a line about the user's
    own request — nanbeige, qwen3_next and gemma4_text all did before this
    guard. Any family added to the set must be added to the label map too.
    """
    label = server_mod._reasoning_answer_pass_family_label
    for fam in server_mod._REASONING_ANSWER_PASS_FAMILIES:
        if fam in ("qwen3_5", "qwen3_5_moe"):
            continue  # these two legitimately own the "Qwen3.5" fallback
        assert label(fam) != "Qwen3.5", (
            f"{fam} is in _REASONING_ANSWER_PASS_FAMILIES but has no entry in "
            "_reasoning_answer_pass_family_label, so it silently logs as Qwen3.5"
        )
