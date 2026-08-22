# SPDX-License-Identifier: Apache-2.0
"""Once an image enters a chat, text-only follow-ups must still reuse cache.

The two halves of the old gate disagreed by construction:

  _media_context      is TOKEN-based  -> True for the REST of the conversation
                                         once a placeholder is in the prompt
  _media_cache_allowed needs req._cache_extra_keys, which is PAYLOAD-derived
                                      -> None the moment the user stops
                                         re-attaching the picture

so `(not _media_context or _media_cache_allowed)` was False on every text-only
turn after an image, and the fetch was skipped OUTRIGHT — not even the
pure-text prefix that had been stored unsalted on turn 1. Cost in a VL
document chat: a full re-prefill of the entire history, every single turn,
forever.

Everything strictly before the first media placeholder is pure text, is
token-deterministic, and was stored under the unsalted key. Reusing exactly
that region cannot pair recurrent state with any image, because the boundary
precedes every placeholder by construction.
"""

import inspect

from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator


def _process_prompts_source():
    return inspect.getsource(MLLMBatchGenerator._process_prompts)


def test_gate_admits_a_text_only_turn_whose_history_holds_media():
    src = _process_prompts_source()
    assert "_media_text_prefix_only" in src
    assert "or _media_text_prefix_only" in src, (
        "the fetch gate still skips text-only turns after an image"
    )


def test_the_flag_requires_the_absence_of_a_payload():
    """A turn that DOES carry pixels must never take the capped path.

    Its salted lane is correct and complete; routing it here would reuse a
    text prefix under an unsalted key while the rest of the turn is salted.
    """
    src = _process_prompts_source()
    start = src.index("_media_text_prefix_only = bool(")
    body = src[start : start + 400]
    assert "not _media_payload_present" in body
    assert "not _media_cache_allowed" in body
    for field in (
        "pixel_values",
        "video_pixel_values",
        "audio_codes",
        "audio_embeds",
        "audio_features",
    ):
        assert field in src, "payload probe misses %s" % field


def test_the_cap_is_the_first_placeholder_and_the_key_is_unsalted():
    src = _process_prompts_source()
    start = src.index("if _media_text_prefix_only:")
    body = src[start : start + 1800]
    assert "_media_safe_capture_limit" in body, (
        "the cap must be the first media placeholder, not an arbitrary length"
    )
    assert "_cache_extra_keys = None" in body, (
        "the pre-media region was STORED unsalted, so it must be READ "
        "unsalted — a salted read can never find it"
    )
    assert "block_size" in body, (
        "a hit shorter than one block is not worth a fetch"
    )


def test_remaining_is_recomputed_against_the_FULL_prompt():
    """The capped fetch returns a tail that stops at the cap.

    Using it verbatim would silently drop every token from the cap to the end
    of the prompt — the media region included — so the turn would answer from
    a truncated prompt. `remaining` must be rebuilt from the full token list.
    """
    src = _process_prompts_source()
    start = src.index("if _media_prefix_cap is not None and block_table is not None:")
    body = src[start : start + 700]
    assert "remaining = list(token_list[_hit_len:])" in body
    assert 'getattr(block_table, "num_tokens", 0)' in body, (
        "the hit length must come from the block table, not from the cap"
    )


def test_media_safe_capture_limit_is_the_index_of_the_first_placeholder():
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen._media_placeholder_token_ids = lambda: {99}

    # Deliberately NOT a multiple of 64 or 256: the aligned lengths are where
    # an off-by-one in the cap would be invisible. Token ids start at 1000 so
    # none of the filler can collide with the placeholder id itself — the
    # first version of this test used range(1003), which contains 99 at index
    # 99, and the helper correctly reported 99.
    tokens = list(range(1000, 2003)) + [99] * 37 + list(range(3000, 3211))
    assert gen._media_safe_capture_limit(tokens) == 1003
    assert gen._media_safe_capture_limit([99] + list(range(1000, 1050))) == 0
    assert gen._media_safe_capture_limit(list(range(1000, 1050))) == 50


def test_no_placeholders_means_the_whole_prompt_is_safe():
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen._media_placeholder_token_ids = lambda: set()
    assert gen._media_safe_capture_limit(list(range(777))) == 777
