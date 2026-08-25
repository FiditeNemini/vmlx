# SPDX-License-Identifier: Apache-2.0
"""The media clean boundary must be BLOCK-ALIGNED, or nothing ever pairs.

Measured live on Qwen3.8-27B VL (M5 Max, SSD-only tier, block_size 64), before
this fix:

    turn 2 stores: KV 7607 tokens, SSM companion at 7607
    turn 3 fetches: paged cache hit 117 blocks = 7488 tokens
    turn 3 asks for a companion at 7488 -> nothing (it is at 7607)
    "VLM prefix cache MISS: 7488 KV blocks found but no SSM companion
     state - full prefill required"

The paged chain can only ever be MATCHED on block boundaries, so a companion
stored at an unaligned N-1 is invisible to every future turn: a found
7,488-token hit was discarded on turn 3, again on turn 4, and TTFT climbed
35s -> 60s -> 85s until the prompt tripped a hard prefill guard and the
conversation was dead.

Giving up at most block_size-1 tokens of stored prefix buys all of that back.
"""

import inspect
from types import SimpleNamespace

from vmlx_engine.mllm_batch_generator import (
    MLLMBatchGenerator,
    MLLMBatchResponse,
)


class _Gen:
    """Bare generator exposing only the alignment helper under test."""

    def __init__(self, block_size=64):
        self.gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        from types import SimpleNamespace

        self.gen.block_aware_cache = SimpleNamespace(block_size=block_size)

    def aligned(self, boundary):
        return self.gen._ssm_block_aligned_boundary(boundary)


def test_the_helper_floors_to_a_block_and_steps_back_when_exact():
    g = _Gen(64)
    # 7607 is the real N-1 from the live failure. 7607 // 64 = 118 -> 7552.
    assert g.aligned(7607) == 7552
    # An EXACTLY aligned boundary steps back one block, so a future request
    # that diverges inside the final block still finds a checkpoint.
    assert g.aligned(7552) == 7488
    # Nothing usable below one block.
    assert g.aligned(64) == 0
    assert g.aligned(10) == 0


def test_media_kv_only_miss_teaches_the_next_clean_boundary():
    """A learned KV-only boundary must win once it covers all media.

    The live Qwen tool continuation matched 4,672 KV tokens, but the media
    lane stored its companion only at the later 6,144-token terminal boundary.
    Text-only prefills already honor ``_ssm_required_checkpoint_tokens``;
    media prefills need the same repair when the learned boundary is
    block-aligned and lies strictly after every media placeholder run.
    """
    generator = _Gen(64).gen
    generator._media_placeholder_token_ids = lambda: {99}
    request = SimpleNamespace(_ssm_required_checkpoint_tokens=4672)
    tokens = [1] * 4000 + [99] * 128 + [2] * 2200

    assert generator._media_clean_cache_boundary_for(request, tokens) == 4672


def test_media_required_boundary_never_cuts_through_media():
    generator = _Gen(64).gen
    generator._media_placeholder_token_ids = lambda: {99}
    request = SimpleNamespace(_ssm_required_checkpoint_tokens=4032)
    tokens = [1] * 4000 + [99] * 128 + [2] * 2200

    terminal = generator._ssm_block_aligned_boundary(len(tokens) - 1)
    assert generator._media_clean_cache_boundary_for(request, tokens) == terminal


def test_capture_uses_the_aligned_length_not_n_minus_1():
    src = inspect.getsource(MLLMBatchGenerator._process_prompts)
    assert "_clean_media_len = self._media_clean_cache_boundary_for(" in src, (
        "the clean media capture is back on the unaligned N-1 boundary"
    )
    assert "_media_tokens[:_clean_media_len]" in src, (
        "the prefill still runs over N-1 rather than the aligned prefix"
    )
    assert "req._media_clean_prefix_len = _clean_media_len" in src


def test_the_companion_store_uses_the_same_length_as_the_capture():
    src = inspect.getsource(MLLMBatchGenerator._process_prompts)
    idx = src.index("stored clean media SSM")
    window = src[max(0, idx - 1400) : idx]
    assert '_media_clean_prefix_len' in window, (
        "the companion is stored at a different length than the cache "
        "actually covers — it would claim state it does not have"
    )


def test_the_response_carries_a_separate_clean_store_key():
    """usage.prompt_tokens is derived from prompt_token_ids.

    The scheduler sets request.num_prompt_tokens = len(prompt_token_ids), and
    that becomes the user-visible usage.prompt_tokens. Shortening it to the
    aligned length would misreport every media request, so the aligned key
    travels on its own field.
    """
    assert "clean_store_token_ids" in MLLMBatchResponse.__dataclass_fields__
    assert (
        MLLMBatchResponse.__dataclass_fields__["clean_store_token_ids"].default
        is None
    )

    src = inspect.getsource(MLLMBatchGenerator._next)
    assert "clean_store_token_ids=_clean_store_tokens" in src
    assert "_orig[: _clean_len + 1]" in src, (
        "the store key must be aligned+1 so the N-1 payload contract holds"
    )


def test_the_scheduler_prefers_the_clean_store_key():
    import vmlx_engine.mllm_scheduler as sched

    src = inspect.getsource(sched)
    occurrences = src.count('"clean_store_token_ids", None)')
    assert occurrences >= 2, (
        "both _extracted_tokens assignment sites must prefer the aligned key; "
        "fixing one of two is the default failure mode here (found %d)"
        % occurrences
    )


def test_a_hit_that_does_not_cover_the_media_span_is_declined():
    """Dropping pixel_values is only safe when the hit COVERS the image.

    A warm media turn is cheap precisely because the hybrid hit path sets
    req.pixel_values = None -- the vision tower is skipped since the image's
    KV is already in the restored prefix. That reasoning holds only if the
    cached prefix actually contains the image. If the tail still carries
    placeholders, forwarding it with no pixels embeds them as ordinary text
    tokens and the model answers fluently about an image it never saw. Nothing
    raises, nothing logs, and the answer looks fine.

    So the hit is refused in that case. A full prefill is slow; a confidently
    wrong answer is worse.
    """
    import inspect

    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

    src = inspect.getsource(MLLMBatchGenerator._process_prompts)
    # The current implementation uses a media-tail admission helper rather
    # than the old local `_tail_has_media` variable. A paged hit may clear
    # processor payloads only after the tail is proven media-free or a
    # family-specific whole-item re-encode is admitted; otherwise it must
    # discard the hit and cold-prefill.
    assert "_tokens_contain_media_placeholders" in src
    assert "_prepare_muse_media_tail_for_cache_hit" in src
    assert "_prepare_dots3_media_tail_for_cache_hit" in src
    assert 'reason="media_placeholders_in_uncached_tail"' in src
    assert "_discard_request_cache_hit" in src
    assert "_clear_mllm_request_media_payloads(req)" in src
