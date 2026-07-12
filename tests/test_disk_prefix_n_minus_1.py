# SPDX-License-Identifier: Apache-2.0
"""Regression: disk L2 longest-prefix hit must honor the N-1 payload contract.

Disk prompt L2 entries are stored with:
  * key    = the full (gen-prompt-stripped) prompt, length N
  * payload = KV state for N-1 tokens  (``_truncate_cache_to_prompt_length``
              truncates to ``prompt_len - 1`` so the last prompt token is
              re-fed on a cache hit; docstring: "forward prefix match:
              remaining has extra tokens INCLUDING THE Nth token").

So on ANY disk hit the restored cache offset is ``len(matched) - 1`` and the
scheduler must re-feed ``matched[-1]`` before the uncached tail. The exact-hit
helper (``_prefix_hit_tail_and_cached_tokens``) already does this. The
longest-prefix helper (``_disk_prefix_hit_tail_and_cached_tokens``) must agree,
or a warm disk-prefix hit silently DROPS ``matched[-1]`` from the fed sequence
(positional skew / warm-vs-cold divergence) on non-hybrid models.

Bug prior to fix: the prefix helper returned ``cached=len(matched)`` with
``tail=fetch[len(matched):]`` — skipping ``matched[-1]`` and over-counting the
cache offset by one.
"""

from vmlx_engine.scheduler import Scheduler


def test_exact_hit_helper_refeeds_last_token():
    """Baseline: the exact-hit helper re-feeds the last key token (N-1)."""
    # key = [10,11,12,13,14]; payload covers 4 tokens; token 14 must be re-fed.
    tail, cached = Scheduler._prefix_hit_tail_and_cached_tokens(
        fetch_tokens=[10, 11, 12, 13, 14],
        remaining=[],
        gen_prompt_suffix=[99],
    )
    assert cached == 4, f"exact-hit cached must be N-1=4, got {cached}"
    assert tail == [14, 99], f"exact-hit must re-feed matched[-1]=14, got {tail}"


def test_disk_prefix_helper_refeeds_last_matched_token():
    """The disk longest-prefix helper must ALSO re-feed matched[-1] (N-1)."""
    # fetch = [10..16] (7 tokens); disk matched the first 5 = [10,11,12,13,14].
    # payload covers 4 tokens (matched len - 1) -> token 14 must be re-fed,
    # then the uncached tail [15,16], then the gen-prompt suffix [99].
    tail, cached = Scheduler._disk_prefix_hit_tail_and_cached_tokens(
        fetch_tokens=[10, 11, 12, 13, 14, 15, 16],
        matched_tokens=[10, 11, 12, 13, 14],
        gen_prompt_suffix=[99],
    )
    assert cached == 4, (
        f"disk-prefix cached must be len(matched)-1=4 (payload is N-1), got {cached}"
    )
    # matched[-1]=14 MUST be present and MUST lead the tail (positional order).
    assert 14 in tail, f"disk-prefix must re-feed matched[-1]=14, got tail={tail}"
    assert tail == [14, 15, 16, 99], (
        f"disk-prefix tail must be [matched[-1], *uncached, *suffix], got {tail}"
    )


def test_disk_prefix_helper_matches_exact_helper_on_full_match():
    """When matched==fetch the disk helper must equal the exact-hit helper."""
    fetch = [10, 11, 12, 13, 14]
    exact_tail, exact_cached = Scheduler._prefix_hit_tail_and_cached_tokens(
        fetch_tokens=fetch, remaining=[], gen_prompt_suffix=[99]
    )
    # Caller routes full-match to the exact helper, but the disk helper's own
    # fallback (matched==fetch, tail-empty path) must not disagree.
    disk_tail, disk_cached = Scheduler._disk_prefix_hit_tail_and_cached_tokens(
        fetch_tokens=fetch, matched_tokens=fetch, gen_prompt_suffix=[99]
    )
    assert (disk_tail, disk_cached) == (exact_tail, exact_cached)


def test_disk_prefix_helper_no_suffix():
    """Without a gen-prompt suffix the tail is just [matched[-1], *uncached]."""
    tail, cached = Scheduler._disk_prefix_hit_tail_and_cached_tokens(
        fetch_tokens=[10, 11, 12, 13, 14, 15, 16],
        matched_tokens=[10, 11, 12, 13, 14],
        gen_prompt_suffix=[],
    )
    assert cached == 4
    assert tail == [14, 15, 16]


def test_disk_prefix_helper_single_matched_token():
    """Degenerate single-token match: offset floors at 0, re-feed that token."""
    tail, cached = Scheduler._disk_prefix_hit_tail_and_cached_tokens(
        fetch_tokens=[10, 11, 12],
        matched_tokens=[10],
        gen_prompt_suffix=[99],
    )
    assert cached == 0, f"single-token match cannot claim any cached offset, got {cached}"
    assert tail == [10, 11, 12, 99], f"got {tail}"
