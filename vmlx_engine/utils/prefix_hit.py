# SPDX-License-Identifier: Apache-2.0
"""Canonical prefix-hit tail arithmetic, shared by BOTH schedulers.

Every prefix-cache payload in this engine stores KV state through
``len(key_tokens) - 1``, never ``len(key_tokens)``. Both store sides truncate:
the text lane through ``Scheduler._truncate_cache_to_prompt_length`` and the
MLLM lane through ``MLLMScheduler._truncate_hybrid_cache`` (plain VLM path) or
by re-prefilling ``token_list[:prompt_len - 1]`` (the mixed-SWA / ZAYA clean
store) -- and both then write that N-1 payload under the FULL N-token key. So
a reader that treats the restored offset as ``len(matched)`` skips exactly one
token: its KV is never computed and every later position lands one slot early.

This module exists because that arithmetic was copy-pasted into two files and
then fixed in only one of them (12ee1c8ee fixed the text lane; the MLLM copy
kept the pre-fix semantics AND a docstring asserting the opposite store
contract). Delegating both call sites here is the only shape in which the two
lanes cannot silently disagree again.
"""

from typing import List, Sequence, Tuple

__all__ = [
    "prefix_hit_tail_and_cached_tokens",
    "disk_prefix_hit_tail_and_cached_tokens",
]


def prefix_hit_tail_and_cached_tokens(
    *,
    key_tokens: Sequence[int],
    remaining: Sequence[int],
    gen_prompt_suffix: Sequence[int],
) -> Tuple[List[int], int]:
    """Return prefill tail + cached-token count for an N-1 prefix hit.

    On an exact key hit the cache layer may report ``remaining=[]`` because its
    key matched the full stripped prompt, but the caller must still re-feed the
    last stripped prompt token before any generation-prompt suffix. Otherwise
    thinking/chat-template suffixes get processed without the final user token
    in cache context.
    """
    keys = list(key_tokens or [])
    cache_remaining = list(remaining or [])
    suffix = list(gen_prompt_suffix or [])
    if suffix and not cache_remaining and keys:
        cache_remaining = [keys[-1]]
    if not cache_remaining and keys:
        cached_tokens = max(len(keys) - 1, 0)
    else:
        cached_tokens = max(len(keys) - len(cache_remaining), 0)
    return cache_remaining + suffix, cached_tokens


def disk_prefix_hit_tail_and_cached_tokens(
    *,
    fetch_tokens: Sequence[int],
    matched_tokens: Sequence[int],
    gen_prompt_suffix: Sequence[int],
) -> Tuple[List[int], int]:
    """Return prefill tail + cached count for a disk L2 partial-prefix hit.

    ``matched_tokens`` is the stored key the disk layer matched, and its payload
    covers ``matched[:-1]``. The restored offset is therefore
    ``len(matched) - 1`` and ``matched[-1]`` must LEAD the uncached tail so its
    KV is recomputed -- identical to the exact-hit helper above.

    This also matters for the boundary-swap match (a stored key whose final
    token differs from the query's, e.g. a thinking sentinel toggling between
    turns): ``matched[-1]`` is then the CURRENT prompt's token, so re-feeding it
    is what makes the toggle take effect. Dropping it silently served the
    previous turn's mode.
    """
    keys = list(fetch_tokens or [])
    matched = list(matched_tokens or [])
    suffix = list(gen_prompt_suffix or [])
    if matched:
        offset = max(len(matched) - 1, 0)
        # keys[offset:] == [matched[-1], *uncached_tail]
        tail = list(keys[offset:]) + suffix
        if tail:
            return tail, offset
    return prefix_hit_tail_and_cached_tokens(
        key_tokens=matched or keys,
        remaining=[],
        gen_prompt_suffix=suffix,
    )
