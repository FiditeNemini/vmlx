# SPDX-License-Identifier: Apache-2.0
"""Canonicalize variant-suffixed special tags at the token→text boundary.

Some bundles ship special tokens carrying a build-variant suffix. The shipped
``tencent/Hy3`` tokenizer is the motivating case: it defines *only* the
suffixed spellings —

    <think:opensource>   </think:opensource>
    <tool_calls:opensource>  <tool_sep:opensource>  <arg_key:opensource>  …
    <｜hy_eos:opensource｜>  <｜hy_User:opensource｜>  <｜hy_Assistant:opensource｜>

— with no bare `<think>` / `<｜hy_eos｜>` token anywhere. vMLX's reasoning
split, think-tag stripping, tool-call parse, and role-boundary stop guards all
match the bare spelling (52 sites in ``server.py`` alone), so every one of them
silently no-ops on such a bundle: the reasoning parser never finds `</think>`,
so the model's actual answer is misfiled as ``reasoning_content`` and
``content`` comes back empty.

Rather than teach 52 call sites a second dialect, normalize once where tokens
become text. Encoding is untouched, so prompts still carry the real special
tokens; only the *decoded* text is rewritten.

The rewrite is literal, never inferred: the suffix is discovered from the
tokenizer's own ``</think:SUFFIX>`` token, and the replacement map is built
from the added-token table. A bundle without such a token gets an empty map and
the identity transform.

Safety: every affected tag is a single atomic special token, so it can never be
split across streaming deltas, and the substitution is idempotent.
"""

from __future__ import annotations

import re
from typing import Any

_VARIANT_THINK_CLOSE = re.compile(r"</think:([A-Za-z0-9_-]+)>\Z")


def _added_token_contents(tokenizer: Any) -> list[str]:
    decoder = getattr(tokenizer, "added_tokens_decoder", None) or {}
    contents: list[str] = []
    for value in decoder.values():
        content = getattr(value, "content", None)
        if content is None and isinstance(value, dict):
            content = value.get("content")
        if content is None and isinstance(value, str):
            content = value
        if isinstance(content, str) and content:
            contents.append(content)
    return contents


def detect_tag_variant(tokenizer: Any) -> str | None:
    """Return the variant suffix (e.g. ``"opensource"``), or None.

    Keyed off the reasoning close tag specifically: that is the token whose
    misspelling breaks correctness, and its presence is what makes the whole
    added-token table suspect.
    """
    for content in _added_token_contents(tokenizer):
        match = _VARIANT_THINK_CLOSE.match(content)
        if match:
            return match.group(1)
    return None


def build_canonical_map(tokenizer: Any) -> dict[str, str]:
    """Map every suffixed special token to its bare spelling."""
    variant = detect_tag_variant(tokenizer)
    if not variant:
        return {}
    marker = f":{variant}"
    mapping: dict[str, str] = {}
    for content in _added_token_contents(tokenizer):
        if marker in content:
            bare = content.replace(marker, "")
            if bare and bare != content:
                mapping[content] = bare
    return mapping


def compile_canonicalizer(mapping: dict[str, str]):
    """Build a text -> text rewriter, or None when there is nothing to do."""
    if not mapping:
        return None
    # Longest-first so no key can be consumed as a prefix of another
    # (`<tool_call:x>` vs `<tool_calls:x>`).
    keys = sorted(mapping, key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(k) for k in keys))
    return lambda text: pattern.sub(lambda m: mapping[m.group(0)], text) if text else text


def make_streaming_detokenizer(tokenizer: Any) -> Any:
    """A ``NaiveStreamingDetokenizer`` that emits canonical tag spellings."""
    from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer

    canon = compile_canonicalizer(build_canonical_map(tokenizer))
    if canon is None:
        detok = NaiveStreamingDetokenizer(tokenizer)
        detok.reset()
        return detok

    class _CanonicalTagDetokenizer(NaiveStreamingDetokenizer):
        @property
        def text(self) -> str:
            # The base property has bookkeeping side effects; call it exactly
            # once. `last_segment` slices `self.text` by `self.offset`, so the
            # offsets stay in canonical-text space and remain consistent.
            return canon(NaiveStreamingDetokenizer.text.fget(self))

    detok = _CanonicalTagDetokenizer(tokenizer)
    detok.reset()
    return detok
