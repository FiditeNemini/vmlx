# SPDX-License-Identifier: Apache-2.0
"""Reasoning parser for Muse Glimmer.

Muse Glimmer separates reasoning from the answer by RECIPIENT, not by an
inline ``<think>`` pair and not by a named channel. The prompt ends at
``<|start|>assistant`` and the model itself picks who the message is for:

    <|start|>assistant to=self<|message|>  ...reasoning...  <|eom|>
    <|start|>assistant to=user<|message|>  ...answer...     <|eot|>

``to=self`` is the reasoning rail. ``to=user`` — or no recipient at all, which
the template treats as ``user`` — is the visible answer. A recipient naming a
tool namespace carries an ATEM call and belongs to the tool parser, so this
parser leaves that text in ``content`` untouched rather than guessing.

``<|eom|>`` ends one message and more follow; ``<|eot|>`` ends the turn.

Two details that follow from the bundle's own contract:

* ``think_in_template`` is false, so no reasoning prefix is injected into the
  prompt. Unlike the Harmony path, output does NOT begin mid-reasoning — the
  first thing the model emits is its recipient. Text before any recipient
  marker is therefore treated as answer content, not reasoning.
* The generation prompt already emitted ``<|start|>assistant``, so the first
  recipient arrives as a bare `` to=self<|message|>`` with no leading
  ``<|start|>``. Both forms are accepted.
"""

import re

from .base import DeltaMessage, ReasoningParser

_START_TAG = "<|start|>"
_MESSAGE_TAG = "<|message|>"
_EOM_TAG = "<|eom|>"
_EOT_TAG = "<|eot|>"

_SELF_RECIPIENT = "self"
_USER_RECIPIENT = "user"

# One assistant message header. The <|start|>assistant portion is optional
# because the generation prompt already emitted it for the first message.
_HEADER_RE = re.compile(
    r"(?:<\|start\|>\s*assistant)?\s*(?:to=(?P<recipient>[^\s<|]+))?\s*<\|message\|>",
)


_ALL_MARKERS = (_START_TAG, _MESSAGE_TAG, _EOM_TAG, _EOT_TAG)

# A header is only ever `<|start|>assistant to=<recipient><|message|>`, so it
# can contain nothing but these characters. The moment the pending run shows
# anything else — punctuation, a newline — it is ordinary prose and must be
# released, or plain replies would never stream at all.
_HEADER_CHARS_RE = re.compile(r"^[A-Za-z0-9_<|>=\- ]*$")
_MAX_HEADER_LEN = 64


def _viable_header_prefix(pending: str) -> bool:
    return len(pending) <= _MAX_HEADER_LEN and bool(_HEADER_CHARS_RE.match(pending))


def _stable_length(text: str) -> int:
    """How much of ``text`` can be interpreted without seeing more tokens.

    Streaming hands us the message a few characters at a time, and both the
    markers and the ``to=<recipient>`` header arrive in pieces. Interpreting a
    partial piece is unrecoverable here: emitted counters only move forward, so
    anything mis-classified early stays wrong for the rest of the turn. Live in
    the app that surfaced as ``to=self`` and bare ``<|message|`` printed as
    answer text.

    So hold back any trailing run that could still grow into a marker or a
    header, and let the next delta re-decide. Whatever is held is emitted as
    soon as the ambiguity resolves, and the final non-streaming pass sees the
    whole text anyway.
    """
    hold = 0

    # 1. A trailing PROPER PREFIX of anything that opens a marker or header:
    #    "<", "<|", "<|mess", "t", "to", "assis", ... Holding one of these
    #    costs a single delta of latency and prevents "to" or "<|eom|" being
    #    printed as prose.
    for opener in _ALL_MARKERS + ("to=", "assistant"):
        for size in range(len(opener) - 1, 0, -1):
            if text.endswith(opener[:size]):
                hold = max(hold, size)
                break

    # 2. A header that has OPENED but not yet reached its <|message|>. The
    #    recipient decides which rail the following body belongs to, so nothing
    #    from the header start onward can be classified until it terminates.
    #
    #    A header may only begin where a message can begin: at the very start,
    #    or immediately after <|eom|> / <|eot|> / <|start|>. Anchoring on those
    #    is what keeps ordinary prose containing "assistant" or "to=" from
    #    being mistaken for a header and held back forever.
    boundary = 0
    for opener in (_EOM_TAG, _EOT_TAG, _START_TAG):
        found = text.rfind(opener)
        if found >= 0:
            boundary = max(
                boundary,
                found if opener is _START_TAG else found + len(opener),
            )
    pending = text[boundary:]
    if _MESSAGE_TAG not in pending and _viable_header_prefix(pending):
        # This message's header is still arriving.
        hold = max(hold, len(pending))

    return max(0, len(text) - hold)


def _segments(text: str) -> list[tuple[str, str]]:
    """Split model output into ``(recipient, body)`` pairs.

    Any text preceding the first header is returned under the ``user``
    recipient: with no reasoning prefix injected, unmarked leading text is
    answer content.
    """
    out: list[tuple[str, str]] = []
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return [(_USER_RECIPIENT, text)] if text else []

    lead = text[: matches[0].start()]
    # Bare "<|start|>assistant" with no <|message|> yet is just the header the
    # prompt already emitted; it is not answer text.
    lead_clean = lead.replace(_START_TAG, "").replace("assistant", "").strip()
    if lead_clean:
        out.append((_USER_RECIPIENT, lead))

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end]
        for terminator in (_EOM_TAG, _EOT_TAG):
            cut = body.find(terminator)
            if cut >= 0:
                body = body[:cut]
        recipient = (match.group("recipient") or _USER_RECIPIENT).strip()
        out.append((recipient, body))
    return out


class MuseGlimmerReasoningParser(ReasoningParser):
    """Split Muse Glimmer output into reasoning (``to=self``) and answer."""

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self.reset_state()

    def reset_state(self, **kwargs) -> None:
        """Reset per-request counters.

        Must accept **kwargs: the server passes ``think_in_prompt`` and
        ``harmony_active`` to every reasoning parser. Neither applies to Muse —
        its rail is chosen by recipient at generation time, not opened by the
        prompt — but refusing the keywords turns every chat request into a 500.
        """
        self._emitted_reasoning = 0
        self._emitted_content = 0

    def _split(self, model_output: str) -> tuple[str, str]:
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        for recipient, body in _segments(model_output):
            if recipient == _SELF_RECIPIENT:
                reasoning_parts.append(body)
            else:
                # Tool-recipient bodies stay in content; the ATEM tool parser
                # owns them and must see the markup verbatim.
                content_parts.append(body)
        return "".join(reasoning_parts), "".join(content_parts)

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        if not model_output:
            return None, None
        reasoning, content = self._split(model_output)
        return (reasoning.strip() or None), (content.strip() or None)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """Emit only what is newly resolved.

        Re-splitting the whole accumulated text each delta keeps a header that
        straddles a chunk boundary from being mistaken for body text; the
        emitted counters then make the result incremental.

        Only the STABLE prefix is split. Emitted counters are monotonic, so a
        character classified from a half-arrived header can never be taken
        back — holding the ambiguous tail is the only way to stay correct.
        """
        reasoning, content = self._split(current_text[: _stable_length(current_text)])

        new_reasoning = reasoning[self._emitted_reasoning:]
        new_content = content[self._emitted_content:]
        self._emitted_reasoning = len(reasoning)
        self._emitted_content = len(content)

        if not new_reasoning and not new_content:
            return None
        return DeltaMessage(
            reasoning=new_reasoning or None,
            content=new_content or None,
        )
