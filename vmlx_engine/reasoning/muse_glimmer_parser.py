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

    def reset_state(self) -> None:
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
        """
        reasoning, content = self._split(current_text)

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
