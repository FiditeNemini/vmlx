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

# Residue of a MESSAGE HEADER that survived into visible answer text — the
# `<|eom|>assistant to=user<|message|>` class that rendered mid-reply. Matching
# the whole header shape, not individual tokens, is deliberate:
#
#   * A lone control token is NOT proof of a leak. A well-formed answer whose
#     subject is the template itself ("Muse frames turns with the <|start|>
#     token") legitimately spells one, and deleting it silently corrupts a
#     correct reply.
#   * An ATEM call can ride the ``to=user`` rail (the template routes a
#     tool-less recipient there), so a token-level scrub can also reach inside
#     `<atem:parameter>` values and corrupt a tool body.
#
# Requiring a boundary/START marker AND at least one following header element
# (`assistant`, `to=<recipient>`, or the `<|message|>` terminator) means only
# genuine structure is removed. Prose keeps its text.
_HEADER_RESIDUE = (
    r"(?:<\|eom\|>|<\|eot\|>|<\|start\|>)"
    r"(?:\s*assistant)?"
    r"(?:\s*to=[^\s<|]*)?"
    r"\s*(?:<\|message\|>)?"
)

# Residue is only scrubbed at a message BOUNDARY — the start or the end of a
# body. A real leak can only appear there: structure that outran the segmenter
# arrives at the head of the next body, and a generation cut mid-header leaves
# its fragment at the tail. Header shape in the MIDDLE of a body is prose.
#
# Scanning the whole body instead deleted correct answers. Live, Muse replied
# "It opens with the <|start|>assistant control token" and the user was shown
# "It opens with the control token" — the sentence a user gets whenever they ask
# how the chat template works, which is exactly when they will spell a header.
_LEADING_RESIDUE_RE = re.compile(r"^\s*" + _HEADER_RESIDUE)
_TRAILING_RESIDUE_RE = re.compile(_HEADER_RESIDUE + r"\s*$")


def _is_header_residue(text: str) -> bool:
    """True when the match carried real header structure, not a bare token."""
    tail = re.sub(r"^\s*(?:<\|eom\|>|<\|eot\|>|<\|start\|>)", "", text, count=1)
    return bool(tail.strip())


def _strip_answer_markers(body: str) -> str:
    """Remove leaked message-header structure from ANSWER (``to=user``) text.

    On well-formed output the segmenter has already consumed every header, so
    this is a no-op. It is a backstop for the malformed / max-tokens-cut case the
    monotonic streaming counters cannot walk back. Applied only to the user rail:
    reasoning keeps its text, and a NAMED tool recipient's body is never routed
    through here at all, so the ATEM parser still sees it byte-for-byte.
    """
    leading = _LEADING_RESIDUE_RE.match(body)
    if leading and _is_header_residue(leading.group(0)):
        body = body[leading.end():]
    trailing = _TRAILING_RESIDUE_RE.search(body)
    if trailing and _is_header_residue(trailing.group(0)):
        body = body[: trailing.start()]
    return body


def _is_prefix_of(text: str, target: str) -> bool:
    return target.startswith(text)


def _viable_header_prefix(pending: str) -> bool:
    """Could ``pending`` still grow into ``<|start|>assistant to=X<|message|>``?

    Matched against the GRAMMAR, not a character class. A character class is
    the wrong tool twice over: too narrow and it rejects the header's own
    ``<|message|>`` terminator (or a dotted recipient like
    ``to=atem.get_weather``) and streams a real header as prose; too wide and
    ordinary prose looks header-shaped, gets held, and — because nothing
    flushes the tail at stream end — is silently DROPPED.

    Only text that genuinely starts like a header is ever held, so a plain
    reply streams immediately and can never lose characters.
    """
    rest = pending

    # optional <|start|> (or a growing prefix of it)
    if rest and _START_TAG.startswith(rest):
        return True
    if rest.startswith(_START_TAG):
        rest = rest[len(_START_TAG):]

    rest = rest.lstrip(" \t")
    # optional literal `assistant` (or a growing prefix of it)
    if rest and "assistant".startswith(rest):
        return True
    if rest.startswith("assistant"):
        rest = rest[len("assistant"):]
    rest = rest.lstrip(" \t")

    # optional `to=<recipient>`; the recipient matches _HEADER_RE's [^\s<|]+
    if rest and "to=".startswith(rest):
        return True
    if rest.startswith("to="):
        rest = rest[3:]
        cut = 0
        while cut < len(rest) and rest[cut] not in " \t<|":
            cut += 1
        rest = rest[cut:].lstrip(" \t")

    # Nothing left yet just means the header is still arriving (the recipient
    # may still be growing, or <|message|> has not started); anything that IS
    # left must be growing into <|message|>.
    return not rest or _is_prefix_of(rest, _MESSAGE_TAG)


def _stable_length(text: str) -> int:
    """How much of ``text`` can be interpreted without seeing more tokens.

    Streaming hands us the message a few characters at a time, and both the
    markers and the ``to=<recipient>`` header arrive in pieces. Interpreting a
    partial piece is unrecoverable here: emitted counters only move forward, so
    anything mis-classified early stays wrong for the rest of the turn. Live in
    the app that surfaced as ``to=self`` and bare ``<|message|`` printed as
    answer text.

    So hold back any trailing run that could still grow into a marker or a
    header, and let the next delta re-decide.

    HOLD ONLY WHAT YOU CAN AFFORD TO LOSE. There is no finish hook in the
    parser contract and the server's terminal chunk reuses only what was
    already emitted, so anything still held when generation stops is dropped
    from the user's view — it is NOT recovered by a later pass. Hold therefore
    has to be provably transient: a marker prefix always resolves on the next
    character, and a header region always ends at its ``<|message|>``.
    """
    hold = 0

    # 1. A trailing PROPER PREFIX of a MARKER: "<", "<|", "<|mess", ...
    #    Every marker starts with "<", which is vanishingly rare in prose, so
    #    holding these costs one delta and drops nothing in practice.
    #
    #    `to=` / `assistant` are deliberately NOT held here. They are only
    #    header openers INSIDE a header region (rule 2); everywhere else they
    #    are ordinary words. Holding them globally meant any reply ending in a
    #    letter run — "...Malta", "...a result", "...to" — kept its last
    #    characters back, and since nothing flushes the tail at stream end
    #    (the parser has no finish hook and the server's terminal chunk reuses
    #    only what was already emitted) those characters were LOST from the
    #    user's view. Every earlier test ended in punctuation or a marker,
    #    which is exactly the set that hides this.
    for opener in _ALL_MARKERS:
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
    #    The hold runs from the BOUNDARY MARKER ITSELF, not merely from the text
    #    after it. A terminator is only structure if a header follows it; if
    #    ordinary prose follows, _strip_answer_markers keeps it as text. That
    #    verdict is not available until the header either completes or fails, so
    #    releasing `<|eom|>` the moment one character lands behind it publishes a
    #    marker the monotonic counters can never retract — the
    #    `<|eom|>assistant to=user` leak seen live in chat.
    boundary = 0
    marker_start = 0
    for opener in (_EOM_TAG, _EOT_TAG, _START_TAG):
        found = text.rfind(opener)
        if found >= 0:
            after = found if opener is _START_TAG else found + len(opener)
            # Strictly greater, so that `<|eom|><|start|>` — where the terminator
            # and the start tag resolve to the SAME boundary — keeps the hold
            # anchored on the terminator. Ties broken the other way released the
            # `<|eom|>` sitting behind the start tag straight into chat.
            if after > boundary:
                boundary = after
                marker_start = found
    pending = text[boundary:]
    if _MESSAGE_TAG not in pending and _viable_header_prefix(pending):
        # This message's header is still arriving.
        hold = max(hold, len(text) - marker_start)

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
        if not text:
            return []
        # A generation cut off mid-header (max_tokens inside `to=self<|mess`)
        # has no body to show. Emitting the half-written header as prose is a
        # leak, and it made the non-streaming path disagree with streaming,
        # which correctly shows nothing.
        if _viable_header_prefix(text):
            return []
        return [(_USER_RECIPIENT, text)]

    lead = text[: matches[0].start()]
    # Leading text is a message body like any other, so a terminator ends it.
    # Here the terminator is unambiguously structure — a header follows it — so
    # cutting is safe, and not cutting rendered `<|eom|>` mid-chat.
    for terminator in (_EOM_TAG, _EOT_TAG):
        cut = lead.find(terminator)
        if cut >= 0:
            lead = lead[:cut]
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
            elif recipient == _USER_RECIPIENT:
                # The visible answer rail. Scrub any control markup that reached
                # the stable prefix before it could be classified so a recipient
                # header never renders in chat (streaming counters are monotonic
                # and cannot retract an already-emitted leak).
                content_parts.append(_strip_answer_markers(body))
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
