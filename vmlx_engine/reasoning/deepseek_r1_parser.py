# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for DeepSeek-R1 models.

DeepSeek-R1 uses <think>...</think> tags for reasoning content.
The model may sometimes start outputting reasoning without the explicit
<think> tag, so this parser is more lenient than Qwen3.
"""

from .base import DeltaMessage
from .think_parser import BaseThinkingReasoningParser


class DeepSeekR1ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for DeepSeek-R1 model.

    DeepSeek-R1 uses <think>...</think> tokens to denote reasoning text.
    This parser is more lenient than Qwen3:
    - The <think> tag may not be explicitly generated (model assumes it)
    - If only </think> is found, everything before it is reasoning

    Example:
        Input: "<think>Step 1: analyze...\nStep 2: solve...</think>The answer is 42."
        Output: reasoning="Step 1: analyze...\nStep 2: solve...", content="The answer is 42."

        Input: "reasoning content</think>final answer"  # No opening tag
        Output: reasoning="reasoning content", content="final answer"
    """

    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        # Base parser defaults _think_in_prompt to False, but DeepSeek-R1 has
        # a historical lenient complete-output contract: a bare </think> with
        # text before it is implicit reasoning unless the request explicitly
        # configured the direct rail via reset_state(think_in_prompt=False).
        self._think_in_prompt_explicit = False

    def reset_state(self, think_in_prompt: bool = False, **kwargs):
        super().reset_state(think_in_prompt=think_in_prompt, **kwargs)
        self._think_in_prompt_explicit = True

    def _explicit_direct_rail(self) -> bool:
        return self._think_in_prompt_explicit and not self._think_in_prompt

    def _strip_owned_reasoning_closes(self, text: str) -> str:
        """Remove only DeepSeek-parser close aliases from visible text."""
        cleaned = text
        for _, end_marker in self.reasoning_marker_pairs:
            cleaned = cleaned.replace(end_marker, "")
        return cleaned

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning from DeepSeek-R1 output.

        More lenient than Qwen3 - handles cases where start tag is implicit.

        Args:
            model_output: Complete model output text.

        Returns:
            (reasoning, content) tuple.
        """
        model_output = self._normalize_reasoning_markers(model_output)

        # DeepSeek-R1 may omit the opening tag in generated text. Preserve the
        # lenient parser contract for standalone/default use, while letting the
        # server explicitly mark direct-rail requests where a stray close marker
        # must not hide visible content as reasoning.
        if self.end_token in model_output and self.start_token not in model_output:
            reasoning, _, content = model_output.partition(self.end_token)
            if self._explicit_direct_rail():
                return (
                    None,
                    self._strip_owned_reasoning_closes(model_output).strip()
                    or None,
                )
            content = self._strip_owned_reasoning_closes(content)
            reasoning = reasoning.strip() or None
            content = content.strip() or None
            if reasoning is None:
                return None, content
            return reasoning, content

        # If neither token present:
        #   - think_in_prompt=True: <think> was in template, so everything the
        #     model emits before </think> is reasoning. Without </think> the whole
        #     output is unclosed reasoning — return it as reasoning, empty content.
        #   - otherwise: plain content (model chose not to think).
        if self.end_token not in model_output and self.start_token not in model_output:
            if self._think_in_prompt:
                return (model_output.strip() or None), None
            return None, model_output

        # Use base class for standard/multi-rail parsing, then enforce the
        # DeepSeek-owned invariant that redundant closing aliases are never
        # visible content.
        reasoning, content = super().extract_reasoning(model_output)
        if content is not None:
            content = self._strip_owned_reasoning_closes(content) or None
        return reasoning, content

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """
        Extract reasoning from streaming delta.

        Handles DeepSeek-R1's pattern where <think> may be implicit.

        Args:
            previous_text: Text accumulated before this delta.
            current_text: Text including this delta.
            delta_text: Just the new text.

        Returns:
            DeltaMessage with reasoning/content, or None to skip.
        """
        previous_text, current_text, delta_text = self._safe_normalized_marker_view(
            previous_text,
            current_text,
        )
        if not delta_text:
            return None

        start_in_delta = self.start_token in delta_text
        past_reasoning_boundary = (
            self.end_token in previous_text
            and previous_text.rfind(self.end_token)
            > previous_text.rfind(self.start_token)
        )
        if past_reasoning_boundary and not start_in_delta:
            # On an explicit direct rail, a stray close after already-visible
            # prose is redundant markup, not a new implicit reasoning phase.
            # Preserve the following delta's leading whitespace exactly; the
            # Base implicit path intentionally lstrips only a structural
            # separator after a genuine private rail.
            if self._explicit_direct_rail():
                content_part = self._strip_owned_reasoning_closes(delta_text)
                return DeltaMessage(content=content_part or None)
            if self.end_token in delta_text:
                content_part = self._strip_owned_reasoning_closes(delta_text)
                _, _, previous_content = previous_text.partition(self.end_token)
                if not previous_content.strip():
                    content_part = content_part.lstrip()
                return DeltaMessage(content=content_part or None)

        # First try base class logic
        result = super().extract_reasoning_streaming(
            previous_text, current_text, delta_text
        )

        # Handle DeepSeek-R1 special case: no start token seen but end token appears
        if result is not None:
            start_in_prev = self.start_token in previous_text
            start_in_delta = self.start_token in delta_text
            end_in_delta = self.end_token in delta_text

            # If end token in delta but we never saw start token. Route the
            # prefix to reasoning in DeepSeek-R1's lenient/default mode, but
            # keep it visible when the request explicitly configured the direct
            # rail.
            if (
                not start_in_prev
                and not start_in_delta
                and end_in_delta
                and self.end_token not in previous_text
            ):
                idx = delta_text.find(self.end_token)
                if self._explicit_direct_rail():
                    content_part = self._strip_owned_reasoning_closes(
                        delta_text[:idx]
                        + delta_text[idx + len(self.end_token) :]
                    )
                    return DeltaMessage(
                        content=content_part if content_part else None,
                    )
                reasoning_part = delta_text[:idx]
                # Match complete extraction and the shared think-tag stream:
                # whitespace directly after the close is a structural rail
                # separator, not visible assistant content.
                content_part = self._strip_owned_reasoning_closes(
                    delta_text[idx + len(self.end_token) :]
                ).lstrip()
                if not reasoning_part:
                    return DeltaMessage(
                        content=content_part if content_part else None,
                    )
                return DeltaMessage(
                    reasoning=reasoning_part if reasoning_part else None,
                    content=content_part if content_part else None,
                )

            # Note: DeepSeek-R1 may omit <think> but still be in reasoning mode.
            # However, we can't reliably detect implicit reasoning without context,
            # so we default to treating unmarked content as regular content.

        if result is not None and result.content is not None:
            result.content = self._strip_owned_reasoning_closes(result.content) or None
        return result
