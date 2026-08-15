# SPDX-License-Identifier: Apache-2.0
"""Declared-context bookkeeping shared by both schedulers.

The field-failure class this prevents (2026-08-15 directive, observed live
in another inference stack): prompt 5,875 + generated 26,893 tokens hit the
model's context ceiling exactly, the run was reported as a truncation with
no explanation, and the user diagnosed a GPU failure. Max OUTPUT and max
CONTEXT are separate budgets; when a request's output budget would push
generation past the model's declared positional ceiling, the only sound
behaviors are a clamp WITH a logged notice (default) or the caller fixing
the request. Generating past the declared ceiling is silent RoPE
out-of-distribution garbage.

The server stamps the loaded bundle's declared ceiling after every load;
both the text scheduler and the MLLM scheduler clamp through the ONE
function below at admission, where the true tokenized prompt length is
known (route-level character estimates are forbidden — see the DSV4
false-rejection note in server.py).

``VMLX_CONTEXT_OUTPUT_CLAMP=0`` disables the clamp (diagnostics only —
accepts past-ceiling generation quality risk). Families that legitimately
extend context (rope-scaled serving) declare the extended ceiling in their
bundle config, which is what the server reads.
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_declared_context_tokens = 0


def set_declared_context_tokens(value: int | None) -> None:
    """Record the loaded bundle's declared context ceiling (0 = unknown)."""
    global _declared_context_tokens
    try:
        normalized = max(0, int(value or 0))
    except (TypeError, ValueError):
        normalized = 0
    with _lock:
        _declared_context_tokens = normalized


def get_declared_context_tokens() -> int:
    with _lock:
        return _declared_context_tokens


def _clamp_enabled() -> bool:
    raw = os.environ.get(
        "VMLX_CONTEXT_OUTPUT_CLAMP",
        os.environ.get("VMLINUX_CONTEXT_OUTPUT_CLAMP", "1"),
    )
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def clamp_output_to_declared_context(
    prompt_tokens: int,
    requested_max_tokens: int | None,
    *,
    request_id: str = "?",
) -> int | None:
    """Bound an output budget by (declared context − prompt length).

    Returns the possibly-clamped max_tokens. A binding clamp is LOGGED —
    the whole point is that context exhaustion must never be silent. When
    the declared ceiling is unknown (0) or the clamp is disabled, the
    requested value passes through unchanged. A prompt already at/over the
    ceiling is left to the prompt-limit guard (this function never returns
    a smaller-than-1 budget on its own).
    """
    if requested_max_tokens is None:
        return None
    declared = get_declared_context_tokens()
    if declared <= 0 or not _clamp_enabled():
        return requested_max_tokens
    try:
        prompt_count = int(prompt_tokens or 0)
        requested = int(requested_max_tokens)
    except (TypeError, ValueError):
        return requested_max_tokens
    remaining = declared - prompt_count
    if remaining <= 0:
        return requested_max_tokens
    if requested <= remaining:
        return requested
    logger.warning(
        "Request %s: prompt (%d tokens) + max_tokens (%d) exceeds the "
        "model's declared context of %d tokens; output budget clamped to "
        "%d. finish_reason=length at that point means CONTEXT EXHAUSTION, "
        "not an output-limit stop. Shorten the prompt/history or serve a "
        "larger-context bundle. VMLX_CONTEXT_OUTPUT_CLAMP=0 disables this "
        "clamp (accepts past-ceiling generation quality risk).",
        request_id,
        prompt_count,
        requested,
        declared,
        remaining,
    )
    return remaining
