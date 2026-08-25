# SPDX-License-Identifier: Apache-2.0
"""Token-level sampling controls shared by text and multimodal generators."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import mlx.core as mx
import numpy as np


def normalize_logit_bias(
    value: Mapping[str | int, float | int] | None,
) -> dict[int, float]:
    """Validate and canonicalize an OpenAI ``logit_bias`` mapping."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("logit_bias must be an object mapping token IDs to biases")

    normalized: dict[int, float] = {}
    for raw_token, raw_bias in value.items():
        try:
            token_id = int(raw_token)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"logit_bias token ID {raw_token!r} is not an integer"
            ) from exc
        if str(raw_token).strip() != str(token_id):
            raise ValueError(
                f"logit_bias token ID {raw_token!r} is not a canonical integer"
            )
        if token_id < 0:
            raise ValueError("logit_bias token IDs must be non-negative")
        try:
            bias = float(raw_bias)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"logit_bias for token {token_id} must be numeric"
            ) from exc
        if not np.isfinite(bias) or bias < -100.0 or bias > 100.0:
            raise ValueError(
                f"logit_bias for token {token_id} must be between -100 and 100"
            )
        normalized[token_id] = bias
    return normalized


def _token_list(tokens: Any) -> list[int]:
    if tokens is None:
        return []
    if isinstance(tokens, list):
        return [int(token) for token in tokens]
    if isinstance(tokens, tuple):
        return [int(token) for token in tokens]
    return [int(token) for token in np.asarray(tokens).reshape(-1)]


def make_openai_token_penalty_processor(
    *,
    logit_bias: Mapping[str | int, float | int] | None = None,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> Callable[[Sequence[int], mx.array], mx.array] | None:
    """Build the OpenAI token-bias and count-penalty transform.

    ``frequency_penalty`` subtracts ``penalty * count`` and
    ``presence_penalty`` subtracts the penalty once for every token already
    present. The sparse adjustment is retained and updated incrementally so a
    long context is not rescanned or expanded to one-hot form every token.
    """
    normalized_bias = {
        token_id: bias
        for token_id, bias in normalize_logit_bias(logit_bias).items()
        if bias != 0.0
    }
    frequency = float(frequency_penalty or 0.0)
    presence = float(presence_penalty or 0.0)
    if not normalized_bias and frequency == 0.0 and presence == 0.0:
        return None
    if not np.isfinite(frequency) or frequency < -2.0 or frequency > 2.0:
        raise ValueError("frequency_penalty must be between -2 and 2")
    if not np.isfinite(presence) or presence < -2.0 or presence > 2.0:
        raise ValueError("presence_penalty must be between -2 and 2")

    state: dict[str, Any] = {
        "width": None,
        "dtype": None,
        "length": 0,
        "last_token": None,
        "counts": Counter(),
        "adjustment": None,
    }

    def _rebuild(token_ids: list[int], logits: mx.array) -> None:
        width = int(logits.shape[-1])
        invalid = [token for token in normalized_bias if token >= width]
        if invalid:
            raise ValueError(
                f"logit_bias token ID {min(invalid)} is outside vocabulary size {width}"
            )
        invalid_context = [token for token in token_ids if token < 0 or token >= width]
        if invalid_context:
            raise ValueError(
                f"token context ID {invalid_context[0]} is outside vocabulary size {width}"
            )

        counts = Counter(token_ids)
        deltas = dict(normalized_bias)
        if frequency != 0.0 or presence != 0.0:
            for token, count in counts.items():
                deltas[token] = deltas.get(token, 0.0) - (
                    frequency * count + presence
                )
        deltas = {token: delta for token, delta in deltas.items() if delta != 0.0}
        adjustment = mx.zeros((width,), dtype=logits.dtype)
        if deltas:
            ordered = sorted(deltas.items())
            indices = mx.array([token for token, _ in ordered], dtype=mx.int32)
            values = mx.array(
                [delta for _, delta in ordered], dtype=logits.dtype
            )
            adjustment = adjustment.at[indices].add(values)

        state.update(
            width=width,
            dtype=logits.dtype,
            length=len(token_ids),
            last_token=token_ids[-1] if token_ids else None,
            counts=counts,
            adjustment=adjustment,
        )

    def _processor(tokens: Sequence[int], logits: mx.array) -> mx.array:
        token_ids = _token_list(tokens) if (frequency or presence) else []
        needs_rebuild = (
            state["adjustment"] is None
            or state["width"] != int(logits.shape[-1])
            or state["dtype"] != logits.dtype
        )
        if frequency or presence:
            prior_length = int(state["length"] or 0)
            if len(token_ids) < prior_length or (
                prior_length and token_ids[prior_length - 1] != state["last_token"]
            ):
                needs_rebuild = True

        if needs_rebuild:
            _rebuild(token_ids, logits)
        elif (frequency or presence) and len(token_ids) > state["length"]:
            width = int(state["width"])
            appended = token_ids[int(state["length"]):]
            invalid = [token for token in appended if token < 0 or token >= width]
            if invalid:
                raise ValueError(
                    f"token context ID {invalid[0]} is outside vocabulary size {width}"
                )
            updates: dict[int, float] = {}
            counts = state["counts"]
            for token in appended:
                delta = -frequency
                if counts[token] == 0:
                    delta -= presence
                counts[token] += 1
                updates[token] = updates.get(token, 0.0) + delta
            updates = {token: delta for token, delta in updates.items() if delta != 0.0}
            if updates:
                ordered = sorted(updates.items())
                indices = mx.array([token for token, _ in ordered], dtype=mx.int32)
                values = mx.array(
                    [delta for _, delta in ordered], dtype=logits.dtype
                )
                state["adjustment"] = state["adjustment"].at[indices].add(values)
            state["length"] = len(token_ids)
            state["last_token"] = token_ids[-1] if token_ids else None

        return logits + state["adjustment"]

    _processor._vmlx_openai_token_penalties = True
    return _processor
