# SPDX-License-Identifier: Apache-2.0
"""Opt-in target-distribution controls for DFlash2 verification.

The drafter keeps its own proposal distribution. Rejection sampling must use
the requested TARGET distribution for every verified row, including the bonus
row. No object is constructed on the existing default sampling path.
"""
import math
import mlx.core as mx

from .utils.token_logits_processors import make_openai_token_penalty_processor


class DFlash2SamplingControls:
    def __init__(self, *, min_p=0.0, logit_bias=None, repetition_penalty=1.0,
                 frequency_penalty=0.0, presence_penalty=0.0):
        self.min_p = float(min_p)
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError("min_p must be between 0 and 1")
        if not math.isfinite(repetition_penalty) or repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be finite and positive")
        self.repetition = None
        if repetition_penalty != 1.0:
            from mlx_lm.sample_utils import make_repetition_penalty
            self.repetition = make_repetition_penalty(repetition_penalty)
        self.penalty = make_openai_token_penalty_processor(
            logit_bias=logit_bias, frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        self.history_dependent = bool(self.repetition or frequency_penalty or presence_penalty)

    def process(self, logits, history, draft_tokens=None):
        if not self.history_dependent:
            return self.penalty([], logits) if self.penalty is not None else logits
        proposed = draft_tokens[0].tolist() if draft_tokens is not None else []
        rows = []
        for i in range(logits.shape[1]):
            context = history + proposed[:i]
            row = logits[:, i, :]
            if self.repetition is not None:
                row = self.repetition(mx.array(context[-20:]), row)
            if self.penalty is not None:
                row = self.penalty(context, row)
            rows.append(row)
        return mx.stack(rows, axis=1)

    def probabilities(self, runtime, logits, temperature, top_p, top_k):
        probabilities = runtime._sampling_probs(logits, temperature, top_p, top_k)
        if self.min_p > 0:
            threshold = self.min_p * mx.max(probabilities, axis=-1, keepdims=True)
            probabilities = mx.where(probabilities >= threshold, probabilities, 0)
            probabilities = probabilities / mx.sum(probabilities, axis=-1, keepdims=True)
        return probabilities
