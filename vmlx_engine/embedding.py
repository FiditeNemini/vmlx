# SPDX-License-Identifier: Apache-2.0
"""
Embedding engine using mlx-embeddings.

Provides lazy-loaded model management and batch embedding generation
for the OpenAI-compatible /v1/embeddings endpoint.
"""

import logging
import time

import mlx.core as mx

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Wrapper around mlx-embeddings for text embedding generation.

    Supports lazy model loading and batch embedding with proper
    tokenization and pooling.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the embedding model and tokenizer."""
        from mlx_embeddings import load

        logger.info(f"Loading embedding model: {self.model_name}")
        start = time.perf_counter()
        self._model, self._tokenizer = load(self.model_name)
        elapsed = time.perf_counter() - start
        logger.info(f"Embedding model loaded in {elapsed:.2f}s: {self.model_name}")

    def _ensure_loaded(self) -> None:
        if not self.is_loaded:
            self.load()

    # Only used when neither the tokenizer nor the model config states a
    # usable limit. It is NOT a product decision about how much text an
    # embedding model can read (vmlx#255: this was hardcoded, so a
    # 32k-context embedding model silently indexed the first 512 tokens of
    # every chunk and returned vectors that ignored the rest).
    _FALLBACK_MAX_SEQUENCE_LENGTH = 512
    # transformers uses a huge sentinel for "no stated limit".
    _IMPLAUSIBLE_MAX_SEQUENCE_LENGTH = 1_000_000

    def _max_sequence_length(self) -> int:
        cached = getattr(self, "_cached_max_sequence_length", None)
        if cached:
            return cached

        candidates: list[int] = []

        def _consider(raw: object) -> None:
            try:
                value = int(raw)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return
            if 0 < value < self._IMPLAUSIBLE_MAX_SEQUENCE_LENGTH:
                candidates.append(value)

        tok = getattr(self._tokenizer, "_tokenizer", self._tokenizer)
        _consider(getattr(tok, "model_max_length", None))

        config = getattr(self._model, "config", None)
        for attr in (
            "max_position_embeddings",
            "max_seq_length",
            "n_positions",
            "seq_length",
        ):
            if config is not None:
                _consider(getattr(config, attr, None))
                if isinstance(config, dict):
                    _consider(config.get(attr))

        # The smallest stated limit is the one that actually holds: exceeding
        # either the tokenizer's or the model's ceiling is what breaks.
        resolved = min(candidates) if candidates else self._FALLBACK_MAX_SEQUENCE_LENGTH
        self._cached_max_sequence_length = resolved
        logger.info(
            "Embedding model %s max sequence length resolved to %d tokens%s",
            self.model_name,
            resolved,
            "" if candidates else " (no stated limit found; using fallback)",
        )
        return resolved

    # Padded tokens per forward. A group costs roughly
    # len(group) x longest_in_group, and that product is what the model
    # materialises. Override with VMLX_EMBED_MAX_PADDED_TOKENS.
    _DEFAULT_MAX_PADDED_BATCH_TOKENS = 262144

    def _max_padded_batch_tokens(self) -> int:
        import os

        try:
            value = int(os.environ.get("VMLX_EMBED_MAX_PADDED_TOKENS", "0"))
        except ValueError:
            value = 0
        return value if value > 0 else self._DEFAULT_MAX_PADDED_BATCH_TOKENS

    def _plan_batches(self, texts: list[str], max_length: int) -> list[list[int]]:
        """Group input indices so each forward stays within the token budget.

        Lengths are estimated from characters rather than by tokenizing
        twice: the estimate only needs to bound the group, and
        over-estimating is safe (it yields smaller groups). Order is
        preserved so the caller's vectors line up with their inputs.
        """
        budget = self._max_padded_batch_tokens()
        groups: list[list[int]] = []
        current: list[int] = []
        current_max = 0
        for index, text in enumerate(texts):
            # ~3 chars/token is deliberately conservative for most tokenizers.
            estimate = min(max(1, len(text) // 3), max_length)
            candidate_max = max(current_max, estimate)
            if current and candidate_max * (len(current) + 1) > budget:
                groups.append(current)
                current = [index]
                current_max = estimate
            else:
                current.append(index)
                current_max = candidate_max
        if current:
            groups.append(current)
        return groups or [[]]

    def _warn_on_truncation(
        self,
        tokenizer: object,
        texts: list[str],
        max_length: int,
        attention_mask: object = None,
    ) -> None:
        """Truncation must never be silent — the caller's vector would
        simply ignore the tail of their document (vmlx#255).

        Rows that did not fill the window cannot have been truncated, so
        only those are re-tokenized. The row's REAL length comes from the
        attention mask, not from the padded input_ids width — with
        padding=True every row shares the widest row's width, so using the
        ids would make one oversize chunk re-tokenize the whole batch.
        Embedding callers index thousands of chunks; the all-fits case must
        add no tokenization work at all.
        """
        try:
            suspects = list(range(len(texts)))
            if attention_mask is not None:
                suspects = [
                    i
                    for i, row in enumerate(attention_mask)
                    if int(sum(row)) >= max_length
                ]
            if not suspects:
                return
            over = []
            for i in suspects:
                length = len(tokenizer(texts[i], truncation=False)["input_ids"])  # type: ignore[operator]
                if length > max_length:
                    over.append(length)
        except Exception:
            return
        if not over:
            return
        logger.warning(
            "Embedding input truncated: %d of %d input(s) exceed the model's "
            "%d-token limit (longest %d). Content past the limit does not "
            "affect the returned vector — split the text into chunks that fit.",
            len(over),
            len(texts),
            max_length,
            max(over),
        )

    def embed(self, texts: str | list[str]) -> list[list[float]]:
        """
        Generate embeddings for one or more texts.

        Args:
            texts: A single string or list of strings.

        Returns:
            List of embedding vectors (one per input text).
        """
        self._ensure_loaded()

        if isinstance(texts, str):
            texts = [texts]

        # Tokenize directly instead of using mlx_embeddings.generate(),
        # which has compatibility issues with newer tokenizers (e.g.
        # GemmaTokenizer lacks batch_encode_plus, and the model's __call__
        # expects positional `inputs` not `input_ids` as a kwarg).
        inner_tok = getattr(self._tokenizer, "_tokenizer", self._tokenizer)
        max_length = self._max_sequence_length()

        # One padded forward over the whole request is only safe while inputs
        # are short. Once the real model limit is honoured (vmlx#255 lifted a
        # hardcoded 512), a batch of long chunks becomes
        # count x max_len x hidden — measured: 100 chunks of 40k tokens asked
        # Metal for 200GB and failed the request. Split into sub-batches with
        # a bounded padded-token budget instead: same vectors, same order,
        # bounded memory, and no refusal.
        results: list[list[float]] = []
        for group in self._plan_batches(texts, max_length):
            batch = [texts[i] for i in group]
            encoded = inner_tok(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="np",
            )
            # Only pay for a second tokenization when a row actually filled
            # the window — RAG callers embed thousands of chunks, and the
            # common case (everything fits) must cost nothing extra.
            self._warn_on_truncation(
                inner_tok, batch, max_length, encoded.get("attention_mask")
            )
            input_ids = mx.array(encoded["input_ids"])
            attention_mask = mx.array(encoded["attention_mask"])
            output = self._model(input_ids, attention_mask=attention_mask)
            embeds: mx.array = output.text_embeds
            results.extend(embeds.tolist())
            del output, embeds, input_ids, attention_mask
            mx.clear_cache()
        return results

    def count_tokens(self, texts: str | list[str]) -> int:
        """Approximate token count for usage reporting."""
        self._ensure_loaded()

        if isinstance(texts, str):
            texts = [texts]

        # Use the same inner tokenizer as embed() for consistency
        inner_tok = getattr(self._tokenizer, "_tokenizer", self._tokenizer)
        total = 0
        for text in texts:
            try:
                tokens = inner_tok.encode(text)
                if isinstance(tokens, list):
                    total += len(tokens)
                elif hasattr(tokens, "__len__"):
                    total += len(tokens)
                else:
                    total += tokens.size
            except Exception:
                # Fallback: rough estimate of ~4 chars per token
                total += max(1, len(text) // 4)
        return total
