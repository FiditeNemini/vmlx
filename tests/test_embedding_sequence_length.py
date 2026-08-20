"""vmlx#255: the embeddings endpoint hardcoded max_length=512.

A Qwen3-Embedding 0.6B (and every other long-context embedding model)
silently indexed only the first 512 tokens of each input, so RAG chunks
larger than that returned vectors that ignored their tail — with no error,
no warning, and no way for the caller to notice.
"""

import logging

from vmlx_engine.embedding import EmbeddingEngine


class _Tok:
    def __init__(self, model_max_length=None):
        if model_max_length is not None:
            self.model_max_length = model_max_length

    def __call__(self, text, truncation=False, **kwargs):
        n = max(1, len(text.split()))
        return {"input_ids": list(range(n))}


class _Cfg:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _Model:
    def __init__(self, config=None):
        self.config = config


def _embedder(tokenizer, model):
    em = EmbeddingEngine.__new__(EmbeddingEngine)
    em.model_name = "test/embed"
    em._tokenizer = tokenizer
    em._model = model
    return em


class TestMaxSequenceLength:
    def test_uses_the_models_real_limit_not_512(self):
        em = _embedder(_Tok(32768), _Model(_Cfg(max_position_embeddings=32768)))
        assert em._max_sequence_length() == 32768

    def test_smallest_stated_limit_wins(self):
        em = _embedder(_Tok(32768), _Model(_Cfg(max_position_embeddings=8192)))
        assert em._max_sequence_length() == 8192

    def test_transformers_sentinel_is_ignored(self):
        # transformers uses ~1e30 for "no stated limit"
        em = _embedder(_Tok(1000000000000000019884624838656), _Model(_Cfg(max_position_embeddings=4096)))
        assert em._max_sequence_length() == 4096

    def test_dict_config_is_read(self):
        em = _embedder(_Tok(None), _Model({"max_position_embeddings": 2048}))
        assert em._max_sequence_length() == 2048

    def test_fallback_only_when_nothing_is_stated(self):
        em = _embedder(_Tok(None), _Model(None))
        assert em._max_sequence_length() == 512

    def test_result_is_cached(self):
        em = _embedder(_Tok(4096), _Model(None))
        assert em._max_sequence_length() == 4096
        em._tokenizer = _Tok(128)  # would resolve differently if recomputed
        assert em._max_sequence_length() == 4096


class TestTruncationIsLoud:
    def test_warns_when_input_exceeds_the_limit(self, caplog):
        em = _embedder(_Tok(512), _Model(None))
        with caplog.at_level(logging.WARNING):
            em._warn_on_truncation(em._tokenizer, ["word " * 900], 512)
        assert any("truncated" in r.getMessage() for r in caplog.records)

    def test_silent_when_everything_fits(self, caplog):
        em = _embedder(_Tok(512), _Model(None))
        with caplog.at_level(logging.WARNING):
            em._warn_on_truncation(em._tokenizer, ["short text"], 512)
        assert not [r for r in caplog.records if "truncated" in r.getMessage()]


def test_no_hardcoded_512_max_length_remains():
    import inspect

    from vmlx_engine import embedding

    src = inspect.getsource(embedding.EmbeddingEngine.embed)
    assert "max_length=512" not in src, (
        "embed() pins max_length again; it must use _max_sequence_length()"
    )


class TestTruncationCheckIsCheap:
    """RAG callers embed thousands of chunks; the all-fits case must not
    pay for a second tokenization pass, and one oversize chunk must not
    drag the whole batch into re-tokenization (padding makes every row the
    same width, so suspects come from the attention mask, not the ids)."""

    class _CountingTok:
        model_max_length = 8192

        def __init__(self):
            self.calls = 0

        def __call__(self, texts, truncation=False, **kw):
            import numpy as np

            self.calls += 1
            if isinstance(texts, str):
                return {"input_ids": list(range(max(1, len(texts.split()))))}
            maxlen = kw.get("max_length", 10**9)
            rows = [list(range(min(len(t.split()), maxlen))) for t in texts]
            w = max(len(r) for r in rows)
            return {
                "input_ids": np.array([r + [0] * (w - len(r)) for r in rows]),
                "attention_mask": np.array(
                    [[1] * len(r) + [0] * (w - len(r)) for r in rows], dtype="int32"
                ),
            }

    def _engine(self):
        import mlx.core as mx

        class _M:
            config = type("C", (), {"max_position_embeddings": 8192})()

            def __call__(self, ids, attention_mask=None):
                return type("O", (), {"text_embeds": mx.zeros((ids.shape[0], 4))})()

        e = EmbeddingEngine.__new__(EmbeddingEngine)
        e.model_name = "t"
        e._tokenizer = self._CountingTok()
        e._model = _M()
        return e

    def test_all_fits_costs_one_tokenizer_call(self):
        e = self._engine()
        e.embed(["short text"] * 200)
        assert e._tokenizer.calls == 1

    def test_one_oversize_chunk_only_retokenizes_that_chunk(self):
        e = self._engine()
        e.embed(["word " * 9000] + ["short one"] * 199)
        assert e._tokenizer.calls == 2
