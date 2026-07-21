from pathlib import Path
from types import SimpleNamespace

from vmlx_engine.loaders.load_laguna import _load_laguna_tokenizer


def test_laguna_tokenizer_preserves_shipped_regex_for_reference_parity(monkeypatch):
    """Do not rewrite only the vMLX side of Laguna tokenization."""
    calls = []

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append((path, kwargs))
            return "tokenizer"

    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=_AutoTokenizer),
    )

    tokenizer = _load_laguna_tokenizer(Path("/models/Laguna-S-2.1-JANG_2L"))

    assert tokenizer == "tokenizer"
    assert calls == [
        (
            "/models/Laguna-S-2.1-JANG_2L",
            {"trust_remote_code": True},
        )
    ]
