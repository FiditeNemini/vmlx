from pathlib import Path
from types import SimpleNamespace

import pytest

from vmlx_engine.loaders.load_laguna import (
    _load_laguna_tokenizer,
    _require_mixed_affine_runtime,
    _uses_mixed_affine_modules,
)


def _mixed_affine_config():
    return {
        "model_type": "laguna",
        "quantization": {
            "bits": 8,
            "group_size": 64,
            "model.embed_tokens": {
                "bits": 6,
                "group_size": 64,
                "mode": "affine",
            },
            "model.layers.1.mlp.switch_mlp.gate_proj": {
                "bits": 4,
                "group_size": 64,
                "mode": "affine",
            },
        },
    }


def test_laguna_mixed_affine_runtime_contract_rejects_stale_wheel():
    cfg = _mixed_affine_config()
    assert _uses_mixed_affine_modules(cfg) is True

    with pytest.raises(RuntimeError, match=r"jang>=2\.5\.34") as exc:
        _require_mixed_affine_runtime(
            Path("/models/Laguna-S-2.1-JANG_4M"),
            cfg,
            SimpleNamespace(__file__="/stale/jang_tools/laguna/runtime.py"),
        )

    assert "/stale/jang_tools/laguna/runtime.py" in str(exc.value)
    assert "Laguna-S-2.1-JANG_4M" in str(exc.value)


def test_laguna_mixed_affine_runtime_contract_accepts_capable_wheel():
    cfg = _mixed_affine_config()
    _require_mixed_affine_runtime(
        Path("/models/Laguna-S-2.1-JANG_4M"),
        cfg,
        SimpleNamespace(LAGUNA_MIXED_AFFINE_RUNTIME_VERSION=1),
    )


def test_laguna_non_affine_formats_do_not_require_affine_marker():
    for weight_format in ("mxtq", "mxfp4"):
        cfg = _mixed_affine_config() | {"weight_format": weight_format}
        assert _uses_mixed_affine_modules(cfg) is False
        _require_mixed_affine_runtime(
            Path("/models/Laguna"), cfg, SimpleNamespace()
        )


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
