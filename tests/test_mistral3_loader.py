from __future__ import annotations

import json
from types import SimpleNamespace


def test_jangtq2_uses_strict_runtime_without_unstable_override(
    tmp_path, monkeypatch
):
    """Official Mistral3 JANGTQ2 bundles must reach the strict runtime loader.

    The historical fail-closed gate predated the current dense prefill flattening
    and all-required-module hydration checks. Runtime correctness now belongs to
    those executable contracts, not an artifact-name/bit-width rejection.
    """
    config = {
        "model_type": "mistral3",
        "weight_format": "mxtq",
        "mxtq_bits": 2,
        "text_config": {"model_type": "ministral3"},
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    import jang_tools.mistral3.runtime as runtime
    from transformers import AutoTokenizer
    from vmlx_engine.loaders.load_mistral3 import load_mistral3_model

    fake_model = SimpleNamespace(config=None)
    fake_config = SimpleNamespace(
        text_config=SimpleNamespace(num_hidden_layers=88, vocab_size=131072),
        vision_config=SimpleNamespace(num_hidden_layers=48),
    )
    calls: list[str] = []

    def fake_load(path: str):
        calls.append(path)
        return fake_model, fake_config, "jangtq"

    fake_tokenizer = object()
    monkeypatch.delenv("VMLX_ALLOW_UNSTABLE_MISTRAL35_JANGTQ", raising=False)
    monkeypatch.setattr(runtime, "load", fake_load)
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: fake_tokenizer,
    )

    model, tokenizer = load_mistral3_model(tmp_path)

    assert calls == [str(tmp_path)]
    assert model is fake_model
    assert model.config == config
    assert tokenizer is fake_tokenizer
