import json


class _DummyTokenizer:
    def __init__(self, template):
        self.chat_template = template
        self.has_chat_template = True


def test_unresolved_chat_template_include_uses_sidecar_jinja(tmp_path):
    from vmlx_engine.utils.tokenizer import (
        _chat_template_is_unresolved_include,
        _inject_chat_template_if_missing,
    )

    include = "{% include 'chat_template.jinja' %}"
    sidecar = "<user>{{ messages[0].content }}</user><assistant><think>"
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": include}),
        encoding="utf-8",
    )
    (tmp_path / "chat_template.jinja").write_text(sidecar, encoding="utf-8")
    tokenizer = _DummyTokenizer(include)

    assert _chat_template_is_unresolved_include(tokenizer.chat_template)
    assert _inject_chat_template_if_missing(tokenizer, tmp_path) == "chat_template.jinja"
    assert tokenizer.chat_template == sidecar


def test_resolved_tokenizer_config_template_is_preserved(tmp_path):
    from vmlx_engine.utils.tokenizer import _inject_chat_template_if_missing

    baked = "<user>{{ messages[0].content }}</user><assistant>"
    sidecar = "<user>sidecar</user><assistant><think>"
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": baked}),
        encoding="utf-8",
    )
    (tmp_path / "chat_template.jinja").write_text(sidecar, encoding="utf-8")
    tokenizer = _DummyTokenizer(baked)

    assert _inject_chat_template_if_missing(tokenizer, tmp_path) is None
    assert tokenizer.chat_template == baked
