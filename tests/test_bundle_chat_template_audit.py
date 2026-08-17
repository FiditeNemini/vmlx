import json
from pathlib import Path

import pytest

from vmlx_engine.utils.jang_loader import (
    audit_bundle_chat_template_sources,
    log_bundle_chat_template_audit,
)

TPL = "{% for m in messages %}{{ m.content }}{% endfor %}"


def _bundle(tmp_path, *, jinja=None, embedded=None):
    if jinja is not None:
        (tmp_path / "chat_template.jinja").write_text(jinja)
    cfg = {}
    if embedded is not None:
        cfg["chat_template"] = embedded
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(cfg))
    return tmp_path


def test_divergent_embedded_template_is_reported(tmp_path):
    """The Zaya-8B shape: the .jinja carries the fix, the embedded copy is stale."""
    b = _bundle(tmp_path, jinja=TPL + "{# fixed #}", embedded=TPL)
    audit = audit_bundle_chat_template_sources(b)
    assert audit["status"] == "divergent"
    assert audit["jinja"] != audit["embedded"]


def test_jinja_only_is_reported(tmp_path):
    """The LFM2.5 shape: tokenizer_config.json carries no template at all."""
    audit = audit_bundle_chat_template_sources(_bundle(tmp_path, jinja=TPL))
    assert audit["status"] == "jinja_only"
    assert audit["embedded"] is None


def test_identical_copies_are_consistent(tmp_path):
    audit = audit_bundle_chat_template_sources(_bundle(tmp_path, jinja=TPL, embedded=TPL))
    assert audit["status"] == "consistent"


def test_whitespace_only_difference_is_not_divergence(tmp_path):
    b = _bundle(tmp_path, jinja=TPL + "\n\n", embedded="  " + TPL)
    assert audit_bundle_chat_template_sources(b)["status"] == "consistent"


def test_no_template_anywhere(tmp_path):
    assert audit_bundle_chat_template_sources(_bundle(tmp_path))["status"] == "no_template"


def test_audit_never_raises_on_a_broken_bundle(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text("{not json")
    assert audit_bundle_chat_template_sources(tmp_path)["status"] == "unreadable"
    assert audit_bundle_chat_template_sources(tmp_path / "nope")["status"] in {
        "no_template",
        "unreadable",
    }


def test_divergence_logs_a_warning(tmp_path, caplog):
    b = _bundle(tmp_path, jinja=TPL + "{# fixed #}", embedded=TPL)
    with caplog.at_level("WARNING"):
        log_bundle_chat_template_audit(b)
    assert any("DIVERGENT" in r.message for r in caplog.records)


def test_consistent_bundle_logs_no_warning(tmp_path, caplog):
    b = _bundle(tmp_path, jinja=TPL, embedded=TPL)
    with caplog.at_level("WARNING"):
        log_bundle_chat_template_audit(b)
    assert not [r for r in caplog.records if r.levelname == "WARNING"]
