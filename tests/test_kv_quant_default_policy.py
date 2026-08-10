# SPDX-License-Identifier: Apache-2.0
"""Auto KV-quantization default policy.

JANGTQ / JANG-affine bundles carry aggressively quantized weights; q4 stored-KV
compounds that error on prefix-cache reuse (live-proven on MiniMax-M2.7 JANGTQ
2026-08-10: q4 deep reuse diverged from its own cold run, q8 stayed
self-consistent and answer-stable, fp16 was byte-exact). These bundles default
to q8; everything else keeps q4. An explicit env override always wins.
"""

import json

from vmlx_engine.cli import _bundle_declares_mxtq_jangtq, _bundle_is_jang_affine


def _write_bundle(tmp_path, config=None, jang_config=None):
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    if config is not None:
        (root / "config.json").write_text(json.dumps(config))
    if jang_config is not None:
        (root / "jang_config.json").write_text(json.dumps(jang_config))
    return str(root)


def test_mxtq_bits_bundle_detected_as_jangtq(tmp_path):
    path = _write_bundle(
        tmp_path,
        config={"model_type": "minimax_m2"},
        jang_config={"mxtq_bits": {"0": 2}, "mxtq_seed": 42},
    )
    assert _bundle_declares_mxtq_jangtq(path) is True


def test_affine_format_bundle_detected(tmp_path):
    for fmt in ("affine", "jang", "jjqf", "mxq"):
        path = _write_bundle(
            tmp_path / fmt,
            config={"model_type": "qwen3"},
            jang_config={"format": fmt},
        )
        assert _bundle_is_jang_affine(path) is True, fmt


def test_plain_bundle_not_detected(tmp_path):
    path = _write_bundle(
        tmp_path,
        config={"model_type": "qwen3"},
    )
    assert _bundle_declares_mxtq_jangtq(path) is False
    assert _bundle_is_jang_affine(path) is False


def test_non_affine_jang_config_not_affine(tmp_path):
    path = _write_bundle(
        tmp_path,
        config={"model_type": "qwen3"},
        jang_config={"format": "gguf"},
    )
    assert _bundle_is_jang_affine(path) is False


def test_auto_kv_default_q8_for_jangtq_q4_otherwise(tmp_path, monkeypatch):
    """The auto default resolves q8 for JANGTQ/affine bundles, q4 for others,
    and VMLX_DEFAULT_KV_CACHE_QUANTIZATION overrides both."""
    jang_path = _write_bundle(
        tmp_path / "jang",
        config={"model_type": "minimax_m2"},
        jang_config={"mxtq_bits": {"0": 2}},
    )
    plain_path = _write_bundle(tmp_path / "plain", config={"model_type": "qwen3"})

    def resolve_default(model_path):
        # Mirrors the cli auto-KV block's resolution order.
        import os

        hi_fi = _bundle_declares_mxtq_jangtq(model_path) or _bundle_is_jang_affine(
            model_path
        )
        fallback = "q8" if hi_fi else "q4"
        value = os.environ.get("VMLX_DEFAULT_KV_CACHE_QUANTIZATION", fallback)
        if value not in ("none", "q4", "q8"):
            value = fallback
        return value

    monkeypatch.delenv("VMLX_DEFAULT_KV_CACHE_QUANTIZATION", raising=False)
    assert resolve_default(jang_path) == "q8"
    assert resolve_default(plain_path) == "q4"

    monkeypatch.setenv("VMLX_DEFAULT_KV_CACHE_QUANTIZATION", "none")
    assert resolve_default(jang_path) == "none"
    assert resolve_default(plain_path) == "none"

    monkeypatch.setenv("VMLX_DEFAULT_KV_CACHE_QUANTIZATION", "bogus")
    assert resolve_default(jang_path) == "q8"
    assert resolve_default(plain_path) == "q4"
