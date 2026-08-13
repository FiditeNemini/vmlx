"""Mixed-SWA bundles must store EXACT KV, or cache hits change the answer.

Live-proven on Laguna-S-2.1-JANG_4M-CRACK 2026-08-12 at temp 0, same prompt
cold then on a confirmed cache hit:

    stored q8 (old default)       cold sha bb040715 -> hit sha 633c133d
    exact stored KV (this fix)    cold sha bb040715 -> hit sha bb040715

Only the quantized read-back diverged; the cold answer was identical in both
arms.
"""

import json

import pytest

from vmlx_engine.cli import _bundle_has_mixed_swa_layout
from vmlx_engine.utils.jang_loader import has_mixed_attention_layout


class TestMixedAttentionDetector:
    def test_detects_interleaved_sliding_and_full(self):
        assert has_mixed_attention_layout(
            {"layer_types": ["sliding_attention", "full_attention"]}
        )

    def test_uniform_full_attention_is_not_mixed(self):
        assert not has_mixed_attention_layout(
            {"layer_types": ["full_attention", "full_attention"]}
        )

    def test_uniform_sliding_is_not_mixed(self):
        # One kind only -- there is no full-attention slot to protect.
        assert not has_mixed_attention_layout(
            {"layer_types": ["sliding_attention", "sliding_attention"]}
        )

    def test_reads_nested_text_config(self):
        assert has_mixed_attention_layout(
            {"text_config": {"layer_types": ["sliding_attention", "full_attention"]}}
        )

    def test_missing_or_malformed_layer_types(self):
        assert not has_mixed_attention_layout({})
        assert not has_mixed_attention_layout({"layer_types": "sliding"})


class TestBundleMixedSwaLayout:
    def _bundle(self, tmp_path, cfg):
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        return str(tmp_path)

    def test_mixed_bundle_detected(self, tmp_path):
        p = self._bundle(
            tmp_path, {"layer_types": ["sliding_attention", "full_attention"]}
        )
        assert _bundle_has_mixed_swa_layout(p)

    def test_dense_bundle_not_detected(self, tmp_path):
        p = self._bundle(tmp_path, {"layer_types": ["full_attention"]})
        assert not _bundle_has_mixed_swa_layout(p)

    def test_missing_path_and_missing_config_are_false(self, tmp_path):
        assert not _bundle_has_mixed_swa_layout(None)
        assert not _bundle_has_mixed_swa_layout(str(tmp_path))

    def test_unreadable_config_does_not_raise(self, tmp_path):
        (tmp_path / "config.json").write_text("{not json")
        assert not _bundle_has_mixed_swa_layout(str(tmp_path))


class TestSharedDetectorIsNotDuplicated:
    """The loader and the CLI must answer this question with one function.

    They previously could not drift because only the loader asked. Now both do,
    and a re-inlined copy in either place is how they start disagreeing.
    """

    def test_loader_no_longer_defines_a_private_copy(self):
        import inspect

        from vmlx_engine.utils import jang_loader

        src = inspect.getsource(jang_loader)
        assert "def _has_mixed_attention_layout" not in src, (
            "jang_loader re-inlined a private mixed-attention detector; "
            "call has_mixed_attention_layout instead"
        )

    def test_cli_delegates_rather_than_reimplementing(self):
        import inspect

        from vmlx_engine import cli

        src = inspect.getsource(cli._bundle_has_mixed_swa_layout)
        assert "has_mixed_attention_layout" in src
        assert "layer_types" not in src, (
            "cli re-implemented the layer_types walk; delegate to the "
            "canonical detector"
        )


@pytest.mark.parametrize(
    "mixed,jang_hi_fi,expected",
    [
        (True, True, "none"),  # mixed-SWA outranks the JANG q8 default
        (True, False, "none"),
        (False, True, "q8"),
        (False, False, "q4"),
    ],
)
def test_stored_quant_default_precedence(mixed, jang_hi_fi, expected):
    """Execute the CLI's own precedence rather than restating it.

    This test used to re-implement the if/elif chain and assert it against its
    own parametrization, so swapping the branches in cli.py left it green. It
    now calls the function the CLI actually uses.
    """
    from vmlx_engine.cli import _stored_kvq_fallback

    assert _stored_kvq_fallback(mixed, jang_hi_fi) == expected


def test_cli_routes_the_default_through_the_precedence_helper():
    """Pin the ordering at its real call site, not only inside the helper."""
    import inspect

    from vmlx_engine import cli

    src = inspect.getsource(cli)
    assert "_fallback_kvq = _stored_kvq_fallback(" in src, (
        "cli no longer routes the stored-codec default through "
        "_stored_kvq_fallback; the precedence is unpinned again"
    )


def test_bench_uses_the_same_precedence_as_serve():
    """A bench that measures a codec serve will never use is a false baseline."""
    import inspect

    from vmlx_engine import cli

    src = inspect.getsource(cli)
    assert 'args.kv_cache_quantization = "q4"' not in src, (
        "bench_command hardcodes q4 again; it must share _stored_kvq_fallback "
        "so it does not benchmark the answer-corrupting stored codec"
    )
