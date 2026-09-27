"""Restore diagnostics must inspect target logits, never native MTP hidden state."""

from types import SimpleNamespace
import json

import mlx.core as mx
import pytest

from vmlx_engine.mllm_batch_generator import _diag_logits_fp
from vmlx_engine.mllm_batch_generator import (
    MLLMBatchGenerator, _diag_position_window, _diag_position_logits,
)


@pytest.mark.parametrize("contract", ["array", "object", "native_mtp"])
def test_target_logit_fingerprint_contracts(contract):
    logits = mx.array([[[9.0, 2.0, 1.0], [1.0, 2.0, 5.0]]], dtype=mx.float16)
    hidden = mx.array([[[999.0, -999.0]]])
    output = {
        "array": logits,
        "object": SimpleNamespace(logits=logits),
        "native_mtp": (logits, hidden),
    }[contract]
    result = _diag_logits_fp(output)
    assert result.startswith("argmax=2 margin=3.000000e+00 ")
    assert "fp-error" not in result
    assert result == _diag_logits_fp(logits)


@pytest.mark.parametrize("raw,expected", [
    (None, None), ("118:124", (118, 124)), ("0:31", (0, 31)),
    ("0:32", (-1, -1)), ("-1:2", (-1, -1)), ("4:3", (-1, -1)),
    ("1", (-1, -1)), ("a:b", (-1, -1)),
])
def test_position_window_is_bounded(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("VMLX_DIAG_RESTORE_POSITION", raising=False)
    else:
        monkeypatch.setenv("VMLX_DIAG_RESTORE_POSITION", raw)
    assert _diag_position_window() == expected


def test_native_verifier_target_positions_and_margins():
    logits = mx.array([[[8., 3., 1.], [2., 9., 4.], [3., 4., 7.]]])
    rows = json.loads(_diag_position_logits((logits, None), 122, (121, 122)))
    assert [r["target_position"] for r in rows] == [121, 122]
    assert [r["row"] for r in rows] == [1, 2]
    assert [r["top_ids"] for r in rows] == [[1, 2], [2, 1]]
    assert [r["margin"] for r in rows] == [5., 3.]


def test_position_window_preserves_four_call_budget(monkeypatch, caplog):
    import vmlx_engine.mllm_batch_generator as generator

    offsets = []
    def fingerprint(cache, positions, upto):
        offsets.append(upto)
        return "native-state"
    monkeypatch.setattr(generator, "_diag_cache_fingerprint", fingerprint)
    monkeypatch.setenv("VMLX_DIAG_RESTORE_POSITION", "118:124")
    host = SimpleNamespace(_hybrid_kv_positions=[1])
    request = SimpleNamespace(request_id="position-test")
    cache = [SimpleNamespace(), SimpleNamespace(offset=40)]
    output = (mx.array([[[8., 3., 1.], [2., 9., 4.], [3., 4., 7.]]]), None)
    caplog.set_level("INFO")
    for _ in range(10):
        MLLMBatchGenerator._diag_decode_fingerprint(host, None, request, cache, output, "verify")
    assert not hasattr(request, "_diag_decode_steps")
    assert "restore fingerprint DECODE" not in caplog.text
    cache[1].offset = 122
    for _ in range(10):
        MLLMBatchGenerator._diag_decode_fingerprint(host, None, request, cache, output, "verify")
    assert request._diag_decode_steps == 4
    assert offsets == [122] * 4
    assert caplog.text.count("restore fingerprint DECODE step") == 4
    assert '"target_position":122' in caplog.text
    assert "logits-fp-error" not in caplog.text


def test_position_window_does_not_guess_unknown_offset(monkeypatch, caplog):
    monkeypatch.setenv("VMLX_DIAG_RESTORE_POSITION", "0:10")
    host = SimpleNamespace(_hybrid_kv_positions=[0])
    request = SimpleNamespace(request_id="unknown-position")
    output = mx.array([[[1., 2.]]])
    MLLMBatchGenerator._diag_decode_fingerprint(
        host, None, request, [SimpleNamespace()], output, "verify",
    )
    assert not hasattr(request, "_diag_decode_steps")


def test_position_window_excludes_broad_layer_tracing(monkeypatch):
    from vmlx_engine.models.qwen4_exp import language

    monkeypatch.setenv("VMLX_DIAG_RESTORE_FINGERPRINT", "1")
    monkeypatch.setattr(language, "_LAYER_FP_STEPS", {"n": 0})
    monkeypatch.setenv("VMLX_DIAG_RESTORE_POSITION", "122:122")
    inputs = SimpleNamespace(shape=(1, 1))
    assert not language._layer_fingerprint_enabled(inputs)
    assert language._LAYER_FP_STEPS["n"] == 0
    monkeypatch.delenv("VMLX_DIAG_RESTORE_POSITION")
    assert language._layer_fingerprint_enabled(inputs)
    assert language._LAYER_FP_STEPS["n"] == 1
