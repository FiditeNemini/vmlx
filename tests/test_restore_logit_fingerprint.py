"""Restore diagnostics must inspect target logits, never native MTP hidden state."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from vmlx_engine.mllm_batch_generator import _diag_logits_fp


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
