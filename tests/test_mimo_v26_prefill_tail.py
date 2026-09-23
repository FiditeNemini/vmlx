"""CPU control-flow proof for bounded cold/restored MiMo prefill; no model loads."""
from types import SimpleNamespace

import numpy as np
import pytest

from vmlx_engine import mllm_batch_generator as module


@pytest.mark.parametrize("cached,original_len,suffix", [(0, 13360, 3), (4286, 13360, 3), (13359, 13360, 3), (475, 827, 3), (0, 827, 0)])
def test_cold_and_restored_tails_preserve_tokens_and_exact_checkpoint(monkeypatch, cached, original_len, suffix):
    generator = module.MLLMBatchGenerator.__new__(module.MLLMBatchGenerator)
    request = SimpleNamespace(_original_token_ids=list(range(original_len)), _cached_tokens=cached,
                              request_id="bounded-tail", _prefill_tokens_done=0)
    tokens = np.arange(cached, original_len + suffix)[None, :]
    state = SimpleNamespace(tokens=list(range(cached)))
    calls, captures, materialized = [], [], []

    def forward(ids, cache, position_ids):
        # Model surrogate rejects the exact resource mistake: a tail larger
        # than the configured chunk, without needing actual GPU allocation.
        assert ids.shape[1] <= 512
        assert np.array_equal(ids, position_ids)
        assert ids[0, 0] == len(state.tokens)
        state.tokens.extend(ids[0].tolist())
        calls.append(ids[0].tolist())
        return {"last_token": state.tokens[-1]}

    generator.language_model = forward
    generator._maybe_capture_mixed_swa_boundary = lambda req, cache: captures.append(list(state.tokens))
    monkeypatch.setattr(module, "_materialize_prefill_cache_state", lambda cache: materialized.append(len(state.tokens)))
    monkeypatch.setattr(module.mx, "clear_cache", lambda: None)
    monkeypatch.setattr(module, "_raise_if_prefill_cancelled", lambda req: None)
    output = generator._prefill_mimo_v26_text(request, tokens, [state],
        {"cache": [state], "position_ids": tokens.copy()}, step=512)
    assert state.tokens == list(range(original_len + suffix))
    assert output == {"last_token": original_len + suffix - 1}
    assert [t for call in calls for t in call] == list(range(cached, original_len + suffix))
    if cached < original_len - 1:
        assert captures == [list(range(original_len - 1))]
        assert materialized[-1] == original_len - 1
    else:
        assert not captures and not materialized  # Existing durable checkpoint already covers N-1.


def test_cancellation_stops_between_bounded_tail_spans(monkeypatch):
    generator = module.MLLMBatchGenerator.__new__(module.MLLMBatchGenerator)
    request = SimpleNamespace(_original_token_ids=list(range(5000)), _cached_tokens=1000)
    calls = []
    generator.language_model = lambda ids, **kwargs: calls.append(ids.shape[1])
    generator._maybe_capture_mixed_swa_boundary = lambda *args: pytest.fail("captured after cancellation")
    monkeypatch.setattr(module, "_materialize_prefill_cache_state", lambda cache: None)
    monkeypatch.setattr(module.mx, "clear_cache", lambda: None)
    def cancelled(req):
        raise RuntimeError("cancelled")
    monkeypatch.setattr(module, "_raise_if_prefill_cancelled", cancelled)
    with pytest.raises(RuntimeError, match="cancelled"):
        generator._prefill_mimo_v26_text(request, np.arange(1000, 5003)[None, :], [], {"cache": []}, step=512)
    assert calls == [512]
