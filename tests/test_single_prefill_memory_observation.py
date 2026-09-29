"""The native generator must feed completed allocations into admission."""
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from vmlx_engine.utils import single_batch_generator as single
from vmlx_engine.utils.prefill_admission import PrefillAdmissionError

GIB = 1024**3


def fixture(monkeypatch, active, peaks, *, family="naive_n05_flash", enabled=True):
    monkeypatch.setattr(single, "_prefill_valve_enabled", lambda: enabled)
    monkeypatch.setattr(single, "_prefill_valve_min_margin_bytes", lambda: 2 * GIB)
    monkeypatch.setattr(single, "get_effective_metal_working_set_bytes", lambda _: (80 * GIB, 100 * GIB))
    active, peaks = iter(active), iter(peaks)
    events = []
    monkeypatch.setattr(mx, "get_active_memory", lambda: next(active) * GIB)
    monkeypatch.setattr(mx, "reset_peak_memory", lambda: events.append("reset"))

    def peak():
        assert events[-1] == "sync"
        events.append("peak")
        return next(peaks) * GIB

    monkeypatch.setattr(mx, "get_peak_memory", peak)
    monkeypatch.setattr(single, "_prefill_keep_alloc_enabled", lambda: True)
    gen = single.SingleBatchGenerator(SimpleNamespace(model_type=family), prefill_step_size=2)
    req = SimpleNamespace(cache=[SimpleNamespace(state=(mx.zeros((1,)),))], context_tokens=[], logits_processors=[])
    calls = []
    gen._model_call = lambda chunk, request: calls.append(list(chunk))
    gen._sync = lambda: events.append("sync")
    return gen, req, calls, events


def test_completed_peak_rejects_next_chunk_before_model_submission(monkeypatch):
    gen, req, calls, events = fixture(monkeypatch, [80, 81], [96])
    with pytest.raises(PrefillAdmissionError):
        gen._prefill([1, 2, 3, 4], req)
    assert calls == [[1, 2]]
    assert req.context_tokens == [1, 2]
    assert events == ["reset", "sync", "peak"]


def test_smaller_later_peak_does_not_forget_largest_transient(monkeypatch):
    gen, req, calls, _ = fixture(monkeypatch, [80, 80, 86], [92, 81])
    with pytest.raises(PrefillAdmissionError):
        gen._prefill([1, 2, 3, 4, 5, 6], req)
    assert calls == [[1, 2], [3, 4]]
    assert req.context_tokens == [1, 2, 3, 4]


@pytest.mark.parametrize("family,enabled", [("other", True), ("naive_n05_flash", False)])
def test_other_owners_and_disabled_admission_keep_existing_behavior(monkeypatch, family, enabled):
    gen, req, calls, events = fixture(monkeypatch, [80, 81], [], family=family, enabled=enabled)
    gen._prefill([1, 2, 3, 4], req)
    assert calls == [[1, 2], [3, 4]]
    assert events == ["sync", "sync"]


def test_unknown_active_reading_does_not_become_a_false_transient(monkeypatch):
    gen, req, calls, events = fixture(monkeypatch, [0, 81], [82])
    gen._prefill([1, 2, 3, 4], req)
    assert calls == [[1, 2], [3, 4]]
    assert events == ["reset", "sync", "reset", "sync", "peak"]
