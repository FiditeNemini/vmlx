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


def test_completed_peak_flags_next_chunk_and_still_runs_it(monkeypatch, caplog):
    """The completed chunk's peak (16GB transient) projects the next chunk past
    the limit. Since 2026-10-08 (ISSUES I-36: estimates ADVISE, never REFUSE)
    the projection is logged and the chunk RUNS."""
    import logging

    caplog.set_level(logging.WARNING)
    gen, req, calls, events = fixture(monkeypatch, [80, 81], [96, 97])
    gen._prefill([1, 2, 3, 4], req)
    assert calls == [[1, 2], [3, 4]]
    assert req.context_tokens == [1, 2, 3, 4]
    assert events == ["reset", "sync", "peak", "reset", "sync", "peak"]
    assert "Prefill admission ADVISORY" in caplog.text


def test_smaller_later_peak_does_not_forget_largest_transient(monkeypatch, caplog):
    """The third chunk is flagged only if the 12GB transient of chunk 1 is still
    the projection input after chunk 2's 1GB transient (a forgetting valve
    would project 86 + 2 = 88GB and stay quiet)."""
    import logging

    caplog.set_level(logging.WARNING)
    gen, req, calls, _ = fixture(monkeypatch, [80, 80, 86], [92, 81, 87])
    gen._prefill([1, 2, 3, 4, 5, 6], req)
    assert calls == [[1, 2], [3, 4], [5, 6]]
    assert req.context_tokens == [1, 2, 3, 4, 5, 6]
    assert caplog.text.count("Prefill admission ADVISORY") == 1


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
