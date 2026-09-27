"""Host diagnostic semantics, not model runtime or performance acceptance."""

from types import SimpleNamespace

import pytest

from vmlx_engine.models.qwen4_exp.host_profile import profile_decode_forward


@pytest.fixture(autouse=True)
def isolated_row_selection(monkeypatch):
    monkeypatch.delenv("VMLX_QWEN4_HOST_PROFILE_ROWS", raising=False)
    monkeypatch.delenv("VMLX_QWEN4_HOST_PROFILE_MODE", raising=False)


def test_disabled_is_original_function(monkeypatch):
    monkeypatch.delenv("VMLX_QWEN4_HOST_PROFILE", raising=False)
    def original(self, inputs):
        return inputs
    assert profile_decode_forward(original) is original


def test_bounded_rows_passthrough_and_single_report(monkeypatch, caplog):
    monkeypatch.setenv("VMLX_QWEN4_HOST_PROFILE", "1")
    calls = []
    @profile_decode_forward
    def original(self, inputs, *, value):
        calls.append(inputs.shape)
        return value
    caplog.set_level("INFO")
    result = object()
    for shape in [(1, 8), (2, 1)] + [(1, 1)] * 40:
        assert original(None, SimpleNamespace(shape=shape), value=result) is result
    assert len(calls) == 42
    assert caplog.text.count("QWEN4_HOST_PROFILE calls=32") == 1
    assert "gpu_fences_added=false" in caplog.text


def test_original_error_propagates(monkeypatch):
    monkeypatch.setenv("VMLX_QWEN4_HOST_PROFILE", "1")
    error = ValueError("original failure")
    @profile_decode_forward
    def original(self, inputs):
        raise error
    with pytest.raises(ValueError) as caught:
        original(None, SimpleNamespace(shape=(1, 1)))
    assert caught.value is error


@pytest.mark.parametrize("rows", [2, 3, 4])
def test_verifier_selection_excludes_decode_and_prefill(monkeypatch, caplog, rows):
    monkeypatch.setenv("VMLX_QWEN4_HOST_PROFILE", "1")
    monkeypatch.setenv("VMLX_QWEN4_HOST_PROFILE_ROWS", str(rows))
    calls = []

    @profile_decode_forward
    def original(self, inputs, *, result, return_hidden=False):
        calls.append(inputs.shape)
        return result

    caplog.set_level("INFO")
    result = object()
    for shape in [(1, 1), (1, 2048), (2, rows)] * 40:
        assert original(None, SimpleNamespace(shape=shape), result=result) is result
    assert "QWEN4_HOST_PROFILE" not in caplog.text
    for _ in range(40):
        assert original(None, SimpleNamespace(shape=(1, rows)), result=result) is result
    assert "QWEN4_HOST_PROFILE" not in caplog.text
    for _ in range(40):
        assert original(None, SimpleNamespace(shape=(1, rows)), result=result,
                        return_hidden=True) is result
    assert len(calls) == 200
    assert caplog.text.count("QWEN4_HOST_PROFILE calls=32") == 1
    assert f"rows={rows}" in caplog.text


@pytest.mark.parametrize("rows", ["0", "5", "2048", "all", "", "-1"])
def test_invalid_selection_disables_instrumentation(monkeypatch, rows):
    monkeypatch.setenv("VMLX_QWEN4_HOST_PROFILE", "1")
    monkeypatch.setenv("VMLX_QWEN4_HOST_PROFILE_ROWS", rows)

    def original(self, inputs):
        return inputs

    assert profile_decode_forward(original) is original


def test_worker_calls_excluded_and_thread_hook_restored(monkeypatch, caplog):
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setenv("VMLX_QWEN4_HOST_PROFILE", "1")
    monkeypatch.setenv("VMLX_QWEN4_HOST_PROFILE_ROWS", "3")

    def worker_only_marker():
        time.sleep(0.001)
        return 7

    @profile_decode_forward
    def original(self, inputs, *, return_hidden):
        return pool.submit(worker_only_marker).result()

    caplog.set_level("INFO")
    with ThreadPoolExecutor(max_workers=2) as pool:
        for _ in range(32):
            assert original(None, SimpleNamespace(shape=(1, 3)), return_hidden=True) == 7
            assert sys.getprofile() is None
    assert "worker_only_marker" not in caplog.text
    assert "backend=thread_profile" in caplog.text


@pytest.mark.parametrize("mode", ["python", "phases"])
def test_existing_thread_profiler_is_preserved(monkeypatch, mode):
    import sys

    monkeypatch.setenv("VMLX_QWEN4_HOST_PROFILE", "1")
    monkeypatch.setenv("VMLX_QWEN4_HOST_PROFILE_MODE", mode)
    @profile_decode_forward
    def original(self, inputs):
        return inputs

    def observer(frame, event, arg):
        pass

    previous = sys.getprofile()
    try:
        sys.setprofile(observer)
        value = SimpleNamespace(shape=(1, 1))
        assert original(None, value) is value
        assert sys.getprofile() is observer
    finally:
        sys.setprofile(previous)


def test_phase_budget_selection_and_thread_exclusion(monkeypatch, caplog):
    import json
    from concurrent.futures import ThreadPoolExecutor
    from vmlx_engine.models.qwen4_exp.host_profile import profile_submission

    monkeypatch.setenv("VMLX_QWEN4_HOST_PROFILE", "1")
    monkeypatch.setenv("VMLX_QWEN4_HOST_PROFILE_ROWS", "3")
    monkeypatch.setenv("VMLX_QWEN4_HOST_PROFILE_MODE", "phases")
    result = object()

    def submit(value):
        assert value is result
        return value

    @profile_decode_forward
    def original(self, inputs, return_hidden=False):
        assert pool.submit(profile_submission, submit, result).result() is result
        return profile_submission(submit, result)

    caplog.set_level("INFO")
    with ThreadPoolExecutor(max_workers=1) as pool:
        for _ in range(40):
            assert original(None, SimpleNamespace(shape=(1, 1))) is result
            assert original(None, SimpleNamespace(shape=(1, 3))) is result
        assert "QWEN4_HOST_PHASES" not in caplog.text
        for _ in range(40):
            assert original(None, SimpleNamespace(shape=(1, 3)), return_hidden=True) is result
    reports = [r.message for r in caplog.records if "QWEN4_HOST_PHASES" in r.message]
    assert len(reports) == 1
    assert "calls=32 rows=3" in reports[0]
    report = json.loads(reports[0].split("data=", 1)[1])
    assert report["submissions"] == 32
    assert report["failed_calls"] == 0
    for clock in ("wall", "cpu"):
        assert report[f"forward_{clock}_s"] >= report[f"submission_{clock}_s"] >= 0
        assert report[f"other_host_{clock}_s"] >= 0
    assert "backend=thread_profile" not in caplog.text


def test_phase_exception_restores_context(monkeypatch):
    from vmlx_engine.models.qwen4_exp.host_profile import profile_submission, _phase_local

    monkeypatch.setenv("VMLX_QWEN4_HOST_PROFILE", "1")
    monkeypatch.setenv("VMLX_QWEN4_HOST_PROFILE_MODE", "phases")
    error = RuntimeError("submission failed")

    def fail(value):
        raise error

    @profile_decode_forward
    def original(self, inputs):
        return profile_submission(fail, inputs)

    with pytest.raises(RuntimeError) as caught:
        original(None, SimpleNamespace(shape=(1, 1)))
    assert caught.value is error
    assert getattr(_phase_local, "active", None) is None
    value = object()
    assert profile_submission(lambda x: x, value) is value


def test_invalid_mode_preserves_original(monkeypatch):
    monkeypatch.setenv("VMLX_QWEN4_HOST_PROFILE", "1")
    monkeypatch.setenv("VMLX_QWEN4_HOST_PROFILE_MODE", "unknown")
    def original(self, inputs):
        return inputs
    assert profile_decode_forward(original) is original
