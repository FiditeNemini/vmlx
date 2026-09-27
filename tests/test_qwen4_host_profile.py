"""Host diagnostic semantics, not model runtime or performance acceptance."""

from types import SimpleNamespace

import pytest

from vmlx_engine.models.qwen4_exp.host_profile import profile_decode_forward


@pytest.fixture(autouse=True)
def isolated_row_selection(monkeypatch):
    monkeypatch.delenv("VMLX_QWEN4_HOST_PROFILE_ROWS", raising=False)


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
